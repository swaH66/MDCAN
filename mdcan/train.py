"""Reproducible training entry point for MDCAN.

The entry point trains MDCAN only and stores model parameters as a state dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models import create_mdcan
from utils import load_checkpoint


LOGGER = logging.getLogger("mdcan.train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(inputs, targets, reduction="none")
        probability = torch.exp(-cross_entropy)
        loss = (1 - probability) ** self.gamma * cross_entropy
        if self.alpha is not None:
            loss = self.alpha[targets] * loss if isinstance(self.alpha, torch.Tensor) else self.alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SkinLesionFolder(Dataset):
    """Class-folder dataset with deterministic class and filename ordering."""

    def __init__(self, data_path: str | Path, transform=None) -> None:
        self.data_path = Path(data_path)
        self.transform = transform
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.data_path}")

        self.classes = sorted(path.name for path in self.data_path.iterdir() if path.is_dir())
        if not self.classes:
            raise ValueError(f"No class directories found in {self.data_path}")
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.data_path / class_name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No supported images found in {self.data_path}")
        self.labels = [label for _, label in self.samples]
        self.cls_num_list = np.bincount(self.labels, minlength=len(self.classes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    return train_transform, evaluation_transform


def train_one_epoch(model, criterion, optimizer, loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        count += batch_size
        progress.set_postfix(loss=f"{total_loss / count:.4f}", acc=f"{correct / count:.4f}")
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate(model, loader, device, description="Evaluating"):
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in tqdm(loader, desc=description, leave=False):
        outputs = model(images.to(device, non_blocking=True))
        labels_all.extend(labels.numpy().tolist())
        predictions_all.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
    labels_array = np.asarray(labels_all)
    predictions_array = np.asarray(predictions_all)
    accuracy = accuracy_score(labels_array, predictions_array)
    class_labels = list(range(model_num_classes(model)))
    per_class_f1 = f1_score(
        labels_array,
        predictions_array,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    weighted_f1 = f1_score(labels_array, predictions_array, average="weighted", zero_division=0)
    per_class_sensitivity = []
    for class_index in range(model_num_classes(model)):
        mask = labels_array == class_index
        per_class_sensitivity.append(float((predictions_array[mask] == class_index).mean()) if mask.any() else 0.0)
    return accuracy, per_class_f1, per_class_sensitivity, weighted_f1


def model_num_classes(model: nn.Module) -> int:
    classifier = model.classifier
    if isinstance(classifier, nn.Sequential):
        return classifier[-1].out_features
    return classifier.out_features


def warmup_cosine_multiplier(epoch, warmup_epochs, total_epochs, min_ratio):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def create_optimizer(model, args):
    if not args.layered_lr:
        return optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    backbone, classifier = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (classifier if "classifier" in name else backbone).append(parameter)
    parameter_groups = []
    if backbone:
        parameter_groups.append({"params": backbone, "lr": args.lr * 0.1, "group_name": "backbone"})
    if classifier:
        parameter_groups.append({"params": classifier, "lr": args.lr, "group_name": "classifier"})
    return optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def create_scheduler(optimizer, args):
    if args.warmup:
        return lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: warmup_cosine_multiplier(
                epoch, args.warmup_epochs, args.epochs, args.min_lr / args.lr
            ),
        )
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1, best_acc, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "best_acc": best_acc,
            "args": vars(args),
        },
        path,
    )


def main(args) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device.")
    seed_everything(args.seed)

    train_transform, evaluation_transform = build_transforms()
    train_dataset = SkinLesionFolder(Path(args.data_path) / "train", train_transform)
    val_dataset = SkinLesionFolder(Path(args.data_path) / "val", evaluation_transform)
    test_dataset = SkinLesionFolder(Path(args.data_path) / "test", evaluation_transform)
    if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
        raise ValueError("train/val/test class directories must be identical.")
    if len(train_dataset.classes) != args.num_classes:
        raise ValueError(
            f"--num_classes={args.num_classes}, but dataset contains "
            f"{len(train_dataset.classes)} classes: {train_dataset.classes}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = create_mdcan(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    ).to(device)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "classifier" in name

    weights = None
    if args.weighted_loss:
        inverse_frequency = 1.0 / (train_dataset.cls_num_list.astype(float) + 1e-6)
        normalized = inverse_frequency / inverse_frequency.sum() * len(inverse_frequency)
        weights = torch.tensor(normalized, dtype=torch.float32, device=device)
    cross_entropy = nn.CrossEntropyLoss(weight=weights)
    focal_loss = FocalLoss(alpha=weights, gamma=args.gamma)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    start_epoch, best_f1, best_acc = 0, 0.0, 0.0
    if args.resume:
        metadata = load_checkpoint(model, args.resume, device=device, strict=True)
        if args.resume_optimizer and "optimizer" in metadata:
            optimizer.load_state_dict(metadata["optimizer"])
            if "scheduler" in metadata:
                scheduler.load_state_dict(metadata["scheduler"])
            start_epoch = int(metadata.get("epoch", -1)) + 1
            best_f1 = float(metadata.get("best_f1", 0.0))
            best_acc = float(metadata.get("best_acc", 0.0))

    (save_dir / "class_to_idx.json").write_text(
        json.dumps(train_dataset.class_to_idx, indent=2), encoding="utf-8"
    )
    (save_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = create_optimizer(model, args)
            scheduler = create_scheduler(optimizer, args)

        criterion = cross_entropy if epoch < args.switch_loss_epoch else focal_loss
        train_loss, train_accuracy = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, scaler
        )
        val_accuracy, per_class_f1, _, val_weighted_f1 = evaluate(
            model, val_loader, device, f"Validation {epoch + 1}"
        )
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("F1_weighted/val", val_weighted_f1, epoch)
        for class_index, value in enumerate(per_class_f1):
            writer.add_scalar(f"F1/class_{class_index}", value, epoch)

        improved_f1 = val_weighted_f1 > best_f1
        improved_acc = val_accuracy > best_acc
        if improved_f1:
            best_f1 = val_weighted_f1
            patience_counter = 0
            save_checkpoint(
                save_dir / "best_model_f1.pth", model, optimizer, scheduler,
                epoch, best_f1, max(best_acc, val_accuracy), args,
            )
        else:
            patience_counter += 1
        if improved_acc:
            best_acc = val_accuracy
            save_checkpoint(
                save_dir / "best_model_acc.pth", model, optimizer, scheduler,
                epoch, max(best_f1, val_weighted_f1), best_acc, args,
            )
        save_checkpoint(
            save_dir / "last.pth", model, optimizer, scheduler,
            epoch, best_f1, best_acc, args,
        )
        LOGGER.info(
            "epoch=%d train_loss=%.6f train_acc=%.6f val_acc=%.6f val_weighted_f1=%.6f",
            epoch + 1, train_loss, train_accuracy, val_accuracy, val_weighted_f1,
        )
        if args.patience > 0 and patience_counter >= args.patience:
            LOGGER.info("Early stopping after %d epochs without F1 improvement", args.patience)
            break

    best_path = save_dir / "best_model_f1.pth"
    if not best_path.exists():
        best_path = save_dir / "last.pth"
    load_checkpoint(model, best_path, device=device, strict=True)
    test_accuracy, test_f1, test_sensitivity, test_weighted_f1 = evaluate(
        model, test_loader, device, "Testing"
    )
    results = {
        "checkpoint": str(best_path),
        "accuracy": test_accuracy,
        "weighted_f1": test_weighted_f1,
        "per_class_f1": test_f1.tolist(),
        "per_class_sensitivity": test_sensitivity,
        "classes": train_dataset.classes,
    }
    (save_dir / "test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    LOGGER.info("test_metrics=%s", json.dumps(results))
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="MDCAN skin-lesion classification")
    parser.add_argument("--data_path", type=str, default="D:\Masterjob\论文\MSCAN\mdcan代码\data\ham10000")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument(
        "--model",
        choices=("mdcan",),
        default="mdcan",
        help="Model name; only MDCAN is included in this repository",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--switch_loss_epoch", type=int, default=100)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_epoch", type=int, default=10)
    parser.add_argument("--layered_lr", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="", help="MDCAN checkpoint path")
    parser.add_argument(
        "--resume_optimizer", action="store_true",
        help="Also restore optimizer/scheduler/epoch when present",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = get_args()
    Path(arguments.save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(arguments.save_dir) / "training.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    main(arguments)
