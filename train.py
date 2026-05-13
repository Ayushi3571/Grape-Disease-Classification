"""
Leaf Disease Transformer Training and Edge Export Pipeline.

Usage examples:
    python train.py list-models
    python train.py train --data_dir ./coloured --output_dir ./output_6class
    python train.py train --data_dir ./coloured --output_dir ./output_edge --model_preset edge
    python train.py predict --image ./leaf.jpg --checkpoint ./output_edge/checkpoints/best.pt
    python train.py export --checkpoint ./output_edge/checkpoints/best.pt --output ./output_edge/edge/model_edge.ts
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


MODEL_PRESETS = {
    "base": {
        "model_name": "google/vit-base-patch16-224-in21k",
        "image_size": 224,
        "description": "Best accuracy baseline using a full ViT backbone.",
    },
    "edge": {
        "model_name": "facebook/deit-tiny-patch16-224",
        "image_size": 224,
        "description": "Small transformer that is much easier to deploy on CPU-class edge devices.",
    },
    "edge_plus": {
        "model_name": "facebook/deit-small-patch16-224",
        "image_size": 224,
        "description": "Middle ground between the baseline and the tiny edge preset.",
    },
}

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL_PRESET = "base"

LESION_HSV_LOWER = np.array([5, 60, 60])
LESION_HSV_UPPER = np.array([30, 255, 200])
DARK_MARK_LOWER = np.array([0, 0, 0])
DARK_MARK_UPPER = np.array([180, 80, 60])
MIN_LESION_AREA = 20


def _leaf_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def _edges(bgr, mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)
    return cv2.bitwise_and(edges, edges, mask=mask)


def _lesions(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lesions = cv2.inRange(hsv, LESION_HSV_LOWER, LESION_HSV_UPPER)
    lesions = cv2.bitwise_and(lesions, lesions, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesions = cv2.morphologyEx(lesions, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(lesions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(lesions)
    for contour in contours:
        if cv2.contourArea(contour) >= MIN_LESION_AREA:
            cv2.drawContours(out, [contour], -1, 255, -1)
    return out


def _dark_marks(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    marks = cv2.inRange(hsv, DARK_MARK_LOWER, DARK_MARK_UPPER)
    marks = cv2.bitwise_and(marks, marks, mask=mask)
    contours, _ = cv2.findContours(marks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(marks)
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_LESION_AREA <= area <= 500:
            cv2.drawContours(out, [contour], -1, 255, -1)
    return out


def preprocess_to_rgb(image_path: str, size: int = 224) -> np.ndarray:
    """
    Convert a raw RGB leaf image into a three-channel disease-focused tensor:
    grayscale texture, lesion mask, and leaf-edge structure.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)

    leaf_mask = _leaf_mask(bgr)
    edge_map = _edges(bgr, leaf_mask)
    lesion_mask = _lesions(bgr, leaf_mask)
    dark_marks = _dark_marks(bgr, leaf_mask)
    disease = cv2.bitwise_or(lesion_mask, dark_marks)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    stacked = np.stack([gray, disease, edge_map], axis=-1)
    stacked = cv2.resize(stacked, (size, size))
    return stacked.astype(np.uint8)


def resolve_model_config(model_preset: str, model_name: str | None, image_size: int | None) -> dict:
    preset_name = model_preset or DEFAULT_MODEL_PRESET
    if preset_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset '{preset_name}'. Use one of: {', '.join(MODEL_PRESETS)}")

    preset = MODEL_PRESETS[preset_name]
    return {
        "model_preset": preset_name,
        "model_name": model_name or preset["model_name"],
        "image_size": image_size or preset["image_size"],
    }


def load_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_normalization_stats(model_name: str) -> tuple[list[float], list[float]]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    image_mean = list(processor.image_mean)
    image_std = list(processor.image_std)
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError(f"Expected 3-channel normalization for '{model_name}', got {image_mean} / {image_std}")
    return image_mean, image_std


def normalize_image(image: np.ndarray, image_mean: list[float], image_std: list[float]) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def upgrade_state_dict_keys(model_state: dict) -> dict:
    upgraded = {}
    for key, value in model_state.items():
        if key.startswith("vit."):
            upgraded[key.replace("vit.", "backbone.", 1)] = value
        else:
            upgraded[key] = value
    return upgraded


def build_label_map(data_dir: str) -> tuple[dict, dict]:
    classes = sorted(
        entry.name for entry in Path(data_dir).iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


class LeafDiseaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        label2id: dict,
        image_mean: list[float],
        image_std: list[float],
        augment: bool = False,
        image_size: int = 224,
    ):
        self.samples = []
        self.label2id = label2id
        self.image_mean = image_mean
        self.image_std = image_std
        self.augment = augment
        self.image_size = image_size

        root_path = Path(root)
        for class_dir in sorted(root_path.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if label not in label2id:
                log.warning("Skipping folder not found in label map: %s", label)
                continue
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(image_path), label2id[label]))

        self._aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor(),
        ]) if augment else None

        log.info(
            "Dataset loaded: %s images, %s classes from %s",
            len(self.samples),
            len(label2id),
            root_path,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = preprocess_to_rgb(image_path, self.image_size)

        if self._aug is not None:
            augmented = self._aug(image)
            image = (augmented.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        pixel_values = normalize_image(image, self.image_mean, self.image_std)
        return pixel_values, torch.tensor(label, dtype=torch.long)


class LeafDiseaseTransformer(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Backbone '{model_name}' does not expose hidden_size; use a ViT/DeiT-style encoder preset."
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(features))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss


def evaluate(model, loader, device) -> tuple[float, float, list, list]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits, loss = model(pixel_values, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            count += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / count, correct / count, all_preds, all_labels


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_history(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    log.info("Training curves saved to %s", output_path)


def split_dataset(dataset, val_split: float) -> tuple[Subset, Subset]:
    num_val = max(1, int(len(dataset) * val_split))
    num_train = len(dataset) - num_val
    if num_train <= 0:
        raise ValueError("Not enough samples to create a training split. Add more data or reduce --val_split.")
    return random_split(
        dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )


def create_model_from_checkpoint(checkpoint: dict, num_labels: int) -> tuple[LeafDiseaseTransformer, dict]:
    model_preset = checkpoint.get("model_preset", DEFAULT_MODEL_PRESET)
    model_name = checkpoint.get("model_name")
    image_size = checkpoint.get("image_size")
    config = resolve_model_config(model_preset, model_name, image_size)
    model = LeafDiseaseTransformer(config["model_name"], num_labels=num_labels)
    return model, config


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    config = resolve_model_config(args.model_preset, args.model_name, args.image_size)
    image_mean, image_std = load_normalization_stats(config["model_name"])

    log.info("Device: %s", device)
    log.info("Training preset: %s (%s)", config["model_preset"], config["model_name"])

    label2id, id2label = build_label_map(args.data_dir)
    num_labels = len(label2id)
    log.info("Classes (%s): %s", num_labels, label2id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as handle:
        json.dump({"label2id": label2id, "id2label": id2label}, handle, indent=2)

    base_dataset = LeafDiseaseDataset(
        args.data_dir,
        label2id,
        image_mean,
        image_std,
        augment=False,
        image_size=config["image_size"],
    )
    train_subset, val_subset = split_dataset(base_dataset, args.val_split)

    train_dataset = LeafDiseaseDataset(
        args.data_dir,
        label2id,
        image_mean,
        image_std,
        augment=True,
        image_size=config["image_size"],
    )
    train_subset.dataset = train_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = LeafDiseaseTransformer(config["model_name"], num_labels=num_labels).to(device)
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model_state = upgrade_state_dict_keys(checkpoint["model_state"])
        old_num_labels = model_state["classifier.weight"].shape[0]

        if old_num_labels == num_labels:
            model.load_state_dict(model_state, strict=True)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            history = checkpoint.get("history", history)
            log.info("Resumed full checkpoint from %s at epoch %s", args.resume, start_epoch)
        elif old_num_labels < num_labels:
            model.load_state_dict(model_state, strict=False)
            old_id2label = {int(k): v for k, v in checkpoint.get("id2label", {}).items()}
            with torch.no_grad():
                for old_idx in range(old_num_labels):
                    class_name = old_id2label.get(old_idx)
                    if class_name in label2id:
                        new_idx = label2id[class_name]
                        model.classifier.weight[new_idx].copy_(model_state["classifier.weight"][old_idx])
                        model.classifier.bias[new_idx].copy_(model_state["classifier.bias"][old_idx])
            log.info("Expanded classifier from %s to %s classes", old_num_labels, num_labels)
        else:
            backbone_only = {
                key: value for key, value in model_state.items()
                if not key.startswith("classifier.")
            }
            model.load_state_dict(backbone_only, strict=False)
            log.info("Loaded only the backbone from %s because the class count shrank", args.resume)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=0.01)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for pixel_values, labels in progress:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, loss = model(pixel_values, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(-1) == labels).sum().item()
            epoch_count += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / epoch_count
        train_acc = epoch_correct / epoch_count
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        log.info(
            "Epoch %3s/%s  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "label2id": label2id,
            "id2label": id2label,
            "history": history,
            "model_preset": config["model_preset"],
            "model_name": config["model_name"],
            "image_size": config["image_size"],
            "image_mean": image_mean,
            "image_std": image_std,
        }
        save_checkpoint(checkpoint_state, output_dir / "checkpoints" / "last.pt")
        if is_best:
            save_checkpoint(checkpoint_state, output_dir / "checkpoints" / "best.pt")
            log.info("New best val_acc=%.4f. Saved checkpoint.", best_val_acc)

    log.info("\nFinal Validation Report")
    print(classification_report(
        val_labels,
        val_preds,
        target_names=[id2label[idx] for idx in sorted(id2label)],
    ))
    plot_history(history, args.output_dir)
    log.info("Done. Best val accuracy: %.4f", best_val_acc)


def predict(image_path: str, checkpoint_path: str, device_str: str = "cpu", top_k: int = 3):
    device = torch.device(device_str)
    checkpoint = load_checkpoint(checkpoint_path, device)
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
    num_labels = len(id2label)

    model, config = create_model_from_checkpoint(checkpoint, num_labels)
    model.load_state_dict(upgrade_state_dict_keys(checkpoint["model_state"]), strict=True)
    model = model.to(device)
    model.eval()

    image_mean = checkpoint.get("image_mean")
    image_std = checkpoint.get("image_std")
    if image_mean is None or image_std is None:
        image_mean, image_std = load_normalization_stats(config["model_name"])

    image = preprocess_to_rgb(image_path, config["image_size"])
    pixel_values = normalize_image(image, image_mean, image_std).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(pixel_values)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    top_k = max(1, min(top_k, num_labels))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    pred_index = int(top_indices[0].item())
    print(f"\nPrediction: {id2label[pred_index]}")
    for score, index in zip(top_probs.tolist(), top_indices.tolist()):
        print(f"  {id2label[int(index)]:35s}: {score:.4f}")
    return id2label[pred_index], probs.tolist()


def export_for_edge(args):
    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
    num_labels = len(id2label)

    model, config = create_model_from_checkpoint(checkpoint, num_labels)
    model.load_state_dict(upgrade_state_dict_keys(checkpoint["model_state"]), strict=True)
    model.eval().cpu()

    image_mean = checkpoint.get("image_mean")
    image_std = checkpoint.get("image_std")
    if image_mean is None or image_std is None:
        image_mean, image_std = load_normalization_stats(config["model_name"])

    export_path = Path(args.output)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "torchscript":
        export_model = model
        if args.quantize:
            export_model = torch.quantization.quantize_dynamic(export_model, {nn.Linear}, dtype=torch.qint8)
            log.info("Applied dynamic INT8 quantization to linear layers for CPU inference.")

        class EdgeWrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, pixel_values):
                logits, _ = self.wrapped(pixel_values)
                return logits

        wrapped_model = EdgeWrapper(export_model).eval()
        example = torch.randn(1, 3, config["image_size"], config["image_size"], dtype=torch.float32)
        traced = torch.jit.trace(wrapped_model, example)
        optimized = torch.jit.optimize_for_inference(traced)
        optimized.save(str(export_path))
        log.info("Saved TorchScript edge artifact to %s", export_path)
    else:
        if args.quantize:
            raise ValueError("INT8 export is only supported for --format torchscript in this project.")

        class OnnxWrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, pixel_values):
                logits, _ = self.wrapped(pixel_values)
                return logits

        wrapped_model = OnnxWrapper(model).eval()
        example = torch.randn(1, 3, config["image_size"], config["image_size"], dtype=torch.float32)
        torch.onnx.export(
            wrapped_model,
            example,
            str(export_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        log.info("Saved ONNX edge artifact to %s", export_path)

    metadata = {
        "artifact_path": str(export_path),
        "format": args.format,
        "model_preset": config["model_preset"],
        "model_name": config["model_name"],
        "image_size": config["image_size"],
        "image_mean": image_mean,
        "image_std": image_std,
        "id2label": id2label,
        "quantized": bool(args.quantize and args.format == "torchscript"),
        "preprocessing": {
            "channels": [
                "grayscale texture",
                "lesion and dark-mark mask",
                "edge and vein structure",
            ],
            "normalization": "pixel_values = (image / 255.0 - mean) / std",
        },
    }
    metadata_path = export_path.with_suffix(export_path.suffix + ".json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    log.info("Saved edge metadata to %s", metadata_path)


def list_models():
    print("Available presets:\n")
    for preset_name, config in MODEL_PRESETS.items():
        print(f"- {preset_name:10s} {config['model_name']}")
        print(f"  {config['description']}")


def build_parser():
    parser = argparse.ArgumentParser(description="Leaf Disease Transformer Trainer and Edge Exporter")
    subparsers = parser.add_subparsers(dest="cmd")

    train_parser = subparsers.add_parser("train", help="Train a classifier")
    train_parser.add_argument("--data_dir", required=True, help="Root folder with class subfolders")
    train_parser.add_argument("--output_dir", default="./output", help="Directory for checkpoints and logs")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--val_split", type=float, default=0.2)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    train_parser.add_argument("--model_preset", choices=sorted(MODEL_PRESETS), default=DEFAULT_MODEL_PRESET)
    train_parser.add_argument("--model_name", default=None, help="Optional Hugging Face model override")
    train_parser.add_argument("--image_size", type=int, default=None, help="Optional preprocessing size override")
    train_parser.add_argument("--cpu_only", action="store_true", help="Force CPU training")

    predict_parser = subparsers.add_parser("predict", help="Run inference on a single image")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--top_k", type=int, default=3)

    export_parser = subparsers.add_parser("export", help="Export a checkpoint for edge inference")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True, help="Artifact path, for example ./edge/model_edge.ts")
    export_parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    export_parser.add_argument("--quantize", action="store_true", help="Apply dynamic INT8 quantization to TorchScript export")
    export_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")

    subparsers.add_parser("list-models", help="Show the built-in model presets")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args.image, args.checkpoint, args.device, args.top_k)
    elif args.cmd == "export":
        export_for_edge(args)
    elif args.cmd == "list-models":
        list_models()
    else:
        parser.print_help()
