"""
Leaf Disease ViT Training Pipeline
------------------------------------
Directory structure expected:
    data/
      healthy/          ← label derived from folder name
        img1.jpg
        img2.jpg
      powdery_mildew/
        img1.jpg
      black_rot/
        img1.jpg

Usage:
    python train.py train --data_dir ./data --epochs 20 --batch_size 16
    python train.py train --data_dir ./coloured --epochs 20 --resume output/checkpoints/best.pt  # add new classes
"""

import os
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Preprocessing (from leaf_preprocess.py)
# ════════════════════════════════════════════════════════════════════════════

LESION_HSV_LOWER  = np.array([5,  60,  60])
LESION_HSV_UPPER  = np.array([30, 255, 200])
DARK_MARK_LOWER   = np.array([0,  0,   0])
DARK_MARK_UPPER   = np.array([180, 80, 60])
MIN_LESION_AREA   = 20


def _leaf_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([25,30,30]), np.array([95,255,255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=2)
    return m

def _edges(bgr, mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    e = cv2.Canny(blur, 30, 100)
    return cv2.bitwise_and(e, e, mask=mask)

def _lesions(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LESION_HSV_LOWER, LESION_HSV_UPPER)
    m = cv2.bitwise_and(m, m, mask=mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_LESION_AREA:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def _dark_marks(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, DARK_MARK_LOWER, DARK_MARK_UPPER)
    m = cv2.bitwise_and(m, m, mask=mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:
        a = cv2.contourArea(c)
        if MIN_LESION_AREA <= a <= 500:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def preprocess_to_rgb(image_path: str, size: int = 224) -> np.ndarray:
    """
    Load an image and return a (size, size, 3) uint8 RGB array where:
      - channel 0: grayscale of original (normalised texture)
      - channel 1: lesion + dark-mark binary mask (disease signal)
      - channel 2: edge map (leaf structure)

    This 3-channel layout is directly compatible with the pretrained ViT
    which expects 3-channel input — no architecture changes needed.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)

    lm   = _leaf_mask(bgr)
    e    = _edges(bgr, lm)
    les  = _lesions(bgr, lm)
    dm   = _dark_marks(bgr, lm)

    # Combine lesion types into one disease-signal channel
    disease = cv2.bitwise_or(les, dm)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Stack as 3-channel
    stacked = np.stack([gray, disease, e], axis=-1)          # (H, W, 3)
    stacked = cv2.resize(stacked, (size, size))
    return stacked.astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class LeafDiseaseDataset(Dataset):
    """
    Folder-per-class dataset.  Label = parent folder name.
    Preprocesses images on-the-fly with the leaf preprocessing pipeline.
    """

    def __init__(self, root: str, label2id: dict, feature_extractor,
                 augment: bool = False, img_size: int = 224):
        self.samples          = []   # list of (path, label_id)
        self.label2id         = label2id
        self.feature_extractor = feature_extractor
        self.augment          = augment
        self.img_size         = img_size

        root = Path(root)
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name
            if label not in label2id:
                log.warning(f"Folder '{label}' not in label map — skipping.")
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(img_path), label2id[label]))

        log.info(f"Dataset loaded: {len(self.samples)} images, "
                 f"{len(label2id)} classes from {root}")

        # Augmentation transforms applied to the uint8 numpy array
        self._aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # forces model to see close-ups
            transforms.RandomAdjustSharpness(sharpness_factor=2),  # accentuates subtle marks
            transforms.ToTensor(),
        ]) if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = preprocess_to_rgb(path, self.img_size)   # (H,W,3) uint8

        if self.augment and self._aug:
            tensor = self._aug(img)                    # (3,H,W) float [0,1]
            # feature_extractor expects PIL or numpy; convert back
            img_np = (tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img

        # HuggingFace feature extractor → normalised pixel_values tensor
        encoded = self.feature_extractor(images=img_np, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)  # (3,224,224)
        return pixel_values, torch.tensor(label, dtype=torch.long)


# ════════════════════════════════════════════════════════════════════════════
#  Model  (unchanged from user spec)
# ════════════════════════════════════════════════════════════════════════════

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output  = self.dropout(outputs.last_hidden_state[:, 0])
        logits  = self.classifier(output)
        loss    = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels),
                                         labels.view(-1))
        return logits, loss


# ════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ════════════════════════════════════════════════════════════════════════════

def build_label_map(data_dir: str) -> tuple[dict, dict]:
    """Return (label2id, id2label) from subfolder names, sorted for reproducibility."""
    classes = sorted(
        d.name for d in Path(data_dir).iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


def evaluate(model, loader, device) -> tuple[float, float, list, list]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)
            logits, loss = model(pixel_values, labels)
            total_loss  += loss.item() * labels.size(0)
            preds        = logits.argmax(dim=-1)
            correct     += (preds == labels).sum().item()
            n           += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / n, correct / n, all_preds, all_labels


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_history(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    log.info(f"Training curves saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Label map ────────────────────────────────────────────────────────────
    label2id, id2label = build_label_map(args.data_dir)
    num_labels = len(label2id)
    log.info(f"Classes ({num_labels}): {label2id}")

    # Save for inference use
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # ── Feature extractor ────────────────────────────────────────────────────
    fe = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # ── Datasets ─────────────────────────────────────────────────────────────
    full_ds = LeafDiseaseDataset(args.data_dir, label2id, fe,
                                 augment=False, img_size=224)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Re-wrap train split with augmentation
    train_ds.dataset = LeafDiseaseDataset(args.data_dir, label2id, fe,
                                          augment=True, img_size=224)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    log.info(f"Train: {n_train} samples | Val: {n_val} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ViTForImageClassification(num_labels=num_labels).to(device)

    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        old_state = ckpt["model_state"]
        old_num_labels = old_state["classifier.weight"].shape[0]

        if old_num_labels == num_labels:
            model.load_state_dict(old_state)
            start_epoch  = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            history      = ckpt.get("history", history)
            log.info(f"Resumed from {args.resume} at epoch {start_epoch}")
        elif old_num_labels < num_labels:
            # Expanding model: add new classes, keep learned weights for existing ones
            model.load_state_dict(old_state, strict=False)
            old_id2label = ckpt.get("id2label", {})
            old_id2label = {int(k): v for k, v in old_id2label.items()}
            with torch.no_grad():
                for old_idx in range(old_num_labels):
                    class_name = old_id2label.get(old_idx)
                    if class_name is not None and class_name in label2id:
                        new_idx = label2id[class_name]
                        model.classifier.weight[new_idx].copy_(old_state["classifier.weight"][old_idx])
                        model.classifier.bias[new_idx].copy_(old_state["classifier.bias"][old_idx])
            start_epoch = 0  # Fresh fine-tune for new classes
            best_val_acc = 0.0
            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            log.info(f"Resumed from {args.resume} and expanded {old_num_labels} → {num_labels} classes")
        else:
            # Shrinking: load backbone only, reinit classifier
            model.load_state_dict({k: v for k, v in old_state.items()
                                   if not k.startswith("classifier.")}, strict=False)
            start_epoch = 0
            best_val_acc = 0.0
            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            log.info(f"Resumed backbone from {args.resume}; classifier reinitialized for {num_labels} classes")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    # Use a lower LR for pretrained ViT backbone, higher for the new head
    optimizer = torch.optim.AdamW([
        {"params": model.vit.parameters(),        "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=0.01)

    total_steps    = args.epochs * len(train_loader)
    warmup_steps   = int(total_steps * 0.05)
    scheduler      = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps, pct_start=warmup_steps / total_steps
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)

            optimizer.zero_grad()
            logits, loss = model(pixel_values, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss    += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(-1) == labels).sum().item()
            epoch_n       += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss    / epoch_n
        train_acc  = epoch_correct / epoch_n

        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        log.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        ckpt_state = {
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "label2id":     label2id,
            "id2label":     id2label,
            "history":      history,
        }
        save_checkpoint(ckpt_state,
                        Path(args.output_dir) / "checkpoints" / "last.pt")
        if is_best:
            save_checkpoint(ckpt_state,
                            Path(args.output_dir) / "checkpoints" / "best.pt")
            log.info(f"  ★ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("\n── Final Validation Report ──")
    print(classification_report(val_labels, val_preds,
                                 target_names=[id2label[i] for i in sorted(id2label)]))

    plot_history(history, args.output_dir)
    log.info(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    log.info(f"Checkpoints and label map saved in: {args.output_dir}/")


# ════════════════════════════════════════════════════════════════════════════
#  Inference helper
# ════════════════════════════════════════════════════════════════════════════

def predict(image_path: str, checkpoint: str, device_str: str = "cpu"):
    """Quick single-image inference."""
    device = torch.device(device_str)
    ckpt   = torch.load(checkpoint, map_location=device)
    id2label = ckpt["id2label"]
    num_labels = len(id2label)

    model = ViTForImageClassification(num_labels=num_labels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    fe  = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    img = preprocess_to_rgb(image_path, 224)
    pv  = fe(images=img, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        logits, _ = model(pv)
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred  = int(torch.argmax(logits).item())

    print(f"\nPrediction: {id2label[pred]}")
    for i, p in enumerate(probs):
        print(f"  {id2label[i]:25s}: {p:.4f}")
    return id2label[pred], probs


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leaf Disease ViT Trainer")
    sub = parser.add_subparsers(dest="cmd")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data_dir",    required=True,  help="Root folder with class subfolders")
    p_train.add_argument("--output_dir",  default="./output", help="Where to save checkpoints & logs")
    p_train.add_argument("--epochs",      type=int,   default=20)
    p_train.add_argument("--batch_size",  type=int,   default=16)
    p_train.add_argument("--lr",          type=float, default=2e-4)
    p_train.add_argument("--val_split",   type=float, default=0.2,  help="Fraction for validation")
    p_train.add_argument("--num_workers", type=int,   default=4)
    p_train.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Run inference on a single image")
    p_pred.add_argument("--image",      required=True)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--device",     default="cpu")

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args.image, args.checkpoint, args.device)
    else:
        # Default to train if called with old-style flat args for convenience
        parser.print_help()

from train import train, predict
import argparse

# ── Training ──────────────────────────────────────────────────────────────
args = argparse.Namespace(
    data_dir    = "./coloured",
    output_dir  = "./output_6class",
    epochs      = 20,
    batch_size  = 16,
    lr          = 2e-4,
    val_split   = 0.2,
    num_workers = 0,
    resume      = None,   # or path to checkpoint e.g. "./output/checkpoints/best.pt"
)
train(args)
