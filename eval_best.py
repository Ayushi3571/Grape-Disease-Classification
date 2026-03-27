"""
Evaluate the best checkpoint on the validation set and print statistics.
Uses the same train/val split as training (seed 42) for reproducibility.

Usage:
    python eval_best.py --checkpoint output/checkpoints/best.pt --data_dir ./coloured
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix

from train import LeafDiseaseDataset, ViTForImageClassification, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output_6class/checkpoints/best.pt", help="Path to best.pt")
    parser.add_argument("--data_dir", default="./coloured", help="Root folder with class subfolders")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction (must match training)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    # Ensure id2label has int keys (JSON/torch may use str)
    id2label = {int(k): v for k, v in id2label.items()}
    num_labels = len(id2label)

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?') + 1}")
    print(f"Best val_acc (from training): {ckpt.get('best_val_acc', '?'):.4f}\n")

    # Model
    model = ViTForImageClassification(num_labels=num_labels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Same data setup as training
    fe = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    full_ds = LeafDiseaseDataset(args.data_dir, label2id, fe, augment=False, img_size=224)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Evaluate
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, device)
    class_names = [id2label[i] for i in sorted(id2label)]

    print("── Best Model Validation Statistics ──")
    print(f"Val loss:  {val_loss:.4f}")
    print(f"Val acc:   {val_acc:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()