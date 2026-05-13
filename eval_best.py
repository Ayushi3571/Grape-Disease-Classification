"""
Evaluate the best checkpoint on the validation split used during training.

Usage:
    python eval_best.py --checkpoint ./output_6class/checkpoints/best.pt --data_dir ./coloured
"""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from train import (
    LeafDiseaseDataset,
    create_model_from_checkpoint,
    evaluate,
    load_checkpoint,
    load_normalization_stats,
    split_dataset,
    upgrade_state_dict_keys,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output_6class/checkpoints/best.pt", help="Path to best.pt")
    parser.add_argument("--data_dir", default="./coloured", help="Root folder with class subfolders")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    label2id = checkpoint["label2id"]
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

    full_dataset = LeafDiseaseDataset(
        args.data_dir,
        label2id,
        image_mean,
        image_std,
        augment=False,
        image_size=config["image_size"],
    )
    _, val_dataset = split_dataset(full_dataset, args.val_split)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loss, val_acc, preds, labels = evaluate(model, val_loader, device)
    class_names = [id2label[i] for i in sorted(id2label)]

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?') + 1}")
    print(f"Model preset: {config['model_preset']} ({config['model_name']})")
    print(f"Best val_acc (saved in checkpoint): {checkpoint.get('best_val_acc', 0.0):.4f}\n")
    print("Validation Statistics")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val acc:  {val_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
