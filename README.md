# Leaf Disease ViT Training

This project trains a Vision Transformer (ViT) classifier for grape leaf disease detection using the pipeline in `train.py`.

## Dataset Layout

Place images in class-wise folders under a root directory (for example `coloured`):

```text
coloured/
  Grape__Downey_Mildew/
  Grape__Powdery_Mildew/
  Grape___Black_rot/
  Grape___Esca_(Black_Measles)/
  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
  Grape___healthy/
```

Each class folder should contain image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

Train a new model:

```bash
python train.py train --data_dir ./coloured --output_dir ./output_6class --epochs 20 --batch_size 16
```

Resume from a checkpoint:

```bash
python train.py train --data_dir ./coloured --output_dir ./output_6class --epochs 20 --resume ./output/checkpoints/best.pt
```

`train.py` automatically:
- builds labels from folder names,
- creates a train/validation split,
- saves `last.pt` and `best.pt` checkpoints,
- writes `label_map.json`,
- saves `training_curves.png`.

## Predict Single Image

```bash
python train.py predict --image ./path/to/image.jpg --checkpoint ./output_6class/checkpoints/best.pt --device cpu
```

Use `--device cuda` if CUDA is available.

## Evaluate Best Checkpoint

Use the helper script:

```bash
python eval_best.py --checkpoint ./output_6class/checkpoints/best.pt --data_dir ./coloured
```

This prints:
- validation loss and accuracy,
- classification report,
- confusion matrix.

## Outputs

By default (or by `--output_dir`), artifacts are saved to:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `label_map.json`
- `training_curves.png`

## Notes

- Class names are taken directly from folder names.
- If you add/remove classes, train again so checkpoint head dimensions match current classes.
