# Grape Disease Detection for Edge Deployment

This project is a subset of a precision agriculture capstone. 

The core idea is:

- isolate the leaf from noisy backgrounds,
- highlight likely disease regions,
- preserve structural cues like veins and boundaries,
- package the result into a 3-channel input for a Vision Transformer.

As I kept working on it, I also started adapting it for edge deployment by adding smaller transformer backbones, TorchScript export, and CPU benchmarking.

## Pipeline

The preprocessing flow has four steps:

1. `Leaf isolation`
   HSV masking and morphology remove some background clutter so later stages focus on the leaf.
2. `Disease signal extraction`
   Lesions and dark marks are surfaced with color filtering and contour-based constraints.
3. `Edge and structure cues`
   Edge maps preserve vein layout and lesion boundaries that the disease mask alone would miss.
4. `ViT-ready 3-channel stack`
   Grayscale texture, disease mask, and edge structure are stacked into a `224 x 224 x 3` input.

I liked this approach because it keeps the model compatible with pretrained transformer backbones while still preserving some hand-crafted agricultural signal.

## Model Options

The repo now supports three presets:

- `base`: `google/vit-base-patch16-224-in21k`
- `edge`: `facebook/deit-tiny-patch16-224`
- `edge_plus`: `facebook/deit-small-patch16-224`

I used them with slightly different goals in mind:

- `base` for the strongest baseline,
- `edge` for strict CPU or embedded deployment,
- `edge_plus` as a middle ground.

## Edge Benchmark Snapshot

I benchmarked the available backbones on CPU with batch size `1`, `224 x 224` input, `5` warmup runs, and `20` timed runs.

| Preset | TorchScript Size | INT8 Size | FP32 Latency | INT8 Latency |
| --- | ---: | ---: | ---: | ---: |
| `base` | 327.37 MB | 82.68 MB | 100.17 ms | 41.57 ms |
| `edge` | 20.71 MB | 5.42 MB | 13.06 ms | 15.73 ms |
| `edge_plus` | 82.15 MB | 20.98 MB | 32.48 ms | 20.79 ms |

What stood out to me:

- `edge` is the most practical preset for lightweight deployment.
- `edge_plus` is a useful compromise when the tiny model is too small.
- quantization helped the larger models more than the tiny one in this setup.

Raw results are saved in `benchmark_artifacts/benchmark_results.json`.

## Failure Cases

I wanted to keep these examples in the README because they reflect the kinds of borderline cases that make agricultural vision interesting in the first place.

**Early-stage black rot**

![Early-stage black rot example](assets/failure_cases/early_black_rot_example.jpg)

This case is hard because the lesion signal is small, sparse, and easy to miss compared with the larger healthy leaf area. In a classification setup, that means the model may not assign strong confidence even when the lesion shape is consistent with early black rot. It is a good reminder that early detection is often the most valuable case and also the hardest one.

**Nutritional deficiency**

![Nutritional deficiency example](assets/failure_cases/nutritional_deficiency_example.jpg)

This is a strong non-disease confounder. The interveinal yellowing is visually obvious, but it does not match the compact lesion patterns the disease pipeline was built to emphasize. A model trained mostly on disease classes can still confuse this kind of stress response with pathology if it has not seen enough nutrient-deficiency examples. I think examples like this are important because they push the project beyond simple disease-vs-healthy thinking.

What these examples highlight:

- subtle early disease can be overwhelmed by background healthy tissue,
- non-disease stress can look abnormal without being infectious,
- classification alone is not always enough for field-ready decision support.

They also point to the next improvements I would like to make:

- add more early-stage examples during training,
- include nutrient-deficiency and other abiotic stress classes,
- move beyond image-level classification toward lesion localization or segmentation.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

List model presets:

```bash
python train.py list-models
```

Train the edge-oriented model:

```bash
python train.py train --data_dir ./coloured --output_dir ./output_edge --model_preset edge --epochs 20 --batch_size 32
```

Run prediction:

```bash
python train.py predict --image ./path/to/image.jpg --checkpoint ./output_edge/checkpoints/best.pt --device cpu
```

Export for deployment:

```bash
python train.py export --checkpoint ./output_edge/checkpoints/best.pt --output ./output_edge/edge/model_edge.ts --format torchscript --quantize
```

Benchmark the exported artifact:

```bash
python edge_infer.py --artifact ./output_edge/edge/model_edge.ts --image ./path/to/image.jpg
```

## Dataset Layout

Expected folder structure:

```text
coloured/
  Grape__Downey_Mildew/
  Grape__Powdery_Mildew/
  Grape___Black_rot/
  Grape___Esca_(Black_Measles)/
  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
  Grape___healthy/
```

Supported image types:
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

## Files

Main scripts:

- `train.py`: training, prediction, and export
- `eval_best.py`: evaluation for a saved checkpoint
- `edge_infer.py`: CPU inference for an exported TorchScript model
- `benchmark_models.py`: preset benchmarking for deployment tradeoffs

Common outputs:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `label_map.json`
- `training_curves.png`

## Current Limitations

This is still a classification-first project, so there are a few directions I would be excited to keep exploring:

- failure-case analysis on difficult field images,
- localization or segmentation instead of only image-level classification,
- testing on a real embedded target such as Jetson-class hardware,
- a more explicit retraining loop for new field data.

## Next Additions

The next improvements I would most like to add are:

- broader failure-case coverage across more lighting and growth-stage conditions,
- brief analysis of why those misses happen across disease and non-disease stress cases,
- stronger deployment notes for edge hardware.
