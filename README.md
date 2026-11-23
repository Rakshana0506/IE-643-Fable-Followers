# Disease Aware Diffusion for Fine Grained Crop Disease Localization from Drone Captured Video

IE 643 Deep Learning: Theory and Practice, IIT Bombay
Team **FableFollowers**: Rakshana Sundaram (24b2428), Amal Shadh K (24b0019)

## Overview

Drone footage covers a whole field, but a single disease lesion takes up only a few pixels in a drone frame, and bicubic upscaling cannot bring that detail back. This project magnifies leaf crops from drone video with a fine-tuned SR3 diffusion model and checks, lesion by lesion, whether the magnified image keeps real lesions in place or shifts, drops or invents them.

Pipeline:

```
drone video
  -> frame sampling (blurry frames dropped)
  -> plant segmentation (DeepLabV3, PhenoBench)
  -> plant tracking across frames (camera-motion compensated)
  -> leaf detection (Faster R-CNN, PlantDoc)
  -> 256x256 leaf crops
  -> SR3 magnification (residual target, fine-tuned on plant + drone data)
  -> lesion segmentation (DeepLabV3+, PlantSeg)
  -> severity = lesion area / leaf area
  -> plant and disease classification (DINOv2 ViT-S/14)
```

## Models and datasets

| Stage | Model | Training data | Result |
|---|---|---|---|
| Plant segmentation | DeepLabV3-ResNet50 (ImageNet) | PhenoBench, 1,407 train / 772 val | Dice 0.916, IoU 0.846 |
| Leaf detection | Faster R-CNN ResNet50-FPN v2 (COCO) | PlantDoc, 2,231 images | mAP 0.436, mAP@50 0.619 |
| Lesion segmentation | DeepLabV3+-ResNet50 (ImageNet) | PlantSeg, 5,367 train / 846 val | IoU 0.638, Dice 0.779 |
| Super-resolution | SR3 UNet (FFHQ-pretrained, fine-tuned) | PlantSeg lesion crops + 6,000 PlantVillage images, then drone leaf crops | PSNR 24.04, SSIM 0.601, LPIPS 0.312 |
| Classification | DINOv2 ViT-S/14 | PlantSeg + PlantVillage, about 131 classes | Accuracy 79.2%, macro-F1 0.719 |

### SR3 details

- HR crops are 256x256; LR inputs are 64x64 (4x), created by Gaussian blur, bicubic downsampling, Gaussian noise and JPEG compression.
- The bicubic-upsampled LR image is the condition. The model predicts the residual `2 x (HR - bicubic(LR))`, so bicubic supplies colour and low frequencies and SR3 adds fine detail. Final output = bicubic + predicted residual.
- L1 noise-prediction loss, AdamW (1e-4, cosine), bf16, EMA weights.
- Predicting the full HR image gave PSNR 22.2 dB; the residual target raised it to 24.0 dB.
- Stage 2 fine-tuning used degraded / original leaf-crop pairs from one drone video. Evaluation used a different video.

## Lesion-level evaluation

Each leaf crop from the test video is degraded, then restored by SR3 and by bicubic upscaling. SR3 sees only the degraded crop; the original crop is used only as the reference. Lesions are segmented on all three images and every lesion in the original is classed as:

- **kept**: at least 30% of it is still covered after restoration
- **shifted**: kept, but its centre moved more than 8 px
- **lost**: gone after restoration
- **new**: appears only after restoration (spurious)

Specks under 15 px are ignored, and lesions are counted only inside the plant mask.

### Results (8 plants, 445 reference lesions)

| Method | Lesions found | Kept | Lost | New | Shifted | IoU | Mean abs. severity error |
|---|---|---|---|---|---|---|---|
| Bicubic | 723 | 278 (62%) | 167 | 640 | 222 | 0.096 | 12.44 pp |
| SR3 | 465 | 156 (35%) | 289 | 324 | 76 | 0.162 | 3.42 pp |

SR3 gives a much closer severity estimate and half the spurious lesions, but loses more real lesions than bicubic. Diffusion restoration trades missed lesions for fewer false detections.

**Limitations:** one test video and 8 plants; the original crop is the reference rather than expert-labelled ground truth.

## Repository structure

| File | Purpose |
|---|---|
| `final_pipeline.py` | Full single-image pipeline: plant mask, leaf detection, degradation, SR3, lesion segmentation, severity, DINOv2 |
| `video_pipeline.py` | Runs the pipeline on a video: frame sampling, plant tracking, best-view selection, pooling per plant |
| `plant_tracker.py` | Camera-motion compensated plant tracking across frames |
| `lesion_compare.py` | Kept / shifted / lost / new lesion counts and agreement overlays |
| `sr3_files/sr3_ft.py` | SR3 UNet, degradation, diffusion training loss, DDIM and ancestral samplers, metrics |
| `server.py`, `index.html` | Flask demo with a browser interface |

## Running

Install dependencies:

```
pip install torch torchvision timm scipy pillow opencv-python flask segmentation-models-pytorch
```

Place the checkpoints at the paths set at the top of `final_pipeline.py`:

```
plant_leaf_deeplab/best_deeplabv3_resnet50_plant.pth
rcnn_files/fasterrcnn_best_new.pth
sr3_files/sr3_plantseg_residual.pth
lesion_deeplab_files/deeplabv3plus_degaug_best.pth
dinoV2_classifier/best.pt
dinoV2_classifier/classes.json
```

Checkpoints: [Google Drive link]

Video pipeline (set `VIDEO_PATH` in `video_pipeline.py` first):

```
python video_pipeline.py
```

Outputs go to `video_results/`: per-plant folders with original, normal and SR3 lesion overlays, `summary.csv`, and `tracks_preview.mp4`.

Demo:

```
python server.py
```

Then open http://127.0.0.1:5000.

## References

1. C. Saharia et al., Image Super-Resolution via Iterative Refinement, IEEE TPAMI, 2023.
2. Janspiry, Image-Super-Resolution-via-Iterative-Refinement, GitHub.
3. T.-Y. Wei et al., PlantSeg: A Large-Scale In-the-Wild Dataset for Plant Disease Segmentation, 2024.
4. D. Singh et al., PlantDoc: A Dataset for Visual Plant Disease Detection, 2019.
5. S. P. Mohanty et al., Using Deep Learning for Image-Based Plant Disease Detection, 2016.
6. J. Weyler et al., PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain.
7. R. Polly, E. Anna Devi, Semantic segmentation for plant leaf disease classification and damage detection, Smart Agricultural Technology, 2024.
