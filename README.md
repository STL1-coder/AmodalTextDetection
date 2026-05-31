# AmodalTextDetection

A framework for scene text detection under occlusion and complex backgrounds. 

## Project Structure

* `model.py` – Network architecture (ResNet18, ViT, CRFT, FPN)
* `dataset.py` – Dataset loading and preprocessing
* `engine.py` – Training and evaluation pipeline
* `train.py` – Model training
* `test.py` – Evaluation and visualization

## Installation

```bash
pip install torch torchvision timm opencv-python matplotlib tqdm gdown
```

## Dataset

The implementation follows the ICDAR 2015 dataset format.

```text
data/
├── ch4_training_images
├── ch4_training_localization_transcription_gt
├── ch4_testing_images
└── ch4_testing_localization_transcription_gt
```

Dataset download:

```bash
gdown https://drive.google.com/drive/folders/1sKrDOMmMXDlrVHYZ-PQ76HJLKLiJfXqY?usp=sharing --folder
```

## Pretrained Model

Download the pretrained checkpoint and place it in the `checkpoints/` directory:

https://drive.google.com/file/d/1igLD8GeM_cH5HWLFTYdXuaUvj0HHLbpB/view?usp=sharing

## Training

```bash
python train.py
```

## Evaluation

```bash
python test.py --mode eval
```

## Visualization

```bash
python test.py --mode viz
```


