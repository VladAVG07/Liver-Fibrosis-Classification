# Liver Fibrosis Classification — Vision Transformers vs. CNNs

Comparative analysis of Vision Transformers (MedViT / MedViT-V2) and convolutional
networks (ResNet-18) for **liver fibrosis staging from ultrasound images**.

The task is a 5-class classification problem over the METAVIR fibrosis grades
(**F0–F4**). MedViT-V1 Base reaches the best result with **98.43% accuracy**
(F1 = 0.9772), clearly outperforming a ResNet-18 pretrained baseline
(92% accuracy, F1 = 0.8884). Grad-CAM visualisations show that MedViT attends to
clinically relevant tissue regions.

## Dataset

[Liver Histopathology Fibrosis Ultrasound Images](https://www.kaggle.com/datasets/vibhingupta028/liver-histopathology-fibrosis-ultrasound-images)
— 6,323 images across five fibrosis stages:

| Stage | F0   | F1  | F2  | F3  | F4   |
| ----- | ---- | --- | --- | --- | ---- |
| Count | 2114 | 861 | 793 | 857 | 1698 |

The data is downloaded into `data/` and split into `dataset_split/{train,val,test}`
with a stratified **70 / 15 / 15** ratio (seed 42) by the first notebook.

## Notebooks

| Notebook                      | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `liver_fibrosis.ipynb`        | Data download/split + ResNet-18 (pretrained & from-scratch). |
| `medvit_fibrosis.ipynb`       | MedViT-V1 small / base / large training and evaluation.      |
| `medvit_fibrosis_large.ipynb` | MedViT-V1 large, focused run.                                |
| `medvitv2_fibrosis.ipynb`     | MedViT-V2 small / base training and evaluation.              |
| `gradcam_fibrosis.ipynb`      | Grad-CAM visualisations for the trained MedViT models.       |
| `generate_figures.ipynb`      | Reproduces the comparison figures and confusion matrices.    |

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy pillow \
            kaggle splitfolders gdown opencv-python grad-cam
```

The MedViT notebooks import the upstream model source. Clone the repositories into
the project root next to the notebooks:

```bash
git clone https://github.com/Omid-Nejati/MedViT.git        # MedViT-V1
git clone https://github.com/Omid-Nejati/MedViTV2.git      # MedViT-V2
```

Provide your own Kaggle API credentials (the in-notebook values are placeholders):

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```
