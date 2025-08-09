# Fabric-Defect-Detection
This project focuses on the automated identification of defects in fabric using deep learning models. It aims to eliminate manual inspection processes by applying convolutional neural networks (CNNs) to detect irregularities in fabric patterns.
Our work presents a complete, robust pipeline for identifying defects in woven fabrics, addressing common challenges like dataset imbalance, duplicate images, and noise. The project is applied to the Woven Fabric Defect Detection (WFDD) dataset.

# DATASETS USED FOR IMPLEMENTATION :

<ul>
<li>

**WFDD DATASET(Warping Fault Detection Dataset)**
It is organized into 4 distinct fabric types, each exhibiting a variety of flaw patterns.
I have used augmentation techniques for balancing the dataset and to increase generalisation
| Fabric Class   | Description                            | # Images | Defect Types                   |
| -------------- | -------------------------------------- | -------: | ------------------------------ |
| Grey Cloth     | Plain woven grey fabric                |      1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Grid Cloth     | Woven pattern with grid/check motifs   |      1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Yellow Cloth   | Solid yellow woven fabric              |      1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Pink Flower    | Floral printed cloth from public set   |      1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
</li>
</ul>

<ul>
<li>
 
**Total images**: 4000

</li>
</ul>

<ul>
<li>
 
 **Use cases**:  
 Binary classification (defect vs. non‑defect)  

</li>
</ul>

### Overview
The core contributions are:

-A reproducible, multi-stage preprocessing framework that significantly improves image quality (PSNR/SSIM) and prepares the data for effective model training.

-A carefully balanced dataset of 4,000 defect and 4,000 defect-free images, created through strategic undersampling and augmentation.

-A custom hybrid deep learning model that combines the strengths of EfficientNetB0, multi-scale attention, and transformer encoders.

-Rigorous evaluation using Bayesian hyperparameter optimization, 5-fold stratified cross-validation, and ensemble testing to ensure robust and generalizable results.

### Folder Structure

WFDD/
├── grey_cloth/
│ ├── normal/
│ └── defect/
├── grid_cloth/
│ ├── normal/
│ └── defect/
├── yellow_cloth/
│ ├── normal/
│ └── defect/
└── pink_flower/
├── normal/
└── defect/
