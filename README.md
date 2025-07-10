# Fabric-Defect-Detection
This project focuses on the automated identification of defects in fabric using deep learning models. It aims to eliminate manual inspection processes by applying convolutional neural networks (CNNs) to detect irregularities in fabric patterns.

# DATASETS USED FOR IMPLEMENTATION :

<ul>
<li>

**WFDD DATASET(Warping Fault Detection Dataset)**
| Fabric Class   | Description                            | # Images | Defect Types                   |
| -------------- | -------------------------------------- | -------: | ------------------------------ |
| Grey Cloth     | Plain woven grey fabric                |      ~1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Grid Cloth     | Woven pattern with grid/check motifs   |      ~1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Yellow Cloth   | Solid yellow woven fabric              |      ~1K | • Block‑shape<br>• Line‑type<br>• Point‑like |
| Pink Flower    | Floral printed cloth from public set   |      ~1K | • Block‑shape<br>• Line‑type<br>• Point‑like |



</li>
</ul>


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
