# 📂 Data sources

This project uses two **public, real-world datasets**. The images are **not stored in this repository** (they total ~1.5 GB and belong to their original authors). They are downloaded from their official sources, either automatically by the notebooks or with the script in this folder.

---

## 1. PlantVillage (training, validation and testing)

| | |
|---|---|
| **What it is** | 54,305 colour photos of individual plant leaves, labelled by plant pathology experts |
| **Classes** | 38 (26 diseases + 12 healthy) across 14 crops |
| **Image format** | JPEG, 256 × 256 pixels, RGB |
| **Created by** | David P. Hughes & Marcel Salathé (Penn State University / EPFL), 2015 |
| **Official copy used** | https://github.com/spMohanty/PlantVillage-Dataset (folder `raw/color`) |
| **Also available at** | TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/plant_village |
| **Paper** | https://arxiv.org/abs/1511.08060 |
| **Used in our notebooks** | Notebooks 1 and 2 (and examples in Notebook 3) |

**Citation**
> Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

The GitHub copy is maintained by S. P. Mohanty, first author of:
> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

<details>
<summary><b>All 38 classes and image counts</b></summary>

| Crop | Class | Images |
|---|---|---|
| Apple | Apple scab | 630 |
| Apple | Black rot | 621 |
| Apple | Cedar apple rust | 275 |
| Apple | Healthy | 1,645 |
| Blueberry | Healthy | 1,502 |
| Cherry | Powdery mildew | 1,052 |
| Cherry | Healthy | 854 |
| Corn (maize) | Cercospora leaf spot / Gray leaf spot | 513 |
| Corn (maize) | Common rust | 1,192 |
| Corn (maize) | Northern leaf blight | 985 |
| Corn (maize) | Healthy | 1,162 |
| Grape | Black rot | 1,180 |
| Grape | Esca (Black measles) | 1,383 |
| Grape | Leaf blight (Isariopsis leaf spot) | 1,076 |
| Grape | Healthy | 423 |
| Orange | Haunglongbing (Citrus greening) | 5,507 |
| Peach | Bacterial spot | 2,297 |
| Peach | Healthy | 360 |
| Pepper (bell) | Bacterial spot | 997 |
| Pepper (bell) | Healthy | 1,478 |
| Potato | Early blight | 1,000 |
| Potato | Late blight | 1,000 |
| Potato | Healthy | 152 |
| Raspberry | Healthy | 371 |
| Soybean | Healthy | 5,090 |
| Squash | Powdery mildew | 1,835 |
| Strawberry | Leaf scorch | 1,109 |
| Strawberry | Healthy | 456 |
| Tomato | Bacterial spot | 2,127 |
| Tomato | Early blight | 1,000 |
| Tomato | Late blight | 1,909 |
| Tomato | Leaf mold | 952 |
| Tomato | Septoria leaf spot | 1,771 |
| Tomato | Spider mites (two-spotted spider mite) | 1,676 |
| Tomato | Target spot | 1,404 |
| Tomato | Yellow leaf curl virus | 5,357 |
| Tomato | Mosaic virus | 373 |
| Tomato | Healthy | 1,591 |
| | **Total** | **54,305** |

</details>

---

## 2. PlantDoc (real-world field test only)

| | |
|---|---|
| **What it is** | ~2,600 real-world leaf photos from fields and the internet (soil, hands, shadows, several leaves per photo) |
| **Classes** | 28 classification folders, all matching a PlantVillage class |
| **Created by** | Singh et al., IIT Gandhinagar, 2020 |
| **Official source** | https://github.com/pratikkayal/PlantDoc-Dataset |
| **Paper** | https://arxiv.org/abs/1911.10317 |
| **Used in our notebooks** | Notebook 3 only. **Never used for training**, only to test real-world performance |

**Citation**
> Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 249–253.

---

## ⬇️ How to get the data

**Option A: automatic (recommended).** Just run the notebooks in Google Colab. They download the data themselves.

**Option B: download into this folder on your own computer**
```bash
# from the project's main folder (requires git)
python data/download_data.py              # both datasets
python data/download_data.py --plantvillage
python data/download_data.py --plantdoc
```

After downloading, the folder looks like this:
```
data/
├── README.md
├── download_data.py
├── PlantVillage-Dataset/raw/color/<38 class folders>/*.JPG
└── PlantDoc-Dataset/{train,test}/<class folders>/*.jpg
```
The downloaded images are ignored by git, so they are never uploaded to GitHub by accident.

**Option C: manual download.** Open the GitHub links above and use **Code → Download ZIP** (large download; it also contains grayscale and segmented versions of PlantVillage that we don't use).

---

## 🔀 Train / validation / test split
We split PlantVillage **70% / 15% / 15%**, stratified by class, with random seed **42** (Notebook 1):

| Split | Images |
|---|---|
| Train | 38,013 |
| Validation | 8,146 |
| Test | 8,146 |

The exact file lists are saved by the notebooks as `split_train.csv`, `split_val.csv` and `split_test.csv`.

## ⚖️ License & usage
Both datasets are published openly for research and education. This repository does not redistribute the images; it links to the original sources and cites the authors as listed above.
