# 🌿 Plant Disease Detection using Convolutional Neural Networks

**Artificial Neural Networks & Deep Learning — PBL (AI253IA), RV College of Engineering**
Swayam Goyal (1RV24CI131) · Yash Bhalotia (1RV24CI142) · Yash Purwar (1RV24CI143)

A deep-learning system that looks at a photo of a plant leaf and predicts whether it is **healthy** or which **disease** it has, out of **38 classes across 14 crops**. We compare a **CNN built from scratch** with **MobileNetV2 transfer learning**, evaluate both with accuracy, precision, recall, F1 and confusion matrices, test them on **real field photos**, and serve the best model in a **web app**.

---

## ▶️ Run it (Google Colab, free GPU)

| Notebook | Phase | Open in Colab |
|---|---|---|
| 1. Dataset & preprocessing | Phase 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/01_dataset_and_preprocessing.ipynb) |
| 2. Training & evaluation | Phase 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/02_train_and_evaluate.ipynb) |
| 3. Web app & field test | Phase 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/03_web_app_and_field_test.ipynb) |

In each notebook: **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Full step-by-step guide: [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md).

---

## 📊 Dataset

| | |
|---|---|
| **Name** | PlantVillage (Hughes & Salathé, 2015) |
| **Images** | 54,305 colour photos of individual leaves, 256 × 256 px |
| **Classes** | 38 (healthy + diseases) across 14 crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato |
| **Source** | [github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (`raw/color`), the copy published by Mohanty et al. (2016) |
| **Split** | 70% train / 15% validation / 15% test, stratified, seed 42 |
| **Field test set** | PlantDoc (Singh et al., 2020): ~2,600 real-world photos from 28 matching classes ([github](https://github.com/pratikkayal/PlantDoc-Dataset)) |

The notebooks download both datasets automatically. Nothing needs to be uploaded by hand.

---

## 🧠 Method

```
Leaf image → Preprocessing → CNN → Feature extraction → Classification → Prediction + confidence
             (resize 224×224,      (conv layers learn       (softmax over
              normalise, augment)   edges→spots→patterns)    38 classes)
```

| | Model A: Basic CNN | Model B: MobileNetV2 |
|---|---|---|
| Approach | Built and trained from scratch | Transfer learning from ImageNet |
| Architecture | 5 × [Conv → BatchNorm → ReLU → MaxPool] (32→256 filters) → GAP → Dense 256 → Softmax 38 | MobileNetV2 base → GAP → Dropout → Softmax 38 |
| Training | Adam 1e-3, up to 25 epochs, early stopping | Stage 1: head only (Adam 1e-3) · Stage 2: fine-tune top 40 layers (Adam 1e-5) |
| Shared | Augmentation (flip, rotate, zoom, contrast), class weights for imbalance, dropout, ReduceLROnPlateau | |

**Evaluation:** accuracy, top-3 accuracy, precision, recall, F1 (macro and weighted), per-class report, normalised confusion matrix, most-confused pairs, lab vs field accuracy.

---

## 📈 Results

> Filled in from `results/model_comparison.csv` and `results/field_test_plantdoc.csv` after running Notebooks 2 and 3.

| Model | Test accuracy | Macro F1 | Field accuracy (PlantDoc) | Parameters |
|---|---|---|---|---|
| Basic CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MobileNetV2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 📁 Repository structure

```
├── notebooks/
│   ├── 01_dataset_and_preprocessing.ipynb   # Phase 1: download, explore, split, preprocess
│   ├── 02_train_and_evaluate.ipynb          # Phase 2: CNN + MobileNetV2, metrics, comparison
│   └── 03_web_app_and_field_test.ipynb      # Phase 3: Gradio app + PlantDoc real-world test
├── app/
│   ├── app.py                               # run the web app on your own laptop
│   └── examples/                            # sample leaf images for the app
├── docs/
│   ├── COLAB_GUIDE.md                       # click-by-click instructions
│   ├── PROJECT_EXPLAINED.md                 # the whole project in simple language
│   └── VIVA_QA.md                           # likely viva questions with answers
├── models/      # trained .keras files go here (download from Drive; not stored in git)
├── results/     # metrics CSVs and figures (copy from Drive after training)
└── requirements.txt
```

## 💻 Run the app locally (optional)

```bash
pip install -r requirements.txt
# copy mobilenetv2.keras, basic_cnn.keras and class_names.json from
# Google Drive (MyDrive/plant-disease-project/models/) into models/
python app/app.py            # opens http://127.0.0.1:7860
```

## ⚠️ Limitations
- PlantVillage images are lab photos (one leaf, plain background), so accuracy on real farm photos is much lower (see the PlantDoc test in Notebook 3).
- The model only knows the 38 classes it was trained on; any other plant or disease is forced into one of them.
- It is a decision-support prototype, not a replacement for an agricultural expert.

## 📚 References
1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.
3. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. *CoDS-COMAD 2020*.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
5. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.
6. Too, E. C., et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272–279.
