# 🌿 Plant Disease Detection using Convolutional Neural Networks

**Artificial Neural Networks & Deep Learning — PBL (AI253IA)**

A deep-learning system that looks at a photo of a plant leaf and predicts whether it is **healthy** or which **disease** it has, out of **38 classes across 14 crops**. We compare a **CNN built from scratch** with **MobileNetV2 transfer learning**, evaluate both with accuracy, precision, recall, F1 and confusion matrices, test them on **real field photos**, and serve the best model in a **web app**.

---

## ▶️ Run it (Google Colab, free GPU)

| Notebook | Phase | Open in Colab |
|---|---|---|
| 1. Dataset & preprocessing | Phase 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/01_dataset_and_preprocessing.ipynb) |
| 2. Training & evaluation | Phase 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/02_train_and_evaluate.ipynb) |
| 3. Web app & field test | Phase 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/03_web_app_and_field_test.ipynb) |
| 4. Field fine-tuning (real-photo improvement) | Extension | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashixb-cyber/plant-disease-detection-cnn/blob/main/notebooks/04_field_finetuning.ipynb) |

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

The notebooks download both datasets automatically. Nothing needs to be uploaded by hand. Full sources, citations, the class list and a download script are in [`data/`](data/README.md).

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

Test set: 8,146 PlantVillage images never seen during training (full numbers in [`results/`](results/)).

| Model | Test accuracy | Top-3 accuracy | Macro precision | Macro recall | Macro F1 | Parameters | Training time (T4) |
|---|---|---|---|---|---|---|---|
| Basic CNN (scratch) | 93.24% | 99.55% | 92.49% | 92.70% | 92.03% | 1.06 M | 59 min |
| **MobileNetV2 (transfer)** | **96.80%** | **99.63%** | **96.32%** | **96.48%** | **96.32%** | 2.31 M | 23 min |

**Key findings**
- Transfer learning wins: **+3.6 points accuracy and +4.3 points macro F1**, in **less than half the training time**.
- MobileNetV2 scores **F1 ≥ 0.96 on 27 of 38 classes**, and 4 classes are perfect (1.00).
- The hardest classes are diseases with **similar-looking symptoms on the same crop**, exactly the challenge in our problem statement:

| Hardest classes (MobileNetV2) | F1 | Basic CNN F1 |
|---|---|---|
| Tomato – Early blight | 0.815 | 0.824 |
| Tomato – Target spot | 0.829 | 0.740 |
| Corn – Cercospora / Gray leaf spot | 0.871 | 0.725 |
| Tomato – Septoria leaf spot | 0.903 | 0.920 |
| Tomato – Late blight | 0.908 | 0.877 |

- Top-3 accuracy is above 99.5% for both models. When the top prediction is wrong, the correct disease is almost always among the next two guesses, so the app shows the top 3.

### Lab vs real-world field test (PlantDoc)
We tested both models, **without retraining**, on PlantDoc: real photos from fields and the internet with soil, hands, shadows and several leaves per photo.

| Model | Lab accuracy (PlantVillage) | Field accuracy (PlantDoc) | Field top-3 | Field macro F1 | Correct crop |
|---|---|---|---|---|---|
| Basic CNN | 93.24% | 7.29% | 22.65% | 6.03% | 24.67% |
| **MobileNetV2** | **96.80%** | **20.09%** | **39.45%** | **18.57%** | **46.16%** |

- Accuracy falls sharply on field photos (**domain shift**). The models partly learned the *lab photo style* (one leaf, plain grey background, even lighting), not only the disease. Mohanty et al. (2016) reported the same effect.
- Transfer learning **generalises far better**: MobileNetV2 is **2.8× more accurate** than the scratch CNN on field photos (20.1% vs 7.3%), because its ImageNet features were learned from varied, real-world images.
- Both are well above random guessing (1 in 38 = 2.6%). MobileNetV2 identifies the crop correctly 46% of the time and has the right disease in its top 3 for 39% of photos.
- **Conclusion:** the system is a strong lab-validated prototype. Real farm deployment needs field images in training.

### Improving real-world accuracy (Notebook 4)
To reduce domain shift, MobileNetV2 is fine-tuned on a mix of **PlantDoc training photos** (real field/internet images) and **150 PlantVillage photos per class** (so lab knowledge is kept), then re-tested on the **separate PlantDoc test folder**.

_Results: run Notebook 4 to fill in (`results/field_finetune_results.csv`)._

---

## 📁 Repository structure

```
├── notebooks/
│   ├── 01_dataset_and_preprocessing.ipynb   # Phase 1: download, explore, split, preprocess
│   ├── 02_train_and_evaluate.ipynb          # Phase 2: CNN + MobileNetV2, metrics, comparison
│   ├── 03_web_app_and_field_test.ipynb      # Phase 3: Gradio app + PlantDoc real-world test
│   └── 04_field_finetuning.ipynb            # Extension: fine-tune on real field photos
├── app/
│   ├── app.py                               # run the web app on your own laptop
│   └── examples/                            # sample leaf images for the app
├── docs/
│   ├── COLAB_GUIDE.md                       # click-by-click instructions
│   └── PROJECT_EXPLAINED.md                 # the whole project in simple language
├── data/
│   ├── README.md                            # dataset sources, citations, class list
│   └── download_data.py                     # download the datasets into this folder
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

## 🔭 Future work
1. Collect more real field images (local farms) to extend the field fine-tuning in Notebook 4.
2. Detect and crop the leaf first (object detection / segmentation), then classify.
3. Stronger augmentation: random backgrounds, blur, shadows and lighting changes.
4. Longer fine-tuning of more MobileNetV2 layers, or larger models such as EfficientNet.
5. Convert to TensorFlow Lite for an offline mobile app.

## ⚠️ Limitations
- PlantVillage images are lab photos (one leaf, plain background), so accuracy on real farm photos is much lower: 20.1% for MobileNetV2 on PlantDoc vs 96.8% in the lab.
- The model only knows the 38 classes it was trained on; any other plant or disease is forced into one of them.
- It is a decision-support prototype, not a replacement for an agricultural expert.

## 📚 References
1. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.
3. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. *CoDS-COMAD 2020*.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
5. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.
6. Too, E. C., et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272–279.
