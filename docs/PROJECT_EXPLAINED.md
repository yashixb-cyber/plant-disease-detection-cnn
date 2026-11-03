# 📖 The whole project, explained simply

This guide assumes **zero** background. Read it top to bottom once and you'll understand every part of the project.

---

## 1. The problem in one paragraph
Farmers spot plant diseases by looking at leaves. This is slow, needs experience, and many diseases look alike (brown spots, yellowing). We built a program that **looks at a leaf photo and names the disease** (or says "healthy"), along with how confident it is. It covers 14 crops and 38 classes.

---

## 2. The big ideas (no maths)

### What is a neural network?
A program that **learns from examples** instead of being given rules. Think of teaching a child to recognise fruits: you don't explain "an apple is round and red"; you show many apples until they get it. We show the network ~38,000 labelled leaf photos, it guesses, we tell it how wrong it was, and it adjusts itself a tiny bit. This repeats thousands of times.

### What is an image to a computer?
A grid of numbers. A 224×224 colour photo is 224 × 224 × 3 numbers (Red, Green, Blue), each between 0 and 255.

### What is a CNN (Convolutional Neural Network)?
A neural network designed for images. Its key tool is the **filter** (or kernel): a tiny 3×3 window that slides across the image looking for one specific pattern.
- **First layers** find simple things: edges, colour changes, dots
- **Middle layers** combine them into textures, spot shapes, vein patterns
- **Last layers** combine those into whole disease signatures, e.g. "concentric brown rings = Early blight"

The CNN **learns which filters to use by itself**. Nobody programs "look for brown spots". That's the main advantage over older methods (slide 5 of our presentation).

### The layers we use, in plain words
| Layer | Plain meaning |
|---|---|
| **Conv2D** | Slides filters over the image to detect patterns |
| **ReLU** | Keeps positive signals ("pattern found") and zeroes out the rest |
| **MaxPooling** | Shrinks the image by half, keeping the strongest signals, like zooming out |
| **BatchNormalization** | Keeps numbers in a stable range so training is faster and smoother |
| **GlobalAveragePooling** | Turns each pattern-map into a single number: "how much of this pattern is in the photo" |
| **Dense** | Combines all the pattern scores to make a decision |
| **Dropout** | Randomly switches off some neurons during training so the network can't memorise |
| **Softmax** | Converts the final scores into 38 probabilities that add up to 100% |

---

## 3. The dataset
**PlantVillage**: 54,305 real photos of leaves, labelled by plant-disease experts at Penn State University. Each photo shows one leaf on a plain background. It's the most widely used dataset in this field and was used by **Mohanty et al. (2016)**, the first reference on our slides.

- 14 crops, 38 classes (e.g. `Tomato___Early_blight`, `Apple___healthy`)
- **Imbalanced**: the largest class has 5,507 images and the smallest has 152

We also use **PlantDoc** (IIT Gandhinagar, 2020): ~2,600 messy, real-world photos from fields and the internet. We **don't train** on it; we only use it to test how well the model copes with real-farm conditions.

---

## 4. The procedure, step by step

### Step 1: Split the data (Notebook 1)
Like a student with a textbook, practice tests and a final exam:
- **Train 70%**: the textbook the model studies
- **Validation 15%**: practice tests after each study round, used to decide when to stop
- **Test 15%**: the final exam, opened only once at the very end

"**Stratified**" means each part keeps the same mix of classes, so even the rare classes appear in the exam.

### Step 2: Preprocess (Notebook 1)
1. **Resize** every photo to 224×224, because the network needs equal-size inputs
2. **Normalise**: divide pixel values by 255 so they're between 0 and 1. Small numbers help learning. (We do this inside the model, so the app can pass raw photos.)
3. **Augment**: during training, randomly flip, rotate, zoom and change contrast. The model sees a "new" version of each photo every round, so it learns the disease rather than the exact photo.

### Step 3: Train Model A, the basic CNN (Notebook 2)
We stack 5 conv blocks (32 → 64 → 128 → 256 → 256 filters), then a small decision part. Training settings:
- **Loss function** (sparse categorical cross-entropy): a score for "how wrong was the guess". Training tries to make it smaller.
- **Optimizer** (Adam): the method that nudges the network's numbers in the right direction after each batch
- **Learning rate** (0.001): how big each nudge is
- **Epoch**: one pass through all training images. We allow up to 25.
- **Batch size** (32): the model looks at 32 photos, then updates itself
- **Early stopping**: if the validation score hasn't improved for 5 epochs, stop and keep the best version
- **ReduceLROnPlateau**: if progress stalls, halve the learning rate to take smaller, more careful steps
- **Class weights**: mistakes on rare classes count more, so they aren't ignored

### Step 4: Train Model B, MobileNetV2 with transfer learning (Notebook 2)
**Transfer learning** = reusing knowledge. MobileNetV2 was already trained by Google on 1.4 million everyday photos (ImageNet), so it already knows edges, textures and shapes. We:
1. **Freeze** it (don't change its knowledge) and train only a new final layer for our 38 classes
2. **Fine-tune**: unfreeze its top 40 layers and train with a learning rate 100× smaller, so it adapts gently to leaves

It's like hiring an experienced photographer and teaching them plant diseases, instead of teaching someone from scratch. MobileNetV2 is also small and fast, built for mobile phones.

### Step 5: Evaluate (Notebook 2)
On the test set:
- **Accuracy**: % correct overall
- **Precision**: when it says "Late blight", how often is it actually Late blight?
- **Recall**: of all real Late blight leaves, how many did it catch?
- **F1-score**: one number combining precision and recall
- **Macro average**: average over all 38 classes equally, so small classes matter as much as big ones
- **Confusion matrix**: a 38×38 grid. Row = true class, column = predicted class. A perfect model has everything on the diagonal.

Example of why accuracy alone is misleading: if a model always said "Orange – citrus greening" (the biggest class), it would still get ~10% "accuracy" while being useless. Per-class recall would expose this immediately.

### Step 6: Real-world test (Notebook 3)
We run both models on PlantDoc field photos. Accuracy drops a lot, because the model got used to lab photos (plain background, even lighting). This is called **domain shift** and is the main honest limitation of the project. Mohanty et al. reported the same thing.

### Step 7: Web app (Notebook 3)
Built with **Gradio**: upload a photo and see the predicted disease, confidence and top-3 guesses. It gives a public link that works on phones.

---

## 5. Reading the training graphs
- **Both accuracy lines go up and stay close** → good learning
- **Training accuracy keeps rising but validation goes down** → **overfitting** (memorising). Early stopping, dropout and augmentation fight this.
- **Both stay low** → **underfitting** (model too simple or not trained long enough)
- In the MobileNetV2 graph, the dashed line marks where fine-tuning starts. There's often a small extra jump in accuracy after it.

---

## 6. Tools used
| Tool | Role |
|---|---|
| Python | programming language |
| Google Colab | free cloud computer with a GPU |
| TensorFlow / Keras | building and training the neural networks |
| tf.data | fast image loading pipeline |
| NumPy, pandas | numbers and tables |
| scikit-learn | data split, class weights, metrics |
| Matplotlib, Seaborn | graphs and confusion matrices |
| Gradio | web app |
| Git / GitHub | storing and sharing the code |
