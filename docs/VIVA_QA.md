# 🎤 Viva preparation: likely questions and simple answers

## About the project
**Q1. What does your project do?**
It takes a photo of a plant leaf and predicts its class with a confidence score. There are 38 classes across 14 crops: 26 diseases plus 12 "healthy" classes. We use CNNs and compare a model built from scratch with a transfer-learning model (MobileNetV2).

**Q2. Why is this useful?**
Early disease detection helps farmers act sooner. Manual inspection is slow and inconsistent, and an app gives a fast first-level opinion. It doesn't replace experts.

**Q3. Which dataset did you use and why?**
PlantVillage: 54,305 expert-labelled leaf images, 38 classes. It's the standard benchmark in this field and was used by Mohanty et al. (2016), our main reference, so our results are comparable with published work.

**Q4. Is the dataset balanced?**
No. Classes range from 152 images (Potato healthy) to 5,507 (Orange citrus greening). We handled this with a stratified split, class weights during training, and macro-averaged and per-class metrics.

## Data & preprocessing
**Q5. Why split into train, validation and test?**
Train is for learning, validation is for tuning and early stopping, and test is used once for an unbiased final score. If we tuned on the test set, the score would be over-optimistic.

**Q6. What is a stratified split?**
Each split keeps the same class proportions as the full dataset, so rare classes appear in all three.

**Q7. Why resize to 224×224?**
The network needs a fixed input size, and 224×224 is the standard size MobileNetV2 was designed for. The originals are 256×256, so little detail is lost.

**Q8. Why normalise?**
Pixel values 0–255 are large. Scaling to 0–1 (or −1 to 1 for MobileNetV2) keeps the numbers small and training stable. We put normalisation inside the model so the deployed app can feed raw images.

**Q9. What is data augmentation and why use it?**
Random flips, rotations, zooms and contrast changes applied only during training. It creates variety, reduces overfitting and makes the model robust to leaf orientation and lighting.

## CNN theory
**Q10. Why CNN and not a normal (dense) neural network?**
A dense network would connect every pixel to every neuron (150,528 inputs), which means far too many parameters, and it ignores the fact that nearby pixels are related. CNNs use small shared filters that detect local patterns anywhere in the image, giving fewer parameters and better results.

**Q11. What does a convolution layer do?**
It slides small learnable filters (3×3) over the image. Each filter produces a feature map showing where its pattern (an edge, a spot) occurs.

**Q12. What is pooling?**
Max pooling takes the maximum in each 2×2 region, halving width and height. It reduces computation and makes features slightly position-independent.

**Q13. What is ReLU?**
An activation function: output = max(0, x). It adds non-linearity so the network can learn complex patterns, and it trains faster than sigmoid/tanh.

**Q14. What is softmax?**
It converts the final 38 scores into probabilities that sum to 1. The highest probability is the prediction, and its value is the confidence.

**Q15. What is batch normalisation?**
It normalises the outputs of a layer within each batch, which stabilises and speeds up training.

**Q16. What is dropout?**
It randomly sets a fraction of neurons to zero during training (40% and 30% in our model), forcing the network not to rely on any single neuron. This reduces overfitting.

**Q17. What is global average pooling?**
It averages each feature map into one number. It replaces a huge Flatten layer, uses far fewer parameters, and reduces overfitting.

## Training
**Q18. What loss function and optimiser did you use?**
Sparse categorical cross-entropy (the standard for multi-class problems with integer labels) and the Adam optimiser (adaptive learning rate per parameter, fast convergence).

**Q19. What are epoch, batch size and learning rate?**
An epoch is one full pass through the training data. Batch size is the number of images per update (32). The learning rate is the step size of each update (0.001, and 0.00001 for fine-tuning).

**Q20. How did you prevent overfitting?**
Data augmentation, dropout, batch normalisation, early stopping (patience 5, restoring the best weights) and validation monitoring.

**Q21. What is overfitting? How do you see it?**
The model memorises training data and does worse on new data. On the curves, training accuracy keeps rising while validation accuracy stalls or falls.

## Transfer learning
**Q22. What is transfer learning?**
Reusing a model pre-trained on a large dataset (ImageNet, 1.4M images) for a new task. Its early layers already detect general features, so we only adapt the later layers. This gives better accuracy with less training.

**Q23. Why MobileNetV2?**
It's lightweight (~2.3M parameters) and fast, designed for mobile devices, which suits a farmer's phone app, while still being accurate. It uses depthwise-separable convolutions and inverted residual blocks.

**Q24. Why two stages (freeze, then fine-tune)?**
If we unfroze everything at once, the large errors from the random new head would destroy the pretrained features. So we first train the head, then gently fine-tune the top layers with a 100× smaller learning rate.

**Q25. Why keep BatchNorm layers frozen during fine-tuning?**
Their saved statistics come from ImageNet. Updating them with small batches makes training unstable, so freezing them is standard practice (recommended in the Keras transfer-learning guide).

## Evaluation
**Q26. Why not just accuracy?**
With imbalanced classes, accuracy can hide poor performance on rare classes. Precision, recall, F1 (macro) and the confusion matrix show per-class behaviour.

**Q27. Precision vs recall?**
Precision: of the leaves predicted as disease X, how many really are X (avoids false alarms). Recall: of the real X leaves, how many were found (avoids missed diseases). For disease detection, recall is especially important, because a missed disease can spread.

**Q28. Macro vs weighted average?**
Macro gives every class equal weight. Weighted gives each class weight by its number of samples. Macro is stricter for imbalanced data.

**Q29. What does the confusion matrix show?**
Rows are true classes and columns are predicted classes. The diagonal is correct predictions; off-diagonal cells show which classes get confused, usually diseases of the same crop with similar symptoms.

**Q30. What is top-3 accuracy?**
The share of images where the true class is among the model's 3 most likely predictions. It's useful for an app that shows the top 3 guesses.

## Results & limitations
**Q31. Which model performed better and why?**
Fill in from your results (usually MobileNetV2): it starts from ImageNet knowledge, converges faster and generalises better.

**Q32. Your accuracy is very high. Is the model really that good?**
On PlantVillage, yes, but these are lab images with uniform backgrounds. Our PlantDoc field test shows a large accuracy drop on real-world photos (domain shift), which Mohanty et al. also observed. So the model is a strong prototype but needs field data for real deployment.

**Q33. How could you improve it?**
Train on mixed lab and field data, detect and crop the leaf before classifying, use stronger augmentation (backgrounds, blur, lighting), collect local farm images, try other models (EfficientNet, ResNet50), and deploy as a TensorFlow Lite mobile app.

**Q34. What happens if someone uploads a photo of a plant not in the dataset?**
The model still picks one of the 38 classes, because softmax always outputs one. That's a limitation. Our app warns when confidence is low. A future fix is an "unknown" class or out-of-distribution detection.

## Tools
**Q35. Why TensorFlow/Keras?** It has a high-level, readable API, built-in pretrained models, an efficient tf.data pipeline, and easy saving and deployment.
**Q36. Why Google Colab?** It provides a free GPU. Training on a CPU would take many hours.
**Q37. What is Gradio?** A Python library that turns a function into a web interface with a few lines of code.
