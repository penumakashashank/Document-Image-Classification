# 🧾 Document Image Classification using CNN

This project is focused on classifying images into different categories using a Convolutional Neural Network (CNN). Instead of using large document datasets like RVL-CDIP, this implementation uses the **Oxford-IIIT Pet Dataset** (available in `torchvision`) to simulate document image classification.

---

## 📌 Objective

- Build a CNN model that classifies input images into multiple categories.
- Use evaluation metrics like accuracy, precision, recall, and F1 score to assess performance.

---

## 📂 Dataset

- **Oxford-IIIT Pet Dataset**
- 37 pet categories (dog and cat breeds)
- Automatically downloaded using `torchvision.datasets`
- Used to simulate document category classification (e.g., invoice, form, article)

---

## 🧠 Model Used

- **ResNet18** (pretrained on ImageNet)
- Final classification layer changed to output 37 categories
- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Training on 80% data, validating on 20%

---

## 🛠️ Tools & Libraries

- Python  
- PyTorch  
- Torchvision  
- Scikit-learn  
- Google Colab

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics are calculated using `classification_report` from `sklearn.metrics`.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Run the cells in order
3. The dataset will be downloaded automatically
4. After training for 5 epochs, a classification report is printed

---

## ✅ Results

- The model achieves good accuracy within a few epochs
- Performance can be improved further with more epochs or fine-tuning

---

## 📌 Sample Code Overview

```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 37)
