# Baby Cry Classification using Deep Learning

![baby-cry](https://img.shields.io/badge/Baby%20Cry-Classifier-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![DL](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-red)

This project aims to classify baby cries into categories like hunger, pain, discomfort, or tiredness using deep learning and audio signal processing. It helps build smarter baby monitoring systems.

---

## 🔍 Problem Statement

Recognizing different types of baby cries using deep learning models trained on audio spectrograms and MFCC features. Real-time detection can assist in infant care and early health interventions.

---

## 🧠 Methodology

The solution combines **signal processing** and **deep learning**:

### 🔉 Preprocessing:

* Convert `.wav` audio files into:

  * **Mel-Spectrograms** (for CNN/ViT pipeline)
  * **MFCCs** (for Bi-LSTM pipeline)

### 🧪 Models Used:

#### Pipeline 1:

* `ResNet-50` + `Vision Transformer (ViT)`
* Input: Mel-Spectrogram images
* Fusion via concatenated embeddings

#### Pipeline 2:

* `Bi-LSTM` model
* Input: MFCC time-series data

### 🔗 Fusion:

* Final classification layer takes combined outputs of both pipelines
* Softmax activation used for multi-class output

### 🧠 Loss Function:

* `Categorical Crossentropy`

### 📈 Metrics:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-Score

---

## 📁 Dataset

* Audio recordings of baby cries categorized into different emotional states
* Custom collected + open-source datasets

---

## 🛠️ Tech Stack

* Python 3.10+
* TensorFlow / Keras
* Librosa (for audio feature extraction)
* Matplotlib & Seaborn (visualizations)
* NumPy, Pandas

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/MohiteYash/baby
cd baby
```

### 2. Setup virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Preprocess the audio data

```bash
python preprocess.py
```

### 5. Train the model

```bash
python train.py
```

### 6. Evaluate and visualize results

```bash
python evaluate.py
```

---

## 📊 Sample Output

### Confusion Matrix:

* Visual representation included in the repo (see `/results/confusion_matrix.png`)

### Model Accuracy:

* Achieved **over 90% accuracy** on validation data

---

## 🧪 Experiments

* Compared different CNN backbones (ResNet18, ResNet50)
* Swapped ViT with AST for ablation
* Tested different MFCC settings

---

## 🙋‍♂️ Author

**Yash Mohite**
[GitHub Profile](https://github.com/MohiteYash)

Feel free to ⭐ this repo or open issues for collaboration!

---
