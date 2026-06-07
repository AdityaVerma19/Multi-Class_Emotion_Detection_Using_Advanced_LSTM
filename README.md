# Multi-Class Emotion Detection using Advanced LSTM & GloVe

This repository features a robust, deep-learning-based NLP pipeline for classifying text into five distinct emotional categories: **Happiness, Anger, Fear, Sadness, and Surprise**. The project leverages transfer learning through pre-trained word embeddings and a sophisticated sequence modeling architecture.

## 🚀 Project Overview

Traditional sentiment analysis often focuses on binary classification (positive/negative). This project dives deeper by identifying specific emotional nuances. By utilizing the **dair-ai/emotion** dataset and state-of-the-art techniques, the model achieves high accuracy and reliable inference performance.

### Key Features
- **Pre-trained Word Embeddings:** Utilizes GloVe (Global Vectors for Word Representation) 100-dimensional vectors to capture semantic relationships between words.
- **Bidirectional LSTM (Bi-LSTM):** Processes text sequences in both forward and backward directions to better understand context.
- **Custom Data Pipeline:** Implements a PyTorch `Dataset` and `DataLoader` for efficient memory management and batch processing.
- **Weighted Loss Function:** Uses class weights to handle dataset imbalances, ensuring rarer emotions are correctly identified.

---

## 🏗️ Architecture

### 1. Data Preprocessing
- **Tokenization:** Text is lowered, special characters are removed via regex, and strings are split into word tokens.
- **Vocabulary Building:** A custom vocabulary is built from the training corpus, filtering out rare tokens to reduce noise.
- **Padding & Truncating:** Sequences are standardized to a length of 30 tokens for batch processing.

### 2. Model Structure (`AdvancedLSTMEmotionClassifier`)
- **Embedding Layer:** Initialized with pre-trained GloVe weights. Set to `freeze=False` to allow fine-tuning during training.
- **Recurrent Layer:** A Bidirectional LSTM with 64 hidden units.
- **Dropout:** 30% dropout rate to prevent overfitting.
- **Fully Connected Layer:** Maps the 128-dimensional combined hidden states (64 forward + 64 backward) to the 5 target classes.

---

## 📊 Training Performance

The model was trained for 10 epochs using the **Adam Optimizer** and **Cross-Entropy Loss**. 
- **Final Loss:** ~0.02
- **Final Training Accuracy:** ~99.0%
- **Device Support:** Seamlessly switches between NVIDIA CUDA (GPU) and CPU.

---

## 🛠️ Requirements & Setup

- Python 3.x
- PyTorch
- NumPy
- Hugging Face `datasets`
- Matplotlib/Seaborn (for visualization)

### Installation
```bash
pip install torch datasets numpy
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

---

## 🔮 Inference

To predict emotions from custom text:
```python
text = "I am so surprised by how well this works!"
prediction, confidence = predict_advanced_emotion(text, model, vocab)
print(f"Emotion: {prediction} ({confidence*100:.1f}%)")
```
