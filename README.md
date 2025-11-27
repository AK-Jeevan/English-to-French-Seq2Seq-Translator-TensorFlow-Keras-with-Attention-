# 📘 English-to-French Seq2Seq Translator (TensorFlow/Keras — With Attention)

A complete, beginner-friendly **Encoder–Decoder LSTM translation model** that converts English sentences to French **using an Attention mechanism**.  
This project demonstrates every step of the Seq2Seq NLP pipeline — preprocessing, vocabulary building, attention-based decoding, teacher forcing, greedy inference, and saving model components.  
The entire implementation is fully commented for clarity.

---

## 📊 Dataset  
https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset

---

## ⭐ Features

- ✔️ Clean **Seq2Seq Encoder–Decoder architecture**  
- ✔️ **Bahdanau/Luong Attention** for improved context handling  
- ✔️ Custom **tokenization, vocabulary creation, and padding**  
- ✔️ **Teacher forcing** for efficient training  
- ✔️ **Greedy decoding** for inference  
- ✔️ Saves encoder, decoder, & full translation model  
- ✔️ Ideal for learners exploring Attention-based Seq2Seq  

---

## 🧠 Model Architecture

### **1. Encoder**
- Tokenized English input  
- Embedding layer  
- LSTM to generate hidden states at each timestep  
- Encoder outputs + final states passed to Attention + Decoder  

### **2. Attention Mechanism**
- Computes alignment scores between decoder state and encoder outputs  
- Generates context vector for each decoder timestep  
- Helps the model "focus" on relevant English words while producing French output  

### **3. Decoder**
- Embedding layer  
- LSTM cell  
- Attention applied at each timestep  
- Dense softmax layer predicts the next French token  

### **4. Inference**
Greedy decoding loop:
1. Feed `<start>`  
2. Apply attention to encoder outputs  
3. Predict next token  
4. Feed prediction back  
5. Stop at `<end>`  

---

## 🔧 Installation

pip install tensorflow keras numpy pandas nltk scikit-learn

Download NLTK tokenizer:
import nltk
nltk.download('punkt')

## 🎯 Why Attention?

Attention solves key limitations of classic Seq2Seq:

Enables the model to focus on relevant words during translation

Reduces information loss from long sentences

Improves fluency, accuracy, and alignment

Forms the conceptual basis for Transformers and modern NLP

## 🏗️ Extend the Project

You can further enhance this project with:

Luong or Bahdanau Attention variants

Beam search decoding

Subword tokenization (SentencePiece, BPE)

Larger datasets (OpenSubtitles, Tatoeba, etc.)

Transformer-based models

## 📄 License

MIT License

## 🤝 Contributions

Feel free to open:

Issues

Pull requests

Suggestions
