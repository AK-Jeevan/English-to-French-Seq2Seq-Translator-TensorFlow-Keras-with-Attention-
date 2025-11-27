# English to French Translation using Encoder-Decoder Architecture with Attention Mechanism
# This model learns to translate English sentences to French using LSTMs and Attention.

import numpy as np                          # numerical computations
import pandas as pd                         # data manipulation
import tensorflow as tf                     # deep learning framework
from tensorflow import keras                # high-level Keras API
from keras.models import Model              # functional model API
from keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate  # layer types
from keras.preprocessing.sequence import pad_sequences  # sequence padding utility
from keras.callbacks import EarlyStopping   # early stopping callback
from nltk.tokenize import word_tokenize     # tokenize text into words
import nltk                                 # natural language toolkit
from sklearn.model_selection import train_test_split  # train/test split utility

# Download required NLTK data (punkt tokenizer)
nltk.download('punkt', quiet=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# 1. Load the dataset
# ----------------------------
# Load CSV with two columns: English sentences and French translations
pairs = pd.read_csv(r"C:\Users\akjee\Documents\AI\NLP\NLP-DL\Encoder-Decoder\en-fr-with-Attention.csv")
print("Dataset shape:", pairs.shape)  # display number of sentence pairs
print(pairs.head())                     # display first few rows

# Separate English and French sentences into two lists
eng_sentences = pairs.iloc[:, 0].values  # first column (English)
fr_sentences = pairs.iloc[:, 1].values   # second column (French)

# ----------------------------
# 2. Tokenization and Vocabulary Building
# ----------------------------
def tokenize_and_build_vocab(sentences):
    """
    Tokenize sentences into words and build word-to-index mappings.
    
    Parameters:
      sentences: list of sentence strings
    
    Returns:
      tokenized: list of token lists (words per sentence)
      word2idx: dict mapping word -> integer id (includes <pad> and <unk>)
      idx2word: dict mapping integer id -> word
    """
    tokenized = []                          # store tokenized sentences
    vocab = {"<pad>": 0, "<unk>": 1}        # special tokens: pad (id 0), unknown (id 1)
    
    for sent in sentences:                  # iterate through each sentence
        tokens = word_tokenize(sent.lower())  # tokenize into words and lowercase
        tokenized.append(tokens)            # add tokenized sentence to list
        for token in tokens:                # iterate through words
            if token not in vocab:          # if word not already in vocab
                vocab[token] = len(vocab)   # assign new unique id
    
    # create reverse mapping (id -> word)
    idx2word = {idx: word for word, idx in vocab.items()}
    
    return tokenized, vocab, idx2word

# Tokenize both English and French sentences
eng_tok, eng_vocab, eng_idx2word = tokenize_and_build_vocab(eng_sentences)
fr_tok, fr_vocab, fr_idx2word = tokenize_and_build_vocab(fr_sentences)

# Print vocabulary sizes
print("English vocab size:", len(eng_vocab))  # number of unique English words
print("French vocab size:", len(fr_vocab))    # number of unique French words

# ----------------------------
# 3. Encode sentences to integer sequences and pad
# ----------------------------
def encode_sequences(tokenized_sentences, word2idx, max_len):
    """
    Convert tokenized sentences to padded integer sequences.
    
    Parameters:
      tokenized_sentences: list of token lists
      word2idx: dict mapping word -> integer id
      max_len: maximum sequence length (pad/truncate to this)
    
    Returns:
      sequences: numpy array of shape (num_sentences, max_len) with integer ids
    """
    sequences = []                          # store encoded sequences
    
    for tokens in tokenized_sentences:      # iterate through tokenized sentences
        # convert words to ids using word2idx, use <unk> (id 1) for unknown words
        sequence = [word2idx.get(token, 1) for token in tokens]
        sequences.append(sequence)          # add encoded sequence to list
    
    # pad all sequences to max_len (right-pad with 0s if shorter, truncate if longer)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    return padded  # return as numpy array

# Determine max sequence length (use a reasonable fixed value)
max_len = 10

# Encode English and French sentences to integer arrays
X_eng = encode_sequences(eng_tok, eng_vocab, max_len)  # shape: (num_pairs, max_len)
Y_fr = encode_sequences(fr_tok, fr_vocab, max_len)     # shape: (num_pairs, max_len)

print("Encoded English shape:", X_eng.shape)  # (num_pairs, 10)
print("Encoded French shape:", Y_fr.shape)    # (num_pairs, 10)

# ----------------------------
# 4. Train/Validation split
# ----------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X_eng, Y_fr, test_size=0.2, random_state=42  # 80% train, 20% validation
)
print("Train set size:", X_train.shape[0])      # number of training examples
print("Validation set size:", X_val.shape[0])   # number of validation examples

# ----------------------------
# 5. Build Encoder-Decoder Model with Attention
# ----------------------------

# Hyperparameters
latent_dim = 128                            # LSTM hidden units
embedding_dim = 64                          # word embedding dimension
epochs = 30                                 # training epochs
batch_size = 32                             # batch size

# ===== ENCODER =====
# Encoder Input: English sentence (integer sequence)
encoder_input = Input(shape=(max_len,), name="encoder_input")

# Embedding Layer: Convert integer ids to dense vectors
encoder_embedding = Embedding(
    input_dim=len(eng_vocab),               # vocab size (number of unique words)
    output_dim=embedding_dim,               # embedding dimension
    mask_zero=True,                         # mask padding (zeros) in LSTM processing
    name="encoder_embedding"
)(encoder_input)

# LSTM Layer: Process sequence and produce context vectors
# return_state=True: also return final hidden and cell states
# return_sequences=True: return outputs at all time steps (needed for attention)
encoder_lstm = LSTM(
    units=latent_dim,                       # number of LSTM units
    return_state=True,                      # return final states (h, c)
    return_sequences=True,                  # return all time step outputs
    name="encoder_lstm"
)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Store encoder states as initial decoder states
encoder_states = [state_h, state_c]         # list of final hidden and cell states

# ===== DECODER =====
# Decoder Input: French sentence (integer sequence, shifted by 1 for teacher forcing)
decoder_input = Input(shape=(max_len,), name="decoder_input")

# Embedding Layer: Convert French integer ids to dense vectors
decoder_embedding = Embedding(
    input_dim=len(fr_vocab),                # vocab size (French)
    output_dim=embedding_dim,               # embedding dimension
    mask_zero=True,                         # mask padding
    name="decoder_embedding"
)(decoder_input)

# LSTM Layer: Process French sequence with initial states from encoder
# return_state=True: return final states
# return_sequences=True: return outputs at all time steps (for attention)
decoder_lstm = LSTM(
    units=latent_dim,                       # same units as encoder
    return_state=True,                      # return states
    return_sequences=True,                  # return all outputs
    name="decoder_lstm"
)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states            # initialize with encoder's final states
)

# ===== ATTENTION =====
# Attention Layer: Compute attention weights between decoder outputs and encoder outputs
attention_layer = Attention(name="attention")
attention_output = attention_layer([decoder_outputs, encoder_outputs])

# Concatenate: Merge attention output with decoder outputs
# axis=1 means concatenate along the feature dimension
merged = Concatenate(axis=-1, name="concat")([decoder_outputs, attention_output])

# Dense Layer: Project merged vectors down to a manageable size
dense = Dense(latent_dim, activation='relu', name="dense")(merged)

# Output Layer: Softmax over French vocabulary to predict next word
# output shape: (batch_size, max_len, len(fr_vocab))
decoder_output = Dense(len(fr_vocab), activation='softmax', name="output")(dense)

# ===== MODEL INSTANTIATION =====
# Functional model: takes encoder_input and decoder_input, outputs predictions
model = Model([encoder_input, decoder_input], decoder_output)

# Compile: specify optimizer, loss, and metrics
model.compile(
    optimizer='adam',                       # Adam optimizer
    loss='sparse_categorical_crossentropy', # loss for integer targets
    metrics=['accuracy']                    # track accuracy during training
)

# Display model architecture
model.summary()

# ----------------------------
# 6. Train the model with Early Stopping
# ----------------------------
# Early stopping: stop training if validation loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',                     # quantity to monitor
    patience=5,                             # epochs with no improvement before stopping
    restore_best_weights=True,              # restore weights from best epoch
    verbose=1                               # print messages
)

# Train the model
history = model.fit(
    [X_train, Y_train],                     # inputs: encoder input and decoder input
    Y_train,                                # targets: French sequences (decoder outputs)
    validation_data=([X_val, Y_val], Y_val),  # validation set
    epochs=epochs,                          # max epochs
    batch_size=batch_size,                  # batch size
    callbacks=[early_stopping],             # apply early stopping
    verbose=1                               # print progress
)

# ----------------------------
# 7. Plot training history
# ----------------------------
import matplotlib.pyplot as plt             # plotting library

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')      # training loss
plt.plot(history.history['val_loss'], label='Validation Loss')  # validation loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Training History')
plt.show()

# ----------------------------
# 8. Translate function (inference)
# ----------------------------
def translate(model, eng_sentence, eng_vocab, fr_vocab, fr_idx2word, max_len):
    """
    Translate an English sentence to French using the trained model.
    
    Parameters:
      model: trained encoder-decoder model
      eng_sentence: English sentence string
      eng_vocab: English word-to-index dictionary
      fr_vocab: French word-to-index dictionary
      fr_idx2word: French index-to-word dictionary
      max_len: max sequence length
    
    Returns:
      translated: translated French sentence string
    """
    # Tokenize and encode English sentence
    tokens = word_tokenize(eng_sentence.lower())
    sequence = [eng_vocab.get(token, 1) for token in tokens]
    X = pad_sequences([sequence], maxlen=max_len, padding='post')
    
    # Initialize decoder input as all zeros (can also use French start token)
    decoder_input = np.zeros((1, max_len))
    
    # Predict: model produces probability distribution over French vocab
    predictions = model.predict([X, decoder_input], verbose=0)
    
    # Decode predictions: pick most likely word at each position
    translated = []
    for pred_seq in predictions:            # iterate through predicted sequences
        for pred_step in pred_seq:          # iterate through time steps
            idx = np.argmax(pred_step)      # get index of highest probability
            if idx != 0:                    # skip padding (id 0)
                translated.append(fr_idx2word.get(idx, "<unk>"))  # convert id to word
    
    return " ".join(translated).strip()

# ----------------------------
# 9. Test translation
# ----------------------------
# Translate a few English sentences
test_sentences = [
    "hello world",
    "how are you",
    "good morning"
]

print("\n--- Translation Examples ---")
for eng_sent in test_sentences:
    fr_translation = translate(model, eng_sent, eng_vocab, fr_vocab, fr_idx2word, max_len)
    print(f"English: {eng_sent}")
    print(f"French:  {fr_translation}\n")

# ----------------------------
# 10. Save the model
# ----------------------------
model.save("en_fr_attention_model.h5")
print("Model saved as en_fr_attention_model.h5")
