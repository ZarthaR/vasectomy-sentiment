import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ======== BACA DATA =========
df = pd.read_csv("komentar_preprocessed.csv")
df = df.dropna(subset=["komentar_bersih", "sentimen"])

# ======== ENCODE LABEL SENTIMEN =========
label_map = {"negatif": 0, "netral": 1, "positif": 2}
df["label"] = df["sentimen"].map(label_map)

# ======== TOKENISASI =========
max_words = 10000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["komentar_bersih"])

sequences = tokenizer.texts_to_sequences(df["komentar_bersih"])
padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

X = padded
y = to_categorical(df["label"], num_classes=3)

# ======== SPLIT TRAIN & TEST =========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== MODEL LSTM =========
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ======== TRAIN MODEL =========
weights = compute_class_weight(class_weight="balanced", classes=np.unique(df["label"]), y=df["label"])
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, class_weight=class_weights)


# ======== EVALUASI =========
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=["negatif", "netral", "positif"]))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# ======== SIMPAN MODEL (Opsional) =========
model.save("model_sentimen_lstm.h5")
