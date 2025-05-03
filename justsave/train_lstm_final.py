import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load dataset
df = pd.read_csv("komentar_preprocessed_updated.csv")

# Label encoding
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentimen"])

# Tokenisasi dan padding
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
df["komentar_bersih"] = df["komentar_bersih"].fillna("").astype(str)
tokenizer.fit_on_texts(df["komentar_bersih"])
sequences = tokenizer.texts_to_sequences(df["komentar_bersih"])
padded = pad_sequences(sequences, maxlen=100)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# One-hot encoding
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Class weight
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Build LSTM
model = Sequential()
model.add(Embedding(max_words, 128, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weights_dict,
    verbose=1,
)

# Evaluate
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Metrics
print(classification_report(y_test, y_pred_labels, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))
