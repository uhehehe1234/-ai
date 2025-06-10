import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# 데이터 불러오기
df = pd.read_csv("말투파일.csv")
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# 라벨 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# 토크나이저 준비
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>", char_level=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=30, padding='post')

# 모델 구성 및 학습
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, y, epochs=10000, verbose=1)

# 모델 저장
model.save("말투학습모델.h5")

# 토크나이저와 라벨 인코더 저장
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ 모델과 전처리기 저장 완료!")