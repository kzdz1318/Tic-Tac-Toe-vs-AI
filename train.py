import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
import os

# دالة لتحميل ومعالجة البيانات
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    # استبدال القيم المفقودة (NaN) وغير الصالحة (مثل '?') بـ -1
    data = data.fillna(-1)
    data = data.replace('?', -1)  # استبدال '?' بـ -1
    # تحويل أعمدة الحركات إلى أعداد صحيحة
    for col in ['MOVE1', 'MOVE2', 'MOVE3', 'MOVE4', 'MOVE5', 'MOVE6', 'MOVE7']:
        data[col] = data[col].astype(int)
    return data

# دالة لتدريب النموذج أو تحديثه
def train_model(X, y, model_path="tic_tac_toe_model.h5", retrain=False):
    if retrain and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(7,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    model.save(model_path)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return model

# تحميل البيانات الأصلية
try:
    data = load_and_process_data("Tic tac initial results.csv")
except FileNotFoundError:
    print("خطأ: ملف 'Tic tac initial results.csv' غير موجود في المجلد!")
    exit(1)

# تحويل العمود CLASS إلى قيم رقمية
le = LabelEncoder()
data['CLASS'] = le.fit_transform(data['CLASS'])  # win=2, draw=0, loss=1
X = data[['MOVE1', 'MOVE2', 'MOVE3', 'MOVE4', 'MOVE5', 'MOVE6', 'MOVE7']].values
y = data['CLASS'].values

# تدريب النموذج الأولي
model = train_model(X, y)

# حفظ LabelEncoder
joblib.dump(le, "label_encoder.pkl")