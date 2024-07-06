import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Загрузка данных (предполагается, что изображения и метки хранятся в каталогах)
def load_data(data_dir, img_size=(28, 28)):
    images = []
    labels = []
    for label, letter in enumerate(os.listdir(data_dir)):
        letter_dir = os.path.join(data_dir, letter)
        for img_name in os.listdir(letter_dir):
            img_path = os.path.join(letter_dir, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img = np.array(img)
            images.append(img)
            labels.append(label)
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    labels = np.array(labels)
    return images, labels

data_dir = '123'
images, labels = load_data(data_dir)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Создание модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(os.listdir(data_dir)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test, y_test_cat))

# Тестирование модели
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test accuracy: {accuracy}')

# Сохранение модели
model.save('armenian_letter_recognition_model.h5')
