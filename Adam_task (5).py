from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.optimizers import Adam

def load_train():
    # Загружаем данные Fashion MNIST
    (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
    
    # Предобработка данных: нормализация и изменение формы
    features_train = features_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    features_test = features_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return (features_train, target_train), (features_test, target_test)

def create_model(input_shape):
    model = Sequential()
    
    # Добавляем свёрточные слои с максимальным объединением
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Используем Dropout для предотвращения переобучения
    model.add(Dense(10, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001)  # Оптимизируем скорость обучения
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def train_model(model, train_data, test_data, batch_size=32, epochs=10):
    features_train, target_train = train_data
    features_test, target_test = test_data
    
    # Обучаем модель
    model.fit(features_train, target_train,
              validation_data=(features_test, target_test),
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True)

# Основной код
train_data, test_data = load_train()
model = create_model(input_shape=(28, 28, 1))
train_model(model, train_data, test_data)
