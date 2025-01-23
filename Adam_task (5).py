from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import time

def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.,
        validation_split=0.2  # Разделение на обучающую и валидационную выборки
    )

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=32,  # Увеличиваем размер батча
        class_mode='sparse',
        seed=12345,
        subset='training'  # Используем подмножество для обучения
    )

    validation_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse',
        seed=12345,
        subset='validation'  # Используем подмножество для валидации
    )

    return train_datagen_flow, validation_datagen_flow

def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                         weights='imagenet',  # Используем предобученные веса на ImageNet
                         include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))  # Предполагается, что у нас 12 классов

    for layer in backbone.layers:
        layer.trainable = False  # Замораживаем слои бэкбона для начала

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=Adam(learning_rate=0.0001),  # Уменьшаем скорость обучения
                  metrics=['accuracy'])

    return model

def train_model(model, train_data, test_data, validation_data, epochs=10):
    start_time = time.time()  # Засекаем время начала обучения
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_steps=validation_data.samples // validation_data.batch_size,
        verbose=2
    )
    end_time = time.time()  # Засекаем время окончания обучения
    print(f"Training time: {end_time - start_time} seconds")
    
    return history
