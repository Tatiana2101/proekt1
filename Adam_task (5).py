from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.
    )

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=32,  # Увеличен размер батча для ускорения обучения
        class_mode='sparse',
        seed=12345
    )

    return train_datagen_flow

def create_model(input_shape):
    backbone = ResNet50(input_shape=(150, 150, 3),
                         weights='imagenet',  # Используем предобученные веса на ImageNet
                         include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam',  
                  metrics=['accuracy'])

    return model

def train_model(model, train_data, test_data, epochs=10):
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_steps=test_data.samples // test_data.batch_size,
        verbose=2
    )
    return history
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
   



