# Функция загрузки данных
def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.0,
        validation_split=0.2  # Разделим данные на обучающую и валидационную выборки
    )

    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',  # Используем часть для обучения
        seed=12345
    )

    test_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='validation',  # Используем часть для валидации
        seed=12345
    )

    return train_datagen_flow, test_datagen_flow

# Функция создания модели
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Замораживаем слои базовой модели

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='softmax')  # Предполагаем, что у нас 5 классов фруктов
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Функция обучения модели
def train_model(model, train_data, test_data, batch_size=None, epochs=10,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    history = model.fit(train_data,
                        validation_data=test_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=2)

    return model

# Основной код для выполнения
if __name__ == "__main__":
    path = "path_to_your_fruit_dataset"  # Замените на путь к вашему набору данных
    train_data, test_data = load_train(path)
    
    model = create_model()
    
    # Обучаем модель
    model = train_model(model, train_data, test_data, epochs=20)  # Увеличьте количество эпох при необходимости

    # Проверка точности на тестовой выборке
    test_loss, test_acc = model.evaluate(test_data)
    print(f'Test accuracy: {test_acc * 100:.2f}%')
