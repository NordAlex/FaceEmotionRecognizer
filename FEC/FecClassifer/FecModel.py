from tensorflow_core.python.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from FecClassifer.Config.Config import ClassifierMode, FecConfig
from FecClassifer.FecProcessor import FecProcessor
from FecClassifer.Interfaces.Model import Model
from FecClassifer.FecLoader import FecLoader


class FecModel(Model):

    def __init__(self, loader):
        self._loader = loader

        self._num_train = 28709
        self._num_val = 7178
        self._batch_size = 64
        self._num_epoch = 1

        self.create_model()

    def create_model(self):
        self._model = Sequential()

        self._model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self._model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Flatten())
        self._model.add(Dense(1024, activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(7, activation='softmax'))
        self._model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6),
                            metrics=['accuracy'])

    def train_model(self):
        print('Load train data...')
        train_generator = self._loader.prepare_train_data()
        print('Load validation data...')
        validation_generator = self._loader.prepare_validation_data()
        model_info = self._model.fit_generator(
            train_generator,
            steps_per_epoch=self._num_train // self._batch_size,
            epochs=self._num_epoch,
            validation_data=validation_generator,
            validation_steps=self._num_val // self._batch_size)

    def evaluate_model(self):
        print('Load validation data...')
        evaluate_generator = self._loader.prepare_validation_data()

        model_info = self._model.evaluate(evaluate_generator,
                                          steps=self._num_train // self._batch_size,
                                          batch_size=self._batch_size,
                                          epochs=self._num_epoch)
        return model_info

    def save_model(self):
        print('Save model...')
        self._model.save_weights(FecConfig.model_file_name)

    def load_model(self):
        print('Load model...')
        self._model.load_weights(FecConfig.model_file_name)

    def make_prediction(self, input_folder):
        processed_images = self._loader.prepare_data(input_folder)
        print('Predicting data...')

        results = []
        for processed_image in processed_images:
            for face_image in processed_image.face_images:
                result = self._model.predict(face_image)
                results.append(result)

        print('Save results...')
        self._loader.save_data(processed_images, results)

