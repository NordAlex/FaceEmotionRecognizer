import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from FecClassifer.Interfaces.Processor import Processor


class ProcessedImage:

    def __init__(self, original_image=None, face_images=[], face_coordinates=[]):
        self.original_image = original_image
        self.face_images = face_images
        self.face_coordinates = face_coordinates


class FecProcessor(Processor):

    def __init__(self):
        self._dataGeneratorWithAugmentation = None
        self._dataGenerator = None
        self.configure_processor()

    def configure_processor(self):
        self._dataGeneratorWithAugmentation = ImageDataGenerator(rescale=1. / 255)
        self._dataGenerator = ImageDataGenerator()

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'Config/haarcascade_frontalface_default.xml')
        self._facecasc = cv2.CascadeClassifier(filename)

    def process_folder_to_generator(self, directory_path, is_augmentation_required=True):
        generator = None
        if is_augmentation_required:
            generator = self._dataGeneratorWithAugmentation.flow_from_directory(
                directory_path,
                target_size=(48, 48),
                batch_size=64,
                color_mode="grayscale",
                class_mode='categorical')
        else:
            generator = self._dataGenerator.flow_from_directory(
                directory_path,
                target_size=(48, 48),
                batch_size=64,
                color_mode="grayscale",
                class_mode='categorical')

        return generator

    def process_raw_images(self, raw_images):
        result = []
        for image in raw_images:
            processed_images = []
            coordinates = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            print('Faces -', faces)
            if faces == ():
                print('Face not found')
                continue
            for face in faces:
                (x, y, w, h) = face
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                coordinates.append(face)
                processed_images.append(cropped_img)
                print(cropped_img)
            result.append(ProcessedImage(
                original_image=image, face_images=processed_images, face_coordinates=coordinates))
        return result

    def process_target_to_images(self, input_data, target):
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        processed_images = []
        for image_index in range(len(input_data)):
            original_image = input_data[image_index].original_image
            for face_index in range(len(input_data[image_index].face_coordinates)):
                (x, y, w, h) = input_data[image_index].face_coordinates[face_index]
                cv2.rectangle(original_image, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                max_index = int(np.argmax(target[image_index + face_index]))
                cv2.putText(original_image, emotion_dict[max_index], (x + 20, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                processed_images.append(original_image)

        return processed_images
