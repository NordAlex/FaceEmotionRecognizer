from FecClassifer.Config.Config import FecConfig
from FecClassifer.FecProcessor import FecProcessor
from FecClassifer.Interfaces.Loader import Loader

import cv2
import glob
import uuid


class FecLoader(Loader):

    def prepare_train_data(self):
        directory = FecConfig.train_dir
        return self._processor.process_folder_to_generator(directory)

    def prepare_validation_data(self):
        directory = FecConfig.val_dir
        return self._processor.process_folder_to_generator(directory)

    def prepare_data(self, folder_name):
        ext = ['png', 'jpg', 'gif']
        files = []
        [files.extend(glob.glob(folder_name + '*.' + e)) for e in ext]

        raw_images = [cv2.imread(file) for file in files]
        processed_images = self._processor.process_raw_images(raw_images)
        return processed_images

    def save_data(self, input_data, target_data):
        result_folder = FecConfig.input_dir
        processed_images = self._processor.process_target_to_images(input_data, target_data)
        for processed_image in processed_images:
            filename = result_folder + str(uuid.uuid4()) + '.jpg'
            print(processed_image.shape)
            cv2.imwrite(filename, processed_image)
