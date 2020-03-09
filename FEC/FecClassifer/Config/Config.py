from enum import Enum


class ClassifierMode(Enum):
    TRAIN = 1,
    DISPLAY = 2,
    EVALUATE = 3


class FecConfig:

    # Define data generators
    train_dir = '../../Data/Faces/train'
    val_dir = '../../Data/Faces/test'
    target_dir = 'input/'
    input_dir = 'target/'

    model_file_name = 'model.h5'


