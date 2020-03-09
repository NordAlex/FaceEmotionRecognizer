class Processor:
    """
    Transforms data to appropriate by model
    form
    """
    def __init__(self):
        self._processor_model = None

    def configure_processor(self):
        """
        Defines logic of the processor
        :return: nothing
        """
        pass

    def process_folder_to_generator(self, directory_path, is_augmentation_required=True):
        """
         WORKS WITH PREPROCESSED DATA ONLY!!!
         Refers for model data final preparation
                :return: generator
        """
        pass

    def process_raw_images(self, raw_images):
        """
        Refers for data initial processing
        :return: preprocessed data
        """
        pass

    def process_target_to_images(self, raw_images, coordinates, target):
        """
        :param coordinates: face coordinates for original images
        :param raw_images: original images
        :param target: target
        :return: none
        """
        pass