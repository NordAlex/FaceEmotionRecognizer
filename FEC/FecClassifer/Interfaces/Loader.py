class Loader:
    """
    Loader refers to download data determined
    in config.py
    """

    def __init__(self, processor=None):
        self._processor = processor

    def save_data(self, input_data, target_data):
        """
        Saves input and target data to disk
        Save path defined in config.py
        :return: nothing
        """
        pass

    def prepare_train_data(self):
        """
        Calls Processor class instance defined
        in Loader field _processor to transform train
        data to configured form by Processor
        :return:
        """
        pass

    def prepare_validation_data(self):
        """
        Calls Processor class instance defined
        in Loader field _processor to transform validation
        data to configured form by Processor
        :return:
        """
        pass

    def prepare_data(self, folder_name):
        """
        Calls Processor class instance defined
        in Loader field _processor to transform
        data to configured form by Processor
        :return:
        """
        pass