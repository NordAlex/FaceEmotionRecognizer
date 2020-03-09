class Model:
    """
    Main prediction logic interface
    """
    def __init__(self):
        self._model = None
        self._loader = None
        self.create_model()

    def create_model(self):
        """
        This function creates instance of
        Keras Sequential model
        :return: nothing
        """
        pass

    def train_model(self):
        """
        Train model with Loader instance
        :return: nothing
        """
        pass

    def evaluate_model(self):
        """
        Connects to Loader instance and
        evaluate model by ____ metrics.
        :return: ___ metrics score
        """
        pass

    def save_model(self):
        """
        Saves model to ___ file.
        :return: nothing
        """
        pass

    def load_model(self):
        """
        Loads model
        :return:
        """
        pass

    def make_prediction(self, input_folder):
        """
        Uses _model field to call specific
        prediction functions from Keras
        :return: probability of class
        """
        pass