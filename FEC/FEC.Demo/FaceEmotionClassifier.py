from FecClassifer.FecLoader import FecLoader
from FecClassifer.FecModel import FecModel
from FecClassifer.FecProcessor import FecProcessor

processor = FecProcessor()
loader = FecLoader(processor)
model = FecModel(loader)

model.load_model()
model.make_prediction('input/')
