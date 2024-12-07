import numpy as np

class NN_Classifier:
    def __init__(self):
        self.training_data = None

    def train(self, data):
        self.training_data = data