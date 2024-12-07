import numpy as np
from sklearn.preproccessing import MinMaxScaler
import time

class NN_Classifier:
    def __init__(self):
        self.training_data = None
        self.training_labels = None
        #for normalizing feature values
        self.scaler = MinMaxScaler()

    def train(self, data, labels):
        self.training_data = self.scaler.fit_transform(data)
        #storing the labels as numpy array
        self.training_labels = np.array(labels)