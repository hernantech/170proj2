import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

class NNClassifier:
    def __init__(self):

        self.training_data = None
        self.training_labels = None
        #for normalizing feature values
        self.scaler = MinMaxScaler()

    def train(self, data, labels):

        self.training_data = self.scaler.fit_transform(data)
        #storing the labels as numpy array
        self.training_labels = np.array(labels)

    def test(self, instance):

        #normalizing the test instance using the smae scaler
        normalized_instance = self.scaler.transform([instance])[0]

        #calculateing the euclidean distances using np to all training instances
        distances = np.linalg.norm(self.training_data - normalized_instance, axis = 1)
        #getting the index of the nearest neighbor
        n_n_index = np.argmin(distances)

        return self.training_labels[n_n_index]