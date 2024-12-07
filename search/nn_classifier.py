import numpy as np
#easier than manually doing the minmaxscaler
from sklearn.preprocessing import MinMaxScaler
import time

class NNClassifier:
    def __init__(self):

        self.training_data = None
        self.training_labels = None
        #for normalizing feature values
        self.min_vals = None
        self.max_vals = None
        #self.scaler = MinMaxScaler()
        #sklearn's version of it throws off the small dataset score for some reason

    def normalize(self, data):
            # For 2D array (training data)
            if len(data.shape) == 2:
                return (data - self.min_vals) / (self.max_vals - self.min_vals)
            # For 1D array (test instance)
            return (data - self.min_vals) / (self.max_vals - self.min_vals)


    def train(self, data, labels):
        # Convert to numpy array if needed
        data = np.array(data)
        # Store min and max for each feature
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        self.training_data = self.normalize(data)
        #self.training_data = self.scaler.fit_transform(data)
        #storing the labels as numpy array
        self.training_labels = np.array(labels)

    def test(self, instance):
        # Convert to numpy array and normalize
        instance = np.array(instance)
        normalized_instance = self.normalize(instance)
        #normalizing the test instance using the same scaler
        #normalized_instance = self.scaler.transform([instance])[0]

        #calculateing the euclidean distances using np to all training instances
        distances = np.linalg.norm(self.training_data - normalized_instance, axis = 1)
        #getting the index of the nearest neighbor
        n_n_index = np.argmin(distances)

        return self.training_labels[n_n_index]