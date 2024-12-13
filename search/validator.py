import numpy as np
import time
from search.nn_classifier import NNClassifier

class Validator:
    def __init__(self):
        self.classifier = NNClassifier()
        
    def validate(self, feature_subset, classifier, data):
        #data needs to be a np.ndarray dataset where first column is class labels
        #feature_subset needs to be a set type with feature indices to use (1-based indexing)
        #classifier is an instance passed through
        #Returns accuracy as a percentage and time it takes
        labels = data[:, 0]
        
        # Debug prints
        print(f"Data shape before feature selection: {data.shape}")
        print(f"Feature subset: {sorted(feature_subset)}")
        
        features = data[:, [i for i in feature_subset]]
        
        # More debug prints
        print(f"Data shape after feature selection: {features.shape}")
        print(f"First row of selected features: {features[0]}")
        
        total_instances = len(data)
        correct_predictions = 0
        
        print(f"\nStarting leave-one-out validation using features {sorted(feature_subset)}")
        
        for i in range(total_instances):
            # Leave one out as test data
            test_instance = features[i]
            test_label = labels[i]
            
            # Use rest for training
            train_features = np.delete(features, i, axis=0)
            train_labels = np.delete(labels, i, axis=0)
            
            # Time the train/test cycle
            classifier.train(train_features, train_labels)
            
            predicted_label = classifier.test(test_instance)
            
            if predicted_label == test_label:
                correct_predictions += 1
                
        accuracy = (correct_predictions / total_instances) * 100
        
        print(f"\nValidation complete:")
        print(f"Correct predictions: {correct_predictions}/{total_instances}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        return accuracy