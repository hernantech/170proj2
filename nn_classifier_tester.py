import pandas as pd
import numpy as np
from search.nn_classifier import NNClassifier
from search.validator import Validator

def test_nn_classifier_with_dataset(dataset_path, feature_subset=None, dataset_name="Dataset"):
    print(f"\nTesting NNClassifier with the {dataset_name}...")
    
    # Load the dataset
    data = pd.read_csv(dataset_path, delim_whitespace=True, header=None)
    labels = data.iloc[:, 0]
    all_features = data.iloc[:, 1:]
    
    # If feature subset arg is provided, we select just those
    if feature_subset:
        # Converting to 0-based indexing for pandas (need it to avoid indexing error)
        features = all_features.iloc[:, [i-1 for i in feature_subset]]
        print(f"Selected features {sorted(feature_subset)}")
    else:
        features = all_features
        print("ok, using all features")
    
    print(f"Dataset shape: {len(labels)} instances, {len(features.columns)} features")
    
    # Basic classifier test
    print("\nPerforming basic classifier test...")
    classifier = NNClassifier()
    classifier.train(features, labels)
    
    test_instance = features.iloc[0].tolist()
    true_label = labels.iloc[0]
    predicted_label = classifier.test(test_instance)
    
    print(f"Test instance: {test_instance}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    
    # Validation test
    print("\nPerforming leave-one-out validation...")
    validator = Validator()
    
    # Convert full dataset to numpy array for validator
    data_array = data.to_numpy()  # This includes all columns
    
    accuracy = validator.validate(feature_subset, classifier, data_array)
    
    print(f"\nValidation Results for {dataset_name}:")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return accuracy

def test_nn_classifier():
    print("Starting NN Classifier tests...")
    
    # Testing with small dataset and required feature set
    small_accuracy = test_nn_classifier_with_dataset(
        "datasets/small-test-dataset.txt",
        {3, 5, 7},
        "Small Dataset"
    )
    
    # Same thing but with large
    large_accuracy = test_nn_classifier_with_dataset(
        "datasets/large-test-dataset.txt",
        {1, 15, 27},
        "Large Dataset"
    )
    
    # Final summary
    print("\nFinal Results Summary")
    print("=" * 50)
    print("Small Dataset (features {3, 5, 7}):")
    print(f"- Expected accuracy: ~89%")
    print(f"- Achieved accuracy: {small_accuracy:.1f}%")
    
    print("\nLarge Dataset (features {1, 15, 27}):")
    print(f"- Expected accuracy: ~94.9%")
    print(f"- Achieved accuracy: {large_accuracy:.1f}%")

if __name__ == "__main__":
    test_nn_classifier()