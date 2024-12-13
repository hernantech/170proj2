from search import Featuresearch
import time
import pandas as pd
import numpy as np
from search.nn_classifier import NNClassifier
from search.validator import Validator


def main():
    print("Welcome to Alex & Tim's Feature Selection Algorithm.")

    print("\nSelect the dataset you want to use.")
    print("1) Small Dataset (datasets/small-test-dataset.txt)")
    print("2) Large Dataset (datasets/large-test-dataset.txt)")
    print("3) Titanic Dataset (datasets/titanic clean.txt)")

    dataset_choice = int(input("Enter choice (1 or 2): "))

    if dataset_choice == 1:
        dataset_path = 'datasets/small-test-dataset.txt'
    elif dataset_choice == 2:
        dataset_path = 'datasets/large-test-dataset.txt'
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = int(input("Enter choice (1 or 2): "))

    if choice == 1:
        print("\n--- Running Forward Selection ---")
        searchobj = Featuresearch(data=dataset_path)
        searchobj.forward_selection()
        print("\n--- Forward Selection Completed ---")

    elif choice == 2:
        print("\n--- Running Backward Elimination ---")
        searchobj = Featuresearch(data=dataset_path)
        searchobj.backward_elimination()
        print("\n--- Backward Elimination Completed ---")

    else:
        print("Invalid choice. Please enter 1 or 2.")
        return

    print("\n--- Running NN Classifier ---")

    #load the dataset and split it into labels and features
    if dataset_choice == 3:  #titanic is in csv format?
        data = pd.read_csv(dataset_path, sep=',', header=0)
    else:
        data = pd.read_csv(dataset_path, delim_whitespace=True, header=None)

    labels = data.iloc[:, 0]    #first column is labels
    features = data.iloc[:, 1:]  #rest are features

    #using the best feature subset from feature selection
    print(f"Using Best Features for NN Classifier: {best_features}")

    #filter the features if subset is available
    selected_features = features.iloc[:, [i - 1 for i in best_features]]  # Convert to 0-indexed
    data_array = np.column_stack((labels.to_numpy(), selected_features.to_numpy()))  # Combine labels + features

    validator = Validator()
    classifier = NNClassifier()

    start_time = time.time()
    accuracy = validator.validate(best_features, classifier, data_array)
    end_time = time.time()

    print(f"\n--- NN Classifier Completed ---")
    print(f"Accuracy on {dataset_path}: {accuracy:.2f}%")
    print(f"Time for NN Classifier: {end_time - start_time:.2f} seconds")

    print("\n--- Final Results ---")
    print(f"Best Feature Subset: {best_features}")
    print(f"Classifier Accuracy: {accuracy:.2f}%")
    print(f"Total Time for Classification: {end_time - start_time:.2f} seconds")

    print("\nThank you for using Alex & Tim's Feature Selection Algorithm!")


main()
