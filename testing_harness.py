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

    print("\nThank you for using Alex & Tim's Feature Selection Algorithm!")


main()
