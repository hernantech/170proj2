import pandas as pd
from search.nn_classifier import NNClassifier

def test_nn_classifier_with_small_dataset():
    print("Testing NNClassifier with the small dataset...")

    data = pd.read_csv("datasets/small-test-dataset.txt", delim_whitespace=True, header=None)

    #seperating class labels and features
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    #indication for successfully loaded data set
    print(f"The Dataset loaded successfully! \nFeatures:\n{features}\nLabels:\n{labels}")

    classifier = NNClassifier()
    classifier.train(features, labels)
    print("Classifier trained successfully!")

    #testing with a new instance and convert the list for compatibility
    test_instance = features.iloc[0].tolist()
    true_label = labels.iloc[0]
    predicted_label = classifier.test(test_instance)

    print(f"Test instance: {test_instance}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    test_nn_classifier_with_small_dataset()
