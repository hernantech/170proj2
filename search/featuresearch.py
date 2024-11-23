import pandas
import random

class Featuresearch:
    def __init__(self, data=None, num_features=None):
        # Initialize with either data or a number of features
        if data is None:
            if num_features is None:
                raise ValueError("You must provide either 'data' or 'num_features'.")
            print(f"Using mock setup with {num_features} features.")
            self.num_features = num_features
            self.features = set(range(1, self.num_features + 1))  # Feature set {1, 2, ..., num_features}
        else:
            self.data = pandas.read_csv(data, delim_whitespace=True, header=None)
            self.num_features = len(self.data.columns) - 1  # Exclude the label column
            self.features = set(range(1, self.num_features + 1))  # Feature indices start at 1

    def forward_selection(self):
        current_features = set()
        not_used_features = self.features.copy()
        accuracy = self.dummy_evaluate(current_features)
        print(f"Using no features and random evaluation, I get an accuracy of {accuracy:.1f}%")
        print("Beginning search.")
        best_total_accuracy = accuracy
        best_total_features = set()

        while not_used_features:
            best_accuracy = -1
            best_feature = None

            # Try each unused feature
            for feature in not_used_features:
                candidate_features = current_features | {feature}
                accuracy = self.dummy_evaluate(candidate_features)
                print(f"Using feature(s) {sorted(candidate_features)} accuracy is {accuracy:.1f}%")

                # Compare for the best accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

            current_features.add(best_feature)
            not_used_features.remove(best_feature)

            if best_accuracy > best_total_accuracy:
                best_total_accuracy = best_accuracy
                best_total_features = current_features.copy()
            else:
                print("Warning! Accuracy has decreased!")

            print(f"Feature set {sorted(current_features)} was best, accuracy is {best_accuracy:.1f}%")

        print(f"\nFinished search!! The best feature subset is {sorted(best_total_features)}, "
              f"which has an accuracy of {best_total_accuracy:.1f}%")

    def backward_elimination(self):
        current_features = set(range(1, self.num_features + 1))
        best_accuracy = self.dummy_evaluate(current_features)
        print(f"Using all features and \"random\" evaluation, I get an accuracy of {best_accuracy:.1f}%")
        print("Beginning search.")

        while current_features:
            best_subset = None
            best_new_accuracy = -1

            # Test removing each feature
            for feature in current_features:
                test_features = current_features - {feature}
                accuracy = self.dummy_evaluate(test_features)
                print(f"Using feature(s) {sorted(test_features)} accuracy is {accuracy:.1f}%")

                # Track the best subset
                if accuracy > best_new_accuracy:
                    best_new_accuracy = accuracy
                    best_subset = test_features

            # Update current features and accuracy
            if best_subset is not None:
                current_features = best_subset
            if best_new_accuracy < best_accuracy:
                print("(Warning! Accuracy has decreased!)")
            best_accuracy = best_new_accuracy

            # Output for each iteration
            if current_features:
                print(f"Feature set {sorted(current_features)} was best, accuracy is {best_accuracy:.1f}%\n")

        print(f"Finished search!! The best feature subset is {sorted(current_features)}, "
              f"which has an accuracy of {best_accuracy:.1f}%")

    def dummy_evaluate(self, subset_of_features):
        return random.uniform(10, 90)
