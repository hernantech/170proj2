import pandas
import random

class Featuresearch:
    def __init__(self, data = None):
        if self.data == None:
            print("do something here with dummies and mocks")
        else:
            self.data = pandas.read_csv(data, delim_whitespace=True, header=None)
            self.num_features = len(self.data.columns) - 1 #index is off by one
            self.features = set(range(1, self.num_features + 1)) #index is off in the other direction by one
        
    def forward_selection(self):
        print("do things here")

    
    
    def dummy_evaluate(self, subset_of_features):
        return random.uniform(25, 85)
    
