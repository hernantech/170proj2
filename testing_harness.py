from search import featuresearch

def main():
    print("Welcome to Alex & Tim's Feature Selection Algorithm.")
    num_features = int(input("Please enter total num of features: "))
    
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
#    print("2) Backward Elimination")
    
    choice = int(input())
    if choice == 1:
        searchobj = featuresearch(num_features=num_features)
        searchobj.forward_selection()
    if choice == 2:
        searchobj = featuresearch(num_features=num_features)
        searchobj.backwards_elimination()