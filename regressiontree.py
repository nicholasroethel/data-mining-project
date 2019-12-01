
#Adopted and modified based on Towards Data Science decision tree tutorial by Joachin Valente
#https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib.legend_handler import HandlerLine2D

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = 11
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
          
            num_left = [0] * self.n_classes_
          
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i-1]
                num_left[c] += 1
                num_right[c] -= 1
                
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


if __name__ == "__main__":
    
    #Consists total of 15 depths, from number range 1 to 30
    max_depths = np.linspace(1,30,15,endpoint=True)
    dataset = pd.read_csv(sys.argv[1])

    X = dataset.drop('quality', axis=1) #All features
    y = dataset.quality #Just target label(Quality)
    
    #80/20 random split on the dataset for training and testing
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

    #Training data trimming
    train_data = X_train.to_numpy()
    train_data = np.delete(train_data,0,1)
    train_target = y_train.to_numpy()

    
    #Obtain training data accuracies
    train_data_accuracies = []
    for max_depth in max_depths:

        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(train_data, train_target) 

        train_result = []       
        for i in train_data:
            train_result.append(clf.predict([i]))
        
        temp = []
        for sublist in  train_result:
            for item in sublist:
                temp.append(item)

        train_result = np.asarray(temp)
        accuracy = accuracy_score(train_target, train_result)
        train_data_accuracies.append(accuracy)

    #Obtain testing data accuracies
    test_data_accuracies = []
    for max_depth in max_depths:

        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(train_data, train_target) 

        test_data = X_test.to_numpy()
        test_data = np.delete(test_data,0,1)
        test_target = y_test.to_numpy()
        test_result = []
        for i in test_data:
            test_result.append(clf.predict([i]))
        
        temp = []
        for sublist in  test_result:
            for item in sublist:
                temp.append(item)

        test_result = np.asarray(temp)
        accuracy = accuracy_score(test_target, test_result)
        test_data_accuracies.append(accuracy)

    # Plotting the graph with corresponding labels and output formats
    line1, = plt.plot(max_depths, train_data_accuracies, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_data_accuracies, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.title(sys.argv[1])
    plt.ylabel('Accuracy Score')
    plt.xlabel('Tree Depth')
    if sys.argv[1] == "winequality-red-reduced.csv":
        plt.savefig('winequality-red-reduced.png', format='png')
    else:
        plt.savefig("winequality-white-reduced.png",format='png')

    print("Mean Absolute Error is: %1.2f"%mean_absolute_error(test_target, test_result))
   





   