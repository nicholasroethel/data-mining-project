#CanonicalEquation solution was used for linear regression becasue 
# number of attributes was small

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error

WineScore = []
weight = []

def classify(data):
	return np.array(data.loc[:,'quality'])
	

def canonicalCalculation(data,scores): #w = inverse(X.T*X)*X.T*y
	return (np.array(np.linalg.inv(data.T@data)@data.T@scores))
	#take floor to be conservative

def predict(data,w): #y= wx + b b == bias will be set to 0
	y = np.array(w@data.T)
	return y.astype(int)

def accuracy(actual,pred):
	accuracy = (actual == pred)
	return (accuracy.mean())

def weights(data):
	return np.ones(1,data.shape[1])

def visualizePairPlot(data):
	sns.pairplot(data,hue="quality")
	#this plots a figure per script automatically
	from os.path import realpath, basename 
	s = basename(realpath(__file__))
	fig = plt.gcf()
	fig.savefig(s.split('.')[0])
	plt.show()

def visualizePairGrid(data):
	g = sns.PairGrid(data)
	g.map_diag(plt.hist,bins=20)
	from os.path import realpath, basename 
	s = basename(realpath(__file__))
	fig = plt.gcf()
	fig.savefig(s.split('.')[0])
	plt.show()


def main():
	whites = (pd.read_csv('winequality-white-reduced.csv', delimiter = ",", usecols = [1,2,3,4]))
	reds = (pd.read_csv('winequality-red-reduced.csv', usecols = [1,2,3,4]))

	# visualizePairPlot(whites)

	# visualizePairPlot(reds)

	whiteScores = classify(whites)
	redScores = classify(reds)
	
	whites = (whites.drop(['quality'], axis = 1))
	reds = (reds.drop(['quality'], axis = 1))

	whitePreds = predict(whites,canonicalCalculation(whites,whiteScores))
	redPreds = predict(reds,canonicalCalculation(reds,redScores))


	redsAccuracy =accuracy(redPreds,redScores)
	whitesAccuracy = accuracy(whitePreds,whiteScores)

	print("Mean Absolute Error")
	print("Whites: %1.2f"%mean_absolute_error(whiteScores, whitePreds))
	print("Reds: %1.2f"%mean_absolute_error(redScores, redPreds))
	print("")
	print("Accuracy:")
	print("Whites: %2d%%"%(whitesAccuracy*100))
	print("Reds: %2d%%"%(redsAccuracy*100))

	

if __name__ == '__main__':
	main()