#CanonicalEquation solution was used for linear regression becasue 
# number of attributes was small
#

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

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

def normailzeData(data):
	z = (data - np.min(data))/(np.max(data)-np.min(data))


def main():
	whites = (pd.read_csv('winequality-white-reduced.csv', delimiter = ",", usecols = [1,2,3,4]))
	reds = (pd.read_csv('winequality-red-reduced.csv', usecols = [1,2,3,4]))

	whiteScores = classify(whites)
	redScores = classify(reds)

	whites = (whites.drop(['quality'], axis = 1))
	reds = (reds.drop(['quality'], axis = 1))

	whitePreds = predict(whites,canonicalCalculation(whites,whiteScores))
	redPreds = predict(reds,canonicalCalculation(reds,redScores))


	redsAccuracy =accuracy(redPreds,redScores)
	whitesAccuracy = accuracy(whitePreds,whiteScores)

	print("canonical solution")
	print('whites: {0}'.format(whitesAccuracy))
	print('reds: {0}'.format(redsAccuracy))

	print('SKLEARN solution')
	clf = linear_model.SGDRegressor(max_iter=1000,tol=0.001)
	reg1 = clf.fit(reds,redScores)
	reg2 = clf.fit(whites,whiteScores)

	print("whites: {0}".format(reg2.score(whites,whiteScores)))
	print("reds:{0}".format(reg1.score(reds,redScores))) # THIS PRODUCES A WACK NUMBER FOR SOMEREASON


	




if __name__ == '__main__':
	main()