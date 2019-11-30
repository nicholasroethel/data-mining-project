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

def pairgrid_heatmap(x,y,**kws):
	cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
	plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

def visualizePairPlot(data,__name__):
	sns.pairplot(data,hue="quality")
	#this plots a figure per script automatically
	from os.path import realpath, basename 
	fig = plt.gcf()
	fig.savefig(__name__)
	plt.close()

def visualizePairGrid(data,__name__):
	g = sns.PairGrid(data)
	g.map_diag(plt.hist,bins=20)
	g.map_offdiag(pairgrid_heatmap,bins=20,norm=LogNorm())
	from os.path import realpath, basename 
	fig = plt.gcf()
	fig.savefig(__name__)
	plt.close()

def regplot(data,__name__):
	sns.scatterplot(x="predicted",y="actual",data = data)
	from os.path import realpath, basename 
	fig = plt.gcf()
	fig.savefig(__name__)
	plt.close()


def main():
	whites = (pd.read_csv('winequality-white-reduced.csv', delimiter = ",", usecols = [1,2,3,4]))
	reds = (pd.read_csv('winequality-red-reduced.csv', delimiter = ',' ,usecols = [1,2,3,4]))

	visualizePairPlot(whites,"whitePairPlot")
	visualizePairPlot(reds,"redsPairPlot")

	visualizePairGrid(whites,"whiteGridPlot")
	visualizePairGrid(reds,"redsGridPlot")

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