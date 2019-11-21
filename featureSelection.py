# recursive feature elimination

# imports
import numpy
import pandas as pd
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def removeUnselectedReds():
	dataframe = read_csv(filepath_or_buffer="winequality-red.csv",delimiter=";") #removes the unselected features for the whites
	dataframe = dataframe.drop(columns = ["chlorides", "total sulfur dioxide", "pH", "free sulfur dioxide", "residual sugar", "density", "fixed acidity", "citric acid"])
	dataframe.to_csv("winequality-red-reduced.csv")

def removeUnselectedWhites():
	dataframe = read_csv(filepath_or_buffer="winequality-white.csv",delimiter=";") #removes the unselected features for the reds
	dataframe = dataframe.drop(columns =["alcohol", "free sulfur dioxide", "pH", "fixed acidity", "sulphates", "total sulfur dioxide", "chlorides", "citric acid"])
	dataframe.to_csv("winequality-white-reduced.csv")


def featureSelection(path): #selects the 3 best features

	# read in the data
	dataframe = read_csv(filepath_or_buffer=path,delimiter=";")

	# get the features
	features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
	X = dataframe.loc[:, features].values

	# get the target
	Y = dataframe.loc[:,['quality']].values

	#scale the data using a min max scaler
	mmScaler = preprocessing.MinMaxScaler()
	X = mmScaler.fit_transform(X)
	Y = mmScaler.fit_transform(Y)

	# use linear regression as the model
	lr = LinearRegression()

	# rank the features and pick 3
	rfe = RFE(lr, n_features_to_select=3)
	rfe.fit(X,Y.ravel())
	 
	#print the ranks
	print("Features ordered by rank (features with the rank 1 will be kept):")
	print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features)))


def main():
	print("\nComputing attribute reduction for reds...")
	featureSelection("winequality-red.csv")
	print("\nComputing attribute reduction for whites...")
	featureSelection("winequality-white.csv")
	removeUnselectedReds()
	removeUnselectedWhites()

if __name__ == '__main__':
	main()