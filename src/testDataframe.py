from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
import copy
import networkx as nx
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import scipy as sp

def constructionTableau(dataframe,nameFile):
    """
    Construction du tableau de données"""
    #convert dataframe to LATeX table
    print(dataframe.to_latex(index=False))


def loadDatasetratiovariete():
    """
    Load dataset
    """

    #Create a dataframe of the results
    #columns = ['Type',10,25,50,100,200]
    columns = ['Type',"10","25","50","100","200"]
    dataframe = pd.DataFrame(index=range(0,16),columns=columns)
    name = "AnalyseMUTAG.txt"
    #read text file
    file = open(name, "r")
    print(file)
    #read line
    lines = file.readlines()
    #for each line 
    compteur = 0
    for line in lines:
        if compteur<100:
            print(line)
            print(compteur)
            #split according to :
            line = line.split(":")
            #for the first part, split with space
            temp = line[0].split(" ")
            IndExtr = temp[1]
            #splut first part of temp with =   
            temp2 = temp[0].split("=")
            #this value is the value of K
            K = int(temp2[1])
            K=str(K)
            #split the second part of the line with space
            temp3 = line[1].split(" ")
            # take the 2 first digits of temp3
            temp3=temp3[1]
            temp4 = temp3[0:5]
            # convert to float
            temp4 = float(temp4)
            # convert to float the third part of the line
            temp5 = float(line[2])
            #keep only 2 digits after the comma
            temp5 = round(temp5,2)
            print(K)
            print(temp4)
            print(temp5)
            #add 14 rows to the dataframe

            #put the value of temp4 in the dataframe at the line 0 if compteur <8, 1 if compteur <15 etc
            nbLine = 2*(compteur%8)
            print(nbLine)
            print(K)
            dataframe[K][nbLine]= temp4
            dataframe[K][nbLine+1] = temp5
            print(dataframe)
            compteur = compteur+1
    #convert dataframe to LATeX table
    #keep only 2 digits after the comma
    dataframe = dataframe.round(2)

    print(dataframe.to_latex(index=False))
        




loadDatasetratiovariete()