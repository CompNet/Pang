# -*- coding: utf-8 -*- 
import copy, random, itertools
from tools import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
#matplotlib.use("pdf")
import matplotlib.pyplot as plt
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
import graphviz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def read_Sizegraph(fileName):
    """Read the number of graphs in a file.
    Input: fileName (string) : the name of the file
    Ouptut: TAILLE (int) : the number of graphs in the file"""
    
    file = open(fileName, "r")
    nbGraph=0
    for line in file:
       if line[0]=="t":
            nbGraph=nbGraph+1
    return nbGraph

def load_graphs(fileName,TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    nom (list of string) : the list of names of each graph)"""
    
    nbV=[]
    nbE=[]
    numbers = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
                nbV.append(len(labelVertices[compteur-1]))
                nbE.append(len(labelEdges[compteur-1]))
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            #for j in range(1,len(b)-1):
            for j in range(1,len(b)-1):
                if not(b[j]=="#"):
                    val = b[j]
                    val = re.sub("\n","",val)
                    val = int(val)
                    temp.append(val)
            numbers[numero]=temp  
    noms.append(temptre)
    nbV.append(len(labelVertices[compteur-1]))
    nbE.append(len(labelEdges[compteur-1]))
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            #tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,noms

def load_patterns(fileName,TAILLE):
    """ This function loads the post-processed patterns, i.e with occurences.
    fileName (string) : the name of the file
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of patterns
            numbers (list of list of int) : the list of occurences of each graph
            numberoccurences (list of list of int) : the list of occurences of each pattern
    """
    numbers = []
    numberoccurences = []
    numbercoverage = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
        numberoccurences.append([])
        numbercoverage.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            tempOccu = []
            tempCoverage = []
            for j in range(1,len(b)-1):
                val = b[j]
                val = re.sub("\n","",val)
                if not(val=="#" or val==""):
                    val = str(val).split("/")
                    numeroGraph = int(val[0])
                    val = str(val[1]).split(":")
                    coverage=1
                    if len(val)>1:
                        coverage = float(val[1])
                    occurences = int(float(val[0]))
                    temp.append(numeroGraph)
                    tempOccu.append(occurences)
                    tempCoverage.append(coverage)
            numbers[numero]=temp 
            numberoccurences[numero]=tempOccu
            numbercoverage[numero]=tempCoverage
    noms.append(temptre)
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,numberoccurences


def computeScoreMono(keep,labels,id_graphs,TAILLEPATTERN):
    """ this function computes the discrimination score of each pattern using the Binary Criterion"""
    """ Input : keep (list of int) : the list of patterns to keep
                labels (list of int) : the list of labels of the graphs
                id_graphs (list of list of int) : the list of occurences of each pattern
                TAILLEPATTERN (int) : the number of patterns
        Output : diffDelta (list of int) : the list of discrimination scores of each pattern
    """
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                            aa=id_graphs[i][j]
                            if labels[aa]==0:
                                        diffDelta[i]=diffDelta[i]+1
                            elif labels[aa]==1:
                                        diffDelta[i]=diffDelta[i]-1
        diffDelta[i] = abs(diffDelta[i])
    print(diffDelta)
    return diffDelta
    
###################################
            
def computeScoreOccurences(keep,labels,id_graphs,occurences,TAILLEPATTERN):
    """ this function computes the discrimination score of each pattern using the Integer Criterion"""
    """ Input : keep (list of int) : the list of patterns to keep
                occurences (list of list of int) : the list of number occurences of each pattern in each graph
                labels (list of int) : the list of labels of the graphs
                id_graphs (list of list of int) : the list of graphs containing of each pattern
                TAILLEPATTERN (int) : the number of patterns
        Output : diffDelta (list of int) : the list of discrimination scores of each pattern
    """
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    diffCORK=np.zeros(TAILLEPATTERN)
    tailleRed =[]
    tailleNonRed=[]
    for i in range(TAILLEPATTERN):
        tailleRed.append(0)
        tailleNonRed.append(0)
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                    if labels[id_graphs[i][j]]==0:
                            diffDelta[i]=diffDelta[i]+occurences[i][j]
                            tailleNonRed[i]=tailleNonRed[i]+occurences[i][j]
                    elif labels[id_graphs[i][j]]==1:
                            diffDelta[i]=diffDelta[i]-occurences[i][j]
                            tailleRed[i]=tailleRed[i]+occurences[i][j]
        diffDelta[i] = abs(diffDelta[i])
    print(diffDelta)
    return diffDelta
    
def readLabels(fileLabel):
    """ this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1
        
    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""
    
    file=open(fileLabel,"r")
    labels = []
    numero=0
    for line in file:
        line = str(line).split("\t")[0]
        if int(line)==-1:
            labels.append(0)
        elif int(line)>-1:
            labels.append(min(int(line),1))
        numero=numero+1
    return labels

def KVector(keep,K,diff,id_graphs,numberoccurences,LENGTHGRAPH):
    """ this fuction creates the vectorial representation of the graphs
        Input : K (int) : the number of patterns to keep
        diff (list of int) : the list of discrimination scores of each pattern
        id_graphs (list of list of int) : the list of graphs containing of each pattern
        numberoccurences (list of list of int) : the list of number occurences of each pattern in each graph
        LENGTHGRAPH (int) : the number of graphs
        labels (list of int) : the list of labels of the graphs
        
        
        Output : X (list of list of int) : the vectorial representation of the graphs
        Y (list of int) : the list of labels of the graphs""" 
    keepPatterns = []
    for i in range(K):
        if sum(diff)==0:
            break
        bestScore = np.max(diff)
        bestPattern = np.argmax(diff)
        keepPatterns.append(bestPattern)
        diff[bestPattern]=0
    vectorialRep = []
    newLabels = []
    for j in range(LENGTHGRAPH):#330
            vectorialRep.append([])
            for k in keepPatterns:
                if j in id_graphs[k]:
                    for t in range(len(id_graphs[k])):
                        if id_graphs[k][t]==j:
                            if numberoccurences==None:
                                occu=1
                            else:
                                occu = numberoccurences[k][t]
                    vectorialRep[j].append(occu)
                else:
                    vectorialRep[j].append(0)
    X = vectorialRep
    return X

import sys, getopt

def graphKeep(PatternRed,fileLabel,nbV,nbE,MegaRES):
    file2=open("Enlever.txt","r")
    file=open(fileLabel,"r")
    nokeep=[]
    for line in file2:
        nokeep.append(int(line))
    labels = []
    xxV=[]
    yyV=[]
    xxE=[]
    yyE=[]
    c=-1
    for line in file:
        c=c+1
        line = str(line).split("\t")[0]
        if int(line)==-1:
            labels.append(0)
        if int(line)==0:
            labels.append(0)
            xxV.append(nbV[c])
            xxE.append(nbE[c])
        elif int(line)>0:
            labels.append(min(int(line),1))
            yyV.append(nbV[c])
            yyE.append(nbE[c])
        else:
            labels.append(0)
    coefficient = (len(labels)-sum(labels))/sum(labels)
    
    ### Equilibre dataset
    if coefficient>1:
        minorite=1
        NbMino=sum(labels)-len(nokeep)
    else:
        minorite =0
        NbMino=len(labels)-sum(labels)
    keep = []
    compteur=0
    graphes = []
    for i in range(len(labels)):
        if labels[i]==0:
            pass
        if labels[i]==minorite:# and purete[i]>0.90:
            if i not in nokeep:
                keep.append(i)
                graphes.append(PatternRed[i])
                MegaRES.append([i])
    complete=NbMino
    for i in range(len(labels)):   
        if labels[i]!=minorite:
            if compteur<complete:# and purete[i]>0.8:
                compteur=compteur+1
                keep.append(i)
                graphes.append(PatternRed[i])
    print(len(keep))
    return keep



def cross_validation(X,Y,cv,classifier):
    #store for each fold the F1 score of each class
    F1_score0 = []
    F1_score1 = []
    # for each fold
    for train_index, test_index in cv.split(X,Y):
        #split the dataset into train and test
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        for l in train_index:
            X_train.append(X[l])
            y_train.append(Y[l])
        for l in test_index:
            X_test.append(X[l])
            y_test.append(Y[l])
        #fit the classifier on the train set
        estimates = estimate_importance(X_train, y_train) #features_train = [[feat1, feat2, feat3 ....], [feat1, feat2, feat3 ....], [feat1, feat2, feat3 ....]...]
        feature_list = []
        for i, val in enumerate(estimates):
            feature_list.append((i, val))
        #order from least to most important
        feature_list = sorted(feature_list, key=lambda x: x[1])
        print(feature_list)         
        #find the F1 score of each class on the test set
        # use the function for finding the f score between two labels sets 
        # using sklearn.metrics
    #compute the mean and standard deviation of the F1 score of each class
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def pangProcessing(Ks,keep,labels,id_graphs_mono,id_graphs_iso,occurences_mono,occurences_iso,LENGTHPATTERN,LENGTHGRAPHS):
    cv = StratifiedKFold(n_splits=10,shuffle=True)
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,LENGTHPATTERN)
    scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,LENGTHPATTERN)
    scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,LENGTHPATTERN)
    for K in Ks:
        X_GenBin = KVector(keep,K,scoresGenBin,id_graphs_mono,None,LENGTHGRAPHS)
        X_GenOcc = KVector(keep,K,scoresGenOcc,id_graphs_mono,occurences_mono,LENGTHGRAPHS)
        X_IndBin = KVector(keep,K,scoresIndBin,id_graphs_iso,None,LENGTHGRAPHS)
        X_IndOcc = KVector(keep,K,scoresIndOcc,id_graphs_iso,occurences_iso,LENGTHGRAPHS)
        results = np.zeros((4,2,2))
        representations = [[X_GenBin,labels],[X_GenOcc,labels],[X_IndBin,labels],[X_IndOcc,labels]]
        for i in range(len(representations)):
            results[i][0][0],results[i][0][1],results[i][1][0],results[i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=1000))
        outputResults(K,results,"../data/"+str(K)+"results.txt")
    return 0

def outputResults(K,results, output_file):
    """ function to output the results of the cross validation
        Input : K : number of patterns
        results : array containing the results of the cross validation
        output_file : file where the results will be written"""
    f = open(output_file,"w")
    f.write("Results for K = "+str(K)+ " \n")
    for i in range(len(results)):
            f.write("Representation "+str(i)+"\n")
            f.write("F1 score for class 0: "+str(results[i][0][0])+" +/- "+str(results[i][0][1])+"\n")
            f.write("F1 score for class 1: "+str(results[i][1][0])+" +/- "+str(results[i][1][1])+"\n")
    f.close()

def printBestPattern(patterns,labels):
    for i in range(len(patterns)):
        if labels[i]==1:
            print(patterns[i])
def main(argv):
    opts, args = getopt.getopt(argv,"hd:o:k:",["ifile=","ofile="])
    for opt, arg in opts:
      if opt == '-h':
         print ('PANG.py -d <dataset> -k<values of K> -m<mode>')
         sys.exit()
      elif opt in ("-d"):
            folder="../data/"+str(arg)+"/"
            FILEGRAPHS=folder+str(arg)+"_graph.txt"
            FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
            FILEMONOSET=folder+str(arg)+"_mono.txt"
            FILEISOSET=folder+str(arg)+"_iso.txt"
            FILELABEL =folder+str(arg)+"_label.txt"
            TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
            TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
      elif opt in ("-k"):
         Ks=[]
         temp=arg
         temp=temp.split(",")
         for j in range(len(temp)):
             Ks.append(int(temp[j]))
    keep= []
    dele =[]
    for i in range(TAILLEGRAPHE):
        if i not in dele:
            keep.append(i)
            
    """loading graphs"""
    print("Reading graphs")
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    print("Reading patterns")
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)
    
    """loading processed patterns"""
    print("Reading processed patterns")
    xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
    xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
    #keep = graphKeep(PatternsRed,FILELABEL)
    labels = readLabels(FILELABEL)
    print("Processing PANG")
    pangProcessing(Ks,keep,labels,id_graphsMono,id_graphsIso,numberoccurencesMono,numberoccurencesIso,TAILLEPATTERN,TAILLEGRAPHE)

#Features importance estimator
def estimate_importance(features, labels):
	clf = ExtraTreesClassifier()
	clf = clf.fit(features, labels)
	return clf.feature_importances_


def compute_results(features_train, labels_train, features_test, labels_test):
	clf = SVC(class_weight='balanced')
	clf = CalibratedClassifierCV(clf)
	clf.fit(features_train,labels_train)
	res = clf.predict(features_test)

	ok = 0
	fp, fn = 0, 0
	tp, tn = 0, 0
	nAbuses = 0
	f1 = 0
	for r in range(len(res)):
		if res[r] == labels_test[r]:
			ok += 1
		if labels_test[r] == 1:
			if res[r] == 1:
				tp += 1
			if res[r] == 0:
				fn += 1
			nAbuses += 1
		if labels_test[r] == 0:
			if res[r] == 1:
				fp += 1
			if res[r] == 0:
				tn += 1

	rec = tp / float(nAbuses)
	pre = tp / float(tp + fp)
	if pre + rec > 0:
		f1 = 2 * (pre * rec) / float(pre + rec)
	return f1

if __name__ == '__main__':
	names = ['cLength', 'nWords', 'avgwLen', 'uniqueChars', 'uniqueWords', 'collapseN', 'longest', 'cRatio', 'nAlpha', 'rAlpha', 'nDigit', 'rDigit', 'nPunct', 'rPunct', 'nOther', 'rOther', 'nSpace', 'rSpace', 'nCap', 'rCap', 'nBadwords', 'nHiddenBadwords', 'tfidfPosScore', 'tfidfNegScore', 'posScore', 'negScore', 'cPosScore', 'cNegScore', 'NB']
	importance_feat = [0.0]*29
	imp_features = dict(zip(names, importance_feat))
	avg_pos_features = dict(zip(names, importance_feat))


	#estimate importance of features
	estimates = estimate_importance(features_train, labels_train) #features_train = [[feat1, feat2, feat3 ....], [feat1, feat2, feat3 ....], [feat1, feat2, feat3 ....]...]
	feature_list = []
	for i, val in enumerate(estimates):
		feature_list.append((feature_names[i], val))
	#order from least to most important
	feature_list = sorted(feature_list, key=lambda x: x[1])

	#40 iteration
	nb_iteration = 40
	for it in range(nb_iteration):
		print it 
		f1s = [0]*29

		for k in range(10):
			feature_names = list(names)

			#load data/labels/scaler
			features_train = zload("features/features_train_%s.pkl.gz" % k)
			labels_train = zload("labels/labels_train_%s.pkl.gz" % k)
			features_test = zload("features/features_test_%s.pkl.gz" % k)
			labels_test = zload("labels/labels_test_%s.pkl.gz" % k)
			scaler = zload("models/scalers/scaler_%s.pkl.gz" % k)
			features_train = scaler.transform(features_train).tolist()
			features_test = scaler.transform(features_test).tolist()

			pool = Pool(processes=1)
			res = []
			nb_feat = len(feature_names)
			
			while nb_feat >= 1:
				res.append(pool.apply_async(compute_results, args=(copy.deepcopy(features_train), labels_train, copy.deepcopy(features_test), labels_test)))
				#Remove the least important feature
				#index of the feature to remove

				feat_to_remove = feature_list[len(names)-nb_feat][0]
				feat_index = feature_names.index(feat_to_remove)
				feature_names.remove(feat_to_remove)
				for f in features_train:
					del f[feat_index]
				for f in features_test:
					del f[feat_index]
				nb_feat -= 1

			i = 0
			for r in res:
				f1 = r.get()
				#Resultat pour 1 iteration
				f1s[i] += f1
				i += 1
			

			pool.close()
			pool.join()
			
		#Sauvegarde l'ordre dans lequel les features ont été suppr pendant ce run
		f = open("ablation/text/run_%s.txt" % it, "w")
		res_list = [x[0] for x in feature_list]
		f.write(str(res_list))

		f1s = [x / 10 for x in f1s]
		####### TRANSFORMATION POUR FAIRE COMME EP 
		#liste des features dans l'ordre dans lequel elles sont suppr
		f_names = [v[0] for v in feature_list]
		#écart de f-mesure lorsque chaque feature est enlevée
		f1s_loss = [f1s[i] - f1s[i+1] for i in range(len(f1s)-1)]
		feature_importance_update = zip(f_names, f1s_loss)
		#Rajoute la dernière feature (la plus importante qui n'a pas été supprimée)
		feature_importance_update.append((f_names[-1], 65.0))
		#order
		for i in range(it%2, len(feature_importance_update)-1, 2):
			if feature_importance_update[i][1] > feature_importance_update[i+1][1]:
				tmp = feature_importance_update[i]
				feature_importance_update[i] = feature_importance_update[i+1]
				feature_importance_update[i+1] = tmp

		#feature_importance_update = sorted(feature_importance_update, key=lambda x: x[1])
		feature_list = feature_importance_update

		#Sauvegarde la f-mesure moyenne perdue à chaque fois qu'on retire chaque feature
		cpt = 29
		for val in feature_importance_update:
			imp_features[val[0]] += val[1]
			# Saubegarde la position moyenne de la feature
			avg_pos_features[val[0]] += cpt
			cpt -= 1
		##############
		# marker='o'
		plt.plot(f1s, color='red', linewidth=1)
		plt.ylabel('F1-Score')
		plt.xlabel('Number Of Features Removed')
		#plt.xticks(range(0,29,2))
		plt.savefig("ablation/run_%s.png" % it)
		plt.close()


	'''plt.plot(f1s, color='red', linewidth=1)	
	plt.ylabel('F1-Score')
	plt.xlabel('Number Of Features Removed')
	plt.savefig("features/F1score.pdf")'''
		
	print( 'PERTE DE F-MESURE MOYENNE DUE AU RETRAIT DE CETTE FEATURE')
	for key, value in sorted(imp_features.iteritems(), key=lambda (k,v): (v,k), reverse=True):
		print "%s: %s" % (key, value/nb_iteration)
	print( '-------------')
	print( 'POSITION MOYENNE DE LA FEATURE')
	for key, value in sorted(avg_pos_features.iteritems(), key=lambda (k,v): (v,k)):
		print ("%s: %s" % (key, value/nb_iteration))
  
if __name__ == "__main__":
   print(sys.argv)
   main(sys.argv[1:])