
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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
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
    return X,keepPatterns

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


if __name__ == '__main__':
    NBPATTERNS=5
    importance_feat = [0.0]*NBPATTERNS
    arg="FOPPA"
    folder="../data/"+str(arg)+"/"
    FILEGRAPHS=folder+str(arg)+"_graph.txt"
    FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
    FILEMONOSET=folder+str(arg)+"_mono.txt"
    FILEISOSET=folder+str(arg)+"_iso.txt"
    FILELABEL =folder+str(arg)+"_label.txt"
    TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
    TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
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
    labels = readLabels(FILELABEL)
    
    scoresGenBin = computeScoreMono(keep,labels,id_graphsMono,TAILLEPATTERN)
    #scoresGenOcc = computeScoreOccurences(keep,labels,id_graphsMono,numberoccurencesMono,TAILLEPATTERN)
    #scoresIndBin = computeScoreMono(keep,labels,id_graphsIso,TAILLEPATTERN)
    #scoresIndOcc = computeScoreOccurences(keep,labels,id_graphsIso,numberoccurencesIso,TAILLEPATTERN)
    X_GenBin,numbers = KVector(keep,NBPATTERNS,scoresGenBin,id_graphsMono,None,TAILLEGRAPHE)
    #X_GenOcc,xx = KVector(keep,500,scoresGenOcc,id_graphsMono,numberoccurencesMono,TAILLEGRAPHE)
    #X_IndBin,xx = KVector(keep,500,scoresIndBin,id_graphsIso,None,TAILLEGRAPHE)
    X=X_GenBin
    Y=labels
    #X_IndOcc,xx = KVector(keep,500,scoresIndOcc,id_graphsIso,numberoccurencesIso,TAILLEGRAPHE)
    originalnumbers= copy.deepcopy(numbers) 
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    boole=True
    for train_index, test_index in cv.split(X_GenBin,labels):
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
        if boole:
            res=[]
            scoress=[]
            boole=False
            from sklearn.svm import SVC
            for j in range(1,NBPATTERNS):
                X_train =   np.array(X_train)
                X_test =   np.array(X_test)
                base_score = compute_results(X_train, y_train,X_test, y_test)
                scores = []
                scoress.append(base_score)
                print(base_score)
                for i in range(len(X_train[0])):
                    use_column = [ndx != i for ndx in range(len(X_train[0]))]
                    scores.append(compute_results(X_train[:, use_column],
                                            y_train,
                                            X_test[:, use_column],
                                            y_test))
                alpha = sorted(enumerate([base_score - s for s in scores]),
                    key=lambda ndx_score: ndx_score[1],
                    reverse=False)
                print(numbers)
                #delete the features with the lowest importance score                                       
                feat_num = alpha[0][0]
                res.append(numbers[feat_num])
                #delete from numbers the feature
                numbers = np.delete(numbers,feat_num)
                X_train = np.delete(X_train, feat_num, axis=1)
                X_test = np.delete(X_test, feat_num, axis=1)
        #write the results in a file
        namedd= str(arg)+"_results.txt" 
        file = open(namedd,"w")
        #write in the following order : base score, score after deleting the feature, feature deleted
        #warning : for the last feature, there is only the base score and the feature deleted
        for i in range(len(res)):
            if i == len(res)-1:
                file.write(str(scoress[i])+" : "+str(res[i])+"\n")
            else:
                file.write(str(scoress[i])+" : "+str(res[i])+" : "+str(scoress[i+1])+"\n")
                
            #do a plot of the results following the format : 
            # in the x axis : each feature
            # in the y axis : the step where the feature is deleted
        import matplotlib.pyplot as plt
        plot = plt.figure()
        feature = []
        for i in range(len(res)):
            feature.append(i)
        position = []
        print(originalnumbers)
        for i in range(len(res)):
            #check the position of the feature in the list of features deleted  
            position.append(originalnumbers.index(res[i]))
        print(position)
        print(feature)
        plt.plot(feature,position)  
        #Name the x axis : features number
        plt.xlabel('Features number')
        plt.ylabel('Step of deletion')
        plt.title('Features importance')
        # save the plot
        #Name = name of the dataset+number of features+".png"
        nameFile = str(arg)+str(len(res))+".png"
        plt.savefig(nameFile)