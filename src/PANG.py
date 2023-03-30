
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
    return graphes,numbers,noms,nbV,nbE

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
    tailleRed =[]
    tailleNonRed=[]
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                        if id_graphs[i][j] in keep:
                            aa=id_graphs[i][j]
                            if labels[aa]==0:
                                        diffDelta[i]=diffDelta[i]+1
                                        tailleNonRed[i]=tailleNonRed[i]+1
                            elif labels[aa]==1:
                                        diffDelta[i]=diffDelta[i]-1
                                        tailleRed[i]=tailleRed[i]+1
        
        diffDelta[i] = abs(diffDelta[i])
    return diffDelta
    
###################################
            
def computeScoreOccurences(keep,occurences,labels,id_graphs,TAILLEPATTERN):
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
    coverageRed=[]
    coverageNonRed=[]
    for i in range(TAILLEPATTERN):
        tailleRed.append(0)
        tailleNonRed.append(0)
        coverageRed.append(0)
        coverageNonRed.append(0)
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                    if labels[id_graphs[i][j]]==0:
                        if j in keep:
                                diffDelta[i]=diffDelta[i]+occurences[i][j]
                                tailleNonRed[i]=tailleNonRed[i]+occurences[i][j]
                    elif labels[id_graphs[i][j]]==1:
                        if j in keep:
                                diffDelta[i]=diffDelta[i]-occurences[i][j]
                                tailleRed[i]=tailleRed[i]+occurences[i][j]
        diffDelta[i] = abs(diffDelta[i])
        return diffDelta
    
def recupLabel(fileLabel):
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

def KVector(K,diff,id_graphs,numberoccurences,LENGTHGRAPH,labels):
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
    Y = newLabels
    return X,Y

def test(X,Y,classifier):
    test_pred_decision_tree = classifier.predict(X)
    confusion_matrix = metrics.confusion_matrix(Y,  
                                               test_pred_decision_tree)#turn this into a dataframe
    matrix_df = pd.DataFrame(confusion_matrix)#plot the result
    ax = plt.axes()
    plt.figure(figsize=(10,7))
    ax.set_title('Confusion Matrix - Decision Tree')
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.savefig("foo.pdf")
    #plt.show()

def Expe(keep,Ks,fileLabel,out,id_graphs,occurences,coverage,MODE,MotifGardes,TAILLEGRAPHE,TAILLEPATTERN,cv,metric):
    patterns = []
    labels,coefficient =recupLabel(fileLabel)
    file = open(out,"a")
    if MODE == "support":
        diffDelta = computeScoreMono(keep,labels,id_graphs,TAILLEPATTERN,1)
    if MODE =="occurences":
        diffDelta = computeScoreOccurences(keep,occurences,coverage,labels,id_graphs,TAILLEPATTERN,1)
    for K in Ks:
        nom=str(K)+"Prediction.txt"
        fileBis=open(nom,"w")
        comc=0
        if MotifGardes==None:
            if metric =="DELTA":
                temp = copy.deepcopy(diffDelta)
                Xt,Yt,numo,DiscriPat,RED,NONRED = KVector(NbRed,NbNonRed,keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
            if metric =="CORK":
                temp = copy.deepcopy(diffCORK)
                Xt,Yt,numo,DiscriPat,RED,NONRED = KVector(NbRed,NbNonRed,keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        else:
            temp = copy.deepcopy(diffDelta)
            Xt,Yt,numo,DiscriPat,RED,NONRED = KVector(NbRed,NbNonRed,keep,K,id_graphs,occurences,MotifGardes,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        for e in range(len(DiscriPat)):
            stre = "Motif "+ str(DiscriPat[e])+ "score "+str(numo[e])+ "NBRED "+str(RED[e])+" NONRED "+str(NONRED[e])+" \n"
            fileBis.write(stre)
    
        X = []
        Y = []
        for j in range(len(Xt)):
            if not(Yt[j]=="miss"):
                X.append(Xt[j])
                Y.append(Yt[j])
        
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_validate
        from sklearn import tree
        import graphviz
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        nummmm=-1
        classifiers = [0.1,1,10,100,1000,10000,100000]
        from sklearn.model_selection import cross_val_score
        for c in classifiers:
                class1=[0,0,0,0,0,0,0,0,0,0]
                class2=[0,0,0,0,0,0,0,0,0,0]
                class3=[0,0,0,0,0,0,0,0,0,0]
                class0=[0,0,0,0,0,0,0,0,0,0]
                acc=[0,0,0,0,0,0,0,0,0,0]
                f_score=[0,0,0,0,0,0,0,0,0,0]
                from sklearn import metrics
                from sklearn.model_selection import StratifiedKFold
                ccc=0
                for train_index, test_index in cv[0].split(X,Y):
                        clf = SVC(C=c)
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
                        clf = clf.fit(X_train,y_train)
                        y_pred = clf.predict(X_test)
                        ci=0
                        acc[ccc] = metrics.accuracy_score(y_test, y_pred)
                        class3[ccc] = metrics.recall_score(y_test, y_pred, average=None)[0]
                        class1[ccc] = metrics.f1_score(y_test, y_pred, average=None)[1]
                        class2[ccc] = metrics.precision_score(y_test, y_pred, average=None)[0]
                        class0[ccc] =  metrics.f1_score(y_test, y_pred, average=None)[0]
                        f_score[ccc] = metrics.f1_score(y_test, y_pred, average=None)[0]
                        ccc=ccc+1
                stre = str(len(X[0]))+ " & "+ str(np.mean(class0))[0:9]+ " & "+ str(np.std(class0))[0:9]+ " & "+ str(np.mean(class1))[0:9]+ " & "+ str(np.std(class1))[0:9]+ " & "+ str(np.mean(acc))[0:9]+" & "+ str(np.mean(f_score))[0:9]+" \\ \n"
                file.write(stre)
                y_pred = clf.predict(X)
                if c==1000:
                    for teta in range(len(y_pred)):
                            stre = "NumeroGraphe"+str(keep[teta])+ " Realite "+str(Y[teta])+"Predi "+str(y_pred[teta])+" Vecto "+str(sum(X[teta]))+" \n"
                            fileBis.write(stre)
    return numo,patterns
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
    #reset the classifier
    classifier.reset()
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
        #train the classifier
        classifier.train(X_train,y_train)
        #test the classifier
        classifier.test(X_test,y_test)
        #add the F1 score of each class to the array
        F1_score0.append(classifier.F1_score0)
        F1_score1.append(classifier.F1_score1)
    #compute the mean and standard deviation of the F1 score of each class
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def pangProcessing(keep,labels,id_graphs_mono,id_graphs_iso,TAILLEPATTERN):
    cv = StratifiedKFold(n_splits=10,shuffle=True)
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,TAILLEPATTERN,1)
    scoresGenOcc = computeScoreMono(keep,labels,id_graphs_iso,TAILLEPATTERN,1)
    scoresIndBin = computeScoreOccurences(keep,labels,id_graphs_iso,TAILLEPATTERN,1)
    scoresIndOccn = computeScoreOccurences(keep,labels,id_graphs_iso,TAILLEPATTERN,1)
    for K in Ks:
        X_GenBin,Y_GenBin,numo,DiscriPat,RED,NONRED = KVector(keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        X_GenOcc,Y_GenOcc,numo,DiscriPat,RED,NONRED = KVector(keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        X_IndBin,Y_IndBin,numo,DiscriPat,RED,NONRED = KVector(keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        X_IndOcc,Y_IndOcc,numo,DiscriPat,RED,NONRED = KVector(keep,K,id_graphs,occurences,None,temp,TAILLEGRAPHE,TAILLEPATTERN,labels)
        results = np.zeros((4,2,2))
        representations = [[X_GenBin,Y_GenBin],[X_GenOcc,Y_GenOcc],[X_IndBin,Y_IndBin],[X_IndOcc,Y_IndOcc]]:
        for i in range(len(representations)):
            results[i][0][0],results[i][0][1],results[i][1][0],results[i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(c=1000))
        outputResults(K,results,"data/"+str(K)+"results.txt")
    return 0

def outputResults(K,results, output_file):
    """ function to output the results of the cross validation
        Input : K : number of patterns
        results : array containing the results of the cross validation
        output_file : file where the results will be written"""
    with open(output_file, 'w') as f:
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
            folder="data/"
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
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)
    
    """loading processed patterns"""
    xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
    xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
    keep = graphKeep(PatternsRed,FILELABEL)
    
    
    numoA,pattA =Expe(keep,Ks,FILELABEL,"DecoMonoCGZ.txt",id_graphs,None,None,"Support",None,TAILLEGRAPHE,TAILLEPATTERN,cv,"DELTA")
    numoB,pattB =Expe(keep,Ks,FILELABEL,"DecoOccu.txt",id_graphsMono,numberoccurencesMono,coverageMono,"Occurences",None,TAILLEGRAPHE,TAILLEPATTERN,cv,"DELTA")
    numoC,pattC =Expe(keep,Ks,FILELABEL,"DecoIso.txt",id_graphsIso,None,coverageIso,"Support",None,TAILLEGRAPHE,TAILLEPATTERN,cv,"DELTA")
    numoD,pattD =Expe(keep,Ks,FILELABEL,"DecoDouble.txt",id_graphsIso,numberoccurencesIso,coverageIso,"Occurences",None,TAILLEGRAPHE,TAILLEPATTERN,cv,"DELTA")

if __name__ == "__main__":
   print(sys.argv)
   main(sys.argv[1:])