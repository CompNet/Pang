from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def load_graphs(fileName,TAILLE):
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    numbers = []
    compteur=-1
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            labelVertices.append([])
            labelEdges.append([])
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
        if b[0]=="x":
            temp= []
            for j in range(1,len(b)):
                val = b[j]
                val = re.sub("\n","",val)
                val = int(val)
                temp.append(val)  
            numbers.append(temp)  
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
    return graphes,numbers


def countSubgraph(Graphes,Subgraphs,id_graphs):
    file=open("Count.txt","w")
    com = -1
    for subgraph in Subgraphs:
        com=com+1
        if com%1==0:
            print(com)
        results=np.zeros(710)
        compt=-1
        for graph in Graphes:
            compt=compt+1
            count=0
            if compt in id_graphs[com]:
                GM = nx.isomorphism.GraphMatcher(graph,subgraph,node_match=lambda n1,n2:n1['color']==n2['color'],edge_match= lambda e1,e2: e1['color'] == e2['color'])
                v = set()
                for m in GM.subgraph_monomorphisms_iter():
                    liste = []
                    for key in m:
                        liste.append(key)
                    a = frozenset(tuple(liste))
                    v.add(a)
                results[compt] = len(v)
        st="t # "+str(com)+ "\n"
        file.write(st)
        stre="x # "
        for j in range(len(results)):
            if results[j]>0:
                stre=stre+str(j)+ " "
        stre=stre+"\n"
        if len(stre)>5:
            file.write(st)
            file.write(stre)
    return 0


def transformInduced(Graphes,Subgraphs,id_graphs,count):
    file=open("Induced.txt","w")
    nbIso=0
    print(Graphes)
    com = -1
    for subgraph in Subgraphs:
        com=com+1
        if com%5==0:
            print(com)
        results=np.zeros(188)
        compt=-1
        for graph in Graphes:
            compt=compt+1
            if compt in id_graphs[com]:
                GMP = ISMAGS(graph,subgraph,node_match=lambda n1,n2:n1['color']==n2['color'],edge_match= lambda e1,e2: e1['color'] == e2['color'])
                match_list = [m for m in GMP.find_isomorphisms()]
                if len(match_list)>0:
                    results[compt]=1
                    nbIso=nbIso+1
        st="t # "+str(com)+ "\n"
        file.write(st)
        stre="x # "
        for j in range(len(results)):
            if results[j]>0:
                stre=stre+str(j)+ " "
        stre=stre+"\n"
        if len(stre)>5:
            file.write(st)
            file.write(stre)
    return 0

Graphes,useless_var = load_graphs("MutagGraph.txt",188)
Subgraphs,id_graphs = load_graphs("outputMutag.txt",50705)
#countSubgraph(Graphes,Subgraphs,id_graphs)
transformInduced(Graphes,Subgraphs,id_graphs,"False")

def readPatterns():
    c=0
    g = None
    file = open('ResNC1.txt', "r")
    GraphesRed = []
    OccurencesRed = []
    NomsRed = []
    tempSTR=""
    compteur=-1
    vertices = []
    colors=[]
    colorsE = []
    edges = []
    listeID = []
    for line in file:
        c=c+1
        a = line
        b = a.split(" ")
        if b[0]=="t":
            g = igraph.Graph()
            val = b[4]
            val = re.sub("\n","",val)
            val = int(val)
            OccurencesRed.append(val)
        if b[0]=="v":
            num = int(b[1])
            val = b[2]
            g.add_vertices(1)
            val = re.sub("\n","",val)
            val = int(val)
            col="blue"
            if val==0:
                col="red"
            vertices.append(val)
            colors.append(col)
            tempSTR=tempSTR+line
        if b[0]=="e":
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            g.add_edge(num1,num2)
            val = re.sub("\n","",val)
            val = int(val)
            col="red"
            if val==1:
                col="orange"
            if val==2:
                col="green"
            colorsE.append(col)
            edges.append(val)
            tempSTR=tempSTR+line
        if b[0]=="x":
            temp= []
            for j in range(1,len(b)):
                val = b[j]
                val = re.sub("\n","",val)
                val = int(val)
                temp.append(val)    
            g.vs["weight"]=vertices
            g.vs["color"]=colors
            g.es["color"]=colorsE
            g.es["weight"]=edges
            GraphesRed.append(g)
            NomsRed.append(tempSTR)
            listeID.append(temp)
            tempSTR=""
            edges = []
            vertices=[]
            colors=[]

    print("passage")       
    file = open('nci1_label.txt', "r")
    labels = []
    for line in file:
        temp = str(line).split("\t")
        labels.append(int(temp[0]))

    print("passage")
    #Creer les arrays de presence Red
    ResumePresence = []
    for j in range(4110):#330
        print(j)
        ResumePresence.append([])
        val = j
        for k in range(len(listeID)):
            if j in listeID[k]:
                ResumePresence[j].append(1)
            else:
                ResumePresence[j].append(0)
    X = ResumePresence
    Y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
from sklearn import tree
import graphviz
classifier = RandomForestClassifier(n_estimators = 10, random_state = 42)


def train(X,Y,classifier):
    classifier.fit(X_train,y_train)
    return classifier


def test(X,Y,classifier):
    test_pred_decision_tree = classifier.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test,  
                                               test_pred_decision_tree)#turn this into a dataframe
    matrix_df = pd.DataFrame(confusion_matrix)#plot the result
    ax = plt.axes()
    sns.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")#set axis titles
    ax.set_title('Confusion Matrix - Decision Tree')
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.savefig("foo.pdf")
    plt.show()
    print(metrics.classification_report(y_test,test_pred_decision_tree))
    
    


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
classifier = RandomForestClassifier(n_estimators = 10, random_state = 42)
classifier = train(X_train,y_train,classifier)
test(X_test,y_test,classifier)