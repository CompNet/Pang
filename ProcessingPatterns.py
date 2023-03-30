from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
import networkx as nx
import numpy as np
import re
import sys

def load_graphs(fileName,TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    """
    numbers = []
    for i in range(TAILLE):
        numbers.append([])
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
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
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
                if not(val=="#") and not(val==""):
                    val = int(val)
                    temp.append(val)  
            numbers[numero]=temp  
    Graphs = []
    for i in range(len(vertices)):
        dicoNodes = {}
        Graphs.append(nx.Graph())
        for j in range(int(vertices[i])):
            #tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            Graphs[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        Graphs[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return Graphs,numbers


def countSubgraph(Graphs,Subgraphs,id_graphs,mode,name):
    """ Compute number of occurences and induced subgraphs
    Inputs : Graphs (list of networkx graphs) : the list of graphs
    Subgraphs (list of networkx graphs) : the list of patterns
    id_graphs (list of list of int) : list of graphs containing each pattern
    mode (string) : "mono" or "iso" : mono for monomorphism and iso for isomorphism
    """
    file=open(name,"w")
    com = -1
    for subgraph in Subgraphs[0:len(Subgraphs)]:
        com=com+1
        results=np.zeros(len(Graphs))
        cork = np.zeros(len(Graphs))
        compt=-1
        for graph in Graphs:
            compt=compt+1
            count=0
            if compt in id_graphs[com]:
                GM = nx.isomorphism.GraphMatcher(graph,subgraph,node_match=lambda n1,n2:n1['color']==n2['color'],edge_match= lambda e1,e2: e1['color'] == e2['color'])
                listeNode=[]
                temp = []
                v = set()
                node = set()
                vertexpoint=0
                if mode=="mono":
                    for m in GM.subgraph_monomorphisms_iter():
                        new=True
                        liste = []
                        for key in m:
                            liste.append(key)
                            if key not in listeNode:
                                temp.append(key)
                            else:
                                new=False
                            if new:
                                vertexpoint=vertexpoint+1
                                listeNode.append(key)       
                        a = frozenset(tuple(liste))
                        v.add(a)
                results[compt] = len(v)
                if mode=="iso":
                    for m in GM.subgraph_isomorphisms_iter():
                        new=True
                        liste=[]
                        for key in m:
                            liste.append(key)
                            if key not in listeNode:
                                temp.append(key)
                            else:
                                new=False
                            if new:
                                vertexpoint=vertexpoint+1
                                listeNode.append(key)
                        a = frozenset(tuple(liste))
                        v.add(a)
                results[compt] = len(v)
  
        st="t # "+str(com)+ "\n"
        file.write(st)
        stre="x "
        for j in range(len(results)):
            if results[j]>0:
                stre=stre+str(j)+ "/" +str(results[j])+":"+str(cork[j])+" "
        stre=stre+"\n"
        if len(stre)>2:
            file.write(stre)
    return 0


def main(argv):
    FILEGRAPHS="mutag_graph.txt" 
    FILESUBGRAPHS="mutag_pattern.txt" 
    LENGTHGRAPHS=188
    LENGTHSUBGRAPHS=14479
    Graphs,useless_var = load_graphs(FILEGRAPHS,LENGTHGRAPHS)
    Subgraphs,id_graphs = load_graphs(FILESUBGRAPHS,LENGTHSUBGRAPHS)
    countSubgraph(Graphs,Subgraphs,id_graphs,"mono","Mono.txt")
    countSubgraph(Graphs,Subgraphs,id_graphs,"iso","Iso.txt")




if __name__ == "__main__":
   print(sys.argv)
   main(sys.argv[1:])
   
   
