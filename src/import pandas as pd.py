import pandas as pd
import numpy as np


def readData(fileName):
    """
    Read the data
    """
    data = pd.read_csv(fileName, sep = ";")
    return data

df = readData("../results/dfResultsWithK.csv")
print(df)


KVALUE = 100
REPRESENTATION = "BINARY"
PATTERNSTYPE = "EXTRACTED"
DATASET = "FOPPA"
re.sub(r'[^a-zA-Z0-9]', '', mystring)

#create df2, part of the dataframe where K is equal to KVALUE, and the representation is equal to REPRESENTATION and the patternstype is equal to PATTERNSTYPE
df2 = df.loc[(df['K'] == KVALUE) & (df['REPRESENTATION'] == REPRESENTATION) & (df['PATTERNSTYPE'] == PATTERNSTYPE) & (df['DATASET'] == DATASET)]
df2 = df2.reset_index(drop=True)
print(df2)

#create a matrix with each metric as a row and a column 
matrix = np.zeros((8,8))
patterns = []
for i in range(8):
    patterns.append([])
names = ["Growth","Support","Unusualness","Generalization","OddsRatio","TruePostivieRate","FalsePositiveRate","Strength"]
for i in range(len(df2)):
    patterns[i].append(df2['PATTERNSLIST'][i])

for i in range(8):
    for j in range(8):
        #matrix[i][j] is equal to the number of common patterns between the two metrics 
        matrix[i][j] = len(set(patterns[i][0]).intersection(patterns[j][0]))
#display the matrix using matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.heatmap(matrix, linewidth=0.5,xticklabels=names, yticklabels=names)
#save the matrix as a pdf
plt.savefig("../results/heatmap.pdf")



