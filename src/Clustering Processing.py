#Clustering Processing

import pandas as pd
import copy as cp

datas = pd.read_csv("../results/dfResultsWithK.csv",sep=";") 
print(datas)


#keep only 2 digits after the comma 
datas = datas.round(2)
print(datas)


def dataframeToLatex(df,fileName):
    #convert dataframe to LATeX table
    #save only 2 digits after the comma
    df = df.round(2)
    #save on a txt file
    #df.to_csv(fileName+".csv",sep=";",index=False)
    #convert to latex

    #modifications 
    #delete the Dataset column
    df = df.drop(columns=["DATASET"])
    print(df)
    #check the name of the first column
    name = df.columns[0]
    modulo=0
    if name == "METRIC":
        if len(df.columns)==8:
            modulo = 4
        else:
            modulo = 2
        for i in range(len(df)):
            if i%modulo==0:
                df["METRIC"][i] = "\multirow{"+str(modulo)+"}{*}{"+df["METRIC"][i]+"}"
            else:
                df["METRIC"][i] = ""
        name2 = df.columns[1]
        modulo2=2
        if len(df.columns)==8:
            for i in range(len(df)):
                if i%modulo2==0:
                    df[name2][i] = "\multirow{"+str(modulo2)+"}{*}{"+df[name2][i]+"}"
                else:
                    df[name2][i] = ""
    else:
        if len(df.columns)==8:
            modulo = 16
        else:
            modulo = 8
        for i in range(len(df)):
            if i%modulo==0:
                df[name][i] = "\multirow"+str(modulo)+"{*}{"+df[name][i]+"}"
            else:
                df[name][i] = ""
        name2 = df.columns[1]
        modulo2=8
        if len(df.columns)==8:
            for i in range(len(df)):
                if i%modulo2==0:
                    df[name2][i] = "\multirow"+str(modulo2)+"{*}{"+df[name2][i]+"}"
                else:
                    df[name2][i] = ""



    # for the columns 10,25,50,100,200
    if len(df.columns)==8:
        ff=3
    else:
        ff=2
    for i in range(ff,len(df.columns)):
        #convert to numeric
        print(df.columns[i])
        df[df.columns[i]] = pd.to_numeric(df[df.columns[i]])
        df[df.columns[i]]=df[df.columns[i]].astype('float')
        #find the highest value
        max = df[df.columns[i]].max()
        max=round(max,2)
        #find the index of the highest value
        index = df[df.columns[i]].idxmax()
        #find the lowest value
        min = df[df.columns[i]].min()
        min=round(min,2)
        #find the index of the lowest value
        index2 = df[df.columns[i]].idxmin()
        #put the highest value in green
        df[df.columns[i]][index] = "\\textcolor{green}{"+str(max)+"}"
        #put the lowest value in red
        df[df.columns[i]][index2] = "\\textcolor{red}{"+str(min)+"}"
    latex = df.to_latex(index=False,float_format="%.2f")

    #Modification of the latex table




    #keep only 2 digits after the comma
    #save on a txt file
    file = open(fileName+".txt", "w")
    print(latex)
    file.write(latex)


def TableMetric(datas):
    dfFOPPA = pd.DataFrame(index=range(0,32),columns=["DATASET","PATTERNSTYPE","REPRESENTATION","METRIC","10","25","50","100","200"])
    dfMUTAG = pd.DataFrame(index=range(0,32),columns=["DATASET","PATTERNSTYPE","REPRESENTATION","METRIC","10","25","50","100","200"])
    dfPTC = pd.DataFrame(index=range(0,32),columns=["DATASET","PATTERNSTYPE","REPRESENTATION","METRIC","10","25","50","100","200"])
    dfNCI1 = pd.DataFrame(index=range(0,32),columns=["DATASET","PATTERNSTYPE","REPRESENTATION","METRIC","10","25","50","100","200"])
    dfDD = pd.DataFrame(index=range(0,32),columns=["DATASET","PATTERNSTYPE","REPRESENTATION","METRIC","10","25","50","100","200"])

    comptDataset=0
    comptMETRIC=0
    comptREPRESENTATION=0
    comptPATTERNSTYPE=0

    for i in range(len(datas)):
        dfActuel=None
        # case on the value of the column DATASET
        if datas["DATASET"][i] == "MUTAG":
            dfActuel = dfMUTAG
        elif datas["DATASET"][i] == "PTC":
            dfActuel = dfPTC
        elif datas["DATASET"][i] == "NCI1":
            dfActuel = dfNCI1
        elif datas["DATASET"][i] == "DD":
            dfActuel = dfDD
        elif datas["DATASET"][i] == "FOPPA":
            dfActuel = dfFOPPA
        #case on the type of the pattern
        if datas["PATTERNSTYPE"][i] == "EXTRACTED":
            comptPATTERNSTYPE=0
        else:
            comptPATTERNSTYPE=1
        
        #case on the representation*
        if datas["REPRESENTATION"][i] == "BINARY":
            comptREPRESENTATION=0
        else:
            comptREPRESENTATION=1

        #case on the metric
        if datas["METRIC"][i] == "Growth":
            comptMETRIC=0
        elif datas["METRIC"][i] == "Support":
            comptMETRIC=1
        elif datas["METRIC"][i] == "Unusualness":
            comptMETRIC=2
        elif datas["METRIC"][i] == "Generalization":
            comptMETRIC=3
        elif datas["METRIC"][i] == "OddsRatio":
            comptMETRIC=4
        elif datas["METRIC"][i] == "TruePositiveRate":
            comptMETRIC=5
        elif datas["METRIC"][i] == "FalsePositiveRate":
            comptMETRIC=6
        elif datas["METRIC"][i] == "Strength":
            comptMETRIC=7

        lineNumber = comptPATTERNSTYPE*16+comptREPRESENTATION*8+comptMETRIC
        #modify the dataframe on the line lineNumber 
        dfActuel["DATASET"][lineNumber] = datas["DATASET"][i]
        dfActuel["PATTERNSTYPE"][lineNumber] = datas["PATTERNSTYPE"][i]
        dfActuel["REPRESENTATION"][lineNumber] = datas["REPRESENTATION"][i]
        dfActuel["METRIC"][lineNumber] = datas["METRIC"][i]

        if datas["K"][i] == 10:
            #modify the dataframe on the line lineNumber and the column 10
            dfActuel["10"][lineNumber] = datas["F1SCORE"][i]  
        elif datas["K"][i] == 25:   
            dfActuel["25"][lineNumber] = datas["F1SCORE"][i]
        elif datas["K"][i] == 50:
            dfActuel["50"][lineNumber] = datas["F1SCORE"][i]
        elif datas["K"][i] == 100:
            dfActuel["100"][lineNumber] = datas["F1SCORE"][i]
        elif datas["K"][i] == 200:
            dfActuel["200"][lineNumber] = datas["F1SCORE"][i]
        
    #Save each dataframe using the function dataframeToLatex
    dataframeToLatex(dfFOPPA,"../results/NewResul/dfFOPPAMetric")
    dataframeToLatex(dfMUTAG,"../results/NewResul/dfMUTAGMetric")
    dataframeToLatex(dfPTC,"../results/NewResul/dfPTCMetric")
    dataframeToLatex(dfNCI1,"../results/NewResul/dfNCI1Metric")
    dataframeToLatex(dfDD,"../results/NewResul/dfDDMetric")



def TablePattern(datas):
    dfFOPPA = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfMUTAG = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfPTC = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfNCI1 = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfDD = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfFOPPA2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfMUTAG2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfPTC2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfNCI12 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfDD2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])

    #fill the dataframe with zeros
    for i in range(0,16):
        dfDD2["10"][i] = 0
        dfDD2["25"][i] = 0
        dfDD2["50"][i] = 0
        dfDD2["100"][i] = 0
        dfDD2["200"][i] = 0
        dfFOPPA2["10"][i] = 0
        dfFOPPA2["25"][i] = 0
        dfFOPPA2["50"][i] = 0
        dfFOPPA2["100"][i] = 0
        dfFOPPA2["200"][i] = 0
        dfMUTAG2["10"][i] = 0
        dfMUTAG2["25"][i] = 0
        dfMUTAG2["50"][i] = 0
        dfMUTAG2["100"][i] = 0
        dfMUTAG2["200"][i] = 0
        dfNCI12["10"][i] = 0
        dfNCI12["25"][i] = 0
        dfNCI12["50"][i] = 0
        dfNCI12["100"][i] = 0
        dfNCI12["200"][i] = 0
        dfPTC2["10"][i] = 0
        dfPTC2["25"][i] = 0
        dfPTC2["50"][i] = 0
        dfPTC2["100"][i] = 0
        dfPTC2["200"][i] = 0

    comptDataset=0
    comptMETRIC=0
    comptREPRESENTATION=0
    comptPATTERNSTYPE=0

    for i in range(len(datas)):
        dfActuel=None
        dfActuel2=None
        # case on the value of the column DATASET
        if datas["DATASET"][i] == "MUTAG":
            dfActuel = dfMUTAG
            dfActuel2 = dfMUTAG2
        elif datas["DATASET"][i] == "PTC":
            dfActuel = dfPTC
            dfActuel2 = dfPTC2
        elif datas["DATASET"][i] == "NCI1":
            dfActuel = dfNCI1
            dfActuel2 = dfNCI12
        elif datas["DATASET"][i] == "DD":
            dfActuel = dfDD
            dfActuel2 = dfDD2
        elif datas["DATASET"][i] == "FOPPA":
            dfActuel = dfFOPPA
            dfActuel2 = dfFOPPA2
        #case on the type of the pattern
        if datas["PATTERNSTYPE"][i] == "EXTRACTED":
            comptPATTERNSTYPE=0
        else:
            comptPATTERNSTYPE=1
        
        #case on the representation*
        if datas["REPRESENTATION"][i] == "BINARY":
            comptREPRESENTATION=0
        else:
            comptREPRESENTATION=1

        #case on the metric
        if datas["METRIC"][i] == "Growth":
            comptMETRIC=0
        elif datas["METRIC"][i] == "Support":
            comptMETRIC=1
        elif datas["METRIC"][i] == "Unusualness":
            comptMETRIC=2
        elif datas["METRIC"][i] == "Generalization":
            comptMETRIC=3
        elif datas["METRIC"][i] == "OddsRatio":
            comptMETRIC=4
        elif datas["METRIC"][i] == "TruePositiveRate":
            comptMETRIC=5
        elif datas["METRIC"][i] == "FalsePositiveRate":
            comptMETRIC=6
        elif datas["METRIC"][i] == "Strength":
            comptMETRIC=7

        lineNumber = comptPATTERNSTYPE+comptREPRESENTATION*2+comptMETRIC*4
        #modify the dataframe on the line lineNumber 
        dfActuel["DATASET"][lineNumber] = datas["DATASET"][i]
        dfActuel["PATTERNSTYPE"][lineNumber] = datas["PATTERNSTYPE"][i]
        dfActuel["REPRESENTATION"][lineNumber] = datas["REPRESENTATION"][i]
        dfActuel["METRIC"][lineNumber] = datas["METRIC"][i]
        
        lineNumber2 = comptREPRESENTATION+comptMETRIC*2
        dfActuel2["DATASET"][lineNumber2] = datas["DATASET"][i]
        dfActuel2["REPRESENTATION"][lineNumber2] = datas["REPRESENTATION"][i]
        dfActuel2["METRIC"][lineNumber2] = datas["METRIC"][i]
        if datas["K"][i] == 10:
            #modify the dataframe on the line lineNumber and the column 10
            dfActuel["10"][lineNumber] = datas["F1SCORE"][i]  
            #si la representation est binaire on fait + dans le dataset 2
            if comptPATTERNSTYPE == 0:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 25:   
            dfActuel["25"][lineNumber] = datas["F1SCORE"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 50:
            dfActuel["50"][lineNumber] = datas["F1SCORE"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 100:
            dfActuel["100"][lineNumber] = datas["F1SCORE"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 200:
            dfActuel["200"][lineNumber] = datas["F1SCORE"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]-datas["F1SCORE"][i]
        
    #Save each dataframe using the function dataframeToLatex
    dataframeToLatex(dfFOPPA,"../results/NewResul/dfFOPPATypePattern")
    dataframeToLatex(dfMUTAG,"../results/NewResul/dfMUTAGTypePattern")
    dataframeToLatex(dfPTC,"../results/NewResul/dfPTCTypePattern")
    dataframeToLatex(dfNCI1,"../results/NewResul/dfNCI1TypePattern")
    dataframeToLatex(dfDD,"../results/NewResul/dfDDTypePattern")

    dataframeToLatex(dfFOPPA2,"../results/NewResul/dfFOPPATypePatternDIFF")
    dataframeToLatex(dfMUTAG2,"../results/NewResul/dfMUTAGTypePatternDIFF")
    dataframeToLatex(dfPTC2,"../results/NewResul/dfPTCTypePatternDIFF")
    dataframeToLatex(dfNCI12,"../results/NewResul/dfNCI1TypePatternDIFF")
    dataframeToLatex(dfDD2,"../results/NewResul/dfDDTypePatternDIFF")

def TableKValues(datas):
    dfFOPPA = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfMUTAG = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfPTC = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfNCI1 = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfDD = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","REPRESENTATION","PATTERNSTYPE","10","25","50","100","200"])
    dfFOPPA2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfMUTAG2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfPTC2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfNCI12 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])
    dfDD2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","REPRESENTATION","10","25","50","100","200"])

    #fill the dataframe with zeros
    for i in range(0,16):
        dfDD2["10"][i] = 0
        dfDD2["25"][i] = 0
        dfDD2["50"][i] = 0
        dfDD2["100"][i] = 0
        dfDD2["200"][i] = 0
        dfFOPPA2["10"][i] = 0
        dfFOPPA2["25"][i] = 0
        dfFOPPA2["50"][i] = 0
        dfFOPPA2["100"][i] = 0
        dfFOPPA2["200"][i] = 0
        dfMUTAG2["10"][i] = 0
        dfMUTAG2["25"][i] = 0
        dfMUTAG2["50"][i] = 0
        dfMUTAG2["100"][i] = 0
        dfMUTAG2["200"][i] = 0
        dfNCI12["10"][i] = 0
        dfNCI12["25"][i] = 0
        dfNCI12["50"][i] = 0
        dfNCI12["100"][i] = 0
        dfNCI12["200"][i] = 0
        dfPTC2["10"][i] = 0
        dfPTC2["25"][i] = 0
        dfPTC2["50"][i] = 0
        dfPTC2["100"][i] = 0
        dfPTC2["200"][i] = 0

    comptDataset=0
    comptMETRIC=0
    comptREPRESENTATION=0
    comptPATTERNSTYPE=0

    for i in range(len(datas)):
        dfActuel=None
        dfActuel2=None
        # case on the value of the column DATASET
        if datas["DATASET"][i] == "MUTAG":
            dfActuel = dfMUTAG
            dfActuel2 = dfMUTAG2
        elif datas["DATASET"][i] == "PTC":
            dfActuel = dfPTC
            dfActuel2 = dfPTC2
        elif datas["DATASET"][i] == "NCI1":
            dfActuel = dfNCI1
            dfActuel2 = dfNCI12
        elif datas["DATASET"][i] == "DD":
            dfActuel = dfDD
            dfActuel2 = dfDD2
        elif datas["DATASET"][i] == "FOPPA":
            dfActuel = dfFOPPA
            dfActuel2 = dfFOPPA2
        #case on the type of the pattern
        if datas["PATTERNSTYPE"][i] == "EXTRACTED":
            comptPATTERNSTYPE=0
        else:
            comptPATTERNSTYPE=1
        
        #case on the representation*
        if datas["REPRESENTATION"][i] == "BINARY":
            comptREPRESENTATION=0
        else:
            comptREPRESENTATION=1

        #case on the metric
        if datas["METRIC"][i] == "Growth":
            comptMETRIC=0
        elif datas["METRIC"][i] == "Support":
            comptMETRIC=1
        elif datas["METRIC"][i] == "Unusualness":
            comptMETRIC=2
        elif datas["METRIC"][i] == "Generalization":
            comptMETRIC=3
        elif datas["METRIC"][i] == "OddsRatio":
            comptMETRIC=4
        elif datas["METRIC"][i] == "TruePositiveRate":
            comptMETRIC=5
        elif datas["METRIC"][i] == "FalsePositiveRate":
            comptMETRIC=6
        elif datas["METRIC"][i] == "Strength":
            comptMETRIC=7

        lineNumber = comptPATTERNSTYPE+comptREPRESENTATION*2+comptMETRIC*4
        #modify the dataframe on the line lineNumber 
        dfActuel["DATASET"][lineNumber] = datas["DATASET"][i]
        dfActuel["PATTERNSTYPE"][lineNumber] = datas["PATTERNSTYPE"][i]
        dfActuel["REPRESENTATION"][lineNumber] = datas["REPRESENTATION"][i]
        dfActuel["METRIC"][lineNumber] = datas["METRIC"][i]
        
        lineNumber2 = comptREPRESENTATION+comptMETRIC*2
        dfActuel2["DATASET"][lineNumber2] = datas["DATASET"][i]
        dfActuel2["REPRESENTATION"][lineNumber2] = datas["REPRESENTATION"][i]
        dfActuel2["METRIC"][lineNumber2] = datas["METRIC"][i]
        if datas["K"][i] == 10:
            #modify the dataframe on the line lineNumber and the column 10
            dfActuel["10"][lineNumber] = datas["K_OPTIMAL"][i]  
            #si la representation est binaire on fait + dans le dataset 2
            if comptPATTERNSTYPE == 0:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]+datas["K_OPTIMAL"][i]
            else:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]-datas["K_OPTIMAL"][i]
        elif datas["K"][i] == 25:   
            dfActuel["25"][lineNumber] = datas["K_OPTIMAL"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]+datas["K_OPTIMAL"][i]
            else:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]-datas["K_OPTIMAL"][i]
        elif datas["K"][i] == 50:
            dfActuel["50"][lineNumber] = datas["K_OPTIMAL"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]+datas["K_OPTIMAL"][i]
            else:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]-datas["K_OPTIMAL"][i]
        elif datas["K"][i] == 100:
            dfActuel["100"][lineNumber] = datas["K_OPTIMAL"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]+datas["K_OPTIMAL"][i]
            else:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]-datas["K_OPTIMAL"][i]
        elif datas["K"][i] == 200:
            dfActuel["200"][lineNumber] = datas["K_OPTIMAL"][i]
            if comptPATTERNSTYPE == 0:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]+datas["K_OPTIMAL"][i]
            else:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]-datas["K_OPTIMAL"][i]
        
    #Save each dataframe using the function dataframeToLatex
    dataframeToLatex(dfFOPPA,"../results/NewResul/dfFOPPATKMEANS")
    dataframeToLatex(dfMUTAG,"../results/NewResul/dfMUTAGKMEANS")
    dataframeToLatex(dfPTC,"../results/NewResul/dfPTCKMEANS")
    dataframeToLatex(dfNCI1,"../results/NewResul/dfNCI1KMEANS")
    dataframeToLatex(dfDD,"../results/NewResul/dfDDKMEANS")

    dataframeToLatex(dfFOPPA2,"../results/NewResul/dfFOPPAKMEANSDIFF")
    dataframeToLatex(dfMUTAG2,"../results/NewResul/dfMUTAGKMEANSDIFF")
    dataframeToLatex(dfPTC2,"../results/NewResul/dfPTCKMEANSDIFF")
    dataframeToLatex(dfNCI12,"../results/NewResul/dfNCI1KMEANSDIFF")
    dataframeToLatex(dfDD2,"../results/NewResul/dfDDKMEANSDIFF")

def tableRepresentation(datas):
    dfFOPPA = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","PATTERNSTYPE","REPRESENTATION","10","25","50","100","200"])
    dfMUTAG = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","PATTERNSTYPE","REPRESENTATION","10","25","50","100","200"])
    dfPTC = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","PATTERNSTYPE","REPRESENTATION","10","25","50","100","200"])
    dfNCI1 = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","PATTERNSTYPE","REPRESENTATION","10","25","50","100","200"])
    dfDD = pd.DataFrame(index=range(0,32),columns=["DATASET","METRIC","PATTERNSTYPE","REPRESENTATION","10","25","50","100","200"])
    dfFOPPA2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","PATTERNSTYPE","10","25","50","100","200"])
    dfMUTAG2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","PATTERNSTYPE","10","25","50","100","200"])
    dfPTC2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","PATTERNSTYPE","10","25","50","100","200"])
    dfNCI12 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","PATTERNSTYPE","10","25","50","100","200"])
    dfDD2 = pd.DataFrame(index=range(0,16),columns=["DATASET","METRIC","PATTERNSTYPE","10","25","50","100","200"])

    #fill the dataframe with zeros
    for i in range(0,16):
        dfDD2["10"][i] = 0
        dfDD2["25"][i] = 0
        dfDD2["50"][i] = 0
        dfDD2["100"][i] = 0
        dfDD2["200"][i] = 0
        dfFOPPA2["10"][i] = 0
        dfFOPPA2["25"][i] = 0
        dfFOPPA2["50"][i] = 0
        dfFOPPA2["100"][i] = 0
        dfFOPPA2["200"][i] = 0
        dfMUTAG2["10"][i] = 0
        dfMUTAG2["25"][i] = 0
        dfMUTAG2["50"][i] = 0
        dfMUTAG2["100"][i] = 0
        dfMUTAG2["200"][i] = 0
        dfNCI12["10"][i] = 0
        dfNCI12["25"][i] = 0
        dfNCI12["50"][i] = 0
        dfNCI12["100"][i] = 0
        dfNCI12["200"][i] = 0
        dfPTC2["10"][i] = 0
        dfPTC2["25"][i] = 0
        dfPTC2["50"][i] = 0
        dfPTC2["100"][i] = 0
        dfPTC2["200"][i] = 0

    comptDataset=0
    comptMETRIC=0
    comptREPRESENTATION=0
    comptPATTERNSTYPE=0

    for i in range(len(datas)):
        dfActuel=None
        dfActuel2=None
        # case on the value of the column DATASET
        if datas["DATASET"][i] == "MUTAG":
            dfActuel = dfMUTAG
            dfActuel2 = dfMUTAG2
        elif datas["DATASET"][i] == "PTC":
            dfActuel = dfPTC
            dfActuel2 = dfPTC2
        elif datas["DATASET"][i] == "NCI1":
            dfActuel = dfNCI1
            dfActuel2 = dfNCI12
        elif datas["DATASET"][i] == "DD":
            dfActuel = dfDD
            dfActuel2 = dfDD2
        elif datas["DATASET"][i] == "FOPPA":
            dfActuel = dfFOPPA
            dfActuel2 = dfFOPPA2
        #case on the type of the pattern
        if datas["PATTERNSTYPE"][i] == "EXTRACTED":
            comptPATTERNSTYPE=0
        else:
            comptPATTERNSTYPE=1
        
        #case on the representation*
        if datas["REPRESENTATION"][i] == "BINARY":
            comptREPRESENTATION=0
        else:
            comptREPRESENTATION=1

        #case on the metric
        if datas["METRIC"][i] == "Growth":
            comptMETRIC=0
        elif datas["METRIC"][i] == "Support":
            comptMETRIC=1
        elif datas["METRIC"][i] == "Unusualness":
            comptMETRIC=2
        elif datas["METRIC"][i] == "Generalization":
            comptMETRIC=3
        elif datas["METRIC"][i] == "OddsRatio":
            comptMETRIC=4
        elif datas["METRIC"][i] == "TruePositiveRate":
            comptMETRIC=5
        elif datas["METRIC"][i] == "FalsePositiveRate":
            comptMETRIC=6
        elif datas["METRIC"][i] == "Strength":
            comptMETRIC=7

        lineNumber = comptPATTERNSTYPE*2+comptREPRESENTATION+comptMETRIC*4
        #modify the dataframe on the line lineNumber 
        dfActuel["DATASET"][lineNumber] = datas["DATASET"][i]
        dfActuel["PATTERNSTYPE"][lineNumber] = datas["PATTERNSTYPE"][i]
        dfActuel["REPRESENTATION"][lineNumber] = datas["REPRESENTATION"][i]
        dfActuel["METRIC"][lineNumber] = datas["METRIC"][i]
        
        lineNumber2 = comptPATTERNSTYPE+comptMETRIC*2
        dfActuel2["DATASET"][lineNumber2] = datas["DATASET"][i]
        dfActuel2["PATTERNSTYPE"][lineNumber2] = datas["PATTERNSTYPE"][i]
        dfActuel2["METRIC"][lineNumber2] = datas["METRIC"][i]
        if datas["K"][i] == 10:
            #modify the dataframe on the line lineNumber and the column 10
            dfActuel["10"][lineNumber] = datas["F1SCORE"][i]  
            #si la representation est binaire on fait + dans le dataset 2
            if comptREPRESENTATION == 0:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["10"][lineNumber2] = dfActuel2["10"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 25:   
            dfActuel["25"][lineNumber] = datas["F1SCORE"][i]
            if comptREPRESENTATION == 0:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["25"][lineNumber2] = dfActuel2["25"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 50:
            dfActuel["50"][lineNumber] = datas["F1SCORE"][i]
            if comptREPRESENTATION == 0:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["50"][lineNumber2] = dfActuel2["50"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 100:
            dfActuel["100"][lineNumber] = datas["F1SCORE"][i]
            if comptREPRESENTATION == 0:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["100"][lineNumber2] = dfActuel2["100"][lineNumber2]-datas["F1SCORE"][i]
        elif datas["K"][i] == 200:
            dfActuel["200"][lineNumber] = datas["F1SCORE"][i]
            if comptREPRESENTATION == 0:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]+datas["F1SCORE"][i]
            else:
                dfActuel2["200"][lineNumber2] = dfActuel2["200"][lineNumber2]-datas["F1SCORE"][i]
        
    #Save each dataframe using the function dataframeToLatex
    dataframeToLatex(dfFOPPA,"../results/NewResul/dfFOPPATypeRepresentation")
    dataframeToLatex(dfMUTAG,"../results/NewResul/dfMUTAGTypeRepresentation")
    dataframeToLatex(dfPTC,"../results/NewResul/dfPTCTypeRepresentation")
    dataframeToLatex(dfNCI1,"../results/NewResul/dfNCI1TypeRepresentation")
    dataframeToLatex(dfDD,"../results/NewResul/dfDDTypeRepresentation")
    dataframeToLatex(dfFOPPA2,"../results/NewResul/dfFOPPATypeRepresentationDIFF")
    dataframeToLatex(dfMUTAG2,"../results/NewResul/dfMUTAGTypeRepresentationDIFF")
    dataframeToLatex(dfPTC2,"../results/NewResul/dfPTCTypeRepresentationDIFF")
    dataframeToLatex(dfNCI12,"../results/NewResul/dfNCI1TypeRepresentationDIFF")
    dataframeToLatex(dfDD2,"../results/NewResul/dfDDTypeRepresentationDIFF")


TableMetric(datas)
TablePattern(datas)
tableRepresentation(datas)
TableKValues(datas)