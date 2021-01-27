import pandas as pd
from pydpi.pypro import PyPro
import os


def BinaryDescriptor(seq):

    BinaryCode = {

    'A':"1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'C':"0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'D':"0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'E':"0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'F':"0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'G':"0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'H':"0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0",
    'I':"0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0",
    'K':"0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0",
    'L':"0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0",
    'M':"0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0",
    'N':"0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0",
    'P':"0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0",
    'Q':"0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0",
    'R':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0",
    'S':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0",
    'T':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0",
    'V':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0",
    'W':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0",
    'Y':"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1",
    }

    lines = []
    Df = []
     

    for s in seq:
        des = []
        for n in s:
            des.append(BinaryCode[n.upper()])
        lines.append(','.join(des).split(','))

    df = pd.DataFrame(lines)

    return df

def Decriptor_generator(infile, lamda, weight, maxlag, destype, out_file):

    list_pep_name = []
    f = open(infile)
    lines = f.readlines()
    
    for line in lines:
        if ">" in line:
            pass
        else:
            list_pep_name.append(line.strip('\n'))

    out_df = pd.DataFrame()

    for seq in list_pep_name:

        protein = PyPro()
        protein.ReadProteinSequence(seq)

 
        if destype == "GetAAComp":
            DS = protein.GetAAComp()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetDPComp":
            DS = protein.GetDPComp()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetTPComp":
            DS = protein.GetTPComp()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetMoreauBrotoAuto":
            DS = protein.GetMoreauBrotoAuto()
            df  = pd.DataFrame(DS, index=[0])
        elif destype =="GetMoranAuto":
            DS = protein.GetMoranAuto()
            df  = pd.DataFrame(DS, index=[0])
        elif destype =="GetGearyAuto":
            DS = protein.GetGearyAuto()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetCTD":
            DS = protein.GetCTD()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetPAAC":
            DS = protein.GetPAAC(lamda=int(lamda),  weight=float(weight))
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetAPAAC":
            DS = protein.GetAPAAC(lamda=int(lamda), weight=float(weight))
            df  = pd.DataFrame(DS, index=[0])
        elif destype =="GetSOCN":
            DS = protein.GetSOCN(maxlag=int(maxlag))
            df  = pd.DataFrame(DS, index=[0])
        elif destype =="GetQSO":
            DS = protein.GetQSO(maxlag=int(maxlag),  weight=float(weight))
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "GetTriad":
            DS = protein.GetTriad()
            df  = pd.DataFrame(DS, index=[0])
        elif destype == "All":
            DS1 = protein.GetAAComp()
            DS2 = protein.GetDPComp()
            DS3 = protein.GetTPComp()
            DS4 = protein.GetMoreauBrotoAuto()
            DS5 = protein.GetMoranAuto()
            DS6 = protein.GetGearyAuto()
            DS7 = protein.GetCTD()
            DS8 = protein.GetPAAC(lamda=int(lamda),  weight=float(weight))
            DS9 = protein.GetAPAAC(lamda=int(lamda), weight=float(weight))
            DS10 = protein.GetSOCN(maxlag=int(maxlag))
            DS11 = protein.GetQSO(maxlag=int(maxlag),  weight=float(weight))
            DS12 = protein.GetTriad()

            DS = {}

            for D in (DS1,DS2,DS3,DS4,DS5,DS6,DS7,DS8,DS9,DS10,DS11,DS12):
                print(D)
                DS.update(D)
            df  = pd.DataFrame(DS, index=[0])

        if destype == 'BinaryDescriptor':
            out_df = BinaryDescriptor(list_pep_name)
        else:
            out_df = pd.concat([out_df, df], axis=0)


    out_df.to_csv(out_file, index=False, sep='\t')


if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile",
                        required=True,
                        default=None,
                        help="pep file")

    parser.add_argument("-l", "--Lamda",
                        required=False,
                        default=50,
                        help="pep file")

    parser.add_argument("-w", "--Weight",
                        required=False,
                        default=0.5,
                        help="pep file")
    
    parser.add_argument("-m", "--MaxLag",
                        required=False,
                        default=10,
                        help="pep file")  

    parser.add_argument("-t", "--DesType",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   

    parser.add_argument("-O", "--Out_file",
                        required=False,  
                        default="Out.tsv",
                        help="Path to target tsv file")  
                              
    args = parser.parse_args()
    Decriptor_generator(args.InFile, args.Lamda, args.Weight, args.MaxLag, args.DesType, args.Out_file)

   