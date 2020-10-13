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

def Decriptor_generator(InFile, Lamda, Weight, DesType, Out_file):

    list_pep_name = []
    f = open(InFile)
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


        if DesType == 'PAAC':
            DS = protein.GetPAAC(lamda=int(Lamda), weight=float(Weight))
        elif DesType == 'APAAC':
            DS = protein.GetAPAAC(lamda=int(Lamda), weight=float(Weight))
        elif DesType == 'CTD':
            DS = protein.GetCTD()
        elif DesType == 'DPComp':
            DS = protein.GetDPComp()
        elif DesType == 'GearyAuto':
            DS = protein.GetGearyAuto()
        elif DesType == 'MoranAuto':
            DS = protein.GetMoranAuto()
        elif DesType == 'MoreauBrotoAuto':
            DS = protein.GetMoreauBrotoAuto()
        elif DesType == 'QSO':
            DS = protein.GetQSO()
        elif DesType == 'SOCN':
            DS = protein.GetSOCN()
        elif DesType == 'TPComp':
            DS = protein.GetTPComp()
        elif DesType == 'All':
            DS_1 = protein.GetPAAC(lamda=int(Lamda), weight=float(Weight))
            DS_2 = protein.GetAPAAC(lamda=int(Lamda), weight=float(Weight))
            DS_3 = protein.GetCTD()
            DS_4 = protein.GetDPComp()
            DS_5 = protein.GetGearyAuto()
            DS_6 = protein.GetMoranAuto()
            DS_7 = protein.GetMoreauBrotoAuto()
            DS_8 = protein.GetQSO()
            DS_9 = protein.GetSOCN()
            DS_10 = protein.GetTPComp()

            DS = {}

            for D in (DS_1, DS_2, DS_3, DS_4, DS_5, DS_6, DS_7, DS_8, DS_9, DS_10):
                DS.update(D)

            df  = pd.DataFrame(DS, index=[0])
            out_df = pd.concat([out_df, df], axis=0)

        else:
            pass

    if DesType == 'BinaryDescriptor':
        out_df = BinaryDescriptor(list_pep_name)

    out_df.to_csv(Out_file, index=False, sep='\t')


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
                        
    parser.add_argument("-t", "--DesType",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   

    parser.add_argument("-O", "--Out_file",
                        required=False,  
                        default="Out.tsv",
                        help="Path to target tsv file")  
                              
    args = parser.parse_args()
    Decriptor_generator(args.InFile, args.Lamda, args.Weight, args.DesType, args.Out_file)

   