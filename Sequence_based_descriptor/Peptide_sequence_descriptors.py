import pandas as pd
from pydpi.pypro import PyPro
import os


def Decriptor_generator(InFile, Lamda, Weight, DesType, Out_file):

    #df = pd.read_csv(InFile, sep="\t")
    #list_pep_name = df[df.columns.tolist()[0]].tolist()

    list_pep_name = []
    f = open(InFile)
    lines = f.readlines()
    
    for line in lines:
        if ">" in line:
            pass
        else:
            list_pep_name.append(line)

    #try: 
    #    target = df[df.columns.tolist()[1]]
    #    print (target)
    #except:
    #    pass
    #

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
        else:

            print ("Enter correct")
            pass

        df  = pd.DataFrame(DS, index=[0])
        out_df = pd.concat([out_df, df], axis=0)

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
                        default=None,
                        help="pep file")

    parser.add_argument("-w", "--Weight",
                        required=False,
                        default=None,
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

   