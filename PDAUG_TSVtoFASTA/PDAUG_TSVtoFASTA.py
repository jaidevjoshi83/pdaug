
import pandas as pd


def TSVtoFASTA(infile, method, firstdatafile, seconddatafile, outfile, clmpepid, slcclasslabel, peps):


    fn = [firstdatafile, seconddatafile]


    df = pd.read_csv(infile, sep="\t")
    if clmpepid == None:
        pass
    else:
        names = df[clmpepid].tolist()

    peps = df[peps].tolist()
    
    if method == "withoutlabel":
        f = open(outfile,'w')
        if clmpepid is not None:
            for i,n in enumerate(peps):
                f.write(">"+names[i]+'\n')
                f.write(n+'\n')
            f.close()
        else:
            for i,n in enumerate(peps):
                f.write(">"+str(i)+'\n')
                f.write(n+'\n')
            f.close()
                 
    elif method == "withlabel":
        labels = df[slcclasslabel].tolist()

        label = list(set(labels))
        
        if clmpepid is None:
            for i, l in enumerate(label):
                f = open(fn[i],'w')
                print('ok1')
                for i, L in enumerate(labels):
                    if l == L:
                        f.write(">"+str(i)+"_"+str(l)+'\n')
                        f.write(peps[i]+'\n')
            f.close()
        else:
            for i, l in enumerate(label):
                f = open(fn[i],'w')          
                for i, L in enumerate(labels):
                    if l == L:
                        f.write(">"+names[i]+"_"+l+'\n')
                        f.write(peps[i]+'\n')        
            f.close()

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
    parser.add_argument("-F", "--FirstDataFile", required=False, default='FirstDataFile.fasta', help="Path to target tsv file")
    parser.add_argument("-S", "--SecondDataFile", required=False, default='SecondDataFile.fasta', help="Path to target tsv file")
    parser.add_argument("-O", "--OutFile", required=False, default='OutFile.fasta', help="Path to target tsv file")
    parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
    parser.add_argument("-C", "--ClmPepID", required=False, default=None, help="Peptide Column Name")
    parser.add_argument("-L", "--SlcClassLabel", required=False, default="Class_label", help="Class Label Column Name")
    parser.add_argument("-P", "--PeptideColumn", required=True, default=None, help="Class Label Column Name")
    args = parser.parse_args()

    TSVtoFASTA(args.InFile, args.Method, args.FirstDataFile, args.SecondDataFile, args.OutFile, args.ClmPepID, args.SlcClassLabel, args.PeptideColumn)



