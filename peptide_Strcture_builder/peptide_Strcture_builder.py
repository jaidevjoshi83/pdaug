import tempfile
import pandas as pd
import shutil
import sys
import glob
import fragbuilder
from fragbuilder import peptide
import os


def read_pep_file(pep_infile):
    
    pep_infile = pep_infile
    df = pd.read_csv(pep_infile)
    list_pep_name = []
    list_class_label = []
    list_pep_name = df[df.columns[0]]
    return list_pep_name

def structure_gen(pep_seq, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print "Structure being Generated !"
    b = len(pep_seq)

    for seq in pep_seq:
        print seq
        pep = peptide.Peptide(seq, nterm = "charged", cterm = "neutral")
        pep.regularize()
        pep.write_pdb(os.path.join(out_dir, seq+".pdb"))
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, os.path.join(out_dir, seq+".pdb"))  
        mol.AddHydrogens()       
        obConversion.WriteFile(mol, os.path.join(out_dir,seq+".sdf"))
    
    print "Structure Generation Finished !"
    
def main_process(str_pep_file,str_des_out, out_dir):
      
    str_pep_file = str_pep_file
    str_des_out = str_des_out
    
    my_pep = read_pep_file(str_pep_file) 
    structure_gen(my_pep, out_dir)
    
    
if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pep",
                        required=True,
                        default=None,
                        help="pep file")
                         

    parser.add_argument("-o", "--OutDir",
                        required=None,
                        default=os.path.join(os.getcwd(),'OutDir'),
                        help="Path to out file")  
                                               
    args = parser.parse_args()
    Str_DS_class().main_process(args.pep, args.OutDir)
   
    print " Structure Generation and Discriptor Calculation finished successfully"
    
    
   
    
