


def Res(InFile, O):

    f = open(InFile)
    lines = f.readlines()


    w = open('O', 'w')

    for line in lines:
        print (line)
        w.write(line)



if __name__=='__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile",
                        required=True,
                        default=None,
                        help="pep file")

    parser.add_argument("-O",
                    required=False,
                    default="jai.fasta",
                    help="pep file")
             
    args = parser.parse_args()
    Res(args.InFile, args.O)





