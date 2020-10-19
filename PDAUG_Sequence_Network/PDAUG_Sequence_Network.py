import Levenshtein
import matplotlib.pyplot as plt
import networkx as nx
import os


def SeqSimilarityNetwork(InFile, OutFile):

    f = open(InFile)
    lines = f.readlines()

    record = []
    seq = []

    G = nx.Graph()

    for line in lines:

        if ">" in line:
            record.append(line.strip('\n'))
        else:
            seq.append(line.strip('\n'))

    for x, i in enumerate(seq):
        for X, I in enumerate(seq):
            L = Levenshtein.ratio(i, I )
            if  L >= 0.4:
                G.add_edge(record[x], record[X], weight=float(Levenshtein.ratio(i, I )))

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.4]

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=elarge,width=1)
    plt.axis('off')

    plt.savefig(OutFile)



if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    parser.add_argument("-O","--OutFile", required=False, help="HTML out file", default="out.png")
    args = parser.parse_args()

    SeqSimilarityNetwork(args.InFile, args.OutFile)


