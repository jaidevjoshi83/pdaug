import matplotlib.pyplot as plt
import networkx as nx

import pandas as pd





"""
import matplotlib.pyplot as plt
import networkx as nx

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

print df

G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])



G = nx.balanced_tree(3, 5)
pos = graphviz_layout(G, prog='twopi', args='')
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.axis('equal')
plt.show()
"""


# Author: Aric Hagberg (hagberg@lanl.gov)
import matplotlib.pyplot as plt
import networkx as nx

df = pd.read_csv('out1.tsv', sep='\t')
print df.shape
df.drop_duplicates
print df.shape

G = nx.Graph()

file = open('lev.tsv')

lines = file.readlines()[1:]

for line in lines:

	line = line.strip('\n')
	line =  line.split('\t')

	G.add_edge(line[0], line[1], weight=float(line[2]))


elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]


pos = nx.spring_layout(G)#pos = nx.kamada_kawai_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=10)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,width=1)
nx.draw_networkx_edges(G, pos, edgelist=esmall,width=1, alpha=0.1, edge_color='b', style='dashed')

# labels
#nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()