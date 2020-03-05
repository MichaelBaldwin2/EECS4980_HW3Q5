import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Load polblogs, covert to undirected, make new graph to unfreeze
directedG = nx.read_gml('polblogs.gml')
undirectedG = directedG.to_undirected(reciprocal=True, as_view=True)
G = nx.Graph(undirectedG)

# Remove all components that have less than 3 nodes
for component in list(nx.connected_components(G)):
    if len(component) < 3:
        for node in component:
            G.remove_node(node)

# Interactive, start figure, draw to figure
plt.figure()
nx.draw(G, with_labels=False)

# Ratio
m = nx.number_of_edges(G)
n = nx.number_of_nodes(G)
secondEv = nx.fiedler_vector(G)
ratioCutClu = np.zeros(n)
ratioCutClu[np.where(secondEv > 0)] = 1
ratioCutClu[np.where(secondEv <= 0)] = 2
print('Ratio Cut 1: ')
print(np.where(ratioCutClu == 1)[0] + 1)
print('Ratio Cut 2: ')
print(np.where(ratioCutClu == 2)[0] + 1)

# Normalized
lapPB = nx.laplacian_matrix(G).todense()
degPB = np.diag(lapPB)
secondEvNorm = nx.fiedler_vector(G, normalized=True)
secondEvNorm = np.sqrt(degPB) * secondEvNorm
normCutClu = np.zeros(n)
normCutClu[np.where(secondEvNorm > 0)] = 1
normCutClu[np.where(secondEvNorm <= 0)] = 2
print('Normalized Cut 1:')
print(np.where(normCutClu == 1)[0] + 1)
print('Normalized Cut 2:')
print(np.where(normCutClu == 2)[0] + 1)

# Modularity
modularity = nx.adjacency_matrix(G).todense() - np.outer(degPB, degPB) / (2 * m)
print(modularity)
[eigenValue, eigenVector] = np.linalg.eigh(modularity)
print(eigenValue)
print(eigenVector)
EV1 = eigenVector[:, np.argmax(eigenValue)]
print(np.shape(EV1))
EV1 = np.ravel(EV1)
print(np.shape(EV1))
modClusturing = np.zeros(n)
modClusturing[np.where(EV1 > 0)] = 1
modClusturing[np.where(EV1 <= 0)] = 2
print('Modularity 1: ')
print(np.where(modClusturing == 1)[0] + 1)
print('Modularity 2: ')
print(np.where(modClusturing == 2)[0] + 1)
