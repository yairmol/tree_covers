import networkx as nx
from scipy.sparse.csr import csr_matrix
from numpy import linalg as la
import numpy as np

# g: nx.Graph = nx.Graph({(0, 1), (2, 3)})
# g_comp: nx.Graph = nx.complement(g)
# Lg: csr_matrix = nx.linalg.laplacian_matrix(g)
# vecs = [[0.70710678,  0.70710678,  0,          0,        ],
#         [ 0,          0,          0.70710678,  0.70710678],
#         [-0.70710678,  0.70710678,  0,          0,        ],
#         [ 0,          0,         -0.70710678,  0.70710678]]
# print(np.matmul(Lg.todense(), vecs))
# evals, evecs = la.eig(Lg.todense())
# print(evals)
# print(evecs)
# Lg_comp: csr_matrix = nx.linalg.laplacian_matrix(g_comp)
# evals, evecs = la.eig(Lg_comp.todense())
# print(evals)
# print(evecs)
# eigenvalues = sorted(nx.linalg.laplacian_spectrum(g))
# print(eigenvalues)
# print(sorted(nx.linalg.laplacian_spectrum(g_comp)))
# print(sorted([0] + [4 - ev for ev in eigenvalues[1:]]))
# nx.random_regular_graph(3, 5)
# n = 10
# g: nx.Graph = nx.star_graph(n - 1)
# eigenvalues = sorted(list(map(lambda z: z.real, nx.linalg.adjacency_spectrum(g))))
# print(eigenvalues, np.sqrt(5))
# Lg: csr_matrix = nx.linalg.adjacency_matrix(g)
# eigen_vectors = np.transpose(np.array([
#     [1 if j == i else -1 if j == i + 1 else 0 for j in range(n)]
#     for i in range(0, n - 1)
# ]))
# print(eigen_vectors)
# print(np.matmul(Lg.todense(), eigen_vectors))

# x = [
#     [0, 1, 1, 1], 0        0
#     [1, 0, 0, 0], 1     =  0
#     [1, 0, 0, 0], -1  =  0
#     [1, 0, 0, 0]] 0     0
# x1 + x2 = -x3
# [0, 1, -1, 0] \cdot [0, 0, -1, 1]
# g = nx.complete_bipartite_graph(3, 3)
g = nx.Graph({(0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 5)})
A_g = np.array([
    [1 if g.has_edge(i, j) else 0 for j in range(6)]
    for i in range(6)
])
# Ag = nx.linalg.adjacency_matrix(g).todense()
print(A_g)
evals, evecs = la.eig(A_g)
# for j in range(6):
#     min_val = 1000
#     for i in range(6):
#         if evecs[i][j] < 1e-10:
#             continue
#         min_val = min(min_val, abs(evecs[i][j]))
#     print(min_val)
#     for i in range(6):
#         evecs[i][j] = evecs[i][j]/min_val
#         if abs(evecs[i][j] - round(evecs[i][j])) < 1e-3:
#             print("here")
#             evecs[i][j] = int(round(evecs[i][j]))
# for i in range(len(evals)):
#     if abs(evals[i] - round(evals[i])) < 1e-3:
#         evals[i] = round(evals[i])
print(evals)
print(evecs)
