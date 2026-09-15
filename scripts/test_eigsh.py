import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

num_nodes = 1000
row = np.random.randint(0, num_nodes, 2000)
col = np.random.randint(0, num_nodes, 2000)
data = np.ones(len(row))
adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
adj.sum_duplicates()
adj.data = np.ones_like(adj.data)
d = np.array(adj.sum(1)).flatten()
d_inv_sqrt = np.zeros_like(d)
mask = d > 0
d_inv_sqrt[mask] = np.power(d[mask], -0.5)
d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
laplacian = sp.eye(num_nodes) - norm_adj

# create disconnected components
laplacian = laplacian.toarray()
laplacian[:10, :10] = 0
laplacian[10:, :10] = 0
laplacian[:10, 10:] = 0
laplacian = sp.csr_matrix(laplacian)

t0 = time.time()
eigvals1, eigvecs1 = np.linalg.eigh(laplacian.toarray())
print(f"eigh: {time.time() - t0:.4f}s")

t0 = time.time()
try:
    eigvals2, eigvecs2 = eigsh(laplacian.astype(np.float64), k=9, sigma=-1e-3)
    print(f"eigsh (sigma): {time.time() - t0:.4f}s")
except Exception as e:
    print(f"eigsh failed: {e}")
