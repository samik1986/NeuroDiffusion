import time
import numpy as np
import scipy.sparse as sp

num_nodes = 1000
row = np.random.randint(0, num_nodes, 2000)
col = np.random.randint(0, num_nodes, 2000)

t0 = time.time()
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
dense_lap = laplacian.toarray()
eigvals, eigvecs = np.linalg.eigh(dense_lap)
print(f"Sparse overhead + eigh: {time.time() - t0:.4f}s")

t0 = time.time()
adj_dense = np.zeros((num_nodes, num_nodes))
adj_dense[row, col] = 1
adj_dense[col, row] = 1
d_dense = adj_dense.sum(axis=1)
d_inv_sqrt_dense = np.zeros_like(d_dense)
mask_dense = d_dense > 0
d_inv_sqrt_dense[mask_dense] = np.power(d_dense[mask_dense], -0.5)
d_mat_inv_sqrt_dense = np.diag(d_inv_sqrt_dense)
norm_adj_dense = d_mat_inv_sqrt_dense @ adj_dense @ d_mat_inv_sqrt_dense
laplacian_dense = np.eye(num_nodes) - norm_adj_dense
eigvals2, eigvecs2 = np.linalg.eigh(laplacian_dense)
print(f"Dense overhead + eigh: {time.time() - t0:.4f}s")
