import numpy as np
from datamining.clustering import *
from datamining.subspace_clustering import *
import matplotlib.pyplot as plt
import time
import torch
import torchvision.datasets as torch_datasets
import umap

# loading mnist
mnist = torch_datasets.MNIST(root="data", train=True, download=True, transform=None)
data = np.asarray(mnist.data[:, :], dtype=np.float32)
data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

reducer = umap.UMAP()
data = np.asarray(reducer.fit_transform(data), dtype=np.float32, order="c")
a = 0.1
# plt.scatter(data[:, 0], data[:, 1], alpha=a)
# plt.show()

eps = .5  # 2.5
minPts = 400

torch.cuda.synchronize()


t0 = time.time()
C = PROCLUS(data, a=eps, minPts=minPts)
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))

t0 = time.time()
C = GPU_DBSCAN(data, eps=eps, minPts=minPts)
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))
# plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
# plt.show()

t0 = time.time()
C = G_DBSCAN(data, eps=eps, minPts=minPts)
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))
# plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
# plt.show()

# t0 = time.time()
# C = EGG_SynC(data, eps=5)
# t1 = time.time()
# print(t1 - t0, C, np.unique(C, return_counts=True))
# plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
# plt.show()

t0 = time.time()
C = SynC(data, eps=5, lam=1.-10.**(-3))
t1 = time.time()
print(t1 - t0, C, np.unique(C, return_counts=True))
plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
plt.show()


t0 = time.time()
C, D = GPU_FAST_PROCLUS(data, k=10, l=2, A=100, B=50, min_deviation=0.5, termination_rounds=10)
t1 = time.time()
print(t1 - t0, C, D, np.unique(C, return_counts=True))
# plt.scatter(data[:, 0], data[:, 1], c=C, alpha=a)
# plt.show()
