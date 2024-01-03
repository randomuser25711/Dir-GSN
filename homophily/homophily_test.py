import torch
from torch_geometric.datasets import (
    TUDataset,
    WikipediaNetwork
)
from utils import get_effective_homophily_directed, get_effective_homophily_undirected
import torch_geometric.transforms as transforms

# Load chameleon dataset
# dataset = WikipediaNetwork(root='data', name='chameleon', transform=transforms.NormalizeFeatures())
# dataset = dataset._data
# print(get_effective_homophily_directed(dataset.y, dataset.edge_index))
# print(get_effective_homophily_undirected(dataset.y, dataset.edge_index))


# Load NCI1 dataset
dataset = TUDataset(root='data', name='NCI1')
avg_h_dir = 0
avg_h = 0
for i, graph in enumerate(dataset):
    h_dir = get_effective_homophily_directed(graph.x, graph.edge_index)
    avg_h_dir += h_dir
    h = get_effective_homophily_undirected(graph.x, graph.edge_index)
    avg_h += h
    if h != h_dir:
        print(i, h, h_dir)
print("Average effective homophily for directed graphs: ", avg_h_dir / len(dataset))
print("Average effective homophily for undirected graphs: ", avg_h / len(dataset))
print("Average effective homophily gain: ", avg_h_dir / avg_h - 1)

