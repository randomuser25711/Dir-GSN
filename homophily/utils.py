import torch
import torch_geometric.utils as U

def get_weighted_edge_index(adj_mat):
    edge_index = torch.nonzero(adj_mat, as_tuple=False).t()
    edge_weights = []
    for i in range(edge_index.size(1)):
        edge_weights.append(adj_mat[edge_index[0][i]][edge_index[1][i]].item())
    edge_weights = torch.tensor(edge_weights)
    return edge_index, edge_weights

def get_effective_homophily_undirected(y, edge_index):
    """
    Return the effective homophily of a graph for k = 2.
    """
    dense_adj_matrix = U.to_dense_adj(edge_index).squeeze(0)

    # Transpose of A
    t_dense_adj_matrix = dense_adj_matrix.t()

    adj_mat = torch.matmul(dense_adj_matrix, dense_adj_matrix)
    adj_mat += torch.matmul(t_dense_adj_matrix, dense_adj_matrix)
    adj_mat += torch.matmul(dense_adj_matrix, t_dense_adj_matrix)
    adj_mat += torch.matmul(t_dense_adj_matrix, t_dense_adj_matrix)
    adj_mat *= 0.25

    edge_index, edge_weights = get_weighted_edge_index(adj_mat)

    return get_node_homophily(y, edge_index, edge_weights)

def get_effective_homophily_directed(y, edge_index): 
    """
    Return the effective homophily of a graph for k <= 2.
    """
    dense_adj_matrix = U.to_dense_adj(edge_index).squeeze(0)
    
    # Transpose of A
    t_dense_adj_matrix = dense_adj_matrix.t()
    
    h = 0

    # Compute for k = 1
    e_index, edge_weights = get_weighted_edge_index(dense_adj_matrix) # A
    h = max(h, get_node_homophily(y, e_index, edge_weights))
    
    e_index, edge_weights = get_weighted_edge_index(t_dense_adj_matrix) # A^t
    h = max(h, get_node_homophily(y, e_index, edge_weights))

    # Compute for k = 2
    adj_mat = torch.matmul(dense_adj_matrix, dense_adj_matrix) # A^2
    e_index, edge_weights = get_weighted_edge_index(adj_mat)
    h = max(h, get_node_homophily(y, e_index, edge_weights))

    e_index = torch.matmul(t_dense_adj_matrix, dense_adj_matrix) # A^t * A
    e_index, edge_weights = get_weighted_edge_index(adj_mat)
    h = max(h, get_node_homophily(y, e_index, edge_weights))

    adj_mat = torch.matmul(dense_adj_matrix, t_dense_adj_matrix) # A * A^t
    e_index, edge_weights = get_weighted_edge_index(adj_mat)
    h = max(h, get_node_homophily(y, e_index, edge_weights))

    adj_mat = torch.matmul(t_dense_adj_matrix, t_dense_adj_matrix) # A^t * A^t
    e_index, edge_weights = get_weighted_edge_index(adj_mat)
    h = max(h, get_node_homophily(y, e_index, edge_weights))

    return h
    
def get_node_homophily(y, edge_index, edge_weight=None):
    """
    Return the weighted node homophily, according to the weights in the provided adjacency matrix.
    """
    src, dst = edge_index

    lst = []
    sumi = {}
    t_sumi = {}
    
    for i in range(y.size(0)):
        sumi[i] = 0
        t_sumi[i] = 0

    for j in range(edge_index.size(1)):
        t_sumi[src[j].item()] += edge_weight[j].item()
        if torch.equal(y[src[j]], y[dst[j]]):
            sumi[src[j].item()] += edge_weight[j].item()

    for i in range(y.size(0)):
        if t_sumi[i] == 0:
            lst.append(0)
        else:
            lst.append(sumi[i] / t_sumi[i])
        # print(dict[i], wsum, dict[i] / wsum)
    lst = torch.tensor(lst)
    # print(lst.mean().item())
    return lst.mean().item()