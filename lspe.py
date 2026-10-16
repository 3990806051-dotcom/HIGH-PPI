"""Structural and positional encodings for the protein-level PPI graph.

The encodings are computed from the graph supplied by the caller.  Training
code should therefore pass the training-only PPI graph when it wants to avoid
using validation edges in the positional features.
"""

import torch


def _as_edge_index(edge_index):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return edge_index


def _undirected_adjacency(edge_index, num_nodes):
    """Return a binary, symmetric adjacency matrix on CPU."""
    edge_index = _as_edge_index(edge_index)
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel():
        src, dst = edge_index
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def compute_rwpe(edge_index, num_nodes, walk_length=8):
    """Compute diagonal random-walk powers for every graph node.

    Column k is diag(P**(k+1)), where P is the row-normalized adjacency.
    Isolated nodes receive zero values.
    """
    if walk_length < 1:
        raise ValueError("walk_length must be at least 1")
    adjacency = _undirected_adjacency(edge_index, num_nodes)
    degree = adjacency.sum(dim=1)
    transition = torch.zeros_like(adjacency)
    nonisolated = degree > 0
    transition[nonisolated] = adjacency[nonisolated] / degree[nonisolated, None]

    power = torch.eye(num_nodes, dtype=torch.float32)
    columns = []
    for _ in range(walk_length):
        power = transition @ power
        columns.append(torch.diagonal(power))
    return torch.stack(columns, dim=1)


def compute_lappe(edge_index, num_nodes, pe_dim=8):
    """Compute the first non-constant normalized-Laplacian eigenvectors."""
    if pe_dim < 1:
        raise ValueError("pe_dim must be at least 1")
    adjacency = _undirected_adjacency(edge_index, num_nodes)
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.zeros_like(degree)
    nonisolated = degree > 0
    inv_sqrt_degree[nonisolated] = degree[nonisolated].rsqrt()
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    laplacian = torch.eye(num_nodes, dtype=torch.float32) - normalized_adjacency
    _, eigenvectors = torch.linalg.eigh(laplacian)

    available = min(pe_dim, max(0, num_nodes - 1))
    pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float32)
    if available:
        # Skip the smallest, approximately constant eigenvector.
        pe[:, :available] = eigenvectors[:, 1:1 + available]
    return pe


def build_lspe(edge_index, num_nodes, rwpe_dim=8, lappe_dim=8):
    """Return concatenated RWPE and LapPE with shape [num_nodes, pe_dim]."""
    rwpe = compute_rwpe(edge_index, num_nodes, walk_length=rwpe_dim)
    lappe = compute_lappe(edge_index, num_nodes, pe_dim=lappe_dim)
    pe = torch.cat((rwpe, lappe), dim=1)
    # Keep the scale of the two fixed encodings comparable to learned features.
    pe = (pe - pe.mean(dim=0, keepdim=True)) / (
        pe.std(dim=0, keepdim=True, unbiased=False) + 1e-6
    )
    return pe
