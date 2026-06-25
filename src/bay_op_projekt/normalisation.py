import torch

def normalize(X_raw, bounds):
    """Normiert auf [0, 1]"""
    return (torch.tensor(X_raw, dtype=torch.double) - bounds[0]) / (bounds[1] - bounds[0])


def standardize_Y(Y_raw):
    Y = torch.tensor(Y_raw, dtype=torch.double)
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)   # (n,) → (n, 1)
    mean = Y.mean()
    std = Y.std()
    return (Y - mean) / std, mean, std

