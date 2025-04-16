import torch

def extract_upper_triangular_no_diag(tensor, device):
    batch_size = tensor.shape[0]
    n = tensor.shape[1]  # Matrix dimension (9 in your case)
    edge_features = tensor.shape[3]  # Number of edge features (5 in your case)
    
    # Number of elements in upper triangular part without diagonal is n*(n-1)/2
    num_elements = int(n * (n - 1) / 2)  # should be 36 for 9x9 matrix
    
    # Initialize the output tensor
    result = torch.zeros((batch_size, num_elements, edge_features), device=device)
    
    # Fill the output tensor with upper triangular values
    for b in range(batch_size):
        idx = 0
        for i in range(n):
            for j in range(i+1, n):  # start from i+1 to exclude diagonal
                result[b, idx] = tensor[b, i, j]
                idx += 1
    
    return result