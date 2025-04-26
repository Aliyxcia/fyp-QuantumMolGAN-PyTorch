import math
import torch

def extract_upper_triangular_no_diag(tensor, device):
    batch_size = tensor.shape[0]
    n = tensor.shape[1]  # assuming square matrix, so n=9 in your case
    
    # Number of elements in upper triangular part without diagonal is n*(n-1)/2
    num_elements = int(n * (n - 1) / 2)  # should be 36 for 9x9 matrix
    
    # Initialize the output tensor
    result = torch.zeros((batch_size, num_elements), device=device, dtype=tensor.dtype)
    
    # Fill the output tensor with upper triangular values
    for b in range(batch_size):
        idx = 0
        for i in range(n):
            for j in range(i+1, n):  # start from i+1 to exclude diagonal
                result[b, idx] = tensor[b, i, j]
                idx += 1
    
    return result


def restore_from_upper_triangular_no_diag(flattened_tensor, device):
    """
    Restore a batch of symmetric square tensors from their flattened upper triangular values (excluding diagonal).
    
    Args:
        flattened_tensor: Tensor of shape (batch_size, num_elements) containing the flattened upper triangular values
        device: Device to create the tensor on
        
    Returns:
        Tensor of shape (batch_size, n, n) with symmetric values (upper triangle mirrors to lower triangle)
    """
    batch_size = flattened_tensor.shape[0]
    num_elements = flattened_tensor.shape[1]
    
    # Solve n*(n-1)/2 = num_elements for n
    # This gives us: n² - n - 2*num_elements = 0
    # We can use the quadratic formula: n = (1 + sqrt(1 + 8*num_elements))/2
    n = int((1 + math.sqrt(1 + 8 * num_elements)) / 2)
    
    # Create output tensor initialized with zeros
    result = torch.zeros((batch_size, n, n), device=device, dtype=flattened_tensor.dtype)
    
    # Fill the upper triangular part (excluding diagonal)
    for b in range(batch_size):
        idx = 0
        for i in range(n):
            for j in range(i+1, n):  # start from i+1 to exclude diagonal
                result[b, i, j] = flattened_tensor[b, idx]
                # Make it symmetric by filling lower triangular part
                result[b, j, i] = flattened_tensor[b, idx]
                idx += 1
    
    return result