import torch

print("----scalar----")
scalar = torch.tensor(3)
print(scalar)
print("ndim:", scalar.ndim)
print("item:", scalar.item())

print("----vector----")
vector = torch.tensor([2, 4, 6, 8])
print(vector)
print("ndim:", vector.ndim)
print("shape:", vector.shape)

print("----matrix----")
matrix = torch.tensor([[1, 2], [3, -1]])
print(matrix)
print("ndim:", matrix.ndim)
print("shape:", matrix.shape)

print("----tensor----")
tensor = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 6, 8]]])
print(tensor)
print("ndim:", tensor.ndim)
print("shape:", tensor.shape)

print("----random tensor----")
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print("dtype:", random_tensor.dtype)

print("----zeros and ones----")
zeros = torch.zeros(size=(3, 4))
print(zeros)
print("dtype:", zeros.dtype)
ones = torch.ones(size=(3, 4))
print(ones)
print("dtype:", ones.dtype)

print("----range of values in tensor----")
# range has issues and so is depricated, use arange instead
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

print("----zeros or ones of shape as existing tensor----")
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
ten_ones = torch.ones_like(input=zero_to_ten)
print(ten_ones)

print("----tensor type device grad----")
float_32_tensor = torch.tensor([3., 5., 7.],
                               dtype=None,  # default None/float32 can give your type
                               device=None,  # default None/type of tensor
                               requires_grad=False)  # if true, operations on tensor are recorded
print(float_32_tensor.shape)
print(float_32_tensor.dtype)
print(float_32_tensor.device)

print("----transpose of matrix----")
tensor_A = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
print(tensor_A)
tensor_A_T_1 = torch.transpose(tensor_A, 2, 3)
print(tensor_A_T_1)
tensor_A_T_2 = tensor_A.T
print(tensor_A_T_2)

print("----min,max,mean,sum...----")
X = torch.arange(0, 100, 10)
print(f"Min: {X.min()}")
print(f"Max: {X.max()}")
print(f"Mean: {X.type(torch.float32).mean()}")
print(f"Sum: {X.sum()}")

print("----position of min/max----")
print("position/index of min: {X.argmin()}")
print("position/index of max: {X.argmax()}")


# torch.reshape(input, shape)	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
# Tensor.view(shape)	Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
# torch.stack(tensors, dim=0)	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
# torch.squeeze(input)	Squeezes input to remove all the dimenions with value 1.
# torch.unsqueeze(input, dim)	Returns input with a dimension value of 1 added at dim.
# torch.permute(input, dims)	Returns a view of the original input with its dimensions permuted (rearranged) to dims.

# torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
# torch.Tensor.numpy() - PyTorch tensor -> NumPy array.
