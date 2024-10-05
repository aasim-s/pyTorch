import torch

print(f"gpu available? {torch.cuda.is_available()}")
print(f"count if yes: {torch.cuda.device_count()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

tensor = torch.tensor([1., 2.])
print(f"default {tensor}, {tensor.device}")

tensor_on_gpu = tensor.to(device)
print(f"on gpu {tensor_on_gpu}")

print(f"back to cpu {tensor_on_gpu.cpu()}")
