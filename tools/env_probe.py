import torch

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print('Available devices', available_gpus)
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


tensor = tensor.to('cuda')

print(f"Device tensor is now stored on: {tensor.device}")
if torch.cuda.is_available():
    print('CUDA available')
    print('ALL OK')