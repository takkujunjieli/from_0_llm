import torch

float32_info = torch.finfo(torch.float32)  # @inspect float32_info
float16_info = torch.finfo(torch.float16)  # @inspect float16_info
bfloat16_info = torch.finfo(torch.bfloat16)  # @inspect bfloat16_info

print("=== PyTorch Floating Point Data Type Information ===")
print(f"float32: {float32_info}")
print(f"float16: {float16_info}")
print(f"bfloat16: {bfloat16_info}")

print("\n=== Tensor Creation and Properties ===")
x = torch.zeros(32, 32)
print(f"Created tensor shape: {x.shape}")
print(f"Tensor dtype: {x.dtype}")
print(f"Tensor device: {x.device}")
print(f"Tensor requires grad: {x.requires_grad}")

assert x.device == torch.device("cpu")
print("\n✅ Assertion passed: Tensor is on CPU device")

print("\n=== Device Information ===")
if torch.cuda.is_available():
    print("CUDA is available")
    num_gpus = torch.cuda.device_count()  # @inspect num_gpus
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties
        print(f"GPU {i} properties: {properties}")
    
    # Test GPU operations
    print("\n=== GPU Operations ===")
    memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated
    print(f"Memory allocated: {memory_allocated}")
    print("Move the tensor to GPU memory (device 0).")
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)
    print("✅ Tensor successfully moved to GPU")
    
    print("Or create a tensor directly on the GPU:")
    z = torch.zeros(32, 32, device="cuda:0")
    new_memory_allocated = torch.cuda.memory_allocated()  # @inspect new_memory_allocated
    memory_used = new_memory_allocated - memory_allocated  # @inspect memory_used
    print(f"Memory used: {memory_used}")
    assert memory_used == 2 * (32 * 32 * 4)  # 2 32x32 matrices of 4-byte floats
    print("✅ GPU memory allocation test passed")
    
else:
    print("CUDA is not available - running on CPU only")
    print("Number of GPUs: 0")
    print("Memory allocated: 0")
    
    # Test CPU operations instead
    print("\n=== CPU Operations ===")
    print("Creating tensors on CPU:")
    y = x.to("cpu")
    assert y.device == torch.device("cpu")
    print("✅ Tensor successfully created on CPU")
    
    z = torch.zeros(32, 32, device="cpu")
    print("✅ CPU tensor creation successful")
    
    # Test tensor operations on CPU
    result = torch.matmul(x, y)
    print(f"✅ Matrix multiplication successful, result shape: {result.shape}")

print("\n=== Test completed successfully! ===")