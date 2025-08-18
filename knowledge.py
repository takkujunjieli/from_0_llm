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

print("\n=== Test completed successfully! ===")