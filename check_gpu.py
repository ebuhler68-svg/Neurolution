import torch

print(f"torch version  : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name       : {torch.cuda.get_device_name(0)}")
else:
    print("GPU name       : N/A (CUDA not available)")
