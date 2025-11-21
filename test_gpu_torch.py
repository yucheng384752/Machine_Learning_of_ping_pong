import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    a = torch.randn((3000, 3000), device="cuda")
    b = torch.randn((3000, 3000), device="cuda")

    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()

    print("Matmul time:", t1 - t0, "sec")
else:
    print("⚠ CUDA NOT AVAILABLE")
