[Back to Index 🗂️](./README.md)

# 🔥 PyTorch Parallelization tutorial

This guide focuses on **leveraging parallelization in PyTorch** to maximize computational efficiency using **GPUs and multi-core CPUs**.

<br>

## ✅ Ensuring Reproducibility
To obtain consistent results across different runs, set random seeds for PyTorch, NumPy, and Python's random module.

```python
import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Example seed
```

<br>

## 📊 GPU Monitoring (Check Memory & Usage)

### 🏃‍♂️ View GPU Utilization and Running Processes
- Inside Terminal:
    ```bash
    # Show GPU status
    nvidia-smi
    # Refresh every second
    watch -n 1 nvidia-smi
    # Compact view of GPU utilization
    gpustat
    ```

- Inside Python:
    ```python
    import torch

    # Show memory usage
    print(torch.cuda.memory_summary())
    ```

### 📃 List Active Processes Using GPUs
```bash
# Show active PIDs using GPUs
nvidia-smi pmon
```

### ❌ Kill a Specific Process Running on GPU
```bash
kill <PID>
kill -9 <PID>  # Force kill
```

<br>

## 🎛️ Using a GPU in PyTorch
To utilize GPU acceleration, move tensors and models to the GPU.

### ✅ Check for GPU Availability
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only")
```

### ➡️ Move a Model and Data to GPU
```python
# Instance my model
model = MyModel()
# Instance my data
tensor = torch.randn(3, 3)
# Define device as GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model and the data to the device
model.to(device)
tensor = tensor.to(device)
```

### ⚙️ Ensure Computation Happens on GPU
```python
# Will be computed on GPU if tensor is on GPU
result = tensor * 2
```

<br>

## 🏗️ Multi-GPU Training

### ⛓️ Using `DataParallel`
PyTorch’s `DataParallel` helps distribute computations across multiple GPUs.
```python
import torch.nn as nn

# Instance my model
model = MyModel()
# Check if more than one device is available
if torch.cuda.device_count() > 1:
    # Distribute input batches across multiple GPUs
    model = nn.DataParallel(model)
# Move the model to the device
model.to(device)
```

### ⚔️ Using `DistributedDataParallel` (DDP)
For better scalability, use DDP instead of `DataParallel`. DDP creates multiple processes, each handling a subset of the data and synchronizing gradients efficiently using NCCL (pronounced "Nickel"), an optimized GPU collective communication library developed by NVIDIA.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group using NCCL backend
dist.init_process_group("nccl")

# Wrap model with DDP
model = DDP(MyModel().to(device))
```

<br>

**Key Differences:**
| Feature                  | `DataParallel` (DP)            | `DistributedDataParallel` (DDP) |
|--------------------------|--------------------------------|--------------------------------|
| Execution Mode           | Single process, auto batch split | Multi-process, better scaling |
| GPU Communication        | Implicit synchronization        | Explicit, optimized comms via NCCL |
| Scalability              | Limited for large models       | Recommended for multi-node/multi-GPU |


<br>

## 🔀 Choosing GPUs for Computation
You can manually select which GPUs to use for training.

### 📍 Setting GPUs using `CUDA_VISIBLE_DEVICES`
Use this environment variable to restrict PyTorch to specific GPUs: <br>

- In bash:
    ```bash
    # Use only GPU 0 and GPU 1
    export CUDA_VISIBLE_DEVICES=0,1
    ```

- Inside Python:
    ```python
    import os

    # Restrict PyTorch to GPUs 0 and 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    ```

### 📌 Assigning Specific GPU

- Define a specific GPU device:
    ```python
    # Use GPU 1 explicitly
    device = torch.device("cuda:1")
    # Move model to GPU 1
    model.to(device)
    ```

- Manually assign a specific GPU for `.cuda()`:
    ```python
    # Manually choose GPU 1
    torch.cuda.set_device(1)
    # Now this will be created on GPU 1
    tensor = torch.randn(3, 3).cuda()
    ```

### 📶 Controlling GPU Order with `CUDA_DEVICE_ORDER`
By default, CUDA assigns GPU indices based on the order they are detected. To enforce ordering by PCI Bus ID, that is the order of how the GPUs are phisically mounted:
```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
```
This ensures consistent device numbering across multiple runs, avoiding mismatches when assigning GPUs.


<br>

## ⚙️ Multi-Core CPU Processing

### 🈁 Set the Number of Threads for CPU Operations
```python
import torch

# Set to the number of available CPU cores
torch.set_num_threads(8)
```

### 🔄 Efficient Data Loading with `DataLoader`
Using multiple workers speeds up data loading during training.
```python
from torch.utils.data import DataLoader, Dataset

dataloader = DataLoader(
    dataset=MyDataset(),
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Use multiple CPU cores
    pin_memory=True  # Optimize memory transfer for GPU
)
```

<br>

[Back to Index 🗂️](./README.md)