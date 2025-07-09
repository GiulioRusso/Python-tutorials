[Back to Index 🗂️](./README.md)

<center><h1>🔥 Parallel Computation</h1></center>

This guide focuses on leveraging parallelization in Python and PyTorch to maximize computational efficiency using GPUs and multi-core CPUs.

<br>
<br>
<br>

## ↔️ Multicore Parallelism

Serial vs. parallel computing. In serial computing, tasks are executed one after another on a single CPU core. In parallel computing, a large problem is divided into smaller sub-tasks that run simultaneously on multiple CPU cores, completing the work faster than doing tasks sequentially on one core. <br>
Modern CPUs have multiple processing units (cores), so running code in parallel can significantly speed up CPU-intensive tasks. In simple terms, using four cores can be like having four friends help solve a problem instead of doing it alone – the workload is divided and completed more quickly

Before parallelizing work, it’s useful to know how many CPU cores (processors) your system has. In Python use `os.cpu_count()` or `multiprocessing.cpu_count()`, which returns the number of CPUs (cores) in the system.

> Note: This typically counts logical cores, so on a machine with 4 physical cores with hyperthreading, os.cpu_count() might return 8. In such a case, 8 is the number of logical processors, which is 4 physical cores × 2 threads per core.

On Linux you can use system commands to check CPU info:
- `nproc`: Prints the number of processing units available. For example, nproc --all might output “8”, meaning 8 CPU cores (including hyperthreaded cores) are detected.
- `lscpu`: Displays detailed CPU information.

For instance, lscpu might show an output snippet like:
```bash
$ lscpu 
CPU(s):             8  
Thread(s) per core: 2  
Core(s) per socket: 4  
Socket(s):          1  
```

This indicates 8 logical CPUs in total, with 4 physical cores (Core(s) per socket) × 2 threads per core (hyperthreading) on 1 socket. In this example, there are 4 physical cores and hyperthreading makes them appear as 8 logical cores to the OS.
These tools help you decide how many worker processes to use. Often, for CPU-bound work, you’ll use one worker per core (e.g., 4 workers on a 4-core CPU) to maximize parallelism without overwhelming the CPU.

Python’s built-in `multiprocessing` module enables parallel execution by using separate processes for each task. Let’s walk through a simple example of parallel computation using the multiprocessing module. We’ll compute the square of each number in a list, using multiple cores to do it faster:

```python
from multiprocessing import Pool
import os

def square(x):
    """Compute square of a number (our worker task)."""
    return x * x

if __name__ == '__main__':
    # 1. Determine number of available CPU cores (workers)
    num_workers = os.cpu_count()  
    print(f"Running on {num_workers} CPU cores")
    
    # 2. Create a pool of worker processes
    with Pool(num_workers) as pool:
        # 3. Distribute the work across cores and collect results
        numbers = [1, 2, 3, 4, 5]
        results = pool.map(square, numbers)
    
    # 4. Use the results (e.g., print them out)
    print("Squares:", results)
```

Output:
```bash
Running on 8 CPU cores  
Squares: [1, 4, 9, 16, 25]  
```

The code above demonstrates a typical structure for using multiprocessing:
1. **Import needed modules**: We import `Pool` from `multiprocessing` and import os to get CPU count. The Pool class will manage a set of worker processes for us.
2. **Define a worker function**: `square(x)` is the function that will run in each worker process to perform the task (here, simply returning x squared). In a real scenario, this could be a CPU-intensive calculation. Each worker will execute this function on different data.
3. **Main block**: Determine number of workers: We call `os.cpu_count()` to find the number of CPU cores available on the machine. Here we store it in `num_workers`. The term “workers” simply refers to the parallel processes that will do the work. `with Pool(num_workers) as pool:` creates a pool with that many worker processes. Each worker process is a separate Python interpreter ready to run tasks. Using a pool is convenient — it will handle starting the processes and cleaning them up for us. If you omit the number, Pool() by default uses one worker per available CPU
4. **Distribute tasks with pool.map**: We prepare some data `numbers = [1,2,3,4,5]` and use `pool.map(square, numbers)`. This call takes our square function and applies it to each item in the list in parallel. The Pool will split the list items among the worker processes and compute the results concurrently. Under the hood, tasks are dispatched to the workers, each running square(x) on a different core at the same time. The map function then gathers all the return values into the results list, preserving the order of the input. In our example, each number is squared by a worker, and we get back `[1, 4, 9, 16, 25]` corresponding to the input order.
5. **Automatic cleanup**: The` with Pool(...) as pool:` context manager ensures the pool is properly closed when the block exits. This means all worker processes finish their tasks and exit. At this point, the parallel work is done and we can use the results.
6. **Using the results**: Finally, we print the results. In a real use-case, you might further process the results

> Note: While the code runs, you can use commands like `htop` (or `top` in a Linux terminal) to observe CPU usage. You should see multiple Python processes running and the CPU cores all busy, indicating the work is being done in parallel across cores.

Alternatively, it is possible to use also the `concurrent` library:

```python
import concurrent.futures
import os

def square(x: int) -> int:
    """Compute square of a number (our worker task)."""
    return x * x

if __name__ == '__main__':
    # 1. Determine number of available CPU cores (workers)
    num_workers = os.cpu_count()
    print(f"Running on {num_workers} CPU cores")

    numbers = [1, 2, 3, 4, 5]
    results = [None] * len(numbers)

    # 2. Submit tasks to a pool of worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 3. Start all tasks
        futures = {executor.submit(square, x): i for i, x in enumerate(numbers)}
        
        # 4. Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                results[i] = result
            except Exception as exc:
                print(f"Task {i} generated an exception: {exc}")

    # 5. Use the results
    print("Squares:", results)
```

Output:
```
Running on 8 CPU cores
Squares: [1, 4, 9, 16, 25]
```

1. **Import needed modules**: We import `concurrent.futures` for any multiprocessing needed.
2. **Define a worker function**: `square(x)` is the function that will run in each worker process to perform the task.
3. **Main block**: Determine number of workers with `os.cpu_count()` to find the number of CPU cores available on the machine. When we say `with ProcessPoolExecutor(max_workers=num_workers)`, we’re creating a pool of separate Python processes that can each run tasks in parallel. 
4. **Submit Tasks**: Here, we take each item from our input list (the numbers to square) and submit them as individual tasks to the executor. Each call to `executor.submit(square, x)` schedules the function `square(x)` to run in a separate process. We also create a dictionary called `futures`, mapping each `Future` object (which represents the result of an asynchronous computation) to its original index in the input list. This is important for keeping track of where each result should go in our final output.
5. **Get Results**: Now, we want to process results as soon as they’re ready. `as_completed(futures)` gives us an iterator that yields each future in the order it finishes, not necessarily the order we submitted them. This is useful when some tasks take longer than others: instead of waiting for everything to finish, we can handle each result the moment it’s done. When we submitted the tasks earlier, we stored the index `i` of each number alongside the corresponding `Future` object. This way, when a future completes, we know which position in the original list that result belongs to. Without this, we might lose the order of the inputs, since `as_completed()` gives results in completion order, not input order. Calling `.result()` on a `Future` object blocks until the computation is done and gives us the output (i.e., `x * x`). If something goes wrong inside `square(x)` — maybe a division by zero or invalid input — `.result()` will raise the exception that occurred in the worker process. So we wrap this in a try-except block to gracefully handle errors.

This approach is more flexible than `pool.map()`:
* You can handle exceptions per-task.
* You can do something as soon as each result is ready (not wait for all).
* You can schedule tasks with varying input complexity.

<br>
<br>
<br>

## ⏮️ Reproducibility

To achieve consistent results across different runs — especially important for debugging and model comparison — you should set the random seed for all relevant libraries. In PyTorch, this involves not only the standard Python and NumPy seeds but also settings for CUDA behavior to eliminate randomness in GPU computations.

```python
import torch
import random
import numpy as np

def set_seed(seed: int):
    """Set random seed for reproducibility across torch, numpy, and random modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Example seed
```

> Note: Full determinism might slightly reduce performance due to disabling certain CUDA optimizations.

<br>
<br>
<br>

## 🏎️ GPUs

Leverage GPU acceleration to significantly speed up tensor operations and model training. Check if a CUDA-compatible GPU is available and gather basic GPU information:

```python
import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only")
```

To run operations on the GPU, both the model and the tensors must be moved to the same device:

```python
# Define the target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and tensor
model = MyModel()
tensor = torch.randn(3, 3)

# Move both model and tensor to the selected device
model = model.to(device)
tensor = tensor.to(device)
```

If the tensor resides on the GPU, all operations on it will also be executed on the GPU:

```python
result = tensor * 2  # Will be computed on GPU if tensor is on GPU
```

> Note: Always ensure your input tensors and model are on the same device to avoid errors.

<br>

### Parallelize GPUs

Training large models or datasets can benefit from parallelizing workloads across multiple GPUs. `DataParallel` splits your batch across multiple GPUs and merges the output:

```python
import torch.nn as nn

model = MyModel()

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)
```

Alternatively, `DistributedDataParallel` (DDP) provides faster and more efficient parallelism. It uses multiple processes, each tied to one GPU, for true parallel computation.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group for DDP using the NCCL backend
dist.init_process_group("nccl")

# Wrap the model with DDP
model = DDP(MyModel().to(device))
```

Key Differences:

| Feature                | `DataParallel` (DP)              | `DistributedDataParallel` (DDP)          |
| ---------------------- | -------------------------------- | ---------------------------------------- |
| Execution Mode         | Single process, auto batch split | Multi-process, explicit parallelism      |
| Communication Overhead | Higher                           | Lower (uses optimized NCCL backend)      |
| Scalability            | Limited for large models         | Scales efficiently across GPUs and nodes |
| Speed                  | Slower                           | Significantly faster for large workloads |

<br>

### Choosing GPUs for Computation

You can control which GPU(s) are used for training or inference. Limit visibility of GPUs to PyTorch or other CUDA applications can be done in Terminal:
```bash
# Make only GPU 0 and GPU 1 visible
export CUDA_VISIBLE_DEVICES=0,1
```

or from code:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

If you want to explicitly assign a model or tensor to a specific GPU:

```python
# Choose a specific GPU (e.g., GPU 1)
device = torch.device("cuda:1")

# Move model to that GPU
model = model.to(device)
```

You can also set the current active device globally:

```python
torch.cuda.set_device(1)
tensor = torch.randn(3, 3).cuda()  # Tensor is now on GPU 1
```

To maintain consistent device numbering across runs or machines, sort GPUs by PCI Bus ID:

```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
```

> Note: Useful when device IDs change depending on system boot order or hardware layout.

<br>

### Multi-Core CPU Processing

Even without a GPU, you can speed up training and data loading by utilizing multiple CPU cores for the operations supported by PyTorch. By default, PyTorch automatically chooses a number of CPU threads, but you can explicitly set it:

```python
import torch

# Set number of threads to match number of physical CPU cores
torch.set_num_threads(8)
```

When loading data from disk or applying transformations, you can use multiple workers to speed things up. `pin_memory=True` is beneficial when using GPUs, as it allows faster transfer of data from CPU to GPU memory.

```python
from torch.utils.data import DataLoader, Dataset

dataloader = DataLoader(
    dataset=MyDataset(),
    batch_size=32,
    shuffle=True,
    num_workers=4,     # Use 4 parallel CPU workers for data loading
    pin_memory=True    # Speeds up transfer to GPU
)
```

<br>

### Monitoring Devices

Monitoring your GPU's status helps optimize performance and debug memory issues. You can check it on the Terminal:

```bash
# Show current GPU status, memory usage, and active processes
nvidia-smi

# Continuously refresh GPU status every 1 second
watch -n 1 nvidia-smi

# Lightweight alternative with a compact display
gpustat
```

or from code:

```python
import torch

# Print a detailed report of memory usage and stats
print(torch.cuda.memory_summary())
```

To see what programs are actively using the GPU and how much memory they are consuming:

```bash
# Monitor GPU usage per process (PID)
nvidia-smi pmon
```

If a process is consuming too much memory or becomes unresponsive:

```bash
# Gracefully stop a process
kill <PID>

# Forcefully terminate a process
kill -9 <PID>
```

> Note: Killing a process may result in data loss or corruption if it’s actively training or saving.

<br>
<br>
<br>

## ⏱️ Model Summary and FLOPs

When developing or debugging deep learning models, it's useful to understand their structure and computational complexity. This includes knowing:

* Number of parameters (trainable and total)
* Estimated FLOPs (floating-point operations)
* Layer-wise structure and output shapes

These insights help evaluate model efficiency, memory footprint, and runtime performance.

<br>

### Print Model Summary (Layers and Parameters)

Use the `torchinfo` package to print a detailed summary:

```bash
pip install torchinfo
```

Example:

```python
from torchinfo import summary
from my_model import MyModel  # Replace with your actual model

model = MyModel()
summary(model, input_size=(1, 3, 224, 224))  # Example input shape for image models
```

* Layer (type): Name and type of the layer (Conv2d, Linear, etc.)
* Output Shape: Shape of the tensor produced by the layer
* Param #: Number of trainable parameters in each layer
* Total params: Sum of all parameters in the model
* Trainable params: Parameters updated during training

If you are using a GPU, move the model to `cuda()` before calling `summary()`:

```python
model = model.to("cuda")
summary(model, input_size=(1, 3, 224, 224), device="cuda")
```

<br>

### Estimate FLOPs and MACs

FLOPs (Floating Point Operations) and MACs (Multiply-Accumulate Operations) help estimate how computationally intensive a model is. You can use `ptflops` for this purpose:

```bash
pip install ptflops
```

Example:

```python
from ptflops import get_model_complexity_info
from my_model import MyModel

model = MyModel()

with torch.cuda.device(0):  # Optional: ensure device context
    macs, params = get_model_complexity_info(
        model,
        input_res=(3, 224, 224),  # Channels, Height, Width
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

print(f"MACs: {macs}")
print(f"Parameters: {params}")
```

Output:

```
MACs: 4.25 GMac
Parameters: 23.59 M
```

> Note:
>
> * 1 **GMac** = 1 billion multiply-add operations
> * MACs and FLOPs are often used interchangeably, though FLOPs usually count both multiplication and addition separately (so FLOPs ≈ 2 × MACs)

<br>
<br>
<br>

[Back to Index 🗂️](./README.md)

