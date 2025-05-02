[Back to Index 🗂️](./README.md)

<center><h1>🤖 Deep Learning Project Guide</h1></center>

<br>

Welcome to your practical guide for setting up a deep learning project in Python. Whether you're building a research prototype or preparing for production deployment, having a clean, modular, and scalable project structure is essential.
This guide walks you through the foundational steps—from scripting best practices to parameter parsing—so you can focus on model development, not boilerplate code.

## 1️⃣ Python Script Entry Point

To organize your Python scripts more cleanly—especially as they grow in complexity—it's best practice to wrap your execution logic inside a main() function. This makes your code easier to read, test, and reuse.
The special `if __name__ == "__main__":` check ensures that your script only runs when executed directly, not when imported as a module in another script.

```python
def main():
    # Your code here
    pass

# Only execute main() if the script is run directly
if __name__ == "__main__":
    main()
```

## 2️⃣ Parameters

In most projects, you'll need to pass configuration values or runtime options to your scripts. The best way to handle this is through command-line parameter parsing, which allows your code to be easily executed with different configurations—without modifying the source. This approach makes your code more flexible, modular, and user-friendly. Parameters parsing can be added in the main function as follow:

```python
import argparse  # Import the argparse module to handle command-line arguments

# Initialize the ArgumentParser object with a short description of the script
parser = argparse.ArgumentParser(description='Argument Parser')

# Add an argument:
# - `type`: ensures the type of the input
# - `default`: sets the default value if not specified
# - `choices=[...]`: restricts input to a list of valid options
# - `help=...`: provides a helpful message displayed in `--help`

# Example:
#   --backbone ResNet-50 sets the backbone to ResNet-50
#   (no flag passed) defaults to ResNet-18
parser.add_argument('--backbone',
                    type=str,
                    default='ResNet-18',
                    choices=['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
                    help='Backbone network')

# Add a boolean flag:
# - `action='store_true'`: sets the value to True if the flag is present, otherwise False
# - `default` (NOTE: This should be a boolean `True` or `False`, not a string)
# - `help=...`: describes the purpose of the flag
# Common pattern: Let the presence of the flag decide True, and avoid setting default=True, otherwise you can't disable it from the command line.

# Example usage:
#   --pretrained enables pretrained weights (sets True)
#   (no flag passed) pretrained is False
parser.add_argument('--pretrained',
                    action='store_true',
                    help='Pretrained backbone flag')

# Parse the arguments provided via the command line
parser = parser.parse_args()
```

A convenient way to manage all the different parameters that it might be needed is to define a parsing function that return the parser object in the main workflow:

```python
def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # Initialize the ArgumentParser object with a short description of the script
    parser = argparse.ArgumentParser(description='Argument Parser')

    # Your parameters here

    return parser
```

## 3️⃣ Device

When training deep learning models, it's crucial to leverage the best available compute device and ensure your experiments are reproducible. These are two foundational steps in any serious machine learning or research pipeline.

Modern deep learning models are computationally intensive and benefit greatly from GPU acceleration. Automatically selecting the appropriate compute device ensures your code runs optimally on different machines—whether it's a local CPU, a workstation with a single GPU, or a multi-GPU cloud server.

```python
# Automatically choose GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set number of threads for CPU-bound tasks (e.g., using a command-line parameter)
torch.set_num_threads(parser.num_threads)
```

Here a usefull debug function to get the device name:

```python
def get_device_name() -> str:
    """
    Returns the name of the current compute device. If a GPU is available, returns its full device name. Otherwise, returns 'cpu'.

    :returns: Name of the active compute device.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return 'cpu'

# Print GPU model name
print("GPU device name: {}".format(get_GPU_name()))
```

## 4️⃣ Reproducibility

In deep learning, many operations involve randomness:
- Weight initialization
- Data augmentation
- Dropout layers
- Mini-batch shuffling

By fixing random seeds across all major libraries (PyTorch, NumPy, random), we make our experiments deterministic, meaning:
The same code produces the same results every time.
Model performance can be fairly compared across runs or collaborators.
Debugging becomes far easier.
Additionally, the `torch.backends.cudnn` flags ensure that CUDA computations remain deterministic—important when working with GPUs and cuDNN kernels, which may otherwise vary slightly between runs for performance reasons.

```python
def reproducibility(seed: int):
    """
    Set seeds for consistent and reproducible results across runs.

    :param seed: seed for random generation.
    """
    torch.manual_seed(seed)                      # For PyTorch
    np.random.seed(seed)                         # For NumPy
    random.seed(seed)                            # For built-in random
    torch.cuda.manual_seed_all(seed)             # For CUDA devices

    torch.backends.cudnn.deterministic = True    # Ensure deterministic results
    torch.backends.cudnn.benchmark = False       # Disable performance heuristics
    torch.backends.cudnn.enabled = False         # Fully disable cuDNN (optional)
```

## 5️⃣ Paths and Configuration

As your project grows, you will inevitably manage multiple file and folder paths, such as:
- Dataset locations
- Model checkpoints
- Logs and outputs
- Pretrained weights

Hardcoding these paths directly into your code makes the project rigid and difficult to reuse or share. Imagine setting all the paths inside the code and passing the project to someone else. Your successor would have to manually search through the entire codebase to update paths, a tedious and error-prone process. Migrating the project between workstations would also become time-consuming, requiring reconfiguration each time.

Instead, a better approach is to use **external configuration files** and maintain a **centralized structure** to manage these paths. This method allows you to seamlessly switch datasets, update save locations, or adapt to different environments without modifying your source code.

A simple example of a `config.yaml` could be:

```yaml
paths:
  data_dir: ./data
  checkpoint_dir: ./checkpoints
  log_dir: ./logs
  pretrained_weights: ./weights/resnet50.pth
```

> **Note:** Some guides recommend using `.yaml` files for all parameters, including hyperparameters and settings. However, I suggest using command-line arguments for dynamic parameters. Command-line parsing offers more flexibility, especially when running multiple experiments. If you want to launch two experiments with different parameters, you can easily copy and modify the execution command. If everything is inside a `.yaml` file, you would need to wait for the first experiment to finish, manually edit the `.yaml` file, and re-launch the next one — far less efficient.

Reading a `.yaml` file it's very easy with the  `PyYAML` library:

```bash
pip install pyyaml
```

Here’s a simple function to load a configuration file:

```python
import yaml

def load_config(path: str) -> dict:
    """
    Load a `.yaml` configuration file.

    :param path: Path to the `.yaml` file.
    :return: Dictionary containing configuration.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
```

A Python dictionary is returned with the structure of the `.yaml` file. You can also pass the path to the config file as a command-line argument (e.g., `--config=config.yaml`) to make your code even more flexible and adaptable.

```python
# Load configuration
config = load_config(parser.config)

# Access paths
data_dir = config["paths"]["data_dir"]
ckpt_dir = config["paths"]["checkpoint_dir"]
log_dir = config["paths"]["log_dir"]

print(f"Data directory: {data_dir}")
print(f"Checkpoint will be saved to: {ckpt_dir}")
print(f"Logs directory: {log_dir}")
```

## 6️⃣ Dataset Class

Before splitting the data, we need a proper Dataset class. In PyTorch, a Dataset class defines how data samples are loaded and optionally transformed. This flexibility is crucial when working with custom datasets, different file formats, or applying pre-processing steps.

A Dataset class must implement two essential functions:
- `__len__`: defines how many samples are in the dataset.
- `__getitem__`: defines how to retrieve a single data sample.

This gives full control over how you read files, preprocess them, and serve them to your model.

Here's a simple example of a custom Dataset class specifically designed for loading image files:

```python
import os
from torch.utils.data import Dataset
from typing import Callable, Optional
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 filenames: list,
                 transform: Optional[Callable] = None):
        """
        Constructor of the dataset. It saves the base folder and the filenames list, 
        and optionally a transformation pipeline.

        :param data_dir: Path to the directory containing the images.
        :param filenames: List of image filenames (e.g., ['001.tiff', '002.tiff']).
        :param transform: Optional torchvision transform to be applied on the loaded image.
        """

        # Directory where images are stored
        self.data_dir = data_dir
        # List of filenames to load
        self.filenames = filenames
        # Transformations to apply (e.g., Resize, Normalize)
        self.transform = transform


    def __len__(self) -> int:
        """
        Defines the total number of samples in the dataset.
        This makes the dataset compatible with functions that expect the dataset size.

        :return: Number of samples in the dataset.
        """

        return len(self.filenames)


    def __getitem__(self,
                    idx: int):
        """
        Retrieve a single sample by index.
        Opens the image, applies transformations if any, and returns it along with its filename.

        :param idx: Index of the sample to retrieve.
        :return: Tuple (transformed image, filename).
        """

        # Build full path to the image
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return both the processed image and its filename (useful for debugging or tracking)
        return image, self.filenames[idx]
```

You can later extend this class to load multiple modalities (e.g., masks for segmentation), load associated metadata (e.g., labels), or support completely different types of data (e.g., tabular data, audio, text).

After defining your `CustomDataset` class, you can easily instantiate and use it.
Below is a general example:

```python
from torchvision import transforms

# Initialize the dataset (e.g., using a `.yaml` path configuration)
custom_dataset = CustomDataset(
    data_dir=config["paths"]["dataset"],
    filenames=config["paths"]["list"]
)

# Example of how to access a sample
sample_image, sample_filename = custom_dataset[0]
print(f"Loaded sample: {sample_filename}, Image shape: {sample_image.shape}")
```

## 7️⃣ Data Splitting

Properly splitting your dataset into training, validation, and testing sets is fundamental for building robust deep learning models. Good data splitting ensures fair evaluation, helps avoid data leakage, and mirrors real-world deployment scenarios.

There are two main strategies you can follow for splitting your data:

### 1. Manual Split using a CSV reference

If you want complete control over your data split, you can manually prepare a CSV file that assigns each sample to a specific split (train/val/test). This is especially useful when you need reproducibility and when different samples require special treatment.

Here's an example function to read a split CSV:

```python
def read_split(path_split_file: str) -> dict:
    """
    Read data split from a CSV file.

    :param path_split_file: Path to the CSV file containing data splits.
    :return: Dictionary with lists for indices, filenames, and split types.
    """
    # Read the column 'FILENAME' as string
    dtype_mapping = {"FILENAME": str}

    # Read CSV file: restrict to specific columns and define dtype for 'FILENAME'
    data_split = read_csv(
        filepath_or_buffer=path_split_file,
        usecols=["INDEX", "FILENAME", "SPLIT"],
        dtype=dtype_mapping
    ).values

    # Split into arrays
    index = data_split[:, 0]
    filename = data_split[:, 1]
    split = data_split[:, 2]

    # Pack into a dictionary
    split_dict = {
        'index': index.tolist(),
        'filename': filename.tolist(),
        'split': split.tolist()
    }

    return split_dict
```

Example split file:

```csv
INDEX,FILENAME,SPLIT
0,001.tiff,train
1,002.tiff,val
2,003.tiff,test
...
```

You can then create subsets based on the split dictionary:

```python
from torch.utils.data import Dataset, Subset
from typing import Tuple, Dict, Any

def create_subsets_from_split(dataset: Dataset,
                              split_dict: Dict[str, list]) -> Tuple[Subset, Subset, Subset]:
    """
    Create dataset subsets based on split information.

    :param dataset: The full dataset object.
    :param split_dict: Dictionary containing split labels for each sample.
    :return: Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    # Create lists of indices for each split type
    train_indices = [i for i, s in enumerate(split_dict['split']) if s == 'train']
    validation_indices = [i for i, s in enumerate(split_dict['split']) if s == 'val']
    test_indices = [i for i, s in enumerate(split_dict['split']) if s == 'test']

    # Create dataset subsets using torch.utils.data.Subset
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, validation_dataset, test_dataset
```

### 2. Automatic Split Using `train_test_split`

If you prefer to split your data dynamically at runtime, you can use `train_test_split` from `sklearn.model_selection`. This method works for lists, arrays, and also for indices representing custom datasets.

Example for custom datasets:

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset
from typing import Tuple

def split_dataset(dataset: Dataset,
                  test_size: float = 0.2,
                  validation_size: float = 0.1,
                  random_state: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Randomly split a dataset into train, validation, and test subsets.

    :param dataset: The full Dataset object.
    :param test_size: Proportion of the dataset to include in the test split.
    :param validation_size: Proportion of the train set to include in the validation split.
    :param random_state: Random seed to ensure reproducibility.
    :return: Tuple (train_dataset, validation_dataset, test_dataset)
    """

    # Generate a list of all sample indices
    indices = list(range(len(dataset)))

    # First split: train+val vs test
    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: train vs val
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=validation_size,
        random_state=random_state
    )

    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, validation_dataset, test_dataset
```

## 8️⃣ Data Transforms

In deep learning workflows, data transformations are a key part of preparing your inputs before feeding them into the model.
They help with:
- Standardizing inputs (e.g., resizing all images to the same dimensions)
- Normalizing pixel values (e.g., scaling between 0-1)
- Data augmentation (e.g., random flips, rotations) to improve generalization.

PyTorch provides a powerful module called torchvision.transforms to define and compose transformation pipelines. Each transformation is a callable operation (like a function) that takes an input and returns a transformed output. To chain multiple transformations sequentially, you use `transforms.Compose([...])`, which combines them into a single callable. This ensures that the input data is passed through each transformation in the order they are listed.

Example of a basic Transform pipeline:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize images to 224x224
    transforms.ToTensor(),                 # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
])
```

Sometimes the built-in torchvision.transforms are not enough. You may need to create your own transformations tailored to your dataset or task.

Creating a custom transform in PyTorch is very simple:
- Define a class.
- Implement an `__init__` method to set any parameters.
- Implement a `__call__` method that takes a sample and returns the transformed sample.

Here's a general example of a custom transformation that works with a basic custom dataset:

```python
class CustomMultiplyTransform:
    """
    Simple custom transform that multiplies an image tensor by a fixed factor.
    """
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, img):
        return img * self.factor
```

The ``__call__`` method allows the object to behave like a function. You can easily combine this custom transform with standard ones using transforms:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    CustomMultiplyTransform(factor=0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

Each dataset split (train, validation, test) should have its own dedicated set of transformations.
For example:
- The training set can apply data augmentations (such as random flips or rotations) followed by normalization.
- The validation and test sets usually apply only resizing and normalization to ensure consistent evaluation.

When you define separate transforms, you must make sure to apply them individually to each subset (train, val, test). Now we can appreciate the importance of this separation, ensuring, for example, that data augmentation only affects training, and that validation/testing remains stable and comparable.

Here's an example transformation pipeline for training, validation and test data:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

```python
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

```python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

To apply train_transform, validation_transform, or any custom transform to your train, validation, and test subsets, you need to assign the new transform to the underlying dataset inside the Subset. Since Subset objects just wrap indices and the original dataset, you can directly modify the .transform attribute of the dataset object.

```python
train_dataset.dataset.transform = train_transform
validation_dataset.dataset.transform = validation_transform
test_dataset.dataset.transform = test_transform  # Usually same as validation_transform
```

Now your data will be well-prepared for model training!

## 9️⃣ Data loader

Once you have your datasets ready (and optionally split and transformed), you need an efficient way to feed the data into your model during training and evaluation.  This is where the PyTorch `DataLoader` comes into play.

A DataLoader is a utility that wraps a Dataset and provides:
- Batching: Groups samples into mini-batches.
- Shuffling: Randomizes the order of samples each epoch.
- Parallel Loading: Loads batches in parallel using multiple CPU workers.
- Memory Efficiency: Handles large datasets without loading everything into memory at once.

You create a DataLoader by passing a Dataset (or Subset) along with a few important parameters:

```python
from torch.utils.data import DataLoader

dataloader_train = DataLoader(
    dataset=train_dataset,   # Your Dataset or Subset object
    batch_size=32,           # Number of samples per batch
    shuffle=True,            # Shuffle data every epoch (important for training)
    num_workers=4,           # Number of subprocesses for loading data
    pin_memory=True          # Speed up transfer to GPU (recommended if using CUDA)
)

dataloader_validation = DataLoader(
    dataset=validation_dataset,
    batch_size=32,
    shuffle=False,           # No shuffle during validation/testing
    num_workers=4,
    pin_memory=True
)

dataloader_test = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,           # No shuffle during validation/testing
    num_workers=4,
    pin_memory=True
)
```

In your training and validation loops, you iterate over batches returned by the DataLoader:

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
```

The DataLoader automatically fetches the next batch, applies parallel loading, and ensures efficient data feeding to your model. Using DataLoaders properly is crucial for efficient, scalable deep learning workflows.

## 🔟 Model Definition

After preparing your data pipeline, the next step is to define your model architecture.
Organizing your model code properly makes it easier to read, maintain, and extend.

In PyTorch, models are typically defined by subclassing `torch.nn.Module`. This gives you full flexibility to define the layers and the forward pass logic:
  - Use `__init__` to define layers.
  - Use `forward` to define how data flows through layers.

Good weight initialization is crucial for the stable and efficient training of deep neural networks. When you define your model, all layers usually benefit from weight and biases initialization.

Here a custom model example:

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):


    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        """
        Initialize model weights properly for better training stability and convergence.
        """
        for m in self.modules():

            # Convolutional layers initialization with Kaiming technique
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Linear layers initialization with Xavier technique
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
```

Instantiating the Model and move it to the device:

```python
from simple_cnn import SimpleCNN

model = SimpleCNN(num_classes=10)
model.to(device)  # Move model to CPU or GPU
```

For bigger architectures, define reusable building blocks:
- Keep Models Modular: If your architecture has repetitive blocks, implement them as separate sub-modules.
- Group Components into Files: Keep models clean by splitting large architectures into multiple files if needed.
- Support Easy Hyperparameter Changes: Allow parameters like number of classes, number of layers, hidden dimensions, etc., to be passed to the constructor.

For example, let's build a basic block:

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
```

Then assemble multiple blocks inside your final architecture:

```python
from basic_block import BasicBlock

class BigModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(BigModel, self).__init__()
        self.block1 = BasicBlock(3, 64)
        self.block2 = BasicBlock(64, 128)
        self.classifier = nn.Linear(128 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

<br>
<br>
<br>

## 🦾 Training and Evaluation Pipeline

After setting up the model, we move to structuring the training and evaluation phases. A clean pipeline improves reproducibility, debuggability, and extensibility.

The general structure of a deep learning pipeline includes the following phases:

### 1. Optimizer and Scheduler Setup

Before training, define the optimizer and the learning rate scheduler. This allows you to update model weights efficiently and adapt the learning rate dynamically during training.

To ensure reusability and facilitate easy adjustments to the setup, the best approach is to create a `get` function that modifies the optimizer based on a command-line parameter:

```python
# Optimizer
optimizer = get_optimizer(
    net_parameters=model.parameters(),
    parser=parser)

# Scheduler
scheduler = get_scheduler(
    optimizer=optimizer,
    parser=parser)
```

Example `get_optimizer` function:

```python
from torch.optim import Adam, SGD

def get_optimizer(net_parameters: Iterator[Parameter],
                  parser: argparse.Namespace) -> Union[Adam, SGD]:
    """
    Get optimizer

    :param net_parameters: net parameters
    :param parser: parser of parameters-parsing
    :return: optimizer
    """

    if parser.optimizer == 'Adam':
        return Adam(net_parameters, lr=parser.learning_rate)
    elif parser.optimizer == 'SGD':
        return SGD(net_parameters, lr=parser.learning_rate, momentum=parser.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {parser.optimizer}")
```

Example `get_scheduler` function:

```python
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

def get_scheduler(optimizer: Union[Adam, SGD],
                  parser: argparse.Namespace) -> Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts]:
    """
    Get scheduler

    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: scheduler
    """

    if parser.scheduler == 'StepLR':
        return StepLR(optimizer,
                      step_size=parser.lr_step_size,
                      gamma=parser.lr_gamma)
    elif parser.scheduler == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer=optimizer, 
                                 patience=parser.lr_patience,
                                 verbose=True)
    elif parser.scheduler == "CosineAnnealing":
        return CosineAnnealingWarmRestarts(optimizer=optimizer,
                                           T_0=parser.lr_T0)
    else:
        raise ValueError(f"Unsupported scheduler: {parser.scheduler}")
```

This structure ensures that optimizers and schedulers are configured cleanly and consistently across experiments.

### 2. Loss Function

Define your loss function that measures the model's performance:

```python
criterion = get_loss(parser=parser)
```

Example `get_loss` function:

```python
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from net.loss.MyFocalLoss import MyFocalLoss
from net.loss.MySigmoidFocalLoss import MySigmoidFocalLoss

def get_loss(parser: argparse.Namespace) -> Union[CrossEntropyLoss, BCEWithLogitsLoss, MySigmoidFocalLoss, MyFocalLoss]:
    """
    Get loss

    :param loss: loss name
    :param device: device
    :param parser: parser of parameters-parsing
    :return: criterion (loss)
    """

    if parser.loss == 'CrossEntropyLoss':
        return CrossEntropyLoss()
    elif parser.loss == 'BCELoss':
        return BCEWithLogitsLoss()
    elif parser.loss == 'SigmoidFocalLoss':
        return MySigmoidFocalLoss(alpha=parser.alpha,
                                  gamma=parser.gamma)
    elif parser.loss == 'FocalLoss':
        return MyFocalLoss(alpha=parser.alpha,
                           gamma=parser.gamma)
    else:
        raise ValueError(f"Unsupported loss: {parser.loss}")
```

Custom losses can also be designed and added into the `get_loss` function if your task requires it (e.g., segmentation, detection).

### 3. Metrics Initialization

Initialize metrics storage structures to track progress during training and evaluation. A convenient way is via a dictionary:

```python
metrics = {
    'train_loss': [],
    'validation_loss': [],
    'accuracy': []
    # Add your metrics here
}
```

### 4. Training Loop

The training loop involves iterating over epochs and batches, computing the loss, performing backpropagation, and updating the model weights:

```python
for epoch in range(num_epochs):
    model.train()  # Set model to training mode (important for layers like dropout, batchnorm)

    for images, labels in train_loader:
        # Move input data and labels to the selected device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients accumulated from previous steps
        outputs = model(images)  # Forward pass: predict outputs for input images
        loss = criterion(outputs, labels)  # Compute loss between prediction and ground-truth labels
        loss.backward()  # Backpropagation: compute gradients
        optimizer.step()  # Update model parameters

    scheduler.step()  # Update learning rate if a scheduler is used
```

### 5. Validation Loop

After training in each epoch, validate the model to track overfitting and generalization:

```python
model.eval()  # Set model to evaluation mode (deactivate dropout, batchnorm behaves differently)
with torch.no_grad():  # Disable gradient computation for validation
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        validation_loss = criterion(outputs, labels)
```

Record metrics at each epoch to later plot loss and metric curves.

### 6. Checkpoint Saving

Save the model if it achieves a better performance on validation metrics (e.g., best validation accuracy, best AUC):

```python
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def save_checkpoint(epoch: int,
                    model: Module,
                    optimizer: Optimizer,
                    scheduler: _LRScheduler,
                    path: str) -> None:
    """
    Save model, optimizer, scheduler states for resuming.

    :param epoch: Current epoch.
    :param model: Model object.
    :param optimizer: Optimizer object.
    :param scheduler: Scheduler object.
    :param path: File path to save checkpoint.
    """

    # Define the checkpoint dictionary information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)
```

```python
# If the current validation accuracy is better than the best seen so far
if validation_accuracy > best_validation_accuracy:
    # Save the model's weights (state_dict) to disk
    torch.save(model.state_dict(), 'best_model.pth')
    
    # Update the best validation accuracy with the current one
    best_validation_accuracy = validation_accuracy
```

### 7. Resume

In deep learning projects, training can be long and computationally expensive. To avoid starting from scratch if your session is interrupted, it’s important to save and reload:
- Model weights
- Optimizer state
- Scheduler state
- Epoch number
- (Optional) Best validation metric achieved so far. This allows you to resume training exactly where you left off.

At the end of each epoch or the best epoch based on validation performance, you should save the model in case a new training should restart from the point it was interrupted.

When resuming training:

```python
def load_checkpoint(model: Module,
                    optimizer: Optimizer,
                    scheduler: _LRScheduler,
                    path: str) -> int:
    """
    Load model, optimizer, and scheduler states for resuming training from a checkpoint.

    :param model: Model object whose state_dict will be restored.
    :param optimizer: Optimizer object whose state_dict will be restored.
    :param scheduler: Scheduler object whose state_dict will be restored.
    :param path: Path to the checkpoint file (usually a '.pt' or '.pth' file).
    :return: Resumed epoch number (start from the next epoch).
    """
    # Load the checkpoint dictionary from the given file path
    checkpoint = torch.load(path)
    
    # Restore the model parameters from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore the optimizer state (momentum, learning rate, etc.)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore the learning rate scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Retrieve the epoch number and increment it (so training continues from the next epoch)
    start_epoch = checkpoint['epoch'] + 1  
    
    # Return the epoch to resume training correctly
    return start_epoch
```

Add a command-line parameter to control the resume:

```python
# Define the path to the model to resume
parser.add_argument('--resume_checkpoint',
                    type=str,
                    default=None,
                    help='Path to resume checkpoint')
```

Before the training:

```python
# Check if a resume checkpoint path is provided through the parser arguments
if parser.resume_checkpoint:
    # If yes, load the model, optimizer, and scheduler states from the checkpoint
    # and set start_epoch to the next epoch to continue training
    start_epoch = load_checkpoint(model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  path=parser.resume_checkpoint)
else:
    # If no checkpoint is provided, start training from epoch 0
    start_epoch = 0
```

Thus, the training loop will be resumed from the `start_epoch`:

```python
for epoch in range(start_epoch, num_epochs):
    model.train()
    ...
```

### 7. Testing Phase

After training is complete, load the best model and evaluate it on the test set:

```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

Compute final metrics and plot the corresponding curves.

### 8. Plotting and Reporting

This helps visualize the model's learning behavior and performance.

<br>
<br>
<br>

# 📁 Code Organization Best Practices

A clean and modular project structure is essential to scaling deep learning projects. Here's an example organization for a classification project, along with a description of each folder and file:

```
net/
├── __init__.py
├── classifications/
│   ├── classifications_test.py          # Save classification results during test
│   ├── classifications_validation.py    # Save classification results during validation
│   └── utility/
│       └── classifications_concatenation.py  # Utility functions for merging classification between the different folds
├── dataset/
│   ├── MyDataset.py                      # Custom Dataset definition
│   ├── dataset_split.py                  # Splitting dataset into train/validation/test
│   ├── dataset_transforms.py             # Transformation pipelines
│   ├── statistics/
│   │   ├── min_max_statistics.py         # Compute min-max statistics for normalization
│   │   └── standard_statistics.py        # Compute mean-std statistics for normalization
│   ├── transforms/
│   │   ├── Add3ChannelsImage.py          # Add extra channels to input images
│   │   ├── MinMaxNormalization.py        # Min-Max normalization transform
│   │   ├── Padding.py                    # Padding transformation
│   │   ├── StandardNormalization.py      # Standard score normalization
│   │   └── ToTensor.py                   # Convert data to tensor format
│   └── utility/
│       ├── read_split.py                 # Read split files (e.g., CSVs)
│       └── split_index.py                # Utility for splitting indices between train/validation/test
├── evaluation/
│   ├── ROC_AUC.py                        # Functions for ROC and AUC computation
│   └── current_learning_rate.py          # Retrieve current learning rate from scheduler
├── initialization/
│   ├── ID/
│   │   ├── experimentID.py                # Generate unique experiment IDs based on the input parameters
│   │   ├── experimentID_complete.py       # Handle complete experiments ID merging the different folds
│   │   └── experimentID_fold.py           # Retrieve the classifications for each fold
│   ├── dict/
│   │   └── metrics.py                     # Define metrics structure
│   ├── folders/
│   │   ├── dataset_folders.py             # Dataset-specific folder creation
│   │   ├── default_folders.py             # Default experiment folders
│   │   ├── experiment_complete_folders.py # Folders for complete experiments
│   │   └── experiment_folders.py          # Folders for single experiments
│   ├── init.py                            # Central path initialization script
│   ├── path/
│   │   ├── experiment_complete_result_path.py # Paths for results of complete experiments
│   │   └── experiment_results_path.py         # Paths for experiment results
│   └── utility/
│       ├── create_folder_and_subfolder.py # Utility to create directory trees
│       └── parameters_ID.py               # Manage parameters linked to IDs
├── loss/
│   ├── MyFocalLoss.py                     # Custom focal loss
│   ├── MySigmoidFocalLoss.py              # Another custom loss for sigmoid outputs
│   └── get_loss.py                        # Wrapper to select loss function
├── metrics/
│   ├── metrics_test.py                    # Compute metrics during testing
│   ├── metrics_train.py                   # Compute metrics during training
│   ├── show_metrics/
│   │   ├── show_metrics_test.py           # Display or log test metrics
│   │   └── show_metrics_train.py          # Display or log training metrics
│   └── utility/
│       ├── my_notation.py                 # Format metric values (scientific notation, etc.)
│       ├── my_round_value.py              # Custom rounding functions
│       └── timer.py                       # Timing utilities
├── model/
│   ├── MyNetwork.py                       # The main model architecture
│   └── utility/
│       ├── load_model.py                  # Load models from checkpoints
│       └── save_model.py                  # Save models to disk
├── optimizer/
│   └── get_optimizer.py                   # Wrapper to select optimizer (Adam, SGD, etc.)
├── parameters/
│   ├── parameters.py                      # Define general parameters
│   ├── parameters_choices.py              # Define possible choices (for args)
│   ├── parameters_default.py              # Define default values
│   └── parameters_help.py                 # Help strings for argparse parser
├── plot/
│   ├── ROC_AUC_plot.py                    # Functions for ROC and AUC plots
│   ├── coords/
│   │   └── save_coords.py                 # Save coordinates for plots
│   ├── loss_plot.py                       # Plot loss curves
│   └── utility/
│       └── figure_size.py                 # Utilities to define figure sizes dynamically
├── reproducibility/
│   └── reproducibility.py                 # Random seed and reproducibility setup
├── resume/
│   ├── metrics_resume.py                  # Resume training with metrics continuation
│   ├── metrics_train_resume.py            # Resume training-specific metrics
│   ├── resume.py                          # Resume training from checkpoint
│   └── resume_models.py                   # Resume models themselves
├── scheduler/
│   └── get_scheduler.py                   # Wrapper to select learning rate scheduler
├── test.py                                # Test script main
├── train.py                               # Training script main
└── validation.py                          # Validation script main
```

Go check [this repository](https://github.com/GiulioRusso/Deep-Learning-boilerplate) for a Deep Learning boilerplate project.



