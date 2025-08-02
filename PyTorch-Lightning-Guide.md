[Back to Index 🗂️](./README.md)

<center><h1>⚡ PyTorch Lightning Guide</h1></center>

The typical PyTorch deep learning project structure can be implemented wih **PyTorch Lightning**, a framework that removes much of the boilerplate code while keeping flexibility. Lightning is great for scalability, readability, and rapid experimentation.

> **Note**: This markdown assumes you're familiar with traditional PyTorch structure and focuses on **what changes** when you switch to Lightning.

<br>

## 🧱 Main Structural Shift: Everything inside the LightningModule

The **core difference** is that PyTorch Lightning introduces the `LightningModule`, a high-level class that encapsulates:
- Model architecture
- Forward pass
- Training loop
- Validation/Testing logic
- Optimizer andSscheduler configuration
- Logging metrics

## 🔁 Training / Validation / Test Loops: `_step()` Functions

Lightning defines key hooks:

| Function | Role |
|---------|------|
| `forward(self, x)` | Defines inference behavior |
| `training_step(self, batch, batch_idx)` | Called on each training batch. Return value is used to backpropagate |
| `validation_step(self, batch, batch_idx)` | Called on each validation batch. Metrics are automatically aggregated |
| `test_step(self, batch, batch_idx)` | Same for test phase |
| `configure_optimizers(self)` | Setup optimizer (and scheduler if needed) |

These are called automatically during training/validation/testing phases.

```python
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class LitClassifier(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model  # Define the model architecture
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)  # Used for inference (e.g., model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch   # Unpack batch data
        y_hat = self(x)  # Forward pass
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=10)

        self.log("train_loss", loss, prog_bar=True)  # Log to TensorBoard
        self.log("train_acc", acc, prog_bar=True)
        return loss  # Used internally for backpropagation

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=10)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=10)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
```

<br>


## 📊 TensorBoard Logging

TensorBoard is a visualization tool for inspecting model metrics, losses, learning curves, and more during and after training.

In PyTorch Lightning:
- Logging is handled automatically via the built-in `TensorBoardLogger`.
- Metrics such as `train_loss`, `val_loss`, and custom ones you log with `self.log(...)` are stored as events.

```python
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# Set up logger (creates logs/lightning_logs/ by default)
logger = TensorBoardLogger("logs", name="my_experiment")

# Trainer with logger
trainer = Trainer(logger=logger)
```

This will generate a folder like:

```
logs/
└── my_experiment/
    └── version_0/
        ├── events.out.tfevents.1234...
        └── checkpoints/
```

- `events.out.tfevents...`: Contains all metric logs.
- `checkpoints/`: Stores model checkpoint files if `ModelCheckpoint` is used.

<br>

## 🧪 Running Training

Lightning replaces your training loop with a single line:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Instantiate logger
logger = TensorBoardLogger("lightning_logs", name="my_model")

# Instantiate Trainer with options
trainer = Trainer(
    max_epochs=10,
    accelerator="auto",     # Auto GPU/CPU
    logger=logger
)

# Create data loaders
train_loader = ...
val_loader = ...

# Create model
model = LitClassifier(model=MyModel(), learning_rate=1e-3)

# Fit
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

<br>

## 📈 Logging and Metrics

With `self.log()` inside your `_step` methods, Lightning automatically:
- Logs metrics per step and epoch.
- Integrates with TensorBoard, WandB, CSV, etc.
- Handles distributed metrics aggregation.

```python
self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
```

| Flag | Meaning |
|------|---------|
| `prog_bar=True` | Shows in progress bar |
| `on_step=True` | Logs on every batch |
| `on_epoch=True` | Logs average per epoch |

All the saved metrics can be visualized. Launch TensorBoard with the following terminal command inside your project folder:

```bash
tensorboard --logdir=logs/
```

Then go to `http://localhost:6006/` in your browser.

You will see:
- Metric plots (`loss`, `accuracy`)
- Scalar summaries
- Histograms of weights/biases (if configured)

> **Note**: Make sure `tensorboard` is installed via `pip install tensorboard`.

This allows you to monitor model performance in real-time while training or analyze it afterward.


<br>

## 🔧 Optimizers and Schedulers

`configure_optimizers()` lets you return:
- one optimizer
- a tuple (optimizer, scheduler)
- or even multiple optimizers/schedulers for GANs, etc.

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return [optimizer], [scheduler]
```

<br>

## 📦 Extras: Checkpointing, EarlyStopping, Resume

In PyTorch Lightning, features like model checkpointing, early stopping, and training resumption are conveniently handled through callbacks passed to the `Trainer()`.

- ModelCheckpoint: Automatically saves the best model during training based on a chosen metric.
- EarlyStopping: Stops training early if the monitored metric doesn’t improve after a defined number of validation steps.

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

# Define callbacks
checkpoint = ModelCheckpoint(
    monitor="val_acc",  # Save model with best validation accuracy
    mode="max",  # Maximize validation accuracy
    save_top_k=1,  # Keep only the best checkpoint
    filename="best-{epoch}-{val_acc:.2f}"  # Custom filename format
)

early_stop = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    mode="min",  # Minimize validation loss
    patience=3  # Stop if no improvement for 3 epochs
)

# Create Trainer with callbacks
trainer = Trainer(
    callbacks=[checkpoint, early_stop]
)
```

These callbacks create a `checkpoints/` folder by default, which contains your best models saved as `.ckpt` files. You can resume training by passing `resume_from_checkpoint="path/to/best.ckpt"` to the `Trainer()`.

<br>

## 🧪 Test After Training

Once training is complete, you can easily run evaluation on your test set:

```python
trainer.test(model, dataloaders=test_loader)
```

This uses the best checkpoint by default (if `ModelCheckpoint` was used). You don’t need to manually reload the model.

<br>

## 📝 Notes

Debugging using Tensorboard needs some precautions:
- Always use `num_workers=0` in the Dataloaders. Use the debugger and a number of workers greater than 0 can cause thred problems.
- If you log images, a `num_workers` greater than 0 can cause Race Condition and crush. If you use `matplotlib` ensure you specify `matplotlib.use("Agg")` to disable any backend GUI and thus use a number of workers greater than 0, log images and avoid crushes. But if you are debugging, this mode will suppress any GUI interaction (e.g., `plt.show()`), thus make sure to use it only when running a training pipeline and not in debugging mode.

<br>
<br>
<br>

[Back to Index 🗂️](./README.md)