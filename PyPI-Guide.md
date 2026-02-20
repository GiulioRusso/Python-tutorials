[Back to Index 🗂️](./README.md)

<center><h1>📦 Create a Python package installable via PyPI</h1></center>

<br>

## 1. PyPI account 👤
1. Create a PyPI account at: https://pypi.org
2. Add a second email as backup option.
3. Save the Recovery codes.
4. Activate the Two Factor Authentication.

<br>

## 2. Local package project 📂
1. Organize your package directory in the following format:
```bash
.
├── your_package/
│   ├── __init__.py       # Marks this folder as a Python package (can be empty or contain init code)
│   └── ...               # Other modules/files for your package
├── LICENSE               # License information for your package (e.g. MIT, Apache, etc.)
├── README.md             # Explains what your package does and how to use it
├── requirements.txt      # Lists required packages/dependencies for your project
└── setup.py              # Contains metadata and instructions for building/installing your package
```

Below, an example of *setup.py*:
```python
from setuptools import setup, find_packages

setup(
    name='your_package',  # Replace with your package name
    version='0.1.0',  # Initial version number
    packages=find_packages(),  # Automatically finds `__init__.py` files and adds them to the package
    install_requires=[],  # List of dependencies (e.g., 'requests', 'numpy')
    author='Your Name',
    author_email='your_email@example.com',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package',  # Project URL (GitHub, GitLab, etc.)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
```

<br>

## 3. Create API Token 🔑
1. Log in to your PyPI account and navigate to your account settings.
2. Under "API tokens," click "Add API token".
3. Name your token (e.g. "package_upload_token") and create it.
4. Save the token string "*pypi-...*" securely (e.g. in a password manager). You will need it to upload your package.

<br>

## 4. Load your package on PyPI 🔋

**Note**: In these steps some packages need to be installed. No matter where these packages will be installed, if inside a local virtual environment or a global one. Make sure that the environment where you install it is the one used in your terminal.

Inside the package project:

1. Install the necessary packages:
    ```bash
    pip3 install setuptools wheel
    ```

2. Set up your package.
    ```bash
    python3 setup.py sdist bdist_wheel
    ```

**Note**: before distributing your package, its better to test it by deploying it locally on your machine with:
```bash
pip3 install dist/<package_name>-<package_version>-py3-none-any.whl
```
and trying to use it in a separate Python project. You can always check if a package is correctly installed with:
    ```bash
    pip3 show <package_name>
    ```

3. Install the twine package in order to upload it to PyPI:
    ```bash
    pip3 install twine
    ```

4. Upload all the *.tar.gz* and the *.whl* of the package distribution:
    ```bash
    twine check dist/*
    twine upload dist/*
    ```
 **Note**: its not necessary to upload the entire directory. You can select which *.tar.gz* and *.whl* upload to the PyPI account.

5. Insert the API Token asked.

6. Check that your package is listed on your PyPI account.

<br>

**Note**: When you need to update your package, just add the new/modified code and repeat the steps `2.` (build the new package version) and `4.` (upload the new build). Note that it could be needed to delete the old version builds from your `dist` folder.

<br>
<br>
<br>

## 5. Deploy using `pyproject.toml` (modern alternative to `setup.py`) 🆕

`pyproject.toml` is the modern standard (PEP 517/518) and the recommended replacement for `setup.py`. The project structure becomes:

```bash
.
├── your_package/
│   ├── __init__.py
│   └── ...
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml        # Replaces setup.py
```

Below, an example of *pyproject.toml*:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "your_package"
version = "0.1.0"
description = "A short description of your package"
readme = "README.md"
license = { file = "LICENSE" }
authors = [
  { name = "Your Name", email = "your_email@example.com" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.6"
dependencies = []          # List of dependencies (e.g., "requests", "numpy")

[project.urls]
Homepage = "https://github.com/yourusername/your_package"
```

1. Install the necessary packages:
    ```bash
    pip3 install build twine
    ```

2. Build your package:
    ```bash
    python3 -m build
    ```
    This generates the same `dist/` folder with `.tar.gz` and `.whl` files as the `setup.py` approach.

3. Export your API Token as environment variables to avoid entering it manually every time:

    **macOS/Linux** — add to `~/.zshrc` or `~/.bashrc` for persistence:
    ```bash
    export TWINE_USERNAME=__token__
    export TWINE_PASSWORD=pypi-your-api-token-here
    ```
    Apply the changes:
    ```bash
    source ~/.zshrc   # or source ~/.bashrc
    ```

    **Windows** — set via PowerShell:
    ```powershell
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = "pypi-your-api-token-here"
    ```

    > **Note**: Always set `TWINE_USERNAME` to the literal string `__token__` and `TWINE_PASSWORD` to your API token (starting with `pypi-`).

4. Upload to PyPI (credentials will be read automatically from the environment):
    ```bash
    twine check dist/*
    twine upload dist/*
    ```

**Note**: To update your package, bump the `version` field in `pyproject.toml`, rebuild (step `2.`), and re-upload (step `4.`). PyPI does not allow re-uploading the same version, so always increment it first.

<br>
<br>
<br>

## 📦 Exposing the public API with `__init__.py`

The `__init__.py` file defines the **public interface** of your package.
Its purpose is **not** to implement logic, but to control *what users can and should import*.

---

### 🎯 Why `__init__.py` matters

A well-written `__init__.py` allows users to write:

```python
import your_package
your_package.some_function()
```

or:

```python
from your_package import useful_function
```

without needing to know the internal file structure of your package.

It also:

* Improves usability
* Prevents leaking internal implementation details
* Makes refactoring easier without breaking users’ code

---

### ✅ Recommended structure

Assume your package structure is:

```text
your_package/
├── __init__.py
├── preprocessing.py
├── draw.py
├── _version.py
```

---

### 1️⃣ Define the version in a single place

Create a dedicated file for the version:

```python
# your_package/_version.py
__version__ = "0.1.0"
```

This avoids duplication and allows tooling to read the version safely.

---

### 2️⃣ Expose selected objects in `__init__.py`

```python
from ._version import __version__

from .preprocessing import normalize, resize
from .draw import draw_box

__all__ = [
    "__version__",
    "normalize",
    "resize",
    "draw_box",
]
```

Only **public, stable functions or classes** should be exposed here.

---

### 🚫 What to avoid in `__init__.py`

* Heavy imports or expensive computations
* Wildcard imports (`from module import *`)
* Internal helpers not meant for users
* Side effects (prints, file I/O, downloads)

Example of **bad practice**:

```python
from .preprocessing import *
print("Package loaded")  # ❌
```

---

### 🧠 Design guideline

> Think of `__init__.py` as your package’s **API contract**.

If a function is imported in `__init__.py`, you are promising users it will remain stable.

---

### 🧪 Quick check

After installation, the following should work:

```python
import your_package
print(your_package.__version__)
help(your_package)
```

If users can discover and use your package **without reading the source code**, your `__init__.py` is doing its job.


<br>
<br>
<br>

[Back to Index 🗂️](./README.md)