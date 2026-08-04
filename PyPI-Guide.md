[Back to Index 🗂️](./README.md)

<center><h1>📦 Create a Python package installable via PyPI</h1></center>

<br>

## 1. PyPI account 👤
1. Create a PyPI account at: https://pypi.org
2. Add a second email as backup option.
3. Save the Recovery codes.
4. Set up **Two-Factor Authentication** — as of 2024 this is **mandatory** on PyPI (not optional) before you can upload or manage a package.

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
├── requirements.txt      # Dev-time dependencies for working on the package (see note below)
├── .gitignore             # Exclude build/, dist/, *.egg-info/ from version control
└── setup.py              # Contains metadata and instructions for building/installing your package
```

> **Note**: `requirements.txt` here is for **developing** the package (tests, linters, build tools). The dependencies your package needs **at install time** belong in `install_requires` (`setup.py`) or `dependencies` (`pyproject.toml`, see §5) — that's what `pip install your_package` actually reads. Don't rely on `requirements.txt` for that; it isn't packaged or installed alongside your library.

> **Note**: the flat layout above (`your_package/` next to `setup.py`) has a common footgun — if you `import your_package` while inside this directory (e.g. running tests from the repo root), Python may silently import the **source folder** instead of the installed package, hiding packaging bugs until a real user installs the wheel. If this bites you, switch to a **`src/` layout** (`src/your_package/...`), which makes it impossible to import the package without installing it first — the standard fix recommended by the Python Packaging Authority.

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
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package',  # Project URL (GitHub, GitLab, etc.)
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
```
> `setup.py`/`setup.cfg` is the legacy way of declaring package metadata. `pyproject.toml` (§5 below) is the modern standard — prefer it for new packages.

<br>

## 3. Create API Token 🔑
1. Log in to your PyPI account and navigate to your account settings.
2. Under "API tokens," click "Add API token".
3. **Scope the token to the single project** you're publishing, not your whole account — limits the blast radius if it ever leaks.
4. Name your token (e.g. "package_upload_token") and create it.
5. Save the token string "*pypi-...*" securely (e.g. in a password manager). You will need it to upload your package.

> For CI/CD (GitHub Actions etc.), skip API tokens entirely and use **Trusted Publishing** instead — see the note in §5, step 3.

<br>

## 4. Load your package on PyPI 🔋

**Note**: In these steps some packages need to be installed. No matter where these packages will be installed, if inside a local virtual environment or a global one. Make sure that the environment where you install it is the one used in your terminal.

Inside the package project:

1. Install the necessary packages:
    ```bash
    pip3 install setuptools wheel
    ```

2. Clean any previous build, then build your package (PyPI rejects re-uploading an existing version, so stale artifacts in `dist/` from an old version number are harmless, but leftover files from a *failed* build of the *same* version can get uploaded by mistake):
    ```bash
    rm -rf build/ dist/ *.egg-info
    python3 setup.py sdist bdist_wheel
    ```

**Note**: before distributing your package, it's better to test it by deploying it locally on your machine with:
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

4. **Rehearse on TestPyPI first** — a separate index for exactly this purpose, so a mistake doesn't burn a version number on the real index (PyPI never lets you re-upload or delete-and-reuse a version):
    ```bash
    twine check dist/*
    twine upload --repository testpypi dist/*
    ```
    Then verify the install works from there:
    ```bash
    pip install --index-url https://test.pypi.org/simple/ <package_name>
    ```
    You'll need a [separate TestPyPI account and API token](https://test.pypi.org/) — it does not share accounts/tokens with the real PyPI.

5. Once verified, upload for real:
    ```bash
    twine upload dist/*
    ```
    Insert the API Token when asked.

6. Check that your package is listed on your PyPI account.

<br>

**Note**: When you need to update your package, just add the new/modified code and repeat the steps `2.` (build the new package version, after bumping `version`) and step `4`/`5` (rehearse, then upload). PyPI does not allow re-uploading the same version number under any circumstances, so always increment it first.

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
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your_package"
dynamic = ["version"]      # version is read from your_package/_version.py — see the __init__.py section below
description = "A short description of your package"
readme = "README.md"
license = "MIT"                    # SPDX expression (PEP 639) — replaces the old license = {file = "..."} table
license-files = ["LICENSE"]
authors = [
  { name = "Your Name", email = "your_email@example.com" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
requires-python = ">=3.9"
dependencies = []          # List of dependencies (e.g., "requests", "numpy")

[project.urls]
Homepage = "https://github.com/yourusername/your_package"

[tool.setuptools.dynamic]
version = { attr = "your_package._version.__version__" }
```
> The old `license = { file = "LICENSE" }` table form and the `"License :: OSI Approved :: ..."` classifier are both **deprecated** since setuptools 77 / PEP 639 — use the SPDX string + `license-files` shown above instead.

1. Install the necessary packages:
    ```bash
    pip3 install build twine
    ```

2. Clean previous build artifacts, then build your package:
    ```bash
    rm -rf build/ dist/ *.egg-info
    python3 -m build
    ```
    This generates the same `dist/` folder with `.tar.gz` and `.whl` files as the `setup.py` approach.

3. Authenticate for upload — two options:

   **Option A — Trusted Publishing (recommended for CI, e.g. GitHub Actions)**: no token to create or store at all. On PyPI, go to your project → **Publishing** → add a Trusted Publisher pointing at your GitHub repo/workflow. In the workflow, grant `permissions: id-token: write` and use [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish) with no credentials — PyPI issues a short-lived, workflow-scoped token automatically over OIDC. Nothing to leak, nothing to rotate.

   **Option B — API token, for local/manual uploads**: store it in `~/.pypirc`, not in your shell rc file — an rc file is easy to accidentally commit via a dotfiles repo, and a leaked long-lived upload token there can be used to publish malicious versions under your name until you notice and revoke it.
    ```ini
    # ~/.pypirc
    [pypi]
    username = __token__
    password = pypi-your-api-token-here
    ```
    ```bash
    chmod 600 ~/.pypirc
    ```
    `twine` reads this file automatically — no env vars needed. If you do prefer env vars for a one-off upload, export them in the shell session only, never persist them to `~/.zshrc`/`~/.bashrc`:
    ```bash
    export TWINE_USERNAME=__token__
    export TWINE_PASSWORD=pypi-your-api-token-here
    ```

4. Rehearse on TestPyPI, then upload to PyPI (see step 4/5 in §4 above for the same rehearse-first flow):
    ```bash
    twine check dist/*
    twine upload --repository testpypi dist/*
    twine upload dist/*
    ```

**Note**: To update your package, bump the version in `your_package/_version.py` (which `dynamic = ["version"]` reads automatically), rebuild (step `2.`), and re-upload (step `4.`). PyPI does not allow re-uploading the same version, so always increment it first.

> If you're using [`uv`](./Python-Setup-Tutorial.md#-uv-fast-all-in-one) for the project, `uv build` and `uv publish` cover steps 2 and 3/4 with the same Trusted Publishing / token support, without needing `build`/`twine` as separate installs.

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
* Makes refactoring easier without breaking users' code

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

This avoids duplication and allows tooling to read the version safely — it's also what `pyproject.toml`'s `dynamic = ["version"]` + `[tool.setuptools.dynamic]` (§5 above) points at, so you only ever bump the number in this one file.

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

> Think of `__init__.py` as your package's **API contract**.

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
