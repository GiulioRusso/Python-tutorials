[Back to Index 🗂️](./README.md)

# 📦 Create a Python package installable via PyPI

## 1. PyPI account 👤
1. Create a PyPI account at: https://pypi.org
2. Add a second email as backup option.
3. Save the Recovery codes.
4. Activate the Two Factor Authentication.

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

## 3. Create API Token 🔑
1. Log in to your PyPI account and navigate to your account settings.
2. Under "API tokens," click "Add API token".
3. Name your token (e.g. "package_upload_token") and create it.
4. Save the token string "*pypi-...*" securely (e.g. in a password manager). You will need it to upload your package.

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
    twine upload dist/*
    ```
 **Note**: its not necessary to upload the entire directory. You can select which *.tar.gz* and *.whl* upload to the PyPI account.

5. Insert the API Token asked.

6. Check that your package is listed on your PyPI account.

<br>

**Note**: When you need to update your package, just add the new/modified code and repeat the steps `2.` (build the new package version) and `4.` (upload the new build). Note that it could be needed to delete the old version builds from your `dist` folder.

<br>

[Back to Index 🗂️](./README.md)