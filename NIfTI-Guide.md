[Back to Index 🗂️](./README.md)

<center><h1>🧠 Guide to NIfTI Files</h1></center>

<br>

## ℹ️ Introduction

NIfTI (Neuroimaging Informatics Technology Initiative) is a file format commonly used to store medical image data, particularly MRI scans, CTA and more. This guide will explain what .nii.gz files are, why they are used, how to handle them, and common operations you may need to perform.

<br>

## 📁 What is a .nii.gz File?

The .nii.gz file is a compressed version of the NIfTI format. NIfTI files store 3D data `(X, Y, Z)` or 4D data for time series `(X, Y, Z, T)`, commonly from medical imaging techniques. NIfTI files contains different metadata:

**1. Header**: Contains metadata about the image.<br>
- *Voxel Information*: the physical size (typically in millimeters) of a 3D pixel, and the space betwen them.
- *Dimensions*: number of voxels in each direction.
- *Datatype* type of the data pixels (uint8, int16, float32, etc.)
- *Affine Matrix*: 4x4 matrix that maps voxel coordinates to real-world coordinates in millimiters. The first 3x3 submatrix apply rotation and scaling. The last column is a translation vector maps voxel coordinates to world coordinates in millimeters

    <img src="images/NIfTI-Guide/affine.png" alt="https://dartbrains.org/content/Preprocessing.html" width=500px>


**2. Data**: The actual image data, stored in a multi-dimensional array.<br>

**3. Compression**: The .gz compression ensures smaller file sizes while maintaining the integrity of the data. When working with `.nii.gz` files, the compression method used can vary. This affects how tools interpret the file structure. Running:
```bash
file your_nifti.nii.gz
```
You may see different outputs:

- **Standard Gzip Compression**: This means the file was compressed using standard `gzip`, and the `file` command correctly recognizes it as a Gzip archive.

   ```
   your_file.nii.gz: gzip compressed data, original size modulo 2^32 XXXXXX
   ```
- **Unexpected File Type**: This suggests that the file was saved with a different compression method or format. If re-saving the `.nii.gz` file using `nibabel`, the compression format could change, making it recognizable as a standard Gzip archive.

   ```
   your_file.nii.gz: dBase III DBT, version number 0, next free block index 348
   ```

<br>
<img src="./images/NIfTI-Guide/slice-info.png" width=300px>
<br>
<img src="./images/NIfTI-Guide/volume-info.png" width=300px>

<br>
<br>
<br>

## 🐍 Installing Necessary Libraries

Before working with .nii.gz files, you'll need the right libraries. In Python, the most commonly used libraries are:
- Nibabel
    ```
    pip3 install nibabel
    ```
- SimpleITK
    ```
    pip3 install SimpleITK
    ```

<br>
<br>
<br>

<center><h1>🧭 Reference system</h1></center>

NIfTI files has a straighforward reference system with the origin placed behing the `right ear of the patient`:

<img src="./images/NIfTI-Guide/3D-reference-system-CTA.png" width=200px>

<br>

The 3D volume reference system can be visualized like this:

<img src="./images/NIfTI-Guide/3D-reference-system.png" width=200px>

<br>

The 2D slices can be extracted from three different point of views:

<table>
    <tr>
        <td><img src="./images/NIfTI-Guide/axial-3D.png" width=150px></td>
        <td><b>Axial</b> (scrolling along Z-Axis): plane that divide the bottom from the top.</td>
    </tr>
        <tr>
        <td><img src="./images/NIfTI-Guide/coronal-3D.png" width=150px></td>
        <td><b>Coronal</b> (scrolling along Y-Axis): plane that divide the front from the back.</td>
    </tr>
        <tr>
        <td><img src="./images/NIfTI-Guide/sagittal-3D.png" width=150px></td>
        <td><b>Sagittal</b> (scrolling along X-Axis): plane that divide the right from the legt.</td>
    </tr>
</table>

<br>

The 2D reference system on each slice is the following: <br>

<table>
    <tr>
        <td><img src="./images/NIfTI-Guide/axial-2D.png" width=150px></td>
        <td><b>Axial</b>: the slice is seen `from the bottom`.</td>
    </tr>
        <tr>
        <td><img src="./images/NIfTI-Guide/coronal-2D.png" width=150px></td>
        <td><b>Coronal</b>: the slice is seen `from the front`.</td>
    </tr>
        <tr>
        <td><img src="./images/NIfTI-Guide/sagittal-2D.png" width=150px></td>
        <td><b>Sagittal</b>: the slice is seen `from the right side of the patient`</td>
    </tr>
</table>

<br>

The 3D reference system is mapped on the 2D slices in a different way for each view. 

<img src="./images/NIfTI-Guide/3D-projection.png" width=250px>

<img src="./images/NIfTI-Guide/3D-mapping.png" width=250px>

The 3D and 2D axis are mapped as follow:

<table>
    <tr>
        <td><img src="./images/NIfTI-Guide/axial-mapping.png" width=150px></td>
        <td>
        <table>
            <b><center>Axial</center></b>
            <tr><td>2D</td><td>3D</td></tr>
            <tr><td>X</td><td>Y</td></tr>
            <tr><td>Y</td><td>X</td></tr>
            <tr><td>Slice Number</td><td>Z</td></tr>
        </table>
        </td>
    </tr>
    <tr>
        <td><img src="./images/NIfTI-Guide/coronal-mapping.png" width=150px></td>
        <td>
        <table>
            <b><center>Coronal</center></b>
            <tr><td>2D</td><td>3D</td></tr>
            <tr><td>X</td><td>Z</td></tr>
            <tr><td>Y</td><td>X</td></tr>
            <tr><td>Slice Number</td><td>Y</td></tr>
        </table>
        </td>
    </tr>
    <tr>
        <td><img src="./images/NIfTI-Guide/sagittal-mapping.png" width=150px></td>
        <td>
        <table>
            <b><center>Sagittal</center></b>
            <tr><td>2D</td><td>3D</td></tr>
            <tr><td>X</td><td>Z</td></tr>
            <tr><td>Y</td><td>Y</td></tr>
            <tr><td>Slice Number</td><td>X</td></tr>
        </table>
        </td>
    </tr>
</table>

<br>

In summary, every coordinate inside the 3D reference system can be mapped on the 2D reference system of each slice with this matching:

<table>
    <tr>
        <td><b><center>3D</center></b></td>
        <td><b><center>2D</center></b></td>
    </tr>
    <tr>
        <td><b>X</b></td>
        <td>
            <table>
                <tr><td>Axial</td><td>Y</td></tr>
                <tr><td>Coronal</td><td>Y</td></tr>
                <tr><td>Sagittal</td><td>Slice Number</td></tr>
            </table>
        </td>
    </tr>
    <tr>
        <td><b>Y</b></td>
        <td>
            <table>
                <tr><td>Axial</td><td>X</td></tr>
                <tr><td>Coronal</td><td>Slice Number</td></tr>
                <tr><td>Sagittal</td><td>Y</td></tr>
            </table>
        </td>
    </tr>
    <tr>
        <td><b>Z</b></td>
        <td>
            <table>
                <tr><td>Axial</td><td>Slice Number</td></tr>
                <tr><td>Coronal</td><td>X</td></tr>
                <tr><td>Sagittal</td><td>X</td></tr>
            </table>
        </td>
    </tr>
</table>

<br>
<br>
<br>

<center><h1>👩‍⚕️ Common operations on NIfTI files</h1></center>

<br>

### ⚙️ Load and Save a `.nii.gz` File

```python
import nibabel as nib

# Load the .nii.gz file into a NIfTI image object
img = nib.load('toy-CTA.nii.gz')

# Get the data from the file as a NumPy array
data = img.get_fdata()

# Create a NIfTI image
nifti_img = nib.Nifti1Image(data, affine)

# Save the NIfTI image
output_path = "path/to/output_image.nii.gz"
nib.save(nifti_img, output_path)

print(f"NIfTI image saved to: {output_path}")
```

<br>

### 🛠️ Open and Save `.nii.gz` with `gz` compression using SimpleITK

If the `.nii.gz` file is not saved with `gz` compression, `nibabel` can't open the NIfTI file. In this case it's possible to open the file with SimpleITK and save it with the `gz` compression.

```python
import SimpleITK as sitk

# Load the .nii.gz image
input_path = "path/to/your_image.nii.gz"
image = sitk.ReadImage(input_path)

# Save the image with gz compression
output_path = "path/to/compressed_image.nii.gz"
sitk.WriteImage(image, output_path, useCompression=True)

print(f"Compressed image saved to {output_path}")
```

<br>

### 📑 Extracting Header Information from a NIfTI File using Nibabel

```python
import nibabel as nib

# Load the NIfTI file
nifti_file = "path/to/your_image.nii.gz"
img = nib.load(nifti_file)
header = img.header

# Extract relevant metadata
print(f"Image Shape: {header.get_data_shape()}")
print(f"Voxel Dimensions: {header.get_zooms()}")
print(f"Data Type: {header.get_data_dtype()}")
```

<br>

### 🍕 Extracting slices

```python
# Axial slice (Z-Axis)
axial_slice = data[:, :, 50]  # Get the 50th slice along Z

# Coronal slice (Y-Axis)
coronal_slice = data[:, 50, :]  # Get the 50th slice along Y

# Sagittal slice (X-Axis)
sagittal_slice = data[50, :, :]  # Get the 50th slice along X
```

<br>

### 🎙️ Show slice

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(axial_slice, cmap='gray')
plt.title('Axial Slice')

plt.subplot(1, 3, 2)
plt.imshow(coronal_slice, cmap='gray')
plt.title('Coronal Slice')

plt.subplot(1, 3, 3)
plt.imshow(sagittal_slice, cmap='gray')
plt.title('Sagittal Slice')

plt.show()
```

<br>

### 🔄 Loop through slices

```python
# Loop through all slices along the Z-axis
for slice_index in range(data.shape[2]):
    axial_slice = data[:, :, slice_index]
    
# Loop through all slices along the Y-axis
for slice_index in range(data.shape[1]):
    axial_slice = data[:, slice_index, :]
    
# Loop through all slices along the X-axis
for slice_index in range(data.shape[0]):
    axial_slice = data[slice_index, :, :]
    
```

<br>

### ✂️ Cut the volume

```python
# Define the slicing indices (z_min:z_max, y_min:y_max, x_min:x_max)
z_min, z_max = 30, 70
y_min, y_max = 40, 120
x_min, x_max = 50, 150

# Cut the volume
cut_data = data[z_min:z_max, y_min:y_max, x_min:x_max]
```

<br>

### 📊 Clip Volume Data Between Two Values

```python
# Clip values in the volume data between 0 and 200
clipped_data = np.clip(data, 0, 200)

# Verify the range
print(f"Data min: {clipped_data.min()}, Data max: {clipped_data.max()}")
```

<br>

### 🧊 Extract the minimum 3D Bounding Box around the brain

```python
# Load the brain mask NIfTI file
nifti_file = "path/to/toy-CTA.nii.gz"
img = nib.load(nifti_file)
data = img.get_fdata()

# Ensure it's a binary mask (0s and 1s)
binary_mask = (data > 0).astype(np.uint8)

# Get the non-zero voxel indices
z_indices, y_indices, x_indices = np.where(binary_mask)

# Compute the 3D bounding box
z_min, z_max = np.min(z_indices), np.max(z_indices)
y_min, y_max = np.min(y_indices), np.max(y_indices)
x_min, x_max = np.min(x_indices), np.max(x_indices)

print(f"Bounding Box: \nZ: {z_min}-{z_max}, Y: {y_min}-{y_max}, X: {x_min}-{x_max}")

# Extract the region of interest
bounding_box_data = data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

# Save the extracted bounding box as a new NIfTI image
bounding_box_img = nib.Nifti1Image(bounding_box_data, img.affine, img.header)
output_path = "path/to/toy-CTA-bbox.nii.gz"
nib.save(bounding_box_img, output_path)

print(f"Bounding box saved to {output_path}")
```

<br>

### 🙏 Padding
```python
import nibabel as nib
import numpy as np

# Load the brain mask NIfTI file
nifti_file = "path/to/toy-CTA.nii.gz"
img = nib.load(nifti_file)
data = img.get_fdata()

# Ensure binary mask
binary_mask = (data > 0).astype(np.uint8)

# Get non-zero voxel indices
z_indices, y_indices, x_indices = np.where(binary_mask)

# Compute the 3D bounding box
z_min, z_max = np.min(z_indices), np.max(z_indices)
y_min, y_max = np.min(y_indices), np.max(y_indices)
x_min, x_max = np.min(x_indices), np.max(x_indices)

# Define the desired number of slices in each dimension
desired_depth, desired_height, desired_width = 80, 150, 150

# Compute the required padding
pad_z = max(0, desired_depth - (z_max - z_min + 1)) // 2
pad_y = max(0, desired_height - (y_max - y_min + 1)) // 2
pad_x = max(0, desired_width - (x_max - x_min + 1)) // 2

# Apply padding equally on both sides to keep the region centered.
padded_data = np.pad(
    data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1],
    ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)),
    mode='constant', constant_values=0
)

# Save the padded bounding box as a new NIfTI image
padded_img = nib.Nifti1Image(padded_data, img.affine, img.header)
output_path = "path/to/toy-CTA-padded.nii.gz"
nib.save(padded_img, output_path)

print(f"Padded bounding box saved to {output_path}")
```

<br>
<br>
<br>

<center><h1>🔭 Tools for imaging</h1></center>

Let's explore some useful tools when working with images and also with 3D images.

<br>
<br>

<center><h1>🔬 ImageJ</h1></center>

ImageJ is a powerful, open-source image processing tool widely used in scientific research, particularly in medical imaging, microscopy, and bioinformatics. It supports a broad range of image formats, including NIfTI, DICOM, TIFF, and JPEG, and provides extensive analysis tools.

<br>

## ⚙️ Installation
ImageJ is available for Windows, macOS, and Linux on the [Official Website](https://imagej.nih.gov/ij/).

<br>

## 🏞️ ImageJ Interface Overview
When you launch ImageJ, the main toolbar appears with essential tools for interacting with images:

### 📺 **Main Image View**
- Displays the currently opened image.
- Allows zooming, panning, and changing views.
- Supports multi-dimensional image stacks (e.g. time-series, Z-stacks).

### 🪛 **Toolbar**
- Contains essential tools such as selection, zoom, measurement, drawing, and text annotation.
- Users can configure and customize the toolbar.

### 📏 ImageJ Coordinate System
ImageJ follows a pixel-based coordinate system, where:
- (0,0) is the top-left corner of the image.
- X-axis increases horizontally to the right.
- Y-axis increases vertically downward.
- Z-axis (if present) represents slices in a stack (e.g. depth in 3D images).
- T-axis (if present) represents time in dynamic image series.

For ROI (Region of Interest) analysis, coordinates are measured in pixels or physical units (e.g. millimeters) if metadata is available.

<img src="./images/NIfTI-Guide/ImageJ-1.png" width=500px>

<br>
<br>
<br>

<center><h1>🩻 ITK-SNAP</h1></center>

ITK-SNAP is a powerful open-source tool for visualizing, annotating, and segmenting 3D medical images. It is widely used for neuroimaging and other medical image analysis tasks.

<br>

## ⚙️ Installation

ITK-SNAP is available for Windows, macOS, and Linux. Download the latest version from the [Official Website](https://www.itksnap.org/)

<br>

## 🏞️ ITK-SNAP Interface Overview

When you open ITK-SNAP, you will see the following sections:

### 📺 **Main Image View**
- Displays the Axial, Coronal, and Sagittal views of the loaded NIfTI image.
- Allows scrolling through slices along each axis.
- Supports zooming and panning for detailed analysis.

    <img src="./images/NIfTI-Guide/ITK-main.png" width=500px>

### 🎚️ **3D Volume Rendering Window**
- Provides a 3D view of the segmented structures.
- Allows rotation and zooming of the rendered volume.

    <img src="./images/NIfTI-Guide/ITK-volume-rendering-1.png" width=500px>

    <img src="./images/NIfTI-Guide/ITK-volume-rendering-2.png" width=500px>

### 🖌️ **Segmentation Panel**

- Used to manually or automatically segment different regions in the image with different tools:

    <img src="./images/NIfTI-Guide/ITK-segmentation-1.png" width=200px>

    <br>

    - Brushes:

        <img src="./images/NIfTI-Guide/ITK-segmentation-2.png" width=400px>

        The process has to be iterated moving thorugh the slices to select areas at different level inside the volume.

        The result can be saved in `Segmentation` ➡️ `Save Segmentation Image`. The resulting `.nii.gz` can be opened and inspected:

        <img src="./images/NIfTI-Guide/ITK-segmentation-3.png" width=400px>

    <br>

    - Polygon: 

        <img src="./images/NIfTI-Guide/ITK-segmentation-4.png" width=400px>

        <img src="./images/NIfTI-Guide/ITK-segmentation-5.png" width=400px>

        The process of identification of the polygon and accepting has to be iterated moving thorugh the slices to select areas at different level inside the volume.

        The result can be saved in `Segmentation` ➡️ `Save Segmentation Image`. The resulting `.nii.gz` can be opened and inspected:

        <img src="./images/NIfTI-Guide/ITK-segmentation-6.png" width=400px>

### 📑 Loading Multiple Files
ITK-SNAP allows loading multiple images simultaneously for comparison, overlay, and segmentation purposes:

- **Overlay Multiple Images:**
  - ITK-SNAP can display multiple images together by overlapping them.
  - This is useful for comparing anatomical structures, different imaging modalities, or segmentations.

    <img src="./images/NIfTI-Guide/ITK-multiple-1.png" width=400px>

    <img src="./images/NIfTI-Guide/ITK-multiple-2.png" width=400px>

    <img src="./images/NIfTI-Guide/ITK-multiple-3.png" width=400px>

    <img src="./images/NIfTI-Guide/ITK-multiple-4.png" width=400px>

- **Control Opacity of Overlays and Color Maps**
  - Use the Opacity Slider to adjust the transparency of the secondary image.
  - Helps in visualizing differences between datasets without obstructing key structures.
  - ITK-SNAP provides customizable color maps to differentiate structures.
  - Go to Segmentation Panel and assign unique colors to various labels to distinguish between anatomical regions.

    <img src="./images/NIfTI-Guide/ITK-multiple-5.png" width=400px>

<br>

## ℹ️ ITK-SNAP User Guide
For more details, visit the [ITK-SNAP User Guide](https://www.itksnap.org/pmwiki/pmwiki.php?n=Documentation.SNAP3).

<br>
<br>
<br>

<center><h1>⚒️ FSL</h1></center>

FSL (FMRIB Software Library) is a powerful set of tools used for analyzing and processing medical imaging data, particularly neuroimaging data. It includes a variety of programs for image registration, segmentation, and statistical analysis.

<br>

## ⚙️ Installation

FSL can be installed on macOS, Linux, and Windows (via WSL) using the [Official Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

### 📌 Basic Installation Steps (Linux/macOS)
- Download the installer:
    ```bash
    wget -O fslinstaller.py https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
    python3 fslinstaller.py
    ```
- Follow the on-screen instructions to complete the installation.

<br>

## 🎛️ Launching FSL Tools

- Start FSL Main GUI:
  ```bash
  fsl &
  ```

    <img src="./images/NIfTI-Guide/FSL.png" width=400px>

- Launch FSLView for Image Visualization:
  ```bash
  fsleyes &
  ```

    <img src="./images/NIfTI-Guide/FSLeyes.png" width=400px>

    Once loaded a `nii.gz` file (just drop the file on the FSLeyes window or open it via the `File` menu), FSLeyes will show it from all the three different views:

    <img src="./images/NIfTI-Guide/FSLeyes-example.png" width=400px>

<br>

## ℹ️ Obtain metadata info
The `fslhd` command provides detailed metadata about a NIfTI image file.
```bash
fslhd input.nii.gz
```

### **General Information**:
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `filename` | Path to the NIfTI file | `/path/to/file.nii.gz` | Incorrect path (missing or corrupt file) |
| `sizeof_hdr` | Header size (should always be 348 for NIfTI-1) | `348` | Anything other than `348` may indicate a corrupt file |
| `data_type` | The data type (usually INT16 for CT) | `INT16` | If `FLOAT32`, conversion may be needed (`fslmaths -dt int16`) |
| `dim0` | Number of dimensions (3 for 3D images) | `3` | If `4`, may indicate extra time dimension |

### **Image Dimensions**:
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `dim1, dim2, dim3` | Image size in X, Y, and Z directions | `512 x 512 x N` (N = num slices, e.g. `160-320`) | If `dim3` is too low (`< 20`), BET may fail. If too high (`> 400`), resampling may be needed. |
| `dim4` | Extra dimensions (should be `1` for 3D) | `1` | If greater than `1`, may contain a time dimension (use `fslroi` to extract a single volume). |
| `vox_units` | Units for voxel dimensions (`mm` for CT) | `mm` | If missing, file may be non-standard. |
| `time_units` | Units for time (not relevant for CT) | `s` or empty | If filled, check if the image is 4D. |

### **Voxel Size (Resolution):**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `pixdim1, pixdim2` | In-plane resolution (X, Y voxel size) | `0.4 - 0.6 mm` | If **too small** (`< 0.3 mm`), BET may fail. If **too large** (`> 1 mm`), resolution is too low. |
| `pixdim3` | Slice thickness (Z resolution) | `0.5 - 1.0 mm` | If **too small** (`< 0.3 mm`), may cause issues. If **too large** (`> 5 mm`), resample with `flirt`. |
| `pixdim4` | Time resolution | `0.0` | If nonzero, likely an incorrect 4D image. |

### **Intensity Calibration & Scaling:**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `cal_max, cal_min` | Calibration max/min intensity | `0.0` or actual HU range | If both are `0.0`, BET might fail. Fix with `fslmaths -add 0`. |
| `scl_slope` | Scaling factor applied to intensities | `0.003052` (default) | If `0.0`, values may be unreadable. If `NaN`, file is corrupt. |
| `scl_inter` | Intensity intercept value | `100.0` or `0.0` | If too high, image contrast may be incorrect. |
| `datatype` | Storage format for intensity values | `4 (INT16)` | If `16 (FLOAT32)`, convert with `fslmaths -dt int16`. |

### **Image Orientation & Transformation Matrices:**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `qform_code, sform_code` | Q-form and S-form matrix type (scanner coordinates) | `1 (Scanner Anat)` | If `0`, spatial information is missing. Fix with `fslorient -setqformcode 1`. |
| `qto_xyz` | Q-form transformation matrix (maps voxel to real-world space) | Nonzero 4x4 matrix | If all `0.0`, image has no spatial transformation. |
| `qform_xorient` | X-axis orientation | `Right-to-Left` | If incorrect (e.g. `Left-to-Right`), may cause misalignment. |
| `qform_yorient` | Y-axis orientation | `Posterior-to-Anterior` | If incorrect, image may be flipped. |
| `qform_zorient` | Z-axis orientation | `Inferior-to-Superior` | If incorrect, slices may be in the wrong order. |

<br>
<br>
<br>

## 🎮 Skull Stripping commands

<br>

### 🧠 Skull Stripping using `BET` (Brain Extraction Tool)
BET removes non-brain tissue from the 3D images.
```bash
bet input.nii.gz output_brain.nii.gz -m -f 0.5
```
- `-m`: Outputs a binary brain mask.
- `-f`: Fractional intensity threshold (default is `0.5`). Adjust for better extraction. Lower values keep more brain tissue.

<br>

### ⚡ Enhanced BET Skull Stripping with Bias Field Correction
To improve skull stripping by reducing intensity bias and improving robustness, use the following command:
```bash
bet input.nii.gz output_brain.nii.gz -B -f 0.5 -g 0
```
- `-B`: Applies bias field correction for better intensity normalization.
- `-f`: Fractional intensity threshold (default is `0.5`). Adjust for better extraction. Lower values keep more brain tissue.
- `-g 0`: Ensures the center-of-mass is automatically detected without additional adjustments.

<br>

### 🎯 Cropping Field of View using `robustfov`
Automatically crops the volume scan to remove unnecessary background.
```bash
robustfov -i input.nii.gz -r output_fov.nii.gz
```
- `-i`: Input image.
- `-r`: Output image focused on the Field of View.

**Note**: the output volume dimension might change from the input image. Robust FOV define the Volume of Interest.

<br>

### 📏 Image Resampling with `fslreorient2std`
Ensures image orientation follows standard conventions.
```bash
fslreorient2std input.nii.gz output_reoriented.nii.gz
```

<br>

### 🏗️ Image Thresholding with `fslmaths`
Threshold an image at a given intensity level.
```bash
fslmaths input.nii.gz -thr 0 -bin thresholded_mask.nii.gz
```
- `-thr 0`: Set intensity threshold at 0. All voxels with intensities below 0 are set to 0.
- `-bin`: Binarizes the output (values become 0 or 1).

<br>

Clip intensities inside a range.
```bash
fslmaths input.nii.gz -thr 0 -uthr 100  input_th.nii.gz
```
- `-thr 0`: Set lower intensity threshold at 0. All voxels with intensities below 0 are set to 0.
- `-uthr 100`: Set upper intensity threshold at 100. All voxels with intensities below 100 are set to 100.

<br>

## ℹ️ FSL Wiki

For more information, visit the [FSL Wiki](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki).


<br>
<br>
<br>

<center><h1>🧲 Registration Guide</h1></center>

<br>

## 🌎 MNI Template: Standardized Brain Atlas for Medical Imaging

The Montreal Neurological Institute (MNI) template is a widely used standardized brain atlas for medical imaging. It provides a common space for registering and analyzing brain scans obtained from MRI and CTA imaging. The MNI template is essential for comparing anatomical structures across different subjects and studies.

Here an example of the registration meaning:

<img src="images/NIfTI-Guide/registration.png" alt="http://jpeelle.net/mri/_images/registration.png" width="500">

### 📏 Purpose of MNI Template
- **Spatial Normalization**: Aligns brain scans to a common coordinate system.
- **Inter-Subject Comparisons**: Enables comparison of different brains in the same reference space.
- **Group Analysis**: Useful for statistical studies in neuroimaging.
- **Atlas-Based Segmentation**: Helps in identifying brain structures.
- **AI Applications**: Used as a preprocessing step for deep learning models in medical imaging.

### 🗺️ MNI Coordinate System
The MNI template follows a standardized coordinate system to describe brain anatomy.
- **Origin**: The standard coordinate system is based on the **anterior commissure (AC)**.
- **XYZ Axes**:
  - **X-axis**: Left (-) to Right (+).
  - **Y-axis**: Posterior (-) to Anterior (+).
  - **Z-axis**: Inferior (-) to Superior (+).

### 📐 MNI Template Dimensions
The most commonly used MNI template, **MNI152**, has the following dimensions:
- **Voxel size**: 1mm³ isotropic resolution.
- **Matrix size**: 182 × 218 × 182 (X × Y × Z).
- **Field of view**: Covers the entire human brain with precise alignment.

<img src="images/NIfTI-Guide/MNI-space.jpeg" alt="https://dartbrains.org/content/Preprocessing.html" width=500px>



<br>

## 🏗️ Common MNI Templates
Several MNI templates exist, optimized for different imaging modalities:

### 1️⃣ **MNI152 Template** (Most Common)
- Based on **152 averaged brain MRIs**.
- Available at **1mm, 2mm, and 4mm resolutions**.
- Used for **functional MRI (fMRI)** and **statistical parametric mapping (SPM)**.
- Can be downloaded from **[FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases)** and **[SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)**.

### 2️⃣ **MNI305 Template**
- Based on **305 MRI scans**.
- Older version, but still used in some neuroimaging studies.

### 3️⃣ **ICBM152 Nonlinear Asymmetric Template**
- Includes **nonlinear deformations** to match individual brain variations.
- Ideal for **high-precision brain mapping**.

<br>

## 🔄 Registering an Image to MNI Space
To align a CTA or MRI scan to the MNI template, image registration is required.

### 🛠️ Using FSL `FLIRT` (Linear Registration)
```bash
flirt -in input_cta.nii.gz -ref MNI152_T1_1mm.nii.gz -out registered_cta.nii.gz -omat transform.mat
```
- `-in input_cta.nii.gz` → Input brain scan.
- `-ref MNI152_T1_1mm.nii.gz` → MNI template reference.
- `-out registered_cta.nii.gz` → Registered output.
- `-omat transform.mat` → Transformation matrix file.

<br>

### 🔍 Using ANTs (Advanced Normalization)
```bash
antsRegistrationSyNQuick.sh -d 3 -f MNI152_T1_1mm.nii.gz -m input_mri.nii.gz -o output_
```
- `-d 3` → 3D registration.
- `-f` → Fixed image (MNI template).
- `-m` → Moving image (input scan).
- `-o` → Output prefix.

<br>

### 🔢 Using SPM (MATLAB Toolbox)
SPM provides graphical tools to align images to MNI space.
1. Open **SPM12** in MATLAB.
2. Select `Normalize (Est & Reslice)`
3. Choose the MNI152 template as the reference.
4. Select the input image for registration.
5. Run the process and save the output.




<br>
<br>
<br>

<center><h1>👾 Preprocessing pipelines</h1></center>

<br>

<h2>🔨Skulling pipeline for CTAs</h2>

Here's a CTA skulling pipeline example:

### ⚒️ FSL:

1. Crop the Field of View (FOV) to focus on the brain. CTA scans often include the neck and skull base, which interfere with BET.
    ```bash
    robustfov -i input.nii.gz -r input_fov.nii.gz
    ```

<br>

2. Intensity Clipping & Smoothing (CTA images may have high contrast regions like bones or vessels, that can interfere with skull stripping). Reduces noise and sharp intensity transitions, helping BET detect the brain boundary better. Clips intensities again after smoothing (ensures intensity values stay in [0,100]).
    ```bash
    fslmaths input_fov.nii.gz -thr 0 -uthr 100  input_th.nii.gz
    fslmaths input_th.nii.gz -s 1  input_th_sm.nii.gz
    fslmaths input_th_sm.nii.gz -thr 0 -uthr 100  input_th_sm_th.nii.gz
    ```

<br>

3. Brain Extraction Using BET.
    ```bash
    bet input_th_sm_th input_brain_sm -R -f 0.1 -g 0 -m
    ```
    Uses: <br>
    `-R` → Robust brain center estimation (recommended for non-MR images like CTA).<br>
    `-f 0.1` → More conservative skull stripping (keeps more peripheral brain tissue).<br>
    `-g 0` → No vertical gradient applied.<br>
    `-m` → Saves the binary brain mask (*_mask.nii.gz).<br>

<br>

4. Apply the Extracted Brain Mask to the Original CTA
    ```bash
    fslmaths input_fov.nii.gz -mul input_brain_sm_mask.nii.gz input_brain.nii.gz
    ```

<br>

### 🐍 Python:

```python
def fsl_pipeline(args):
    """
    Runs the full FSL pipeline and save the skull stripped image:
    1. FOV cropping.
    2. Intensity clipping and smoothing.
    3. BET.
    4. Apply BET mask to the original image and clip it.

    Parameters:
    - filename (str): Name of the original CTA to skull.
    - input_path (str): Input path where the input CTA is located
    - output_path (str): Output path where the output skulled CTA will be saved
    - f_value (str): Fractional intensity value.
    """
    filename, input_path, output_path, f_value = args

    input_image = os.path.join(input_path, filename)
    base_name = filename.replace(".nii.gz", "")
    fov_image = os.path.join(output_path, f"{base_name}_fov.nii.gz")
    threshold_image = os.path.join(output_path, f"{base_name}_th.nii.gz")
    smooth_image = os.path.join(output_path, f"{base_name}_th_sm.nii.gz")
    threshold_smooth_image = os.path.join(output_path, f"{base_name}_th_sm_th.nii.gz")
    skulled_image = os.path.join(output_path, f"{base_name}.skulled.nii.gz")
    mask_image = os.path.join(output_path, f"{base_name}.skulled_mask.nii.gz")
    final_brain_image = os.path.join(output_path, f"{base_name}_brain.nii.gz")
    output_image = os.path.join(output_path, f"{base_name}.skulled.clipped.nii.gz")

    try:
        # step 1: crop FOV
        subprocess.run(["robustfov", "-i", input_image, "-r", fov_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # step 2: intensity clipping and smoothing
        subprocess.run(["fslmaths", fov_image, "-thr", "0", "-uthr", "100", threshold_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["fslmaths", threshold_image, "-s", "1", smooth_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["fslmaths", smooth_image, "-thr", "0", "-uthr", "100", threshold_smooth_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # step 3: BET skull stripping
        subprocess.run(["bet", threshold_smooth_image, skulled_image, "-R", "-f", str(f_value), "-g", "0", "-m"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # step 4: apply brain mask to the original CTA
        subprocess.run(["fslmaths", input_image, "-mul", mask_image, final_brain_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # step 5: clip the final brain image
        # load the nifti file
        nii_img = nib.load(final_brain_image)
        nii_data = nii_img.get_fdata()
        # clip the values
        clipped_data = np.clip(nii_data, 0, 200)
        # create a new nifti image
        clipped_img = nib.Nifti1Image(clipped_data, nii_img.affine, nii_img.header)
        # save the modified image
        nib.save(clipped_img, output_image)

        return filename, None

    except subprocess.CalledProcessError as e:

        print(f"Error processing {filename}: {e}")

        return filename, str(e)
```

<br>
<br>
<br>

<h2>🏁 Resampling Guide</h2>

Before resampling a NIfTI image, it is important to extract key metadata such as size, spacing, origin, and direction. SimpleITK provides various functions to retrieve these properties:
- **`GetImageFromArray(arr)`**: Returns a sitk image from numpy array.
- **`GetSize()`**: Retrieves the voxel count in each dimension.
- **`GetSpacing()`**: Returns the physical spacing between voxels.
- **`GetOrigin()`**: Specifies the physical location of the first voxel.
- **`GetDirection()`**: Defines the image orientation in world coordinates.
- **`Resample()`**: Adjusts image resolution and spacing while maintaining alignment.

<br>

```python
import SimpleITK as sitk

# Load the NIfTI image
image = sitk.ReadImage("toy-CTA.nii.gz")

# Get the image size (number of voxels in each dimension)
size = image.GetSize()  # Output: (Width, Height, Depth)

# Get the voxel spacing (physical size of each voxel in mm)
spacing = image.GetSpacing()  # Output: (SpacingX, SpacingY, SpacingZ)

# Get the image origin (physical coordinates of the first voxel)
origin = image.GetOrigin()

# Get the image direction (orientation of the image in physical space)
direction = image.GetDirection()

# Get the pixel type (data type of the image)
pixel_type = image.GetPixelID()

# Print information
print(f"Size: {size}")
print(f"Spacing: {spacing}")
print(f"Origin: {origin}")
print(f"Direction: {direction}")
print(f"Pixel Type: {pixel_type}")
```

Let's work in the case we want to adjust one dimension of the image (e.g., the Z-dimension, number of slices) while maintaining the field of view. To do this, we compute the new spacing along the dimension to resample:

```python
# Define the target number of slices (e.g., 320 slices)
target_z_size = 320

# Compute new Z-spacing to maintain the same field of view
new_z_spacing = (spacing[2] * size[2]) / target_z_size
new_spacing = (spacing[0], spacing[1], new_z_spacing)

print(f"New Spacing: {new_spacing}")
```

To resample a CTA image, we use `sitk.Resample()`, ensuring that the other dimensions remain unchanged while adjusting the desired axis. Different interpolation methods exist:


| Interpolation Method         | Description                                        | Best Used For                        |
|-----------------------------|----------------------------------------------------|--------------------------------------|
| `sitk.sitkNearestNeighbor`  | Keeps values strictly 0 or 1                      | Binary masks or annotations         |
| `sitk.sitkLinear`           | Fast, but may introduce slight blurring           | General-purpose resampling          |
| `sitk.sitkBSpline`          | Smooth interpolation, preserves details           | High-quality medical images         |
| `sitk.sitkGaussian`         | Uses Gaussian kernel for smoothing                | Reducing noise, preserving structure |
| `sitk.sitkHammingWindowedSinc` | Sharp edges, minimizes ringing artifacts       | High-quality, artifact-free resampling |
| `sitk.sitkCosineWindowedSinc`  | Cosine-weighted sinc function for smoothness   | Preserving image intensity balance  |
| `sitk.sitkWelchWindowedSinc`   | Welch window function, balances sharpness & smoothness | MRI or CT images with minimal distortion |
| `sitk.sitkLanczosWindowedSinc` | High-precision sinc interpolation              | When fine details must be retained  |



```python
def resample_cta_pipeline(image: sitk.Image, new_z_size=512):
    """
    Resamples a CTA scan to have a uniform number of slices along Z while keeping X-Y unchanged.
    
    Parameters:
    - image (sitk.Image): Input CTA image.
    - new_z_size (int): Target number of slices.
    
    Returns:
    - resampled_img (sitk.Image): Resampled image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    # Compute new Z spacing
    new_z_spacing = (original_spacing[2] * original_size[2]) / new_z_size
    new_spacing = (original_spacing[0], original_spacing[1], new_z_spacing)
    
    # Define new size, keeping X and Y the same
    new_size = [original_size[0], original_size[1], new_z_size]
    
    # Resample the image
    resampled_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),  # Identity transform to prevent shifting
        sitk.sitkBSpline,  # Interpolation method for smooth results
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,  # Background fill value
        image.GetPixelID()
    )

    # To ensure the image does not shift during resampling, always preserve the origin and direction
    resampled_image.SetOrigin(image.GetOrigin())
    resampled_image.SetSpacing(new_spacing)
    resampled_image.SetDirection(image.GetDirection())
    
    return resampled_img

# Example usage
image = sitk.ReadImage("toy-CTA.nii.gz")
resampled_image = resample_cta(image)
sitk.WriteImage(resampled_image, "toy-CTA-resampled.nii.gz")
```

<br>
<br>
<br>

<h2>🔩 Registration pipeline</h2>

The registration pipeline aligns CTA scans to the MNI template.

You can resample the template to your desired shape with:


```python
import SimpleITK as sitk


def resample_template(input_template_path, output_template_path, target_size=(512, 512, 182), target_spacing=None):
    """
    Resamples a given template to the desired dimensions.

    Parameters:
    - input_template_path (str): Path to the input NIfTI template.
    - output_template_path (str): Path to save the resampled template.
    - target_size (tuple): Desired output dimensions (X, Y, Z). Default is (512, 512, 182).
    - target_spacing (tuple): Desired voxel spacing. If None, it is computed automatically.

    Output:
    - Saves the resampled template as a new NIfTI file.
    """

    # Load input template
    template_image = sitk.ReadImage(input_template_path)

    # Get original template size and spacing
    original_size = template_image.GetSize()  # (X, Y, Z)
    original_spacing = template_image.GetSpacing()  # (spacingX, spacingY, spacingZ)

    print(f"Original size: {original_size}, Original spacing: {original_spacing}")

    # Compute new spacing to preserve the same field of view (if not provided)
    if target_spacing is None:
        target_spacing = [
            original_spacing[0] * (original_size[0] / target_size[0]),  # New spacing X
            original_spacing[1] * (original_size[1] / target_size[1]),  # New spacing Y
            original_spacing[2] * (original_size[2] / target_size[2])  # New spacing Z
        ]

    print(f"Target size: {target_size}, Target spacing: {target_spacing}")

    # Define resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetSize(target_size)
    resample.SetOutputSpacing(target_spacing)
    resample.SetOutputDirection(template_image.GetDirection())  # Keep orientation
    resample.SetOutputOrigin(template_image.GetOrigin())  # Keep origin
    resample.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation for smooth resampling
    resample.SetDefaultPixelValue(0)  # Fill new space with 0

    # Apply resampling
    resampled_template = resample.Execute(template_image)

    # Save the resampled template
    sitk.WriteImage(resampled_template, output_template_path)
    print(f"Resampled template saved to: {output_template_path}")


# Example Usage:
input_template_path = "coreTemplate-MNI152lin_T1_1mm.nii.gz"  # Your original template
desired_size = (512, 512, 416)  # Desired volume shape
output_template_path = f"coreTemplate-MNI152lin_T1_1mm_{desired_size[0]}_{desired_size[1]}_{desired_size[2]}_.nii.gz"  # Where to save the resampled template

# Call the resampling function
resample_template(input_template_path, output_template_path, desired_size)
```

The pipeline works as follows:

- **1️. Preprocessing the CTA Image**: Load the CTA scan and apply Gaussian filtering preprocessing and clip intensity values (0-95 intensity range)
- **2️. Loading the Images & Masks**: Load the CTA, CTA mask, MNI template, and MNI template mask. Ensure the CTA and MNI template share the same pixel type. Clip CTA intensities between 0 and 100.
- **3️. Performing the Registration**: Use Mattes Mutual Information as the metric. Initialize the transformation based on image moments. Optimize using Gradient Descent. Execute registration.
- **4️. Applying the Transformation**: Resample the CTA to align with the MNI template. Save the registered CTA and transformation matrix (`.tfm`).

<br>

SimpleITK provides various functions to perform registration:

- **`Clamp(image, lowerBound, upperBound, outputPixelType)`**: Clips intensity values within a specified range (used to constrain CTA intensities).
- **`ImageRegistrationMethod()`**: Creates an instance of the registration method class.
- **`CenteredTransformInitializer(fixed, moving, transform, method)`**: Initializes a transformation based on the center of mass of the images.
Uses `sitk.Euler3DTransform()` for rigid registration (rotation + translation). Uses `sitk.CenteredTransformInitializerFilter.MOMENTS` to improve initial alignment.
- **`SetMetricAsMattesMutualInformation(numberOfHistogramBins)`**: Sets the similarity metric for registration (good for multimodal images).
- **`SetMetricSamplingStrategy(strategy)`**: Chooses how image samples are selected during registration.
- **`SetMetricSamplingPercentage(percentage)`**: Defines the fraction of pixels used for metric computation (speeds up registration).
- **`SetMetricMovingMask(mask)`**: Sets a mask for the moving image to focus registration on relevant regions.
- **`SetMetricFixedMask(mask)`**: Sets a mask for the fixed image to ignore irrelevant areas.
- **`SetInterpolator(interpolation_method)`**: Defines the interpolation method (e.g., sitk.sitkLinear).
- **`SetOptimizerAsGradientDescent(learningRate, numberOfIterations, estimateLearningRate)`**: Optimizes the transformation using a gradient descent approach. numberOfIterations=500: Controls how many optimization steps are performed.
- **`Execute(fixed_image, moving_image)`**: Runs the registration process and returns the final transformation.

<br>

```python
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import os

def register_cta_pipeline(
    cta_image_nib, brain_mask_nib, template_nib, template_mask_nib, output_dir, pat="001"
):
    """
    Pipeline for registering CTA images to an MNI template while preserving 512x512 dimensions.
    
    Parameters:
    - cta_image_nib (nibabel.Nifti1Image): The input CTA image loaded with nibabel.
    - brain_mask_nib (nibabel.Nifti1Image): The corresponding brain mask loaded with nibabel.
    - template_nib (nibabel.Nifti1Image): The MNI template loaded with nibabel.
    - template_mask_nib (nibabel.Nifti1Image): The MNI brain mask loaded with nibabel.
    - output_dir (str): Directory to save the registered images and transformations.
    - filename (str): Patient ID or identifier for file naming.
    
    Outputs:
    - Saves registered CTA images and the transformation matrix in the output directory.
    """

    # Define output file paths
    transformation_path = os.path.join(output_dir, f"{filename}_transformation_cta_clipped_to_mni_template.tfm")
    registered_cta_path = os.path.join(output_dir, f"{filename}_cta_registered_to_mni.nii.gz")
    registered_filtered_cta_path = os.path.join(output_dir, f"{filename}_cta_clipped_registered_to_mni.nii.gz")

    print(f"Preprocessing CTA for subject {filename}...")

    # Convert nibabel images to NumPy arrays
    moving_image_arr = cta_image_nib.get_fdata().astype(np.float32)

    # Apply preprocessing: remove negative values, smooth, and intensity clip
    moving_image_arr[moving_image_arr < 0] = 0  # Remove negative values
    moving_image_arr = gaussian_filter(moving_image_arr, sigma=2.0)  # First Gaussian filter
    moving_image_arr[moving_image_arr > 95] = 0  # Clip high-intensity values
    moving_image_arr = gaussian_filter(moving_image_arr, sigma=3.0)  # Second Gaussian filter

    # Save the preprocessed image
    preprocessed_cta_nib = nib.Nifti1Image(moving_image_arr, cta_image_nib.affine)
    preprocessed_cta_path = os.path.join(output_dir, f"{filename}_cta_0_95_gaussian_filtered.nii.gz")
    nib.save(preprocessed_cta_nib, preprocessed_cta_path)

    print(f"Converting images to SimpleITK format for registration...")

    # Convert nibabel images to SimpleITK images
    inputCTA = sitk.GetImageFromArray(moving_image_arr)
    templateCTA = sitk.GetImageFromArray(template_nib.get_fdata().astype(np.float32))
    templateMask = sitk.GetImageFromArray(template_mask_nib.get_fdata().astype(np.float32))
    maskCTA = sitk.GetImageFromArray(brain_mask_nib.get_fdata().astype(np.float32))

    # Ensure CTA image has the same pixel type as the template
    inputCTA = sitk.Cast(inputCTA, templateCTA.GetPixelID())

    # Clip intensity values to [0,100]
    inputCTA = sitk.Clamp(inputCTA, lowerBound=0, upperBound=100, outputPixelType=inputCTA.GetPixelID())

    print(f"Performing registration for subject {filename}...")

    registration_method = sitk.ImageRegistrationMethod()

    # Initialize transformation using image moments (for better initial alignment)
    initial_transform = sitk.CenteredTransformInitializer(
        templateMask, maskCTA, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Set the metric as Mutual Information (good for different modalities like CTA and MRI)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricMovingMask(maskCTA)
    registration_method.SetMetricFixedMask(templateMask)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.5)

    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=500, estimateLearningRate=registration_method.Once
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Execute the registration
    final_transform = registration_method.Execute(templateCTA, inputCTA)

    print(f"Applying transformation and saving registered images...")

    # Resample the preprocessed CTA to match the template space
    outputCTA_filtered = sitk.Resample(inputCTA, templateCTA, final_transform, sitk.sitkLinear, 0.0)
    
    # Resample the original CTA to match the template space
    original_cta_sitk = sitk.GetImageFromArray(cta_image_nib.get_fdata().astype(np.float32))
    outputCTA = sitk.Resample(original_cta_sitk, templateCTA, final_transform, sitk.sitkLinear, 0.0)

    # Convert SimpleITK images back to nibabel for saving
    outputCTA_filtered_nib = nib.Nifti1Image(sitk.GetArrayFromImage(outputCTA_filtered), template_nib.affine)
    outputCTA_nib = nib.Nifti1Image(sitk.GetArrayFromImage(outputCTA), template_nib.affine)

    # Save the registered images
    nib.save(outputCTA_filtered_nib, registered_filtered_cta_path)
    nib.save(outputCTA_nib, registered_cta_path)

    # Save the transformation matrix
    sitk.WriteTransform(final_transform, transformation_path)

    print(f"Registration complete. Outputs saved in {output_dir}")


# Define file paths
cta_image_path = "toy-CTA.nii.gz"
brain_mask_path = "toy-CTA-mask.nii.gz"
template_path = "coreTemplate-MNI152lin_T1_1mm.nii.gz"
template_mask_path = "coreTemplate-MNI152lin_T1_1mm-mask.nii.gz"
output_dir = "registered_output"

# Load images with nibabel
cta_image_nib = nib.load(cta_image_path)
brain_mask_nib = nib.load(brain_mask_path)
template_nib = nib.load(template_path)
template_mask_nib = nib.load(template_mask_path)

# Import the function
register_cta_pipeline(
    cta_image_nib=cta_image_nib, 
    brain_mask_nib=brain_mask_nib, 
    template_nib=template_nib, 
    template_mask_nib=template_mask_nib, 
    output_dir=output_dir,
    filename="toy-CTA"
)

```

<br>

The transformation can be applied to another CTA or annotation file:

```python
import SimpleITK as sitk

def apply_transformation(input_image_path, reference_cta_path, transform_path, output_path, is_binary_mask=False):
    """
    Applies a saved transformation to an image (e.g., annotation mask, vessel mask) to align it with the registered CTA. Saves the transformed image in the specified output path.

    Parameters:
    - input_image_path (str): Path to the image that needs transformation (e.g., annotation or vessel mask).
    - reference_cta_path (str): Path to the registered CTA that serves as the spatial reference.
    - transform_path (str): Path to the transformation file (.tfm) obtained from CTA registration.
    - output_path (str): Path where the transformed image will be saved.
    - is_binary_mask (bool): If True, uses nearest-neighbor interpolation (preserves binary masks). Default is False.
    """

    # Load the input image to be transformed
    input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)

    # Load the registered CTA as a reference for spatial alignment
    reference_cta = sitk.ReadImage(reference_cta_path, sitk.sitkFloat32)

    # Load the transformation matrix (.tfm) obtained from CTA registration
    transformation = sitk.ReadTransform(transform_path)

    # Choose interpolation method based on the type of input image
    # - Nearest Neighbor (sitk.sitkNearestNeighbor) for binary masks (ensures 0s and 1s remain unchanged)
    # - Linear Interpolation (sitk.sitkLinear) for soft images (ensures smooth transformation)
    interpolator = sitk.sitkNearestNeighbor if is_binary_mask else sitk.sitkLinear

    # Apply the transformation using `Resample`
    transformed_image = sitk.Resample(
        input_image,         # Image to be transformed
        reference_cta,       # Registered CTA (defines the spatial properties)
        transformation,       # Precomputed transformation matrix
        interpolator,        # Selected interpolation method
        0.0,                 # Default pixel value for new regions
        input_image.GetPixelID()  # Preserve original pixel type
    )

    # Save the transformed image
    sitk.WriteImage(transformed_image, output_path)
    print(f"Transformed image saved to: {output_path}")


# Define file paths
annotation_path = "toy-annotation.nii.gz"  # Original annotation mask
registered_cta_path = "toy-CTA_cta_registered_to_mni.nii.gz"  # Already registered CTA
transform_path = "./toy-CTA_transformation_cta_clipped_to_mni_template.tfm"  # Transformation from CTA registration
output_dir = "./registered_output/"

# Apply transformation to annotation (binary mask)
apply_transformation(
    input_image_path=annotation_path,
    reference_cta_path=registered_cta_path,
    transform_path=transform_path,
    output_path=os.path.join(output_dir, "toy-annotation_registered.nii.gz"),
    is_binary_mask=True  # Because it's a segmentation mask
)

```

<br>
<br>
<br>

[Back to Index 🗂️](./README.md)
