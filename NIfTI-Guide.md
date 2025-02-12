[Back to Index 🗂️](./README.md)

# 🧠 Guide to NIfTI Files (.nii.gz) 

## ℹ️ Introduction

NIfTI (Neuroimaging Informatics Technology Initiative) is a file format commonly used to store medical image data, particularly MRI scans, CTA and more. This guide will explain what .nii.gz files are, why they are used, how to handle them, and common operations you may need to perform.

## 📁 What is a .nii.gz File?

The .nii.gz file is a compressed version of the NIfTI format. NIfTI files store 3D data `(X, Y, Z)` or 4D data for time series `(X, Y, Z, T)`, commonly from medical imaging techniques. NIfTI files contains different metadata:

**1. Header**: Contains metadata about the image.<br>
- *Voxel Information*: the physical size (typically in millimeters) of a 3D pixel, and the space betwen them.
- *Dimensions*:number of voxels in each direction.
- *Datatype* (uint8, int16, float32, etc.)
- *Affine Matrix*:
    - First 3x3 submatrix: Rotation and scaling.
    - Last column: Translation vector maps voxel coordinates to world coordinates in millimeters

<br>
<img src="./images/NIfTI-Guide/slice-info.png" width=300px>
<br>
<img src="./images/NIfTI-Guide/volume-info.png" width=300px>
<br>
<br>

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

## 🧭 Reference system

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

## 👩‍⚕️ Common operations on NIfTI files

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

### 🍕 Extracting slices

```python
# Axial slice (Z-Axis)
axial_slice = data[:, :, 50]  # Get the 50th slice along Z

# Coronal slice (Y-Axis)
coronal_slice = data[:, 50, :]  # Get the 50th slice along Y

# Sagittal slice (X-Axis)
sagittal_slice = data[50, :, :]  # Get the 50th slice along X
```

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

### ✂️ Cut the volume

```python
# Define the slicing indices (z_min:z_max, y_min:y_max, x_min:x_max)
z_min, z_max = 30, 70
y_min, y_max = 40, 120
x_min, x_max = 50, 150

# Cut the volume
cut_data = data[z_min:z_max, y_min:y_max, x_min:x_max]
```

### 📊 Clip Volume Data Between Two Values

```python
# Clip values in the volume data between 0 and 200
clipped_data = np.clip(data, 0, 200)

# Verify the range
print(f"Data min: {clipped_data.min()}, Data max: {clipped_data.max()}")
```

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

### 🏁 Resampling Guide

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

When resampling an image, we want to adjust one dimension of the image (e.g., the Z-dimension, number of slices) while maintaining the field of view. To do this, we compute the new spacing along the dimension to resample:

```python
# Define the target number of slices (e.g., 320 slices)
target_z_size = 320

# Compute new Z-spacing to maintain the same field of view
new_z_spacing = (spacing[2] * size[2]) / target_z_size
new_spacing = (spacing[0], spacing[1], new_z_spacing)

print(f"New Spacing: {new_spacing}")
```

To resample a CTA image, we use `sitk.Resample()`, ensuring that the other dimensions remain unchanged while adjusting the desired axis. Different interpolation methods exist:


| Interpolation Method       | Description | Best Used For |
|---------------------------|-------------|--------------|
| `sitk.sitkNearestNeighbor` | Keeps values strictly 0 or 1 | Binary masks or annotations |
| `sitk.sitkLinear`          | Fast, but may introduce slight blurring | General-purpose resampling |
| `sitk.sitkBSpline`         | Smooth interpolation, preserves details | High-quality medical images |



```python
def resample_cta(image: sitk.Image, new_z_size=512):
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

# 🔭 Tools for imaging

Let's explore some useful tools when working with images and also with 3D images.

<br>
<br>

# 🔬 ImageJ

ImageJ is a powerful, open-source image processing tool widely used in scientific research, particularly in medical imaging, microscopy, and bioinformatics. It supports a broad range of image formats, including NIfTI, DICOM, TIFF, and JPEG, and provides extensive analysis tools.

## ⚙️ Installation
ImageJ is available for Windows, macOS, and Linux on the [Official Website](https://imagej.nih.gov/ij/).

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

# 🩻 ITK-SNAP

ITK-SNAP is a powerful open-source tool for visualizing, annotating, and segmenting 3D medical images. It is widely used for neuroimaging and other medical image analysis tasks.

## ⚙️ Installation

ITK-SNAP is available for Windows, macOS, and Linux. Download the latest version from the [Official Website](https://www.itksnap.org/)

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

# ⚒️ FSL

FSL (FMRIB Software Library) is a powerful set of tools used for analyzing and processing medical imaging data, particularly neuroimaging data. It includes a variety of programs for image registration, segmentation, and statistical analysis.

## ⚙️ Installation

FSL can be installed on macOS, Linux, and Windows (via WSL) using the [Official Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

### 📌 Basic Installation Steps (Linux/macOS)
- Download the installer:
    ```bash
    wget -O fslinstaller.py https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
    python3 fslinstaller.py
    ```
- Follow the on-screen instructions to complete the installation.
- Once installed, configure the environment by adding this to your `~/.bashrc` or `~/.zshrc`:
  ```bash
  export FSLDIR=/usr/local/fsl
  source $FSLDIR/etc/fslconf/fsl.sh
  PATH=${FSLDIR}/bin:${PATH}
  export PATH
  ```
- Apply the changes:
  ```bash
  source ~/.bashrc  # or source ~/.zshrc
  ```

## 🎛️ Launching FSL Tools

- **Start FSL Main GUI**:
  ```bash
  fsl &
  ```

    <img src="./images/NIfTI-Guide/FSL.png" width=400px>

- **Launch FSLView for Image Visualization**:
  ```bash
  fsleyes &
  ```

    <img src="./images/NIfTI-Guide/FSLeyes.png" width=400px>

    Once loaded a `nii.gz` file (just drop the file on the FSLeyes window or open it via the `File` menu), FSLeyes will show it from all the three different views:

    <img src="./images/NIfTI-Guide/FSLeyes-example.png" width=400px>

## 🎮 FSL Commands

### ℹ️ Obtain metadata info
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

### **Voxel Size (Resolution)**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `pixdim1, pixdim2` | In-plane resolution (X, Y voxel size) | `0.4 - 0.6 mm` | If **too small** (`< 0.3 mm`), BET may fail. If **too large** (`> 1 mm`), resolution is too low. |
| `pixdim3` | Slice thickness (Z resolution) | `0.5 - 1.0 mm` | If **too small** (`< 0.3 mm`), may cause issues. If **too large** (`> 5 mm`), resample with `flirt`. |
| `pixdim4` | Time resolution | `0.0` | If nonzero, likely an incorrect 4D image. |

### **Intensity Calibration & Scaling**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `cal_max, cal_min` | Calibration max/min intensity | `0.0` or actual HU range | If both are `0.0`, BET might fail. Fix with `fslmaths -add 0`. |
| `scl_slope` | Scaling factor applied to intensities | `0.003052` (default) | If `0.0`, values may be unreadable. If `NaN`, file is corrupt. |
| `scl_inter` | Intensity intercept value | `100.0` or `0.0` | If too high, image contrast may be incorrect. |
| `datatype` | Storage format for intensity values | `4 (INT16)` | If `16 (FLOAT32)`, convert with `fslmaths -dt int16`. |

### **Image Orientation & Transformation Matrices**
| **Parameter** | **Description** | **Correct CT Value** | **Potential Anomalies** |
|--------------|----------------|----------------------|--------------------------|
| `qform_code, sform_code` | Q-form and S-form matrix type (scanner coordinates) | `1 (Scanner Anat)` | If `0`, spatial information is missing. Fix with `fslorient -setqformcode 1`. |
| `qto_xyz` | Q-form transformation matrix (maps voxel to real-world space) | Nonzero 4x4 matrix | If all `0.0`, image has no spatial transformation. |
| `qform_xorient` | X-axis orientation | `Right-to-Left` | If incorrect (e.g. `Left-to-Right`), may cause misalignment. |
| `qform_yorient` | Y-axis orientation | `Posterior-to-Anterior` | If incorrect, image may be flipped. |
| `qform_zorient` | Z-axis orientation | `Inferior-to-Superior` | If incorrect, slices may be in the wrong order. |



### 🧠 Skull Stripping using `BET` (Brain Extraction Tool)
BET removes non-brain tissue from the 3D images.
```bash
bet input.nii.gz output_brain.nii.gz -m -f 0.5
```
- `-m`: Outputs a binary brain mask.
- `-f`: Fractional intensity threshold (default is `0.5`). Adjust for better extraction. Lower values keep more brain tissue.

### ⚡ Enhanced BET Skull Stripping with Bias Field Correction
To improve skull stripping by reducing intensity bias and improving robustness, use the following command:
```bash
bet input.nii.gz output_brain.nii.gz -B -f 0.5 -g 0
```
- `-B`: Applies bias field correction for better intensity normalization.
- `-f`: Fractional intensity threshold (default is `0.5`). Adjust for better extraction. Lower values keep more brain tissue.
- `-g 0`: Ensures the center-of-mass is automatically detected without additional adjustments.

### 🎯 Cropping Field of View using `robustfov`
Automatically crops the volume scan to remove unnecessary background.
```bash
robustfov -i input.nii.gz -r output_fov.nii.gz
```
- `-i`: Input image.
- `-r`: Output image focused on the Field of View.

### 🛠️ Image Registration with `FLIRT`
FLIRT (FMRIB's Linear Image Registration Tool) aligns images.
```bash
flirt -in input.nii.gz -ref reference.nii.gz -out output_registered.nii.gz
```
- `-in`: Input image.
- `-ref`: Reference image.
- `-out`: Output registered image.

### 📏 Image Resampling with `fslreorient2std`
Ensures image orientation follows standard conventions.
```bash
fslreorient2std input.nii.gz output_reoriented.nii.gz
```

### 🏗️ Image Thresholding with `fslmaths`
Threshold an image at a given intensity level.
```bash
fslmaths input.nii.gz -thr 100 -bin thresholded_mask.nii.gz
```
- `-thr 100`: Set intensity threshold at 100. All voxels with intensities below 100 are set to 0.
- `-bin`: Binarizes the output (values become 0 or 1).

<br>

## ℹ️ FSL Wiki

For more information, visit the [FSL Wiki](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki).

<br>
<br>

[Back to Index 🗂️](./README.md)
