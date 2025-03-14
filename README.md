# mg2hk

mg2hk is a lightweight, efficient tool designed to convert and process sji rasters to iris rasters. Built by Juno Kim under Alberto Sainz-Dalda's mentorship at Lockheed Martin 2022-2023.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

To get started with mg2hk on your local machine, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Neontus/mg2hk.git
   cd mg2hk
   ```

2. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

## Usage

Main file is s2alib.py, containing functions to help process AIA data and conversion into SJI format.

Suggested Workflow:

1. Obtain clean OBSIDs from https://iris.lmsal.com/search/ that are processed well by s2alib.s2adatacubeassembly() | manual check needed, or use provided clean OBSID list in /sanhome/juno/data/DATAREADME.md
2. Use clean OBSIDs to run s2adatacubeassembly and save full data cubes (31 layers detailed in DATAREADME.md) or use /sanhome/juno/data/dataset/ ** In this step, it is currently normalizing but as discussed, exclude this code by commenting out lines 179-180 in s2alib.py and replacing 'normalized' with 'masked' in line 182.
3. On the full data cubes, filter out which observation are clean and without artifacts/visual errors. Use s2alib.prep_clean_data_cubes() on the corresponding OBSIDs to obtain data cubes ready for input into the neural network.
4. In improveCNN.ipynb, adjust the cell labeled VARIABLES as needed.
5. Run model (Inputs may need to be adjusted from colab and mounting google drive)
6. Compare Metrics depending on loss function and visualize outputs to evaluate model's performance (obsid is printed for ease-of-access in comparing to original data cubes)

### iris2aia class
- initialize with time of iris raster, coordinates of iris raster, and the aia time.
```python
i2a = iris2aia(iris_time, iris_x, aia_time, iris_y)
```
- Supplement iris raster x coordinates using add_iris_x
```python
i2a.add_iris_x(new_iris_x)
```
- Main use case is correlating multiple iris slit images to aia raster.

### closest_time(iris_time, aia_times)
- Given an input of lists with times for iris slit images and aia images respectively, outputs the closest time for an aia image corresponding to an iris slit.
```python
closest_aia_img_time = closest_time(iris_slit_time, aia_image_times)
```

### sji2aia class
- Initialize with a SJI image, iris time, and aia time index
```python
s2a = sji2aia(sji_img_var, iris_time, aia_time)
```
- Main use case is for correlating SJIs on a timeline of iris and aia indexes.

### template_match(main, aia_index, sji_img)
- Given an AIA image main, index of specific AIA scan, and slit jaw image, use template matching to find the coordinates of the SJI image relative to main
```python
iris_on_aia_x, iris_on_aia_y = template_match(aia_main, aia_index, sji_img)
```

### s2adatacubeassembly(obsid, dir_to_save)
- Given an OBSID for an AIA image with matching IRIS raster, and a path to save output, saves a data cube containing the aligned iris image, aia image at different filters, and extracted physical variables.
```python
s2adatacubeassembly(OBSID, output_path)
```
- Main usage: Utilizes previously defined functions to align AIA and IRIS, builds a synthetic IRIS Raster for the AIA image channels (1700, 304, 193, 171), then performs IRIS2 inversions on original IRIS image. Saves data cubes to be used for training U-net.

### clean_outliers(outlier_obsids, data_cube_directory, dir_to_save)
- Given OBSIDS for outliers, directory containing corresponding data cubes, and output directory, saves new data cubes using normalize mask.
```python
clean_outliers(outlier_OBSIDS, data_cube_dir, output_dir)
```

### prep_clean(clean_obsids, data_cube_directory, dir_to_save)
- Given a data cube via their OBSID and directory, prepares the 'y'/desired output of the U-net of physical variables and saves to output directory.
```python
prep_clean(cleaned_obsids, data_cube_dir, output_dir)
```

### prep_clean_data_cubes(clean_obsids, data_cube_directory, dir_to_save)
- Given clean OBSIDS, data cube directory, and output path: Prepares data cube of x and y for the U-net, x = AIA filters & y = extracted physical variables.
```python
prep_clean_data_cubes(OBSIDS, data_dir, output_dir)
```

### checkpoint(message, variable)
- Used for troubleshooting, printing a message and information of a variable
```python
checkpoint("Troubleshooting variable x", x)
```

## Troubleshooting

For dependency issues, try clean reinstall + pip install of requirements

Code is based on proprietary LM codebase as of 2022/3, may use deprecated functions

Otherwise, feel free to open issues at this link[https://github.com/Neontus/mg2hk/issues]

## License

This project is licensed under the MIT License. See the LICENSE file for details.
