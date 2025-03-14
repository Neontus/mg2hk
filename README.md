# mg2hk

mg2hk is a lightweight, efficient tool designed to convert and process sji rasters to iris rasters. Built by Juno Kim under Alberto Sainz-Dalda's mentorship at Lockheed Martin 2022-2023.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The mg2hk project aims to bridge the gap between legacy MG file formats and modern HK-based systems. Whether you are integrating legacy data into current workflows or experimenting with new formats, mg2hk provides a straightforward interface with a range of options to customize your conversion process.

Built with clear code and a modular design, this tool ensures that users can quickly get started and integrate it into their projects. The repository includes detailed usage examples, configuration options, and tests to help you understand and extend its functionality.

## Features

- **Seamless Conversion:** Automatically converts MG files to HK format with minimal intervention.
- **Customizable Options:** Offers command-line flags to modify conversion parameters (e.g., file paths, conversion presets, verbose logging).
- **Cross-Platform Compatibility:** Runs on major operating systems including Windows, macOS, and Linux.
- **Modular Design:** Easily extend or integrate into larger workflows with clearly documented modules.
- **Robust Error Handling:** Provides informative error messages and logging to troubleshoot conversion issues.
- **Documentation & Examples:** Comes with comprehensive examples and inline documentation to assist new users.

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

## Troubleshooting

For dependency issues, try clean reinstall + pip install of requirements

Code is based on proprietary LM codebase as of 2022/3, may use deprecated functions

Otherwise, feel free to open issues at this link[https://github.com/Neontus/mg2hk/issues]

## License

This project is licensed under the MIT License. See the LICENSE file for details.
