# mg2hk
Key File: s2alib.py

### Workflow from scratch:
1. Obtain clean OBSIDs from https://iris.lmsal.com/search/ that are processed well by s2alib.s2adatacubeassembly() | manual check needed, or use provided clean OBSID list in /sanhome/juno/data/DATAREADME.md
2. Use clean OBSIDs to run s2adatacubeassembly and save full data cubes (31 layers detailed in DATAREADME.md) or use /sanhome/juno/data/dataset/ ** In this step, it is currently normalizing but as discussed, exclude this code by commenting out lines 179-180 in s2alib.py and replacing 'normalized' with 'masked' in line 182.
3. On the full data cubes, filter out which observation are clean and without artifacts/visual errors. Use s2alib.prep_clean_data_cubes() on the corresponding OBSIDs to obtain data cubes ready for input into the neural network.
4. In improveCNN.ipynb, adjust the cell labeled VARIABLES as needed.
5. Run model (Inputs may need to be adjusted from colab and mounting google drive)
6. Compare Metrics depending on loss function and visualize outputs to evaluate model's performance (obsid is printed for ease-of-access in comparing to original data cubes)

### Suggested Fixes:
1. Normalization of arrays from data-cube level --> dataset-wide level
2. Loss function
3. Including more data from /sanhome/juno/data/dataset
