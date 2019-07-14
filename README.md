# SiameseChange

Siamese change detection code for pair-wise medical image comparison.

**Requirements**: 
- Python 3.6.7
- PyTorch 1.1.0
- CUDA 10.0
- GPU support

**How to Run**:

pending...

Run via command line (preferably in a docker with above requirements) using main.py file

`python main.py [action] [data_directory] [csv_file] [model_path]`

Actions: prepare, train, eval, cluster

Data Directory: contains all images

CSV File: has train and test images split with their names, locations, and labels

Model Path: where you want to save the model OR where model is already saved + .epoch#
