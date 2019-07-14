# SiameseChange

Siamese neural network for change detection in pair-wise longitudinal medical image comparison (a manuscript detailing this approach is submitted), in this case for retinopathy of prematurity retinal photographs. 

THESE SCRIPTS ARE A WORK IN PROGRESS. Images and their annotations feature patient data and are not included here.

**Requirements**: 

- Python 3.6.7
- PyTorch 1.1.0
- CUDA 10.0
- GPU support

**How to Run**:

The working directory should contain the following Python scripts:

- main.py (training and validation script)
- eval.py (testing evaluation script)
- siamese_classes.py (python helper classes)

Run the scripts in an interactive shell, like IPython.

The working directory should also contain a 'data/' subdirectory, which contains:

- processed input images 
- csv files containing annotations for the input images

**Citation**:

If you use this code in your work, you can cite... manuscript in submission.

**Acknowledgments**:

The Center for Clinical Data Science at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware supportm including access to graphics processing units. The basis of the siamese network implementation in PyTorch is also indebted to code shared on GitHub by Harshvardhan Gupta (https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch).

Questions? Contact Matt at mdli@mgh.harvard.edu.



