# SiameseChange

Using medical images to evaluate disease severity and change over time is a routine and important task in clinical decision making. Grading systems are often used, but are unreliable as domain experts disagree on disease severity category thresholds. These discrete categories also do not reflect the underlying continuous spectrum of disease severity. To address these issues, we developed a convolutional Siamese neural network approach to evaluate disease severity at single time points and change between longitudinal patient visits on a continuous spectrum. We demonstrate this in two medical imaging domains: retinopathy of prematurity (ROP) in retinal photographs and osteoarthritis in knee radiographs. Key code related to the retinal photographs are shown here; the knee radiograph code is analogous.

This work is published in TO-BE-ANNOUNCED-JOURNAL entitled "Siamese neural networks for continuous disease severity evaluation and change detection in medical imaging" (doi: XXX).

Please refer to the manuscript methodology for details. 

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

If you use this code in your work, please cite: 

PENDING-CITATION

**Acknowledgments**:

The Center for Clinical Data Science (CCDS) at Massachusetts General Hospital and the Brigham and Woman's Hospital provided technical and hardware support including access to graphics processing units. The basis of the siamese neural network implementation in PyTorch is also indebted to code shared on GitHub by Harshvardhan Gupta (https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch).

Questions? Contact us at qtimlab@gmail.com.



