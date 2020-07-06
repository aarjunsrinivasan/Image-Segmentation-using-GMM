# Image Segementation Using Gaussian Mixture Modelling 

This project focuses on color segmentation using concepts of Gaussian Mixture Modelling and Expectation Maximization (EM). We are provided with an underwater video sequence consisting of 3 buoys of different colors; yellow, orange, green. Since the buoys have been captured in an underwater scenario, change in lighting intensities and noise make it difficult to employ the use of conventional segmentation techniques involving color thresholding. Hence, we choose to develop a Gaussian Mixture Model that can learn the
color distributions of the buoys present in the video and use them for segmentation. The remaining sections of the report discuss about the implementations of the above mentioned concepts.

## Pre-requisites

- Python Version: 3.x

## Packages Required:

- Numpy

- OpenCV

- Matplotlib

## How to run

- Run the python files in the current directory which contains all the codes.
- To estimate the GMM's for individual colored buoys, run the jupyter notebooks greenGMM.ipynb, yellowGMM.ipynb, and orangeGMM.ipynb
- To test the modelled GMM's, run the scripts greenvideo.py, yellowvideo.py, and orangevideo.py


