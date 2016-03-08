Edge detection by Canny filter using CUDA
================================================

Edge detection is the name for a set of mathematical methods which aim at identifying points in a digital image at which the image brightness changes sharply or, more formally, has discontinuities. It is one of the fundamental steps in image processing, image analysis, image pattern recognition, and computer vision. 

This project is a CUDA based implementation of a Computational Approach to Edge Detection, by John Canny. Canny presented an accurate, localized method of edge detection. Canny method uses multiple parallelizable matrix and floating point operations, which makes it an algorithm that can potentially have major performance increases if implemented in CUDA and run on an NVIDIA GPU. 

This project demonstrates that using general-purpose GPU computation does in fact result in a faster performing implementation of the canny algorithm.
