### This program has been modified to remove data that is not available publically ###

To set up an environment in which SoraniOCR.py will be able to run, make sure you have performed the following tasks:

These instructions have been validated as of 4/29/2020 on Windows 10 with a GTX 1080 GPU, adjust for newer compatible versions of Python and CUDA for Keras and TensorFlow, if necessary

1) Download and install the latest version of Python 3.7.x from https://www.python.org/downloads/

2) Download and install Anaconda 3 from https://www.anaconda.com/products/individual

### If using an Nvida GPU, follow steps 3-5, otherwise skip to step 6 ###

3) Download and install the latest Nvidia drivers for your GPU

4) Download and install CUDA Toolkit 10.1 from https://developer.nvidia.com/cuda-10.1-download-archive-base

5) Download and install cuDNN v7.6.5 for CUDA 10.1 from https://developer.nvidia.com/rdp/cudnn-download (A free Nvida developer account is required)
	Intallation instructions are provided here: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/

6) Launch Anaconda3 and update Anaconda3 by running the following commands:
	conda update conda
	conda update --all

7) Create and activate a new environment using the following commands:
	conda create -n SoraniOCR pip python=3.6
	conda activate SoraniOCR

8a) For Nvidia GPUs run the following command to install TensorFlow for a GPU:
	pip install tensorflow-gpu

8b) For CPU-only TensorFlow run the following command:
	pip intall tensorflow

9) Intall the other required libraries by running the following commands:
	pip install keras
	pip install pillow
	pip install pydot
	pip install graphviz
	conda install graphviz

10) Navigate to the directory where the Anaconda3 environment was created

11) Move the SoraniOCR folder into the environment directory (In this case also named SoraniOCR)

12) Run SoraniOCR by executing the following command in the activated Anaconda3 environment, replacing [Anaconda3 install directory] with the install directory of the Anaconda3 installation and [Name of Anaconda3 environment] with the name of the Anaconda3 environment created in step 7 (In this case, also called SoraniOCR):
	python [Anaconda3 install directory]\Anaconda3\envs\[Name of Anaconda3 environment]\SoraniOCR\SoraniOCR.py