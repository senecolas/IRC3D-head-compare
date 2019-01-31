# INSTALLATION

This file explains how to correctly install the dependencies of the application, including instructions for using the GPU or CPU

## Start with basic installations

First you have to activate the administrator mode (Linux and Mac) and update the repositories.
```
sudo -s
apt update
```

### Python 3.7 *(Linux and Mac)*
The most indispensable.
```
apt-get install python3.7
```

### pip *(Linux and Mac)*
The PyPA recommended tool for installing Python packages.
```
apt-get install mesa-utils python3-pip build-essential
```

### Python and pip *for Windows*
To install Python3.7 Windows you have to download it from the official website (for windows in particular) : [https://www.python.org/downloads/](https://www.python.org/downloads/windows/)
Then, to configure it and use pip, refer to these instructions [https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation)


## Python Librairies

### Numpy
Numpy is a lib for manipulating large sets of numbers
```
pip install numpy
```

### OpenCV
OpenCV (for Open Computer Vision) is a free graphics library
```
apt-get install python3-opencv
```
or
```
pip install opencv-python
```

And if you want to have some video codecs
```
pip install ffmpeg ffmpeg-python
pip install gstreamer-player
```

### Matplotlib
```
apt-get install python3-matplotlib
```
or
```
pip install matplotlib
```

### Skimage
```
pip install scikit-image
```

### Pillow
```
pip install pillow
```

### Skimage
```
pip install scikit-image
```

### Pandas
```
pip install pandas
```

### Scipy
```
pip install scipy
```

### PyQt
```
pip install pyqt5 pyqt5-tools
```
Or, if you all tools for Linux and Mac 
```
sudo apt-get install python3-pyqt5 qtcreator qttranslations5-l10n qt5-doc-html qtbase5-examples pyqt5-dev-tools
```

### OpenGL 
```
pip install PyOpenGL PyOpenGL_accelerate
```

### Pyglet 
```
pip install pyglet 
```

### PyWavefront
```
pip install PyWavefront
```


## *GPU* installation (you must have an NVIDIA graphics card)

### CUDA

#### CUDA 9 for linux
```
apt-get install build-essential dkms
apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-cuda-toolkit gcc-6
nvcc --version
```

#### CUDA for all platforms
Guided download link: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### cuDNN
Guided download link: [https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

### PyTorch with CUDA
Guided download link (use pip): [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Dlib with cuda
To use DLIB with the GPU (recommended for much faster analysis), you must install it from the [official git](https://github.com/davisking/dlib) with `DDLIB_USE_CUDA` option
```
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
```

For windows, you have to install and use Visual Studio in 64 bit. Exemple for `Visual Studio 15 2017` (put your Visual Studio version instead, with “Win64”)
```
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
```

Check during the installation that dlib can locate and use CUDA and cuDNN. Otherwise read the error messages to fix the problem and recompile.


## *CPU* installation (if you have no NVIDIA graphics card)

### PyTorch without CUDA
Guided download link (use pip and CUDA=None): [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Dlib without CUDA
Just install with pip
```
pip install dlib
```
Or you can install it with the [official git](https://github.com/davisking/dlib) as before but disable the `DDLIB_USE_CUDA` option.
```
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA
```