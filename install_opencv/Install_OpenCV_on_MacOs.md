# Install OpenCV on MacOs



## Required packages

- CMake 3.9 or higher
- Git
- Python 2.7 or later
- Numpy 1.5 or later

## Getting OpenCV Source Code

1. Clone the Git repository:

   ​	`git clone https://github.com/opencv/opencv.git`

2. Configuring. Create temporary directory, named `build_opencv`,

   ​    `mkdir build_opencv`

   ​	`cd build_opencv`

   ​	and run:

   ​    `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=0N <path to the OpenCV source directory>`

3. Build. From build directory execute *make*.

4. Check if OpenCV is installed correctly. Run on terminal:

   `python`

   `import cv2`

   `print cv2.__version__`

   Above command should give output `4.1.2`.

   Instead, if the output is `[numpy.core.multiarray failed to import]`, run `pip install -U numpy` and hopefully it should work now.

