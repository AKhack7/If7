## Required Libraries

To compile this C++ version, you'll need to install:

> OpenCV - For camera and image processing

> Microsoft Speech SDK - For TTS and speech recognition

> cpp-httplib - For HTTP server (single header file)

> nlohmann/json - For JSON parsing (single header file)


## Installation Steps

Install vcpkg (package manager for C++):

bash
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

## Install required packages:

bash
```
.\vcpkg install opencv
.\vcpkg install curl
```

## Download header-only libraries:


## Build with CMake

bash
```
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```


## Notes