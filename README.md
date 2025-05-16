FAISS-GPU Build Helper Scripts

This repository contains helper Python scripts designed to assist in building the GPU-enabled version of the FAISS library from source on Ubuntu systems. Building FAISS with GPU support requires specific dependencies and configuration steps, which these scripts aim to simplify and automate.

Disclaimer: These scripts are provided as helpers and are based on a specific environment (Ubuntu 24.04, CUDA 12.4, cuDNN 9.10.0). While they include checks and workarounds for common issues, build processes can be sensitive to system configurations. Always review the script output and consult the official FAISS documentation if you encounter problems. USE AT YOUR OWN RISK.

Contents

    install_faiss_dependencies.py: A script to check for and help install mandatory dependencies (CUDA Toolkit, cuDNN, CMake, g++, SWIG, BLAS development libraries) on Ubuntu. Includes a dry-run mode.

    build_faiss_gpu.py: A script to clone the FAISS repository, configure the build using CMake, build the library, and install the Python bindings. Includes prerequisite checks and suggestions.

    faiss_gpu_test.py: A simple Python script to test if the installed faiss-gpu package is working correctly and can utilize your GPU(s).

Prerequisites

Before running the build scripts, ensure you have the following installed and configured on your Ubuntu system:

    NVIDIA Driver: A compatible NVIDIA graphics driver for your GPU. Ensure nvidia-smi is working in your terminal and reports a compatible CUDA version (e.g., 12.4).

    CUDA Toolkit: NVIDIA CUDA Toolkit. These scripts target CUDA Toolkit 12.4. Ensure nvcc is in your system's PATH.

    cuDNN: NVIDIA cuDNN library. These scripts target cuDNN 9.10.0 (compatible with CUDA 12.4). cuDNN files should be accessible in your CUDA Toolkit directories or standard system paths.

    CMake: Version 3.18 or higher.

    C++17 Compiler: A C++ compiler that supports C++17 (like g++ or clang++).

    SWIG: Simplified Wrapper and Interface Generator. Required for building Python bindings.

    BLAS Library: A Basic Linear Algebra Subprograms implementation (e.g., OpenBLAS or Intel MKL). This is typically installed as a system package (e.g., libopenblas-dev).


The install_faiss_dependencies.py script is designed to help you check for and install these dependencies on Ubuntu via apt.

Setup

    Clone the repository:

    git clone https://github.com/piyushkumarjiit/FAISS-GPU-Build.git
    cd FAISS-GPU-Build

    Create a Python Virtual Environment (Recommended):
    It's highly recommended to build and install FAISS within a dedicated Python virtual environment (like venv or conda) to avoid conflicts with your system Python packages.

    Using venv:

    python3 -m venv venv
    source venv/bin/activate

Usage

1. Install Dependencies (using install_faiss_dependencies.py)

Run the dependency installation script. It will check for existing dependencies and attempt to install missing ones using apt on Ubuntu.

Important: This script requires sudo privileges to install system packages.

sudo python install_faiss_dependencies.py

Dry Run: You can perform a dry run to see which dependencies the script would attempt to install without actually installing them:

sudo python install_faiss_dependencies.py --dry-run

Review the output to ensure all mandatory dependencies are reported as found or successfully installed.

2. Build FAISS-GPU (using build_faiss_gpu.py)

Once the dependencies are met, run the build script from within your activated Python environment.

python build_faiss_gpu.py

This script will:

    Clone the FAISS repository into the faiss subdirectory (if it doesn't exist).

    Create a build directory inside faiss.

    Configure the build using CMake, enabling GPU support. It attempts to automatically find CUDA, cuDNN, BLAS, and gflags.

    Build the FAISS library using make.

    Install the Python bindings into your active Python environment using pip.

Dry Run: You can perform a dry run for the build script to see the commands it would execute without actually building or installing:

python build_faiss_gpu.py --dry-run

3. Test FAISS-GPU Installation (using faiss_gpu_test.py)

After the build script completes successfully, run the test script from within your activated Python environment to verify the installation.

python faiss_gpu_test.py

This script will attempt to import faiss, check for detected GPUs, and perform a basic index creation, transfer to GPU, and search operation. Look for messages indicating successful import, GPU detection, and test completion.
Troubleshooting

Here are solutions to some common issues you might encounter:

    nvidia-smi command not found or working:

        Ensure your NVIDIA graphics driver is correctly installed. On Ubuntu, you can often use the "Additional Drivers" tool or sudo apt install nvidia-driver-XXX (replace XXX with a recommended version).

        Reboot your system after installing drivers.

    nvcc command not found:

        This means the CUDA Toolkit is not installed or its bin directory is not in your system's PATH.

        Install the CUDA Toolkit from the NVIDIA CUDA Downloads page.

        After installation, you might need to manually add the CUDA bin directory (e.g., /usr/local/cuda/bin) to your PATH environment variable. Add export PATH=/usr/local/cuda/bin:$PATH to your ~/.bashrc or ~/.profile and source the file or open a new terminal.

    CUDA keyring download errors (like 404):

        Ensure your internet connection is stable.

        Verify the Ubuntu version detected by the script matches your system (lsb_release -r). The script attempts to use the correct URL based on this.

    libtinfo5 dependency issues on Ubuntu 24.04 during CUDA installation:

        The install_faiss_dependencies.py script includes an automated workaround for this by temporarily adding the Ubuntu 23.04 (Lunar) repository.

        If the automated workaround fails, you might need to manually install libtinfo5 from the Lunar repository or download the .deb package directly (search for libtinfo5 on packages.ubuntu.com for the Lunar distribution).

    CMake fails to find libraries (cuDNN, BLAS, gflags):

        Ensure the libraries are actually installed on your system.

        If installed in non-standard locations, you might need to set environment variables before running the build_faiss_gpu.py script to help CMake find them. Common variables include CUDA_TOOLKIT_ROOT_DIR, CUDNN_ROOT, BLAS_DIR, OpenBLAS_DIR, gflags_DIR, or CMAKE_PREFIX_PATH. Consult the FAISS build documentation and the CMake output for specific details.

        Example: export CMAKE_PREFIX_PATH=/usr/local or export BLAS_DIR=/opt/openblas.

    SWIG-related errors or AttributeError when importing faiss in Python:

        Ensure you have SWIG installed and it's working correctly (swig -version).

        These errors can sometimes indicate compatibility issues between the specific versions of FAISS, SWIG, and your C++ compiler.

        Review the full build output (DEBUG level in console) for SWIG-related warnings or errors.

        Consult the official FAISS documentation, GitHub issues, or community forums for known compatibility issues and potential workarounds or recommended version combinations.

    Build fails during make:

        Review the build output carefully for specific compilation or linking errors.

        Common causes include missing headers, libraries not being found, or compiler issues. Ensure all dependencies are correctly installed and discoverable.

License

These scripts are licensed under the GNU General Public License (GPL). See the GNU website for the full text of the license.

Contributing

If you'd like to contribute to these scripts, please feel free to open issues or pull requests.
