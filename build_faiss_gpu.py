# build_faiss_gpu.py
# A helper script to build FAISS-GPU from source on Linux.
# Includes checks for prerequisites, handles cloning, building, and installing.
# Added check for existing faiss/faiss-cpu installation and CUDA version compatibility.

import subprocess
import os
import sys
import shutil
import logging
import argparse
import re

# --- Configuration ---
FAISS_REPO_URL = "https://github.com/facebookresearch/faiss.git"
FAISS_DIR = "faiss" # Directory where FAISS source will be cloned
BUILD_DIR = os.path.join(FAISS_DIR, "build") # Directory for the build process

# --- Logging Setup ---
# Configure logging to show INFO level messages and above in the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
# For more detailed output, change level=logging.DEBUG
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
# ---------------------

def run_command(command, cwd=None, shell=False, check=True):
    """Runs a shell command and logs the output."""
    logging.debug(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        # Use capture_output=True and text=True for easier output handling
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=check, # Raise CalledProcessError if command returns non-zero exit code
            text=True,
            capture_output=True
        )
        logging.debug("Command stdout:\n" + result.stdout)
        if result.stderr:
            logging.debug("Command stderr:\n" + result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.critical(f"Command failed with exit code {e.returncode}")
        logging.critical(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}")
        logging.critical(f"Stderr:\n{e.stderr}")
        logging.critical(f"Stdout:\n{e.stdout}")
        raise # Re-raise the exception after logging

def check_prerequisite(command, name, install_instructions=None, version_command=None):
    """Checks if a command exists in the system's PATH."""
    logging.info(f"Checking for {name}...")
    try:
        run_command(command, check=True, shell=True) # Use shell=True to find command in PATH
        logging.info(f"✅ Found {name}")
        if version_command:
            try:
                version_output = run_command(version_command, check=True, shell=True)
                logging.info(f"   {name} Version:\n{version_output.strip()}")
            except Exception as e:
                logging.warning(f"   Could not get {name} version: {e}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error(f"❌ {name} not found.")
        if install_instructions:
            logging.error(f"   Installation instructions: {install_instructions}")
        return False

def find_cuda_toolkit_root():
    """Attempts to find the CUDA Toolkit root directory."""
    logging.info("Attempting to find CUDA Toolkit Root Directory...")
    # Common installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-", # Check for versioned directories like /usr/local/cuda-12.4
        "/opt/cuda",
        "/opt/cuda-"
    ]

    for path in cuda_paths:
        if path.endswith('-'): # Handle versioned paths
            # Find directories starting with the prefix
            base_dir = os.path.dirname(path)
            prefix = os.path.basename(path)
            if os.path.exists(base_dir):
                for entry in os.listdir(base_dir):
                    full_path = os.path.join(base_dir, entry)
                    if os.path.isdir(full_path) and entry.startswith(prefix):
                         # Check if nvcc exists in the bin directory
                        if os.path.exists(os.path.join(full_path, "bin", "nvcc")):
                            logging.info(f"✅ Found CUDA Toolkit Root at: {full_path}")
                            return full_path
        else: # Handle exact paths like /usr/local/cuda
            if os.path.exists(path):
                 # Check if nvcc exists in the bin directory
                if os.path.exists(os.path.join(path, "bin", "nvcc")):
                    logging.info(f"✅ Found CUDA Toolkit Root at: {path}")
                    return path

    logging.error("❌ CUDA Toolkit Root not found in common locations.")
    logging.error("   Please ensure CUDA Toolkit is installed and accessible.")
    logging.error("   You might need to set the CUDA_TOOLKIT_ROOT_DIR environment variable manually.")
    return None

def get_cuda_toolkit_version_from_nvcc():
    """Gets the CUDA Toolkit version from nvcc --version."""
    try:
        # Use shell=True to find nvcc in PATH
        result = run_command("nvcc --version", check=True, shell=True)
        # Example output: Cuda compilation tools, release 12.4, V12.4.131
        match = re.search(r"release (\d+\.\d+)", result)
        if match:
            version = match.group(1)
            logging.info(f"✅ Detected CUDA Toolkit Version from nvcc: {version}")
            return version
        else:
            logging.warning("Could not parse CUDA Toolkit version from nvcc output.")
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("❌ Could not get CUDA Toolkit version from nvcc. Is nvcc in your PATH?")
        return None

def get_cuda_version_from_smi():
    """Gets the CUDA version reported by nvidia-smi (driver compatibility)."""
    try:
        # Use shell=True to find nvidia-smi in PATH
        result = run_command("nvidia-smi", check=True, shell=True)
        # Example output line: | NVIDIA-SMI 535.230.02     Driver Version: 535.230.02     CUDA Version: 12.2     |
        match = re.search(r"CUDA Version: (\d+\.\d+)", result)
        if match:
            version = match.group(1)
            logging.info(f"✅ Detected Driver Compatible CUDA Version from nvidia-smi: {version}")
            return version
        else:
            logging.warning("Could not parse CUDA Version from nvidia-smi output.")
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("❌ Could not run nvidia-smi. Is your NVIDIA driver installed correctly?")
        return None

def check_cudnn(cuda_toolkit_root):
    """Checks for cuDNN headers and libraries."""
    logging.info("Checking for cuDNN...")
    if not cuda_toolkit_root:
        logging.warning("   Skipping cuDNN check as CUDA Toolkit Root was not found.")
        return False

    # Common cuDNN locations relative to CUDA_TOOLKIT_ROOT_DIR
    cudnn_include = os.path.join(cuda_toolkit_root, "include", "cudnn.h")
    cudnn_lib_dir = os.path.join(cuda_toolkit_root, "lib64")

    header_found = os.path.exists(cudnn_include)
    # Check for common library names (libcuDNN.so, libcuDNN.so.8, etc.)
    lib_found = any(re.match(r"libcudnn\.so(\.\d+)*$", f) for f in os.listdir(cudnn_lib_dir)) if os.path.exists(cudnn_lib_dir) else False

    if header_found and lib_found:
        logging.info("✅ Found cuDNN")
        logging.info(f"   Please ensure your installed cuDNN version is compatible with CUDA Toolkit {get_cuda_toolkit_version_from_nvcc()}.")
        logging.info("   Refer to the cuDNN Installation Guide for compatibility matrix.")
        return True
    else:
        logging.error("❌ cuDNN headers or libraries not found in expected locations.")
        logging.error(f"   Looked for: {cudnn_include} and libraries in {cudnn_lib_dir}")
        logging.error("   Ensure cuDNN is installed correctly for your CUDA Toolkit version.")
        return False

def check_cuvs(cuda_toolkit_root):
    """Checks for cuVS headers and libraries."""
    logging.info("Checking for cuVS...")
    if not cuda_toolkit_root:
        logging.warning("   Skipping cuVS check as CUDA Toolkit Root was not found.")
        return False

    # Common cuVS locations relative to CUDA_TOOLKIT_ROOT_DIR or system paths
    cuvs_include = os.path.join(cuda_toolkit_root, "include", "cuvs", "core", "library.hpp")
    cuvs_lib_dir = os.path.join(cuda_toolkit_root, "lib64")

    header_found = os.path.exists(cuvs_include)
    # Check for common library names (libcuvs.so, libcuvs.so.x.y, etc.)
    lib_found = any(re.match(r"libcuvs\.so(\.\d+)*$", f) for f in os.listdir(cuvs_lib_dir)) if os.path.exists(cuvs_lib_dir) else False

    if header_found and lib_found:
        logging.info("✅ Found cuVS")
        return True
    else:
        logging.error("❌ cuVS header (cuvs/core/library.hpp) or library (libcuvs\.so(\.\d+)*$) not found in checked paths.")
        logging.info("\nℹ️ NVIDIA cuVS library not found.")
        logging.info("   cuVS is an optional library that can provide performance improvements for certain algorithms.")
        logging.info("   You can install it if desired, but it's not required to build basic FAISS-GPU.")
        logging.info("   Installation instructions: https://github.com/rapidsai/cuvs/blob/branch-24.12/docs/install.md")
        logging.info("   You can typically install it via pip:")
        logging.info("   `pip install cuvs-cuXX --extra-index-url=https://pypi.nvidia.com`")
        logging.info("   (Replace cuXX with your CUDA version, e.g., cu12 for CUDA 12.x)")

        return False

def is_package_installed(package_name):
    """Checks if a Python package is installed in the current environment."""
    # Use sys.executable to ensure we check the current virtual environment's pip
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            check=True,
            capture_output=True, # This captures stdout and stderr
            text=True,
            # Removed stdout=subprocess.PIPE and stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Build FAISS-GPU from source.")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run (check prerequisites and configure CMake only).")
    args = parser.parse_args()

    logging.info("--- FAISS-GPU Build Helper Script ---")
    logging.info("This script will guide you through building FAISS-GPU from source.")
    logging.info("Ensure you have the prerequisites installed: NVIDIA driver, CUDA Toolkit, cuDNN, CMake, C++17 compiler (g++/clang), and a BLAS library (OpenBLAS/MKL).")
    logging.info(f"Script running from: {os.getcwd()}")
    logging.info("-" * 20)

    # --- Check for existing FAISS installations ---
    logging.info("Checking for existing FAISS installations in the current Python environment...")
    faiss_installed = is_package_installed("faiss")
    faiss_cpu_installed = is_package_installed("faiss-cpu")

    if faiss_installed or faiss_cpu_installed:
        logging.critical("\nFATAL: Conflicting FAISS installation detected!")
        if faiss_installed:
            logging.critical("   'faiss' package is already installed.")
        if faiss_cpu_installed:
            logging.critical("   'faiss-cpu' package is already installed.")
        logging.critical("   Having multiple FAISS installations can cause import errors (like AttributeError).")
        logging.critical("   Please uninstall the conflicting package(s) before building from source:")
        if faiss_installed:
            logging.critical("   `pip uninstall faiss`")
        if faiss_cpu_installed:
            logging.critical("   `pip uninstall faiss-cpu`")
        logging.critical("   Then run this script again.")
        sys.exit(1)
    else:
        logging.info("✅ No conflicting FAISS installations found.")
    logging.info("-" * 20)


    # --- Check Mandatory Prerequisites ---
    logging.info("--- Checking Mandatory Prerequisites ---")
    os_name = run_command("lsb_release -is", check=False, shell=True).strip()
    os_version = run_command("lsb_release -rs", check=False, shell=True).strip()

    if os_name and os_version:
         logging.info(f"✅ Detected OS: {os_name} \"{os_version}\"")
         if os_name.lower() != "ubuntu":
             logging.warning("\n⚠️ This script is primarily tested on Linux (Ubuntu).")
             logging.warning("   You may need to adapt the installation commands for your operating system.")
    else:
        logging.warning("Could not detect OS distribution and version.")


    # Check CUDA Toolkit version from nvcc
    nvcc_cuda_version = get_cuda_toolkit_version_from_nvcc()
    if not nvcc_cuda_version:
        logging.critical("\nFATAL: CUDA Toolkit not found or nvcc not in PATH.")
        logging.critical("   Please install CUDA Toolkit and ensure its bin directory is in your PATH.")
        sys.exit(1)

    # Check Driver Compatible CUDA version from nvidia-smi
    smi_cuda_version = get_cuda_version_from_smi()
    if not smi_cuda_version:
         logging.warning("Could not get Driver Compatible CUDA Version from nvidia-smi.")
         logging.warning("   Skipping CUDA version compatibility check.")
    else:
        # --- Check CUDA Version Compatibility ---
        logging.info("--- Checking CUDA Version Compatibility ---")
        # Simple major.minor version comparison
        nvcc_major_minor = '.'.join(nvcc_cuda_version.split('.')[:2])
        smi_major_minor = '.'.join(smi_cuda_version.split('.')[:2])

        if nvcc_major_minor != smi_major_minor:
            logging.critical("\nFATAL: CUDA Toolkit version and Driver Compatible CUDA version mismatch!")
            logging.critical(f"   Detected CUDA Toolkit version (nvcc): {nvcc_cuda_version}")
            logging.critical(f"   Detected Driver Compatible CUDA version (nvidia-smi): {smi_cuda_version}")
            logging.critical("   These versions must match (at least major.minor) to avoid runtime errors (like PTX compilation issues).")
            logging.critical("   Please install a CUDA Toolkit version that is compatible with your NVIDIA driver.")
            logging.critical("   Refer to NVIDIA's CUDA Compatibility documentation.")
            sys.exit(1)
        else:
            logging.info("✅ CUDA Toolkit version and Driver Compatible CUDA version are compatible (major.minor match).")
        logging.info("-" * 20)


    # Find CUDA Toolkit Root Dir (needed for CMake and cuDNN/cuVS checks)
    cuda_toolkit_root = find_cuda_toolkit_root()
    if not cuda_toolkit_root:
         # If nvcc was found but root wasn't in standard places, try inferring from nvcc path
         try:
             nvcc_path = run_command("which nvcc", check=True, shell=True).strip()
             # Assuming standard structure like /path/to/cuda/bin/nvcc
             inferred_root = os.path.dirname(os.path.dirname(nvcc_path))
             if os.path.exists(inferred_root):
                 cuda_toolkit_root = inferred_root
                 logging.info(f"✅ Inferred CUDA Toolkit Root from nvcc path: {cuda_toolkit_root}")
             else:
                 logging.critical("\nFATAL: Could not determine CUDA Toolkit Root directory.")
                 sys.exit(1)
         except:
             logging.critical("\nFATAL: Could not determine CUDA Toolkit Root directory.")
             sys.exit(1)


    all_prereqs_met = True
    all_prereqs_met &= check_prerequisite("cmake --version", "cmake")
    all_prereqs_met &= check_prerequisite("g++ --version", "g++", install_instructions="sudo apt update && sudo apt install build-essential")
    all_prereqs_met &= check_prerequisite("swig -version", "swig", install_instructions="sudo apt update && sudo apt install swig", version_command="swig -version")

    # Check for BLAS (CMake will find it, but good to remind the user)
    logging.info("\n--- Checking for BLAS ---")
    logging.info("BLAS (Basic Linear Algebra Subprograms) is a mandatory prerequisite.")
    logging.info("Ensure you have a BLAS implementation (e.g., OpenBLAS or Intel MKL) installed system-wide.")
    logging.info("CMake will attempt to find BLAS automatically.")
    logging.warning("If CMake fails to find BLAS, you might need to help it by setting environment variables like BLAS_DIR or OpenBLAS_DIR.")
    logging.info("Example (add to your terminal session or ~/.bashrc):")
    logging.info("`export BLAS_DIR=/usr # Or wherever your BLAS was installed`")
    logging.info("`export OpenBLAS_DIR=/usr # Or wherever OpenBLAS was installed`")
    logging.info("Then run: `source ~/.bashrc` (or ~/.profile) or open a new terminal.")
    logging.info("-" * 20)


    # Check for cuDNN (mandatory for FAISS-GPU)
    all_prereqs_met &= check_cudnn(cuda_toolkit_root)

    # Check for gflags (often a dependency)
    all_prereqs_met &= check_prerequisite("dpkg -s libgflags-dev", "gflags", install_instructions="sudo apt update && sudo apt install libgflags-dev")


    if not all_prereqs_met:
        logging.critical("\nFATAL: Not all mandatory prerequisites are met. Please install them and run the script again.")
        sys.exit(1)

    logging.info("\n--- All mandatory prerequisites seem to be met. ---")
    logging.info(f"Using inferred CUDA Toolkit Root for CMake: {cuda_toolkit_root}")
    logging.info("-" * 20)

    # --- Checking Optional Prerequisites ---
    logging.info("--- Checking Optional Prerequisites ---")
    cuvs_found = check_cuvs(cuda_toolkit_root)
    logging.info("-" * 20)

    if args.dry_run:
        logging.info("\n--- Dry Run Complete ---")
        logging.info("Prerequisites checked and CMake configuration command generated.")
        logging.info("Exiting as requested by --dry-run flag.")
        sys.exit(0)

    # --- Step 2: Cloning FAISS Repository ---
    logging.info("\n--- Step 2: Cloning FAISS Repository ---")
    if os.path.exists(FAISS_DIR):
        logging.info(f"FAISS directory '{FAISS_DIR}' already exists. Skipping clone.")
        logging.info("Consider updating it: cd faiss && git pull")
    else:
        try:
            run_command(["git", "clone", FAISS_REPO_URL], check=True)
            logging.info(f"✅ Cloned FAISS repository into '{FAISS_DIR}'.")
        except Exception as e:
            logging.critical(f"❌ Failed to clone FAISS repository: {e}")
            sys.exit(1)
    logging.info("-" * 20)

    # --- Step 3: Creating Build Directory ---
    logging.info("\n--- Step 3: Creating Build Directory ---")
    os.makedirs(BUILD_DIR, exist_ok=True)
    logging.info(f"Build directory created: {BUILD_DIR}")
    logging.info("-" * 20)

    # --- Step 4: Configuring Build with CMake ---
    logging.info("\n--- Step 4: Configuring Build with CMake ---")
    logging.info("Generating CMake command for GPU build...")

    cmake_command = [
        "cmake",
        "-B", BUILD_DIR, # Build in the specified build directory
        FAISS_DIR,     # Source directory
        "-DFAISS_ENABLE_GPU=ON",
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_toolkit_root}",
        "-DCMAKE_BUILD_TYPE=Release", # Release build for performance
        "-DBUILD_TESTING=OFF", # Usually not needed for just installing the library
        "-DCMAKE_VERBOSE_MAKEFILE=ON", # Show detailed build commands
    ]

    if cuvs_found:
        logging.info("Including CMake flag: -DFAISS_USE_CUVS=ON (cuVS found)")
        cmake_command.append("-DFAISS_USE_CUVS=ON")
    else:
        logging.info("Not including CMake flag: -DFAISS_USE_CUVS=ON (cuVS not found)")


    logging.info("\nGenerated CMake command:")
    logging.info(" ".join(cmake_command))

    logging.info("\nExecuting CMake command...")
    try:
        run_command(cmake_command, check=True)
        logging.info("--- Command finished successfully ---")
        logging.info("\n--- Verifying Makefile Generation at: " + os.path.join(BUILD_DIR, "Makefile") + " ---")
        if os.path.exists(os.path.join(BUILD_DIR, "Makefile")):
            logging.info("✅ Makefile found. CMake configuration and generation appear successful.")
        else:
            logging.critical("❌ Makefile not found. CMake configuration may have failed.")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"❌ CMake configuration failed: {e}")
        sys.exit(1)
    logging.info("-" * 20)


    # --- Step 5: Building FAISS ---
    logging.info("\n--- Step 5: Building FAISS ---")
    logging.info("Generating Make command...")
    # Use -j$(nproc) to utilize all available CPU cores for faster compilation
    make_command = ["make", f"-j{os.cpu_count()}"]

    logging.info("\nGenerated Make command (run from build directory):")
    logging.info(" ".join(make_command))

    logging.info("\nExecuting Make command from build directory...")
    try:
        run_command(make_command, cwd=BUILD_DIR, check=True)
        logging.info("--- Command finished successfully ---")
        logging.info("✅ FAISS build appears successful.")
    except Exception as e:
        logging.critical(f"❌ FAISS build failed: {e}")
        sys.exit(1)
    logging.info("-" * 20)

    # --- Step 6: Installing Python Bindings ---
    logging.info("\n--- Step 6: Installing Python Bindings ---")
    logging.info("Generating pip install command...")

    # The pip install command needs to be run from the python subdirectory within the build directory
    faiss_python_dir = os.path.join(BUILD_DIR, FAISS_DIR, "python")
    pip_install_command = [
        sys.executable, # Use the python executable of the current environment
        "-m", "pip",
        "install", "." # Install the package from the current directory
    ]

    logging.info(f"\nGenerated pip install command (run from '{faiss_python_dir}'):")
    logging.info(" ".join(pip_install_command))

    logging.info(f"\nExecuting pip install command from '{faiss_python_dir}'...")
    try:
        run_command(pip_install_command, cwd=faiss_python_dir, check=True)
        logging.info("--- Command finished successfully ---")
        logging.info("✅ FAISS Python bindings installed.")
    except Exception as e:
        logging.critical(f"❌ Installing FAISS Python bindings failed: {e}")
        sys.exit(1)
    logging.info("-" * 20)


    logging.info("\n--- FAISS-GPU Build and Installation Complete! ---")
    logging.info("You should now be able to import 'faiss' in your Python environment.")
    logging.info("Remember to use a Conda environment for managing CUDA/cuDNN/BLAS dependencies if you haven't already.")
    logging.warning("\n⚠️ If you encounter an AttributeError when importing faiss (e.g., 'module ... has no attribute ...'),")
    logging.warning("   this can sometimes happen with SWIG-generated bindings and specific FAISS/SWIG versions.")
    logging.warning("   Review the full build output (DEBUG level in console) for SWIG-related warnings or errors.")
    logging.warning("   Consult FAISS documentation or community forums for known compatibility issues or workarounds.")
    logging.info("-" * 20)


if __name__ == "__main__":
    main()
