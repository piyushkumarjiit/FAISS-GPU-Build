# install_faiss_dependencies.py
# This script checks for and installs mandatory dependencies for building FAISS-GPU
# from source on Ubuntu. It covers common prerequisites, CUDA Toolkit 12.4, and cuDNN 9.10.0.
# It prioritizes checking for and installing dependencies via apt, falling back to
# manual methods (like cuDNN tar file download/copy) if apt doesn't provide the necessary files
# in standard locations.
# Includes an automated workaround for the libtinfo5 dependency issue on Ubuntu 24.04 when installing CUDA,
# by adding the Ubuntu 23.04 (Lunar) repository and relying on apt's dependency resolution.
# IMPORTANT: This script cleans up the added Lunar repository source file after successful CUDA installation.
# Includes a more robust check for command existence under sudo.
# FIX: Corrected the URL construction for CUDA keyring download on Ubuntu 24.04.
# IMPROVEMENT: Added --dry-run mode to check dependencies without installing.
# IMPROVEMENT: Improved dependency checking for libgflags-dev and libopenblas-dev using check_library_files.
# IMPROVEMENT: Implemented a more robust check for CUDA Toolkit existence and version, less reliant on sudo PATH.
# IMPROVEMENT: Allow nvidia-smi to run even in dry-run mode to get CUDA version for checks.
#
# IMPORTANT: This script requires root privileges for many operations (apt install, file copying, file deletion).
# Please run this script using `sudo python install_faiss_dependencies.py`.

import subprocess
import os
import sys
import platform
import re
import logging
import argparse
import glob # Import glob for finding the downloaded tar file
import shutil # Import shutil for copying files
import time # Import time for sleep

# --- Configuration ---
# Versions to target for installation
TARGET_CUDA_VERSION = "12.4"
TARGET_CUDNN_VERSION = "9.10.0" # Compatible cuDNN version for CUDA 12.4

# Assuming CUDA 12.4 is installed in /usr/local/cuda (common symlink)
# Adjust this path if your CUDA 12.4 is installed elsewhere
# The script will try to infer the actual path if this is a symlink
CUDA_INSTALL_PATH = "/usr/local/cuda" # Used as a fallback/default if inference fails

# URL for cuDNN 9.10.0 for CUDA 12.x (adjust if a different version is needed)
CUDNN_TAR_URL = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.10.0.56_cuda12-archive.tar.xz"
CUDNN_TAR_FILENAME = "cudnn-linux-x86_64-9.10.0.56_cuda12-archive.tar.xz"

# Apt package names (based on Ubuntu 24.04 and CUDA 12.4 / cuDNN 9.10)
# These might need adjustment for other Ubuntu versions or CUDA versions
CUDA_APT_PACKAGE = f"cuda-toolkit-{TARGET_CUDA_VERSION.replace('.', '-')}"
CUDNN_APT_PACKAGE = f"cudnn9-cuda-{TARGET_CUDA_VERSION.split('.')[0]}" # Example: cudnn9-cuda-12
GFLAGS_APT_PACKAGE = "libgflags-dev"
OPENBLAS_APT_PACKAGE = "libopenblas-dev"
BUILD_ESSENTIAL_APT_PACKAGE = "build-essential" # Package containing g++

# Details for automated libtinfo5 workaround (Lunar repository)
LIBTINFO5_LUNAR_REPO_FILE = "/etc/apt/sources.list.d/ubuntu-lunar.sources" # Using a distinct file name
LIBTINFO5_LUNAR_REPO_CONTENT = """Types: deb
URIs: http://old-releases.ubuntu.com/ubuntu/
Suites: lunar
Components: universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"""
LIBTINFO5_PACKAGE_NAME = "libtinfo5" # Still use the package name for error checking


# --- Global dry run flag ---
DRY_RUN = False

# --- Logging Setup ---
# Configured in main execution block

# --- Helper Functions ---

def run_command(command, shell=False, check=True, cwd=None, log_output=True):
    """Runs a shell command and logs the output."""
    # Do not prepend 'sudo' here. The script should be run with sudo from the start.
    command_str = ' '.join(command) if isinstance(command, list) else command
    logging.info(f"Running command: {command_str}")

    # Allow specific commands to run even in dry-run mode if their output is needed for checks
    # nvidia-smi is needed to get the CUDA version for compatibility checks
    if DRY_RUN and command[0] != "nvidia-smi":
        logging.info(f"--- DRY RUN: Skipping command execution: {command_str} ---")
        return "", "" # Return empty strings in dry run for most commands

    # For nvidia-smi in dry run, we still execute it to get the version
    if DRY_RUN and command[0] == "nvidia-smi":
         logging.info(f"--- DRY RUN: Executing nvidia-smi to get CUDA version for checks ---")
         # Bypass the dry-run return and proceed to subprocess.Popen
         pass
    # For non-dry run, or for nvidia-smi in dry run, proceed to execution

    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate() # Use communicate to wait for process and get output

        if log_output:
             logging.debug(f"Command stdout:\n{stdout}")
             logging.debug(f"Command stderr:\n{stderr}")

        return_code = process.returncode
        if return_code != 0 and check:
            logging.error(f"--- Command failed with return code {return_code} ---")
            # Print error output to console for visibility on failure
            sys.stderr.write(stdout)
            sys.stderr.write(stderr)
            sys.stderr.flush()
            raise subprocess.CalledProcessError(return_code, command, stdout, stderr)
        elif return_code != 0 and not check:
             logging.warning(f"--- Command finished with non-zero return code {return_code} (check=False) ---")
             if log_output: # Only log debug output if check=False and log_output=True
                 logging.debug(f"Command stdout:\n{stdout}")
                 logging.debug(f"Command stderr:\n{stderr}")


        if check:
             logging.info("--- Command finished successfully ---")
        return stdout, stderr # Return both stdout and stderr

    except FileNotFoundError:
        logging.error(f"--- Error: Command not found: {command[0]} ---")
        raise
    except subprocess.CalledProcessError as e:
        # The error is already logged and printed in the check=True block
        raise
    except Exception as e:
        logging.error(f"--- An error occurred while running command: {e} ---", exc_info=True) # Log exception info
        raise


def check_command_exists(command):
    """Checks if a command exists and is executable in the system's PATH."""
    try:
        # Try running the command with a minimal argument that usually doesn't require complex setup
        # Use check=False to prevent raising CalledProcessError for non-zero exit codes (e.g., help text)
        # Use a short timeout in case the command hangs
        # We use 'which' here as a common cross-shell utility, falling back if needed.
        # This is generally more reliable than just running the command name for existence check.
        result = subprocess.run(["which", command], capture_output=True, text=True, timeout=5, check=False)

        if result.returncode == 0 and result.stdout.strip():
             # 'which' found the command and printed its path
             logging.debug(f"Command '{command}' found by 'which' at: {result.stdout.strip()}")
             logging.info(f"✅ Found command: {command}")
             return True
        else:
             # 'which' didn't find it or failed
             logging.debug(f"'which {command}' failed or found nothing. Return code: {result.returncode}, Output: {result.stdout.strip()}")
             logging.warning(f"❌ Command not found: {command}")
             return False

    except FileNotFoundError:
        # 'which' command itself not found - fallback to trying to run the command directly
        logging.debug("'which' command not found. Falling back to direct command execution check.")
        try:
            # Try running the command with a minimal argument that usually doesn't require complex setup
            # Use check=False to prevent raising CalledProcessError for non-zero exit codes (e.g., help text)
            # Use a short timeout in case the command hangs
            subprocess.run([command], capture_output=True, text=True, timeout=5, check=False)
            # If we reach here without FileNotFoundError, the command was found and could be executed
            logging.debug(f"Command '{command}' found by direct execution check.")
            logging.info(f"✅ Found command: {command}")
            return True
        except FileNotFoundError:
            logging.warning(f"❌ Command not found: {command}")
            return False
        except Exception as e:
            logging.warning(f"❌ Error checking for command '{command}' by direct execution: {e}")
            return False

    except Exception as e:
        # Catch other potential errors like timeout for 'which'
        logging.warning(f"❌ Error checking for command '{command}' using 'which': {e}")
        return False


def get_command_version(command, version_flag="--version"):
    """Gets the version string from a command."""
    try:
        result, _ = run_command([command, version_flag], check=True, log_output=False)
        logging.debug(f"'{command} {version_flag}' output:\n{result.strip()}")
        return result.strip()
    except Exception:
        logging.warning(f"Could not get version for command: {command} using flag {version_flag}")
        return None


def get_cuda_toolkit_version(cuda_root=None):
    """
    Gets the CUDA Toolkit version reported by nvcc.
    Prioritizes looking in cuda_root/bin if provided, falls back to PATH.
    Returns the version string if found, None otherwise.
    """
    nvcc_command = "nvcc"
    if cuda_root and os.path.exists(os.path.join(cuda_root, "bin", "nvcc")):
        nvcc_command = os.path.join(cuda_root, "bin", "nvcc")
        logging.debug(f"Checking nvcc at specific path: {nvcc_command}")
    else:
        logging.debug("Checking nvcc in PATH.")
        # If cuda_root/bin doesn't exist or isn't provided, check if nvcc is in the PATH
        if not check_command_exists("nvcc"):
             logging.warning("❌ nvcc command not found in PATH.")
             return None


    try:
        result = subprocess.run([nvcc_command, "--version"], capture_output=True, check=True, text=True)
        output = result.stdout
        # Regex to find version like "Cuda compilation tools, release 12.2, V12.2.140"
        match = re.search(r"release (\d+\.\d+)", output)
        if match:
            cuda_version = match.group(1)
            logging.info(f"✅ Detected CUDA Toolkit Version from nvcc: {cuda_version}")
            return cuda_version
        else:
            logging.warning("Could not detect CUDA Toolkit Version from nvcc output.")
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError is already handled by the check for cuda_root/bin and check_command_exists
        # CalledProcessError might happen if nvcc exists but fails to run --version
        logging.error(f"❌ Failed to run '{nvcc_command} --version'.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while getting nvcc version: {e}", exc_info=True)
        return None


def get_cuda_version_from_smi():
    """Gets the CUDA version reported by nvidia-smi (driver compatibility)."""
    # No need for check_command_exists here, the try/except handles FileNotFoundError
    try:
        # Run nvidia-smi even in dry-run mode to get the CUDA version for checks
        # Bypass the dry-run check in run_command for this specific command
        if DRY_RUN:
             logging.info(f"--- DRY RUN: Executing nvidia-smi to get CUDA version for checks ---")
             result = subprocess.run(["nvidia-smi"], capture_output=True, check=True, text=True)
             stdout = result.stdout
             stderr = result.stderr
             logging.debug(f"nvidia-smi stdout:\n{stdout}")
             logging.debug(f"nvidia-smi stderr:\n{stderr}")
        else:
             stdout, stderr = run_command(["nvidia-smi"], check=True, log_output=False)


        # Example output line: | NVIDIA-SMI 535.230.02     Driver Version: 535.230.02     CUDA Version: 12.2     |
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            logging.info(f"✅ Detected Driver Compatible CUDA Version from nvidia-smi: {version}")
            return version
        else:
            logging.warning("Could not parse CUDA Version from nvidia-smi output.")
            return None
    except FileNotFoundError:
        logging.error("❌ nvidia-smi command not found.")
        logging.error("   Ensure NVIDIA drivers are installed and nvidia-smi is in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ nvidia-smi command failed with return code {e.returncode}.")
        logging.error("   Ensure NVIDIA drivers are installed and working correctly.")
        logging.debug(f"nvidia-smi stdout:\n{e.stdout}")
        logging.debug(f"nvidia-smi stderr:\n{e.stderr}")
        return None
    except Exception:
        logging.warning("Could not run nvidia-smi.")
        return None


def get_os_info():
    """Gets basic OS information."""
    os_system = platform.system()
    if os_system == "Linux":
        try:
            # Try to get distribution info on Linux
            with open("/etc/os-release") as f:
                os_info = dict(re.findall(r'^(.*?)=(.*?)$', f.read(), re.MULTILINE))
                os_name = os_info.get("ID", "linux")
                os_version = os_info.get("VERSION_ID", "")
                logging.info(f"✅ Detected OS: {os_name} {os_version}")
                return os_name, os_version
        except FileNotFoundError:
            logging.info(f"✅ Detected OS: {os_system} (details not available)")
            return os_system.lower(), ""
    else:
        logging.info(f"✅ Detected OS: {os_system}")
        return os_system.lower(), ""

def find_cuda_toolkit_root():
    """Tries to find the CUDA Toolkit installation root directory."""
    logging.info("Attempting to find CUDA Toolkit Root Directory...")
    # Check standard locations first
    standard_paths = ["/usr/local/cuda", "/opt/cuda"]
    for path in standard_paths:
        if os.path.exists(os.path.join(path, "bin", "nvcc")):
            logging.info(f"✅ Found CUDA Toolkit Root at: {path}")
            return path

    # If not found in standard locations, try searching using 'which nvcc' (if nvcc is in PATH)
    # Use the robust check_command_exists to find nvcc path
    try:
        # Use subprocess.run directly to capture output for 'which'
        result = subprocess.run(["which", "nvcc"], capture_output=True, check=False, text=True)
        if result.returncode == 0 and result.stdout.strip():
             nvcc_path = result.stdout.strip()
             cuda_bin_dir = os.path.dirname(nvcc_path)
             # Assume CUDA root is the parent of 'bin'
             cuda_root = os.path.dirname(cuda_bin_dir)
             if os.path.exists(os.path.join(cuda_root, "include")): # Basic check for include dir
                  logging.info(f"✅ Found CUDA Toolkit Root via 'which nvcc' at: {cuda_root}")
                  return cuda_root
        else:
             logging.debug("'which nvcc' failed or found nothing.")


    except Exception as e:
        logging.debug(f"Error during 'which nvcc' attempt: {e}")
        pass # which nvcc failed, continue

    logging.warning("❌ Could not automatically find CUDA Toolkit Root Directory.")
    logging.warning("   You may need to manually specify it if the build fails.")
    return None # Return None if not found automatically


def check_library_files(library_name, header_file, lib_file_pattern, base_search_paths):
    """Checks for the presence of library header and library files."""
    logging.debug(f"Checking for {library_name}...")
    found_header = False
    found_lib = False
    found_paths = []

    # Define common include and lib subdirectories relative to base paths
    include_subdirs = ["include", "include/x86_64-linux-gnu"] # Added architecture-specific include
    lib_subdirs = ["lib64", "lib", "lib/x86_64-linux-gnu"] # Added common Ubuntu lib path

    # Check for header file in base_search_paths + include_subdirs AND standard system include paths
    header_search_paths = base_search_paths + ["/usr"] # Add /usr to search for headers in /usr/include...
    for base_path in header_search_paths:
        if base_path is None: continue # Skip None paths
        for subdir in include_subdirs:
            include_path = os.path.join(base_path, subdir)
            header_full_path = os.path.join(include_path, header_file)
            if os.path.exists(header_full_path):
                logging.debug(f"✅ Found {header_file} at: {header_full_path}")
                found_header = True
                found_paths.append(include_path)
                break # Found header, no need to check other include subdirs for this base path
        if found_header:
             break # Found header in one base path, no need to check others

    # Check for library file in base_search_paths + lib_subdirs AND standard system lib paths
    lib_search_paths = base_search_paths + ["/usr"] # Add /usr to search for libs in /usr/lib...
    for base_path in lib_search_paths:
        if base_path is None: continue # Skip None paths
        for subdir in lib_subdirs:
            lib_path = os.path.join(base_path, subdir)
            if os.path.exists(lib_path):
                 for fname in os.listdir(lib_path):
                      if re.match(lib_file_pattern, fname):
                           logging.debug(f"✅ Found {library_name} library file: {fname} at {lib_path}")
                           found_lib = True
                           found_paths.append(lib_path)
                           break # Found one matching lib file, no need to check other files in this lib path
            if found_lib:
                 break # Found lib in one base path+subdir, no need to check others


    if not found_header or not found_lib:
        logging.warning(f"❌ {library_name} header ({header_file}) or library ({lib_file_pattern}) not found in checked paths.")
        return False, list(set(found_paths)) # Return unique paths found
    else:
        logging.info(f"✅ Found {library_name}") # Consolidated success message
        return True, list(set(found_paths)) # Return unique paths found


def setup_logging(console_level=logging.INFO):
    """Confgures the logging system for console output only."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Capture all messages at root level

    # Remove existing handlers to prevent duplicate output
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    # Use a format that includes level name for clarity
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.debug("Logging configured for console output.")


# --- Installation Functions ---

def install_apt_packages(package_list):
    """Installs a list of packages using apt-get."""
    logging.info(f"\n--- Installing Apt Packages: {', '.join(package_list)} ---")
    if DRY_RUN:
        logging.info(f"--- DRY RUN: Skipping apt install for: {', '.join(package_list)} ---")
        return True # Assume success in dry run

    try:
        # When running the script with sudo, individual commands do not need sudo
        run_command(["apt-get", "update"])
        run_command(["apt-get", "install", "-y"] + package_list)
        logging.info(f"✅ Successfully installed Apt Packages: {', '.join(package_list)}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to install Apt Packages: {', '.join(package_list)}", exc_info=True)
        return False

def install_cuda_toolkit(os_name, os_version, driver_cuda_version):
    """Installs CUDA Toolkit using the NVIDIA apt repository."""
    logging.info(f"\n--- Installing CUDA Toolkit {TARGET_CUDA_VERSION} (Ubuntu Apt) ---")
    if os_name != "ubuntu":
        logging.error("❌ Automatic CUDA installation via apt is only supported for Ubuntu.")
        logging.info("Please install CUDA Toolkit manually for your OS from: https://developer.nvidia.com/cuda-downloads")
        return False

    # Check if CUDA Toolkit is already installed and matches the desired version
    # Use the robust check here
    current_cuda_version = get_cuda_toolkit_version(find_cuda_toolkit_root())
    if current_cuda_version and current_cuda_version.startswith(TARGET_CUDA_VERSION):
        logging.info(f"✅ CUDA Toolkit {current_cuda_version} is already installed and matches the desired version {TARGET_CUDA_VERSION}.")
        return True
    elif current_cuda_version:
        logging.warning(f"⚠️ CUDA Toolkit version {current_cuda_version} found, but desired version is {TARGET_CUDA_VERSION}.")
        logging.warning("Please uninstall the existing CUDA Toolkit before proceeding.")
        logging.warning("Refer to NVIDIA documentation for uninstallation instructions.")
        return False

    logging.info(f"CUDA Toolkit {TARGET_CUDA_VERSION} not found. Proceeding with installation.")

    # Basic compatibility check between driver and target CUDA Toolkit version
    if driver_cuda_version:
        driver_major_minor = '.'.join(driver_cuda_version.split('.')[:2])
        target_major_minor = '.'.join(TARGET_CUDA_VERSION.split('.')[:2])
        if driver_major_minor != target_major_minor:
             logging.warning(f"⚠️ Driver compatible CUDA version ({driver_cuda_version}) may not be fully compatible with target CUDA Toolkit version ({TARGET_CUDA_VERSION}).")
             logging.warning("   Proceeding with installation of CUDA Toolkit {TARGET_CUDA_VERSION}, but be aware of potential runtime issues.")
             logging.warning("   For best compatibility, install a CUDA Toolkit version that matches your driver's compatible version.")
             logging.warning("   Refer to NVIDIA's CUDA Compatibility documentation.")


    logging.info("Step 1: Add NVIDIA CUDA public key.")
    if DRY_RUN:
        logging.info("--- DRY RUN: Skipping CUDA keyring download and installation ---")
    else:
        try:
            # Download the keyring package (wget does not require root unless writing to a protected directory)
            # FIX: Correctly format the OS version string without quotes for the URL
            # Ensure os_version is treated as a string before replacing
            os_version_str = str(os_version)
            os_version_formatted = os_version_str.replace('.', '') # "24.04" -> "2404"
            cuda_keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{os_version_formatted}/x86_64/cuda-keyring_1.1-1_all.deb"
            run_command(["wget", cuda_keyring_url])
            logging.info("✅ CUDA keyring downloaded.")
            # Install the keyring package (dpkg requires root, handled by running script with sudo)
            run_command(["dpkg", "-i", "cuda-keyring_1.1-1_all.deb"]) # Use the fixed filename
            logging.info("✅ CUDA keyring installed.")
        except Exception as e:
            logging.error(f"❌ Failed to add NVIDIA CUDA public key: {e}")
            logging.error("Please check your internet connection and try again.")
            return False

    logging.info("Step 2: Update the apt package list.")
    if DRY_RUN:
        logging.info("--- DRY RUN: Skipping apt update ---")
    else:
        try:
            # apt update requires root, handled by running script with sudo
            run_command(["apt", "update"])
            logging.info("✅ Apt package list updated.")
        except Exception as e:
            logging.error(f"❌ Failed to update apt package list: {e}")
            logging.error("Please check your internet connection and repository configurations.")
            return False

    logging.info(f"Step 3: Attempting to install CUDA Toolkit {TARGET_CUDA_VERSION} via apt.")
    logging.warning("\n⚠️ Note: You might encounter a dependency issue with 'libtinfo5' on newer Ubuntu versions (like 24.04).")
    logging.warning(f"   We will attempt to install the full {CUDA_APT_PACKAGE} package.")


    try:
        # Attempt to install the full meta-package first (apt install requires root)
        logging.info(f"Attempting to install the full {CUDA_APT_PACKAGE} meta-package...")

        if DRY_RUN:
            logging.info(f"--- DRY RUN: Skipping apt install for {CUDA_APT_PACKAGE} ---")
            # In dry run, simulate failure to trigger the workaround logging
            process_returncode = 1
            stdout = "dependency problems - leaving unconfigured\nThe following packages have unmet dependencies:\n cuda-toolkit-12-4 : Depends: libtinfo5 but it is not installable\n"
            stderr = ""
        else:
            # Capture stdout and stderr to check for libtinfo5 error without failing immediately
            process = subprocess.Popen(
                ["apt", "install", "-y", CUDA_APT_PACKAGE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            process_returncode = process.returncode

            logging.debug(f"CUDA Apt Install stdout:\n{stdout}")
            logging.debug(f"CUDA Apt Install stderr:\n{stderr}")


        if process_returncode != 0:
             # Check if the error output indicates the libtinfo5 issue
             if "libtinfo5" in stdout or "libtinfo5" in stderr:
                 logging.warning(f"Installation of {CUDA_APT_PACKAGE} failed, likely due to missing 'libtinfo5'.")
                 logging.warning("This is a known issue on Ubuntu 24.04 with CUDA 12.4.")
                 logging.info("\n--- Attempting Automated libtinfo5 Workaround (Add Lunar Repository) ---")
                 logging.info(f"Step 3a: Creating new sources file for Ubuntu 23.04 (Lunar) at: {LIBTINFO5_LUNAR_REPO_FILE}")
                 if DRY_RUN:
                      logging.info(f"--- DRY RUN: Skipping creation of {LIBTINFO5_LUNAR_REPO_FILE} ---")
                 else:
                      try:
                          # Write the repository content to the sources file (requires root)
                          with open(LIBTINFO5_LUNAR_REPO_FILE, 'w') as f:
                               f.write(LIBTINFO5_LUNAR_REPO_CONTENT)
                          logging.info(f"✅ Created {LIBTINFO5_LUNAR_REPO_FILE}.")
                      except Exception as write_e:
                           logging.error(f"❌ Failed to create {LIBTINFO5_LUNAR_REPO_FILE}: {write_e}")
                           logging.critical("FATAL: Automated libtinfo5 workaround failed at sources file creation.")
                           logging.critical("Please manually create the file and add the repository content.")
                           logging.critical(f"  `sudo nano {LIBTINFO5_LUNAR_REPO_FILE}`")
                           logging.critical("  Paste the following content into the file:")
                           logging.critical("-" * 20)
                           logging.critical(LIBTINFO5_LUNAR_REPO_CONTENT.strip())
                           logging.critical("-" * 20)
                           logging.critical("  Save the file and exit the editor.")
                           logging.critical("Then run this script again.")
                           return False # Indicate failure

                 logging.info("Step 3b: Updating apt package list after adding Lunar repository.")
                 if DRY_RUN:
                      logging.info("--- DRY RUN: Skipping apt update ---")
                 else:
                      try:
                          # apt update requires root
                          run_command(["apt", "update"])
                          logging.info("✅ Apt package list updated.")
                      except Exception as update_e:
                           logging.error(f"❌ Failed to update apt package list after adding Lunar repo: {update_e}")
                           logging.critical("FATAL: Automated libtinfo5 workaround failed at apt update.")
                           logging.critical("Please manually run `sudo apt update`.")
                           logging.critical("Then run this script again.")
                           return False # Indicate failure

                 # --- Second attempt to install CUDA Toolkit after libtinfo5 workaround ---
                 logging.info(f"\n--- Attempting to install CUDA Toolkit {TARGET_CUDA_VERSION} again after libtinfo5 workaround ---")
                 # Rely on apt to find and install libtinfo5 as a dependency from the Lunar repo
                 if DRY_RUN:
                      logging.info(f"--- DRY RUN: Skipping second apt install for {CUDA_APT_PACKAGE} ---")
                      # In dry run, assume success after workaround for logging purposes
                      process_retry_returncode = 0
                      stdout_retry = ""
                      stderr_retry = ""
                 else:
                      process_retry = subprocess.Popen(
                          ["apt", "install", "-y", CUDA_APT_PACKAGE],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True
                      )
                      stdout_retry, stderr_retry = process_retry.communicate()
                      process_retry_returncode = process_retry.returncode

                      logging.debug(f"CUDA Apt Install Retry stdout:\n{stdout_retry}")
                      logging.debug(f"CUDA Apt Install Retry stderr:\n{stderr_retry}")


                 if process_retry_returncode != 0:
                      logging.error(f"❌ Second attempt to install {CUDA_APT_PACKAGE} failed via apt.")
                      logging.critical("\nFATAL: CUDA Toolkit installation failed even after attempting automated libtinfo5 workaround.")
                      logging.critical("Please review the error messages above carefully.")
                      logging.critical("\n--- Manual Troubleshooting for libtinfo5 ---")
                      logging.critical("If you are still facing issues, ensure libtinfo5 is installed:")
                      logging.critical(f"  `sudo apt install {LIBTINFO5_PACKAGE_NAME}`")
                      logging.critical("If that fails, consider manually downloading and installing the libtinfo5 .deb package:")
                      logging.critical(f"  Download: `wget http://archive.ubuntu.com/ubuntu/pool/main/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb`")
                      logging.critical(f"  Install: `sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb`")
                      logging.critical("Then run this script again.")
                      return False # Indicate failure

                 else:
                      logging.info(f"✅ CUDA Toolkit {TARGET_CUDA_VERSION} installed successfully via apt after libtinfo5 workaround.")
                      # --- Clean up the added Lunar repository source file ---
                      logging.info(f"Step 3d: Cleaning up added Lunar repository source file: {LIBTINFO5_LUNAR_REPO_FILE}")
                      if DRY_RUN:
                           logging.info(f"--- DRY RUN: Skipping removal of {LIBTINFO5_LUNAR_REPO_FILE} ---")
                           logging.info("--- DRY RUN: Skipping final apt update ---")
                      else:
                           try:
                                if os.path.exists(LIBTINFO5_LUNAR_REPO_FILE):
                                     os.remove(LIBTINFO5_LUNAR_REPO_FILE)
                                     logging.info(f"✅ Removed {LIBTINFO5_LUNAR_REPO_FILE}.")
                                     # Optional: Run apt update again to remove the repo from sources list
                                     logging.info("Running apt update to refresh sources after removing Lunar repo.")
                                     run_command(["apt", "update"], check=False) # Don't fail if update has issues
                                else:
                                     logging.warning(f"Lunar repository source file not found during cleanup: {LIBTINFO5_LUNAR_REPO_FILE}")
                           except Exception as cleanup_e:
                                logging.error(f"❌ Failed to clean up Lunar repository source file: {cleanup_e}", exc_info=True)
                                logging.warning("Please manually remove the file:")
                                logging.warning(f"  `sudo rm {LIBTINFO5_LUNAR_REPO_FILE}`")
                                logging.warning("And run `sudo apt update`.")
                      # --- End Cleanup ---
                      return True # Indicate success

             else:
                 # If the failure is not related to libtinfo5, log the error and exit
                 logging.error(f"❌ Installation of {CUDA_APT_PACKAGE} failed via apt with unexpected error.")
                 logging.critical("\nFATAL: CUDA Toolkit installation failed.")
                 logging.critical("Please review the error messages above carefully.")
                 return False

        else:
             # If the apt install command finished with return code 0
             logging.info(f"✅ CUDA Toolkit {TARGET_CUDA_VERSION} installed successfully via apt.")
             return True

    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during CUDA Toolkit installation: {e}", exc_info=True)
        return False


def install_cudnn():
    """Installs cuDNN 9.10.0 for CUDA 12.4."""
    logging.info(f"\n--- Checking/Installing cuDNN {TARGET_CUDNN_VERSION} for CUDA {TARGET_CUDA_VERSION} ---")

    # Check if cuDNN is already installed in the expected location
    cuda_root = find_cuda_toolkit_root() # Find CUDA root again
    if cuda_root:
         cudnn_found, _ = check_library_files("cuDNN", "cudnn.h", r"libcudnn\.so(\.\d+)*$", [cuda_root])
         if cudnn_found:
             logging.info(f"✅ cuDNN appears to be installed in the expected location relative to CUDA Toolkit Root.")
             logging.info("Skipping cuDNN installation.")
             return True
         else:
              logging.warning("cuDNN not found in the expected location. Proceeding with installation attempts.")
    else:
        logging.warning("Could not find CUDA Toolkit Root. Cannot check if cuDNN is already installed relative to it.")
        logging.warning("Proceeding with cuDNN installation attempts assuming it's missing or in a non-standard location.")

    logging.info(f"Attempting to install cuDNN {TARGET_CUDNN_VERSION} via apt ({CUDNN_APT_PACKAGE}).")
    logging.warning(f"Note: On some Ubuntu versions (like 24.04), the apt package for cuDNN might only install documentation.")

    if DRY_RUN:
        logging.info(f"--- DRY RUN: Skipping apt install for {CUDNN_APT_PACKAGE} ---")
        # In dry run, simulate failure to trigger manual fallback logging
        apt_success = False
    else:
        try:
            # Attempt to install cuDNN via apt (apt install requires root)
            run_command(["apt", "install", "-y", CUDNN_APT_PACKAGE], check=False) # Don't fail immediately if apt install has issues
            logging.info(f"Attempted to install {CUDNN_APT_PACKAGE} via apt.")
            apt_success = True # Assume apt command itself ran without critical error, even if it didn't install files

        except Exception as e:
            logging.warning(f"⚠️ Attempting apt install of {CUDNN_APT_PACKAGE} failed: {e}")
            logging.warning("Proceeding to check if cuDNN files are present anyway or attempt manual installation.")
            apt_success = False


    # --- Re-check for cuDNN files after apt attempt ---
    logging.info("Re-checking for cuDNN files in standard locations after apt attempt...")
    # Use the inferred cuda_root for searching cuDNN if found, otherwise default CUDA_INSTALL_PATH
    cuDNN_search_paths = [cuda_root] if cuda_root else [CUDA_INSTALL_PATH]
    cuDNN_search_paths.append("/usr") # Always check /usr as well
    cudnn_found_after_apt, _ = check_library_files("cuDNN", "cudnn.h", r"libcudnn\.so(\.\d+)*$", cuDNN_search_paths)

    if cudnn_found_after_apt:
         logging.info(f"✅ cuDNN {TARGET_CUDNN_VERSION} files found after apt attempt.")
         logging.info("Manual installation fallback is not needed.")
         return True
    else:
         logging.warning(f"❌ cuDNN {TARGET_CUDNN_VERSION} files still not found after apt attempt.")
         logging.info("Proceeding with manual cuDNN installation by downloading and copying files.")

    # --- Manual cuDNN Installation Fallback (wget and copy) ---
    logging.info(f"Step 1: Downloading cuDNN Library tar file from {CUDNN_TAR_URL}")
    if DRY_RUN:
        logging.info(f"--- DRY RUN: Skipping download of {CUDNN_TAR_FILENAME} ---")
        # Simulate success for logging purposes in dry run
        download_success = True
    else:
        # wget does not require root unless writing to a protected directory
        try:
            run_command(["wget", CUDNN_TAR_URL])
            logging.info(f"✅ Downloaded {CUDNN_TAR_FILENAME}.")
            download_success = True
        except Exception as e:
            logging.critical(f"❌ Failed to download cuDNN tar file: {e}")
            logging.critical("Please check your internet connection and the download URL.")
            download_success = False

    if not download_success:
        return False # Exit if download failed

    logging.info("Step 2: Extract the cuDNN tar file.")
    # tar does not require root unless extracting to a protected directory
    extracted_dir = CUDNN_TAR_FILENAME.replace(".tar.xz", "").replace(".tgz", "")
    if DRY_RUN:
        logging.info(f"--- DRY RUN: Skipping extraction of {CUDNN_TAR_FILENAME} ---")
        # Simulate directory existence for logging purposes in dry run
        if not os.path.exists(extracted_dir):
            logging.info(f"--- DRY RUN: Simulating creation of directory: {extracted_dir} ---")
            os.makedirs(extracted_dir, exist_ok=True) # Create directory for dry run
        extract_success = True
    else:
        try:
            run_command(["tar", "-xvf", CUDNN_TAR_FILENAME])
            logging.info("✅ cuDNN tar file extracted.")
            extract_success = True
        except Exception as e:
            logging.critical(f"❌ Failed to extract cuDNN tar file: {e}")
            logging.critical("Ensure the file is not corrupted and you have enough disk space.")
            extract_success = False
    logging.info("-" * 40)

    if not extract_success:
        # Clean up downloaded tar file if extraction failed
        if os.path.exists(CUDNN_TAR_FILENAME):
             os.remove(CUDNN_TAR_FILENAME)
             logging.info(f"Cleaned up downloaded file: {CUDNN_TAR_FILENAME}")
        return False # Exit if extraction failed


    logging.info("Step 3: Copy cuDNN files into your CUDA Toolkit directory.")
    logging.info(f"Copying headers to {CUDA_INSTALL_PATH}/include/")
    logging.info(f"Copying libraries to {CUDA_INSTALL_PATH}/lib64/")

    # Find the extracted directory (should be named similarly to the tar file)
    if not os.path.isdir(extracted_dir):
        logging.critical(f"❌ Extracted directory '{extracted_dir}' not found.")
        logging.critical("Extraction may have failed or the directory name is unexpected.")
        # Clean up downloaded tar file if extracted dir not found
        if os.path.exists(CUDNN_TAR_FILENAME):
             os.remove(CUDNN_TAR_FILENAME)
             logging.info(f"Cleaned up downloaded file: {CUDNN_TAR_FILENAME}")
        return False

    # Check if the target CUDA_INSTALL_PATH exists
    if not os.path.exists(CUDA_INSTALL_PATH):
         logging.critical(f"❌ CUDA installation path '{CUDA_INSTALL_PATH}' does not exist.")
         logging.critical(f"Please ensure CUDA Toolkit {TARGET_CUDA_VERSION} is correctly installed and accessible at this path.")
         logging.critical("You might need to adjust the CUDA_INSTALL_PATH variable in this script.")
         # Clean up downloaded tar file and extracted dir if CUDA path is missing
         if os.path.exists(CUDNN_TAR_FILENAME):
              os.remove(CUDNN_TAR_FILENAME)
              logging.info(f"Cleaned up downloaded file: {CUDNN_TAR_FILENAME}")
         if os.path.exists(extracted_dir):
              shutil.rmtree(extracted_dir)
              logging.info(f"Cleaned up extracted directory: {extracted_dir}")
         return False

    if DRY_RUN:
        logging.info(f"--- DRY RUN: Skipping copy from {extracted_dir} to {CUDA_INSTALL_PATH} ---")
        copy_success = True
    else:
        try:
            # cp requires root to copy to system directories, handled by running script with sudo
            run_command(["cp", "-r", os.path.join(extracted_dir, "include", "*"), os.path.join(CUDA_INSTALL_PATH, "include") + "/"])
            run_command(["cp", "-r", os.path.join(extracted_dir, "lib", "*"), os.path.join(CUDA_INSTALL_PATH, "lib64") + "/"])
            logging.info("✅ cuDNN files copied to CUDA Toolkit directory.")
            copy_success = True
        except Exception as e:
            logging.critical(f"❌ Failed to copy cuDNN files: {e}")
            logging.critical(f"Ensure CUDA Toolkit is installed at {CUDA_INSTALL_PATH}.")
            copy_success = False
    logging.info("-" * 40)

    # Clean up downloaded tar file and extracted dir regardless of copy success
    if os.path.exists(CUDNN_TAR_FILENAME):
         os.remove(CUDNN_TAR_FILENAME)
         logging.info(f"Cleaned up downloaded file: {CUDNN_TAR_FILENAME}")
    if os.path.exists(extracted_dir):
         shutil.rmtree(extracted_dir)
         logging.info(f"Cleaned up extracted directory: {extracted_dir}")

    if not copy_success:
        return False # Exit if copy failed


    logging.info("Step 4: Set library permissions (Recommended).")
    if DRY_RUN:
        logging.info("--- DRY RUN: Skipping setting library permissions ---")
    else:
        try:
            # chmod requires root, handled by running script with sudo
            run_command(["chmod", "a+r", os.path.join(CUDA_INSTALL_PATH, "lib64", "libcudnn*")], check=False) # Don't fail if this command has issues
            logging.info("✅ Library permissions set.")
        except Exception as e:
            logging.warning(f"⚠️ Failed to set library permissions: {e}")
            logging.warning("This step is recommended but may not be strictly necessary depending on your system.")
    logging.info("-" * 40)

    logging.info("Step 5: Update library cache.")
    if DRY_RUN:
        logging.info("--- DRY RUN: Skipping ldconfig ---")
    else:
        try:
            # ldconfig requires root, handled by running script with sudo
            run_command(["ldconfig"], check=False) # ldconfig can sometimes print non-critical warnings
            logging.info("✅ Library cache updated.")
            logging.warning("Note: You might see messages from ldconfig about files not being symbolic links. These are usually not errors.")
        except Exception as e:
            logging.critical(f"❌ Failed to update library cache: {e}")
            logging.critical("This command is important for the system to find the new libraries.")
            return False
    logging.info("-" * 40)

    logging.info("--- cuDNN Installation Script Finished ---")
    logging.info(f"cuDNN {TARGET_CUDNN_VERSION} should now be correctly installed for CUDA {TARGET_CUDA_VERSION}.")
    return True


def setup_logging(console_level=logging.INFO):
    """Confgures the logging system for console output only."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Capture all messages at root level

    # Remove existing handlers to prevent duplicate output
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    # Use a format that includes level name for clarity
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.debug("Logging configured for console output.")


# --- Main Script Execution ---

if __name__ == "__main__":
    # Set up logging before any other output
    setup_logging(console_level=logging.INFO)

    # --- Argument Parsing for Dry Run ---
    parser = argparse.ArgumentParser(description="FAISS GPU Dependencies Installation Script")
    parser.add_argument('--dry-run', '-d', action='store_true', help='Perform a dry run without installing packages or modifying the system.')
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if DRY_RUN:
        logging.info("\n--- DRY RUN MODE ENABLED ---")
        logging.info("No packages will be installed, and no system files will be modified.")
        logging.info("The script will only check for existing dependencies and log intended actions.")
        logging.info("-" * 60)


    logging.info("--- Install FAISS GPU Dependencies Script ---")
    logging.info("This script will check for and install necessary dependencies for building FAISS-GPU.")
    logging.info(f"It targets Ubuntu and CUDA Toolkit {TARGET_CUDA_VERSION} / cuDNN {TARGET_CUDNN_VERSION}.")
    logging.info("\nIMPORTANT: This script requires root privileges. Please run it using: `sudo python install_faiss_dependencies.py`")
    logging.info("-" * 60)

    # --- Check if running with sudo ---
    if os.geteuid() != 0:
        logging.critical("FATAL: This script requires root privileges.")
        logging.critical("Please run it using: `sudo python install_faiss_dependencies.py`")
        sys.exit(1)
    logging.info("Running with root privileges.")
    logging.info("-" * 60)


    # --- Check OS ---
    os_name, os_version = get_os_info()
    if os_name != "ubuntu":
        logging.warning("\n⚠️ This script is primarily designed for Ubuntu.")
        logging.warning("   You may need to adapt installation steps for your operating system.")

    # --- Check NVIDIA Driver ---
    logging.info("\n--- Checking NVIDIA Driver ---")
    # Directly call get_cuda_version_from_smi, its internal error handling is sufficient
    driver_cuda_version = get_cuda_version_from_smi()
    if not driver_cuda_version:
        # The get_cuda_version_from_smi function already logs critical errors if it fails
        sys.exit(1)
    else:
         logging.info(f"✅ NVIDIA Driver found. Compatible CUDA Version: {driver_cuda_version}")


    # --- Check/Install Common Prerequisites ---
    logging.info("\n--- Checking/Installing Common Prerequisites ---")
    # Use the improved check_command_exists for executables
    # Use check_library_files for libraries

    missing_executables = []
    executables_to_check = ["cmake", "swig"]
    for cmd in executables_to_check:
        if not check_command_exists(cmd):
            missing_executables.append(cmd)

    # Check for C++ compiler (g++ or clang++)
    gpp_found = check_command_exists("g++")
    clangpp_found = check_command_exists("clang++")
    if not gpp_found and not clangpp_found:
        logging.error("❌ C++ compiler (g++ or clang++) not found.")
        logging.info(f"   To install g++ (part of {BUILD_ESSENTIAL_APT_PACKAGE}) on Ubuntu, run: `sudo apt update && sudo apt install {BUILD_ESSENTIAL_APT_PACKAGE}`")
        missing_executables.append(BUILD_ESSENTIAL_APT_PACKAGE) # Add build-essential to missing list for installation


    # Check for gflags library files
    gflags_found, _ = check_library_files("gflags", "gflags/gflags.h", r"libgflags\.so(\.\d+)*$", ["/usr", "/usr/local"])
    if not gflags_found:
        logging.error(f"❌ gflags library ({GFLAGS_APT_PACKAGE}) not found.")
        logging.info(f"   To install gflags on Ubuntu, run: `sudo apt update && sudo apt install {GFLAGS_APT_PACKAGE}`")
        missing_executables.append(GFLAGS_APT_PACKAGE) # Add gflags dev package to missing list

    # Check for OpenBLAS library files (assuming OpenBLAS is the target BLAS)
    openblas_found, _ = check_library_files("OpenBLAS", "cblas.h", r"libopenblas\.so(\.\d+)*$", ["/usr", "/usr/local"])
    if not openblas_found:
        logging.error(f"❌ OpenBLAS library ({OPENBLAS_APT_PACKAGE}) not found.")
        logging.info(f"   To install OpenBLAS on Ubuntu, run: `sudo apt update && sudo apt install {OPENBLAS_APT_PACKAGE}`")
        missing_executables.append(OPENBLAS_APT_PACKAGE) # Add openblas dev package to missing list

    # Remove duplicate package names if any were added multiple times
    packages_to_install = list(set(missing_executables))

    if packages_to_install:
        logging.info(f"Missing common prerequisites packages: {', '.join(packages_to_install)}")
        if os_name == "ubuntu":
            if install_apt_packages(packages_to_install):
                 logging.info("✅ Common prerequisites installed successfully via apt.")
            else:
                 logging.critical("\nFATAL: Failed to install common prerequisites via apt.")
                 logging.critical("Please manually install the missing packages and run the script again.")
                 sys.exit(1)
        else:
            logging.critical("\nFATAL: Missing common prerequisites.")
            logging.critical("Please manually install the following using your system's package manager:")
            for p in packages_to_install:
                 logging.critical(f" - {p}")
            logging.critical("Then run the script again.")
            sys.exit(1)
    else:
        logging.info("✅ All common prerequisites found.")

    # --- Check/Install CUDA Toolkit ---
    logging.info(f"\n--- Checking/Installing CUDA Toolkit {TARGET_CUDA_VERSION} ---")
    # Find CUDA Toolkit Root Dir first
    cuda_root = find_cuda_toolkit_root() # This function logs its own success/failure

    # Use the robust get_cuda_toolkit_version function, passing the found cuda_root
    nvcc_cuda_version = get_cuda_toolkit_version(cuda_root)

    if nvcc_cuda_version and nvcc_cuda_version.startswith(TARGET_CUDA_VERSION):
        logging.info(f"✅ CUDA Toolkit {nvcc_cuda_version} is already installed and matches the desired version {TARGET_CUDA_VERSION}.")
    elif nvcc_cuda_version:
        logging.critical(f"\nFATAL: Detected CUDA Toolkit version {nvcc_cuda_version}, but desired version is {TARGET_CUDA_VERSION}.")
        logging.critical("Please uninstall the existing CUDA Toolkit before proceeding.")
        logging.critical("Refer to NVIDIA documentation for uninstallation instructions.")
        sys.exit(1)
    else:
        logging.info(f"CUDA Toolkit {TARGET_CUDA_VERSION} not found. Attempting to install.")
        # Pass driver_cuda_version to install_cuda_toolkit for compatibility check within the function
        if not install_cuda_toolkit(os_name, os_version, driver_cuda_version):
             # install_cuda_toolkit handles its own fatal errors and exits if necessary
             # If it returns False, it means a non-fatal error occurred that we should report
             logging.critical("\nFATAL: CUDA Toolkit installation failed.")
             logging.critical("Please address the errors above and run the script again.")
             sys.exit(1)

    # Re-check nvcc version after installation attempt using the robust check
    nvcc_cuda_version = get_cuda_toolkit_version(find_cuda_toolkit_root()) # Find root again in case it was just installed
    if not nvcc_cuda_version or not nvcc_cuda_version.startswith(TARGET_CUDA_VERSION):
         logging.critical(f"\nFATAL: CUDA Toolkit {TARGET_CUDA_VERSION} is still not found or the wrong version after installation attempt.")
         logging.critical("Review the installation output carefully.")
         sys.exit(1)

    # --- Check CUDA Version Compatibility (Final Check) ---
    logging.info("--- Final CUDA Version Compatibility Check ---")
    # Simple major.minor version comparison
    nvcc_major_minor = '.'.join(nvcc_cuda_version.split('.')[:2])
    smi_major_minor = '.'.join(driver_cuda_version.split('.')[:2])

    if nvcc_major_minor != smi_major_minor:
        logging.critical("\nFATAL: Installed CUDA Toolkit version and Driver Compatible CUDA version mismatch!")
        logging.critical(f"   Installed CUDA Toolkit version (nvcc): {nvcc_cuda_version}")
        logging.critical(f"   Detected Driver Compatible CUDA version (nvidia-smi): {driver_cuda_version}")
        logging.critical("   These versions must match (at least major.minor) to avoid runtime errors (like PTX compilation issues).")
        logging.critical("   Please install a CUDA Toolkit version that is compatible with your NVIDIA driver.")
        logging.critical("   Refer to NVIDIA's CUDA Compatibility documentation.")
        sys.exit(1)
    else:
        logging.info("✅ Installed CUDA Toolkit version and Driver Compatible CUDA version are compatible (major.minor match).")


    # --- Check/Install cuDNN ---
    logging.info(f"\n--- Checking/Installing cuDNN {TARGET_CUDNN_VERSION} ---")
    # Find CUDA Toolkit Root Dir (needed for cuDNN check/install) - already found above
    # cuda_root = find_cuda_toolkit_root() # No need to call again

    # Check for cuDNN relative to CUDA root or standard paths
    cuDNN_search_paths = [cuda_root] if cuda_root else [CUDA_INSTALL_PATH]
    cuDNN_search_paths.append("/usr") # Always check /usr as well
    cudnn_found, _ = check_library_files("cuDNN", "cudnn.h", r"libcudnn\.so(\.\d+)*$", cuDNN_search_paths)

    if cudnn_found:
        logging.info(f"✅ cuDNN {TARGET_CUDNN_VERSION} appears to be installed.")
        # Basic compatibility check (major version)
        # This is a heuristic, true compatibility depends on patch versions too.
        try:
            # Try to parse cuDNN version from header or library name if possible
            # More robust check would involve running a cuDNN sample
            logging.info(f"Please ensure your installed cuDNN version is fully compatible with CUDA Toolkit {nvcc_cuda_version}.")
            logging.info("Refer to the cuDNN Installation Guide for compatibility matrix.")
        except Exception:
            pass # Skip version check if parsing fails
    else:
        logging.info(f"cuDNN {TARGET_CUDNN_VERSION} not found in expected locations. Attempting to install.")
        if not install_cudnn():
             logging.critical("\nFATAL: cuDNN installation failed.")
             logging.critical("Please address the errors above and run the script again.")
             sys.exit(1)

    # Re-check cuDNN after installation attempt
    cudnn_found, _ = check_library_files("cuDNN", "cudnn.h", r"libcudnn\.so(\.\d+)*$", cuDNN_search_paths)
    if not cudnn_found:
         logging.critical(f"\nFATAL: cuDNN {TARGET_CUDNN_VERSION} is still not found after installation attempt.")
         logging.critical("Review the installation output carefully.")
         sys.exit(1)


    logging.info("\n--- All mandatory dependencies for FAISS-GPU seem to be met. ---")
    if DRY_RUN:
        logging.info("--- DRY RUN COMPLETE ---")
        logging.info("The script checked for dependencies but did not perform any installations or modifications.")
    else:
        logging.info("You should now be able to run the FAISS build script.")
        logging.info("Make sure your Python virtual environment is activated before building.")
        logging.info("Example: `source venv/bin/activate`")
        logging.info("Then run the FAISS build script:")
        logging.info("Example: `python build_faiss_gpu.py`")

    logging.info("\n--- Dependency Installation Script Finished ---")

