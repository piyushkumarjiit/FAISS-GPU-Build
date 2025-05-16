# faiss_gpu_test.py
# This script tests the basic functionality of FAISS with GPU support.
# It attempts to import faiss, check GPU availability, create an index,
# move it to the GPU, and perform a simple add and search operation.

import sys
import time
import numpy as np
import logging

# --- Logging Setup ---
# Configure logging to show INFO level messages and above in the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
# ---------------------

logging.info("--- FAISS GPU Test Script ---")

try:
    logging.info("Step 1: Attempting to import faiss...")
    import faiss
    logging.info("✅ Successfully imported faiss.")

    logging.info(f"FAISS Version: {faiss.__version__}")

    logging.info("Step 2: Checking for GPU availability...")
    num_gpus = faiss.get_num_gpus()
    logging.info(f"Detected {num_gpus} GPUs.")

    if num_gpus == 0:
        logging.warning("⚠️ No GPUs detected by FAISS. The build might not have correctly enabled GPU support, or there's an issue with the CUDA/driver setup.")
        logging.warning("   Proceeding with CPU-only test (if possible).")
        gpu_available = False
    else:
        logging.info("✅ GPUs detected. Proceeding with GPU test.")
        gpu_available = True

    # --- Basic Index Operations (CPU) ---
    logging.info("\nStep 3: Creating a simple IndexFlatL2 index (on CPU)...")
    dimension = 128  # Dimension of vectors
    num_vectors = 100 # Number of vectors to add
    index_cpu = faiss.IndexFlatL2(dimension)
    logging.info(f"✅ Created IndexFlatL2 index on CPU (is_trained={index_cpu.is_trained}, ntotal={index_cpu.ntotal}).")

    # Generate some random vectors
    np.random.seed(1234) # for reproducibility
    vectors = np.random.random((num_vectors, dimension)).astype('float32')
    logging.info(f"Generated {num_vectors} random vectors.")

    logging.info("Step 4: Adding vectors to the CPU index...")
    index_cpu.add(vectors)
    logging.info(f"✅ Added vectors to CPU index (ntotal={index_cpu.ntotal}).")

    # --- GPU Specific Test ---
    if gpu_available:
        logging.info("\nStep 5: Attempting to move the index to the GPU...")
        try:
            # Use the first available GPU (index 0)
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index_cpu)
            logging.info(f"✅ Successfully moved index to GPU (on device 0).")
            logging.info(f"GPU Index: is_trained={gpu_index.is_trained}, ntotal={gpu_index.ntotal}")

            logging.info("Step 6: Performing a simple search on the GPU index...")
            k = 5 # Number of nearest neighbors to search for
            # Use the first vector as a query
            query_vector = vectors[0:1]

            t0 = time.time()
            distances, indices = gpu_index.search(query_vector, k)
            t1 = time.time()

            logging.info(f"✅ Search on GPU index completed in {t1 - t0:.4f} seconds.")
            logging.info(f"Nearest neighbor indices for the first query vector: {indices[0]}")
            logging.info(f"Corresponding distances: {distances[0]}")

        except Exception as e:
            logging.error(f"❌ Error during GPU specific test: {e}", exc_info=True)
            logging.error("This indicates a problem with the FAISS GPU build or CUDA setup.")
            logging.error("Review the build logs (DEBUG level) for GPU/SWIG/linking errors.")

    else:
        logging.info("\nSkipping GPU specific test as no GPUs were detected by FAISS.")
        # Optionally perform a search on the CPU index if no GPU is available
        logging.info("Step 5 (CPU): Performing a simple search on the CPU index...")
        k = 5
        query_vector = vectors[0:1]
        t0 = time.time()
        distances, indices = index_cpu.search(query_vector, k)
        t1 = time.time()
        logging.info(f"✅ Search on CPU index completed in {t1 - t0:.4f} seconds.")
        logging.info(f"Nearest neighbor indices for the first query vector: {indices[0]}")
        logging.info(f"Corresponding distances: {distances[0]}")


except ImportError:
    logging.critical("FATAL: Failed to import faiss.")
    logging.critical("This indicates the FAISS Python package was not installed correctly.")
    logging.critical("Review the output of 'Step 6: Installing Python Bindings' from the build script.")
except AttributeError as ae:
    logging.critical(f"FATAL: AttributeError during faiss operation: {ae}")
    logging.critical("This often indicates a mismatch between the SWIG-generated Python code and the compiled C++ library.")
    logging.critical("Review the build logs (DEBUG level) for SWIG or linking errors.")
    logging.critical("Consider trying a different SWIG version or a specific FAISS commit known to be compatible.")
except Exception as e:
    logging.critical(f"FATAL: An unexpected error occurred during the test: {e}", exc_info=True)
    logging.critical("Review the full error traceback above.")

logging.info("\n--- FAISS GPU Test Script Finished ---")
