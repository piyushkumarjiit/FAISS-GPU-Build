Script to build FAISS-GPU from source.

Currently (at the time of writing) it is not possible to install faiss-gpy in venv by executing `pip install faiss-gpu`, so I ended up building it from source for my use.
As it was not straight forward, I created these scripts in case someone else is in the same boat.
These work for Ubuntu 24.04 with a working Nvidia 1080TI. Please note `nvidia-smi` command needs to be working before any of these scripts can be executed.

There are 3 scripts in total:
install_faiss_dependencies.py --> checks and tries to install dependencies (to the best of its abilities).
build_faiss_gpu.py --> does the actual build of faiss from source (assuming all dependeicies are met).
faiss_gpu_test.py --> runs a set of basic test to confirm that faiss is able to use gpu.

These scripts were created and tested with below Nvidia dependecies :
`nvidia-smi` --> NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4 
`nvcc --version` --> Cuda compilation tools, release 12.4, V12.4.131     Build cuda_12.4.r12.4/compiler.34097967_0
`ls /usr/local/cuda/lib64/libcudnn* | grep -oP 'libcudnn(_\w+)?\.so\.\K\d+\.\d+\.\d+' | cut -d. -f1-2 | head -n 1` --> 9.10

