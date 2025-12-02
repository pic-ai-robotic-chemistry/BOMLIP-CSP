
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode.  
nvidia-smi -i 1 -c EXCLUSIVE_PROCESS # Set GPU 1 to exclusive mode.  
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS # Set GPU 2 to exclusive mode.  
nvidia-smi -i 3 -c EXCLUSIVE_PROCESS # Set GPU 3 to exclusive mode.  
nvidia-smi -i 4 -c EXCLUSIVE_PROCESS # Set GPU 4 to exclusive mode.  
nvidia-smi -i 5 -c EXCLUSIVE_PROCESS # Set GPU 5 to exclusive mode.  
nvidia-smi -i 6 -c EXCLUSIVE_PROCESS # Set GPU 6 to exclusive mode.  
nvidia-smi -i 7 -c EXCLUSIVE_PROCESS # Set GPU 7 to exclusive mode.  
nvidia-cuda-mps-control -d # Start the daemon.