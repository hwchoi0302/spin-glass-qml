# cupy-cuda12x needs libcublas.so.12, but this box's site-packages carries the
# CUDA 13 wheels (nvidia-cublas 13.1.1.3) that torch pulled in, so cupy's BLAS
# calls -- xp.vdot in 03i_gpu_benchmark.py part B -- die with
# "ImportError: libcublas.so.12: cannot open shared object file".
#
# The CUDA 12 wheels are installed OUT OF TREE so they cannot shadow the
# CUDA 13 ones torch links against:
#   pip install --target ~/.local/cuda12-libs nvidia-cublas-cu12 \
#       nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cufft-cu12 \
#       nvidia-curand-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12 \
#       nvidia-cuda-runtime-cu12
#
# Source this before any cupy run:  source scripts/cuda12_env.sh
_C12=$HOME/.local/cuda12-libs/nvidia
export LD_LIBRARY_PATH="$_C12/cublas/lib:$_C12/cusolver/lib:$_C12/cusparse/lib:$_C12/cufft/lib:$_C12/curand/lib:$_C12/nvjitlink/lib:$_C12/cuda_nvrtc/lib:$_C12/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _C12
