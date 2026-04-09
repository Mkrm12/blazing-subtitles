#!/bin/bash
# =============================================================================
# run.sh — Execute the pipeline on Lightning AI
# =============================================================================
set -e

# =============================================================================
# 1. HF TOKEN — Load securely from .env file
# =============================================================================
echo ">>> Loading environment variables..."

if [ -f .env ]; then
    # This command safely reads the .env file and exports the variables
    export $(grep -v '^#' .env | xargs)
else
    echo "FATAL ERROR: .env file not found."
    echo "Please create a .env file in the root directory and add: HF_TOKEN=your_token"
    exit 1
fi

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "hf_your_actual_token_goes_here" ]; then
    echo "FATAL ERROR: HF_TOKEN is empty or invalid in the .env file."
    exit 1
fi

echo ">>> HF_TOKEN successfully loaded securely."

# =============================================================================
# 2. CUDA LIBRARY PATH — defensive multi-strategy detection
# Fixes: libcudnn_ops_infer.so.8 cannot open shared object file
# Tries nvidia pip packages first, then system CUDA paths, then skips safely
# =============================================================================
echo ">>> Configuring CUDA library paths..."

CUDA_LIB_PATHS=""

# Strategy A: nvidia pip packages (installed with torch)
CUBLAS_PATH=$(python -c "
try:
    import nvidia.cublas.lib, os
    print(os.path.dirname(nvidia.cublas.lib.__file__))
except: pass
" 2>/dev/null)

CUDNN_PATH=$(python -c "
try:
    import nvidia.cudnn.lib, os
    print(os.path.dirname(nvidia.cudnn.lib.__file__))
except: pass
" 2>/dev/null)

CUSOLVER_PATH=$(python -c "
try:
    import nvidia.cusolver.lib, os
    print(os.path.dirname(nvidia.cusolver.lib.__file__))
except: pass
" 2>/dev/null)

[ -n "$CUBLAS_PATH" ]   && CUDA_LIB_PATHS="$CUBLAS_PATH:$CUDA_LIB_PATHS"
[ -n "$CUDNN_PATH" ]    && CUDA_LIB_PATHS="$CUDNN_PATH:$CUDA_LIB_PATHS"
[ -n "$CUSOLVER_PATH" ] && CUDA_LIB_PATHS="$CUSOLVER_PATH:$CUDA_LIB_PATHS"

# Strategy B: System CUDA paths (Lightning AI has these)
for sys_path in \
    "/usr/local/cuda/lib64" \
    "/usr/lib/x86_64-linux-gnu" \
    "/usr/local/cuda-12.1/lib64" \
    "/usr/local/cuda-12/lib64"; do
    [ -d "$sys_path" ] && CUDA_LIB_PATHS="$sys_path:$CUDA_LIB_PATHS"
done

# Strategy C: ctranslate2's bundled libs (most reliable for whisper)
CT2_LIB=$(python -c "
try:
    import ctranslate2, os
    print(os.path.dirname(ctranslate2.__file__))
except: pass
" 2>/dev/null)
[ -n "$CT2_LIB" ] && CUDA_LIB_PATHS="$CT2_LIB:$CUDA_LIB_PATHS"

# Apply if we found anything
if [ -n "$CUDA_LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo ">>> LD_LIBRARY_PATH configured with CUDA libs"
else
    echo ">>> WARNING: No extra CUDA lib paths found, relying on system defaults"
fi

# =============================================================================
# 3. Suppress non-fatal warnings that pollute logs
# =============================================================================
export TOKENIZERS_PARALLELISM=false
export ORT_LOGGING_LEVEL=3              # suppress onnxruntime GPU discovery warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# =============================================================================
# 4. Run
# =============================================================================
cd /teamspace/studios/this_studio
echo ">>> Launching pipeline..."
echo ""
python pipeline.py