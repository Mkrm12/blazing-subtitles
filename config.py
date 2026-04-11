"""
config.py — Single source of truth for the entire pipeline.
"""

import os

# --- PATHS ---
WORKSPACE_ROOT = "/teamspace/studios/this_studio"
DATA_DIR       = os.path.join(WORKSPACE_ROOT, "data")
OUTPUT_DIR     = os.path.join(WORKSPACE_ROOT, "output")
CACHE_DIR      = os.path.join(WORKSPACE_ROOT, "cache")
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "checkpoints")

for d in [DATA_DIR, OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

VIDEO_FILENAME = "Blazing Teens_03.mp4"
VIDEO_PATH     = os.path.join(DATA_DIR, VIDEO_FILENAME)

# --- CACHE MANAGEMENT ---
FORCE_CLEAR_CHECKPOINTS = False

# --- TIMELINE & PROCESSING KNOBS ---
MAX_MINUTES                = 21.5
SKIP_INTRO_SECONDS         = 85.0  

# --- OCR MASTER CLOCK SETTINGS ---
OCR_SWEEP_FPS              = 2.0   # Scans the screen 2 times per second
OCR_SIMILARITY_THRESH      = 0.60  # If text is 60% similar, keep extending the block

# --- THE DUAL-BOX SPATIAL CROPS ---
# Box 1: Environmental / Loose Box (Catches location names + subtitles)
OCR_ENV_TOP                = 0.65  
OCR_ENV_BOTTOM             = 0.90  
OCR_ENV_LEFT               = 0.10  
OCR_ENV_RIGHT              = 0.90  

# Box 2: Subtitle Zoom Box (Strictly isolates the burnt-in subtitles)
OCR_SUB_TOP                = 0.72  
OCR_SUB_BOTTOM             = 0.86  
OCR_SUB_LEFT               = 0.12  
OCR_SUB_RIGHT              = 0.88  

OCR_LANGUAGES              = ['ch_sim', 'en']

# --- WHISPER ---
VAD_MIN_SILENCE_MS         = 300   
VAD_SPEECH_PAD_MS          = 100   

# --- MODELS ---
WHISPER_MODEL_SIZE   = "large-v3"
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LANGUAGE     = "zh"        
WHISPER_TRANSLATE    = False

FLORENCE_MODEL_ID       = "microsoft/Florence-2-base"
DIARIZATION_MODEL_ID    = "pyannote/speaker-diarization-3.1"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

OUTPUT_FILENAME = "BlazingTeens_CloudBrain_Dump.txt"
OUTPUT_PATH     = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "pipeline_checkpoint.json")