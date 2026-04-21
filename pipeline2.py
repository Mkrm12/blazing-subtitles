"""
pipeline2.py — Automated Factory Extraction Pipeline
Dual-Box OCR Master Clock Architecture (Ultra v16 - Holy Ground & Dynamic Cropping)
Features: FFMPEG Car Wash, Storage Isolation, Auto-Cleanup, VRAM Memory Armor
"""

import cv2
import os
import gc
import json
import datetime
import subprocess
import sys
import torch
import numpy as np
import re
import difflib
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config

# ============================================================
# UTILITIES & CONFIG OVERRIDES
# ============================================================

HARD_CUTOFF_SECONDS = config.MAX_MINUTES * 60.0 

def log(msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def format_time(seconds: float) -> str:
    td = datetime.timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    mins, secs = divmod(rem, 60)
    
    ms = round((td.total_seconds() - total_seconds) * 1000)
    if ms >= 1000:
        ms = 0
        secs += 1
    return f"{hours:02}:{mins:02}:{secs:02},{ms:03}"

def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def save_checkpoint(phase: str, data):
    checkpoint = {}
    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            with open(config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
        except:
            pass
    checkpoint[phase] = data
    with open(config.CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    log(f"Checkpoint saved: {phase}")

def load_checkpoint(phase: str):
    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            with open(config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            if phase in checkpoint:
                log(f"Checkpoint found: {phase} — skipping recompute.")
                return checkpoint[phase]
        except:
            pass
    return None

def extract_audio_wav(video_path: str, output_path: str):
    cmd = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn"]
    cmd += ["-t", str(HARD_CUTOFF_SECONDS)]
    cmd += [output_path, "-y"]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode(errors='replace')}")

def ffmpeg_car_wash(input_path: str, output_path: str):
    if os.path.exists(output_path):
        log(f"Washed video already exists at {output_path}. Skipping wash.")
        return
    
    log(f"Starting FFMPEG Car Wash for {input_path}...")
    cmd = [
        "ffmpeg", "-err_detect", "ignore_err", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy", "-y", output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFMPEG Car Wash failed:\n{result.stderr.decode(errors='replace')}")
    log("Car Wash complete. Video index rebuilt perfectly.")

def validate_environment():
    errors = []
    if config.FORCE_CLEAR_CHECKPOINTS and os.path.exists(config.CHECKPOINT_FILE):
        os.remove(config.CHECKPOINT_FILE)
        log(">>> FORCE_CLEAR_CHECKPOINTS is ON. Old cache nuked for this episode.")

    if not config.HF_TOKEN:
        errors.append("HF_TOKEN not set.")
    if not torch.cuda.is_available():
        errors.append("CUDA not available.")
    if errors:
        for e in errors:
            log(f"FATAL: {e}")
        sys.exit(1)

# ============================================================
# PHASE 1 & 2: AUDIO & SPEAKERS
# ============================================================

def phase1_transcribe() -> list:
    cached = load_checkpoint("phase1_transcription")
    if cached: return cached
    log("─" * 50)
    log("PHASE 1: Audio Transcription (Background Data)")
    
    from faster_whisper import WhisperModel
    
    try:
        model = WhisperModel(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE, download_root=config.CACHE_DIR)
    except Exception:
        log("Hugging Face timeout detected. Booting Whisper from local offline cache...")
        model = WhisperModel(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE, download_root=config.CACHE_DIR, local_files_only=True)
        
    vad_params = dict(
        min_silence_duration_ms=getattr(config, 'VAD_MIN_SILENCE_MS', 300),
        speech_pad_ms=getattr(config, 'VAD_SPEECH_PAD_MS', 100)
    )
    
    segments, _ = model.transcribe(
        config.VIDEO_PATH, 
        language=config.WHISPER_LANGUAGE, 
        vad_filter=True, 
        vad_parameters=vad_params,
        condition_on_previous_text=False
    )
    
    results = []
    for seg in segments:
        if seg.end <= config.SKIP_INTRO_SECONDS: continue
        start_time = max(seg.start, config.SKIP_INTRO_SECONDS)
        if start_time >= HARD_CUTOFF_SECONDS: break
        end_time = min(seg.end, HARD_CUTOFF_SECONDS)
        results.append({"start": round(start_time, 3), "end": round(end_time, 3), "text": seg.text.strip()})
        
    del model
    flush_gpu()
    save_checkpoint("phase1_transcription", results)
    return results

def phase2_diarize() -> list:
    cached = load_checkpoint("phase2_diarization")
    if cached: return cached
    
    log("─" * 50)
    log("PHASE 2: Speaker Diarization")

    from pyannote.audio import Pipeline
    audio_path = config.DIARIZATION_AUDIO_PATH
    if not os.path.exists(audio_path):
        log(f"Extracting WAV to {audio_path}...")
        extract_audio_wav(config.VIDEO_PATH, audio_path)

    log("Loading pyannote/speaker-diarization-3.1...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.HF_TOKEN, cache_dir=config.CACHE_DIR)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        log("FATAL: Pyannote failed to load.")
        raise e

    log("Processing audio for speaker turns...")
    diarization = pipeline(audio_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end <= config.SKIP_INTRO_SECONDS: continue
        start_time = max(turn.start, config.SKIP_INTRO_SECONDS)
        if start_time >= HARD_CUTOFF_SECONDS: break
        end_time = min(turn.end, HARD_CUTOFF_SECONDS)
        results.append({"start": round(start_time, 3), "end": round(end_time, 3), "speaker": speaker})

    # Strict Memory Wipe for Pyannote
    del pipeline
    del diarization
    flush_gpu()
    
    save_checkpoint("phase2_diarization", results)
    return results

# ============================================================
# PHASE 3: THE OCR DUAL-BOX MASTER CLOCK (V16 HOLY GROUND)
# ============================================================

def phase3_ocr_master_clock() -> list:
    log("─" * 50)
    log("PHASE 3: OCR Master Clock (v16 Holy Ground & Dynamic Crop)")
    cached = load_checkpoint("phase3_ocr_timeline")
    if cached: return cached

    import easyocr
    from paddleocr import PaddleOCR
    import logging

    logging.getLogger("ppocr").setLevel(logging.ERROR)

    log("Booting EasyOCR and PaddleOCR...")
    reader_env = easyocr.Reader(config.OCR_LANGUAGES, gpu=True, model_storage_directory=config.CACHE_DIR, verbose=False)
    reader_sub = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cutoff_frames = int(HARD_CUTOFF_SECONDS * fps)
    max_frames = min(total_frames, cutoff_frames)
    start_frame = int(config.SKIP_INTRO_SECONDS * fps)
    frame_step = int(fps / config.OCR_SWEEP_FPS) 
    
    ocr_timeline = []
    current_block = None

    log(f"Sweeping video at {config.OCR_SWEEP_FPS} FPS...")
    for f in tqdm(range(start_frame, max_frames, frame_step), desc="OCR Sweep"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        current_time = round(f / fps, 3)
        h, w = frame.shape[:2]
        
        current_env_left = max(config.OCR_ENV_LEFT, 0.25) if current_time <= 150.0 else config.OCR_ENV_LEFT
        current_sub_left = max(config.OCR_SUB_LEFT, 0.25) if current_time <= 150.0 else config.OCR_SUB_LEFT
        
        env_crop = frame[
            int(h * config.OCR_ENV_TOP):int(h * config.OCR_ENV_BOTTOM), 
            int(w * current_env_left):int(w * config.OCR_ENV_RIGHT)
        ]
        sub_crop = frame[
            int(h * config.OCR_SUB_TOP):int(h * config.OCR_SUB_BOTTOM), 
            int(w * current_sub_left):int(w * config.OCR_SUB_RIGHT)
        ]
        
        env_texts = []
        env_protected = False
        env_res = reader_env.readtext(env_crop, detail=1, paragraph=True)
        for res in env_res:
            box, text = res[0], res[1]
            cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
            if len(cjk_chars) > 0:
                cx = sum([pt[0] for pt in box]) / 4.0 + (w * current_env_left)
                cy = sum([pt[1] for pt in box]) / 4.0 + (h * config.OCR_ENV_TOP)
                row_idx = int(cy / (h / 9.0))
                col_idx = int(cx / (w / 6.0))
                
                is_holy_ground = (row_idx in [7, 8]) and (col_idx in [2, 3])
                
                if len(cjk_chars) >= 2 or is_holy_ground:
                    env_texts.append(text)
                    if is_holy_ground:
                        env_protected = True
        
        env_text = " ".join(env_texts).strip()
        valid_env = bool(env_text)

        sub_texts = []
        sub_protected = False
        sub_res = reader_sub.ocr(sub_crop, cls=False)
        if sub_res and sub_res[0]:
            for line in sub_res[0]:
                box, text = line[0], line[1][0]
                cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
                if len(cjk_chars) > 0:
                    cx = sum([pt[0] for pt in box]) / 4.0 + (w * current_sub_left)
                    cy = sum([pt[1] for pt in box]) / 4.0 + (h * config.OCR_SUB_TOP)
                    row_idx = int(cy / (h / 9.0))
                    col_idx = int(cx / (w / 6.0))
                    
                    is_holy_ground = (row_idx in [7, 8]) and (col_idx in [2, 3])
                    
                    if len(cjk_chars) >= 2 or is_holy_ground:
                        sub_texts.append(text)
                        if is_holy_ground:
                            sub_protected = True
                        
        sub_text = " ".join(sub_texts).strip()
        valid_sub = bool(sub_text)

        if valid_env and valid_sub:
            if sub_text in env_text or difflib.SequenceMatcher(None, env_text, sub_text).ratio() > 0.85:
                env_text = "[Same as Sub]"
                valid_env = False

        if valid_env or valid_sub:
            is_protected = env_protected or sub_protected
            
            if current_block is None:
                current_block = {
                    "start": current_time, "end": current_time + (1.0/config.OCR_SWEEP_FPS), 
                    "ocr_env": env_text if valid_env else "[None]", 
                    "ocr_sub": sub_text if valid_sub else "[None]",
                    "is_protected": is_protected
                }
            else:
                if valid_sub or current_block["ocr_sub"] != "[None]":
                    similarity = difflib.SequenceMatcher(None, sub_text, current_block["ocr_sub"]).ratio()
                else:
                    similarity = difflib.SequenceMatcher(None, env_text, current_block["ocr_env"]).ratio()

                current_dur = current_time - current_block["start"]
                if similarity > config.OCR_SIMILARITY_THRESH and current_dur < 6.0:
                    if len(env_text) > len(current_block["ocr_env"]) and valid_env:
                        current_block["ocr_env"] = env_text
                    if len(sub_text) > len(current_block["ocr_sub"]) and valid_sub:
                        current_block["ocr_sub"] = sub_text
                    current_block["end"] = current_time + (1.0/config.OCR_SWEEP_FPS)
                    if is_protected:
                        current_block["is_protected"] = True
                else:
                    ocr_timeline.append(current_block)
                    current_block = {
                        "start": current_time, "end": current_time + (1.0/config.OCR_SWEEP_FPS), 
                        "ocr_env": env_text if valid_env else "[None]", 
                        "ocr_sub": sub_text if valid_sub else "[None]",
                        "is_protected": is_protected
                    }
        else:
            if current_block is not None:
                ocr_timeline.append(current_block)
                current_block = None

    if current_block is not None:
        ocr_timeline.append(current_block)

    cap.release()
    del reader_env, reader_sub
    flush_gpu()
    
    save_checkpoint("phase3_ocr_timeline", ocr_timeline)
    return ocr_timeline

# ============================================================
# PHASE 4: DATA MERGING, NLP FILTERING & VISION CONTEXT
# ============================================================

def _clean_audio(text: str) -> str:
    return re.sub(r'\[.*?\]\s*', '', text).strip()

def _get_speaker(w_start: float, w_end: float, diarization_data: list) -> str:
    max_overlap = 0.0
    best_speaker = "SPEAKER_UNKNOWN"
    for turn in diarization_data:
        overlap = max(0.0, min(w_end, turn["end"]) - max(w_start, turn["start"]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = turn["speaker"]
    return best_speaker

def smooth_timeline(enriched_timeline: list) -> list:
    smoothed = []
    current_block = None
    MAX_BLOCK_DURATION = 6.0 

    for block in enriched_timeline:
        if current_block is None:
            current_block = block.copy()
            continue
            
        gap = block["start"] - current_block["end"]
        same_speaker = (current_block["speaker"] == block["speaker"])
        same_sub = (current_block["ocr_sub"] == block["ocr_sub"])
        is_audio_only = (current_block["type"] == "AUDIO_ONLY" or block["type"] == "AUDIO_ONLY")
        current_dur = current_block["end"] - current_block["start"]
        
        if same_speaker and same_sub and gap < 1.0 and not is_audio_only and current_dur < MAX_BLOCK_DURATION:
            current_block["end"] = block["end"]
            
            clean_curr = _clean_audio(current_block["audio_text"])
            clean_new = _clean_audio(block["audio_text"])
            
            if clean_new not in clean_curr and clean_new != "[No localized audio]":
                if current_block["audio_text"] == "[No localized audio]":
                    current_block["audio_text"] = block["audio_text"]
                else:
                    current_block["audio_text"] += " " + block["audio_text"]
        else:
            smoothed.append(current_block)
            current_block = block.copy()

    if current_block:
        smoothed.append(current_block)
        
    return smoothed

def deduplicate_whisper_stutter(timeline: list) -> list:
    deduped = []
    for i, block in enumerate(timeline):
        if i == 0:
            deduped.append(block.copy())
            continue
        
        prev = deduped[-1]
        curr_block = block.copy()
        
        curr_audio = curr_block.get("audio_text", "")
        prev_audio = prev.get("audio_text", "")
        
        is_valid_audio = lambda x: x not in ["[No localized audio]", "", "[None]"]
        
        if not is_valid_audio(curr_audio) or not is_valid_audio(prev_audio):
            deduped.append(curr_block)
            continue
            
        same_speaker = (curr_block["speaker"] == prev["speaker"])
        gap = curr_block["start"] - prev["end"]
        
        if same_speaker and gap < 2.0:
            clean_prev = _clean_audio(prev_audio)
            clean_curr = _clean_audio(curr_audio)
            
            sim = difflib.SequenceMatcher(None, clean_prev, clean_curr).ratio()
            is_stutter = (clean_prev in clean_curr) or (clean_curr in clean_prev) or (sim > 0.80)
            
            if is_stutter:
                if len(clean_curr) > len(clean_prev):
                    prev["audio_text"] = curr_audio
                
                curr_block["audio_text"] = "[No localized audio]"
                
                if curr_block.get("ocr_sub") == prev.get("ocr_sub"):
                    curr_block["ocr_sub"] = "[None]"
                if curr_block.get("ocr_env") == prev.get("ocr_env"):
                    curr_block["ocr_env"] = "[None]"
                
        deduped.append(curr_block)
        
    return deduped

def sweep_trailing_micro_blocks(timeline: list) -> list:
    swept = []
    for i, block in enumerate(timeline):
        if i == 0:
            swept.append(block.copy())
            continue

        prev = swept[-1]
        curr = block.copy()
        
        curr_dur = curr["end"] - curr["start"]
        has_audio = curr.get("audio_text") not in ["[No localized audio]", "", "[None]"]
        
        same_sub = curr.get("ocr_sub") != "[None]" and curr.get("ocr_sub") == prev.get("ocr_sub")
        same_env = curr.get("ocr_env") != "[None]" and curr.get("ocr_env") == prev.get("ocr_env")
        
        if not has_audio and curr_dur < 0.6 and (same_sub or same_env):
            prev["end"] = curr["end"]
        else:
            swept.append(curr)
            
    return swept

def enforce_chronological_bounds(timeline: list) -> list:
    timeline = sorted(timeline, key=lambda x: x["start"])
    for i in range(1, len(timeline)):
        prev = timeline[i-1]
        curr = timeline[i]
        
        if prev["end"] > curr["start"]:
            clamped_end = curr["start"] - 0.01
            if clamped_end <= prev["start"]:
                clamped_end = prev["start"] + 0.10
                curr["start"] = clamped_end + 0.01
            timeline[i-1]["end"] = clamped_end
            
    return timeline

def load_florence_model():
    """Helper to cleanly load Florence and isolate GPU memory calls"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    try:
        vp = AutoProcessor.from_pretrained(config.FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=config.CACHE_DIR)
        vm = AutoModelForCausalLM.from_pretrained(config.FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, cache_dir=config.CACHE_DIR).cuda().eval()
    except Exception:
        vp = AutoProcessor.from_pretrained(config.FLORENCE_MODEL_ID, trust_remote_code=True, cache_dir=config.CACHE_DIR, local_files_only=True)
        vm = AutoModelForCausalLM.from_pretrained(config.FLORENCE_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, cache_dir=config.CACHE_DIR, local_files_only=True).cuda().eval()
    return vp, vm

def phase4_data_merger(ocr_timeline: list, zh_data: list, diarization_data: list) -> list:
    log("─" * 50)
    log("PHASE 4: Audio Merging & Safe Pre-Diarization")
    cached = load_checkpoint("phase4_enriched")
    if cached: return cached

    mapped_whisper_indices = set()
    enriched_timeline = []

    for block in ocr_timeline:
        block["block_audio_list"] = []

    for w in zh_data:
        w["speaker"] = _get_speaker(w["start"], w["end"], diarization_data)

    for i, w in enumerate(zh_data):
        w_start, w_end = w["start"], w["end"]
        segment_dur = w_end - w_start
        
        best_block = None
        max_overlap = 0.0
        
        for block in ocr_timeline:
            overlap = max(0.0, min(block["end"], w_end) - max(block["start"], w_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_block = block
                
        if best_block and (max_overlap > 0.5 or (segment_dur > 0 and (max_overlap / segment_dur) > 0.3)):
            best_block["block_audio_list"].append((w["speaker"], w["text"]))
            mapped_whisper_indices.add(i)

    for block in ocr_timeline:
        audio_lines = []
        current_spk = None
        
        for spk, txt in block["block_audio_list"]:
            if spk != current_spk:
                audio_lines.append(f"[{spk}] {txt}")
                current_spk = spk
            else:
                audio_lines.append(txt)
                
        audio_joined = " ".join(audio_lines)
        block_primary_speaker = block["block_audio_list"][0][0] if block["block_audio_list"] else "[ENVIRONMENT]"
        
        ocr_sub_final = block["ocr_sub"]
        
        if block.get("is_protected", False) and ocr_sub_final not in ["[None]", "[No text on screen]"]:
            if len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', ocr_sub_final)) <= 3:
                ocr_sub_final = f"[POTENTIAL MICRO-REACTION] {ocr_sub_final}"

        enriched_timeline.append({
            "type": "ON_SCREEN_TEXT",
            "start": block["start"],
            "end": block["end"],
            "speaker": f"[{block_primary_speaker}]" if audio_joined else "[ENVIRONMENT]",
            "audio_text": audio_joined if audio_joined else "[No localized audio]",
            "ocr_env": block["ocr_env"],
            "ocr_sub": ocr_sub_final,
            "is_protected": block.get("is_protected", False)
        })

    for i, w in enumerate(zh_data):
        if i not in mapped_whisper_indices:
            enriched_timeline.append({
                "type": "AUDIO_ONLY",
                "start": w["start"],
                "end": w["end"],
                "speaker": f"[{w['speaker']}]",
                "audio_text": f"[{w['speaker']}] {w['text']}",
                "ocr_env": "[No text on screen]",
                "ocr_sub": "[No text on screen]",
                "is_protected": False
            })

    enriched_timeline = sorted(enriched_timeline, key=lambda x: x["start"])
    enriched_timeline = smooth_timeline(enriched_timeline)
    enriched_timeline = deduplicate_whisper_stutter(enriched_timeline)
    enriched_timeline = sweep_trailing_micro_blocks(enriched_timeline)

    pruned_timeline = []
    for i, block in enumerate(enriched_timeline):
        has_audio = block.get("audio_text") not in ["[No localized audio]", "", "[None]"]
        has_sub = block.get("ocr_sub") not in ["[None]", "[No text on screen]"]
        has_env = block.get("ocr_env") not in ["[None]", "[No text on screen]"]
        
        if not has_audio and not has_sub and not has_env:
            continue 
            
        sub_len = len(block["ocr_sub"]) if has_sub else 0
        env_len = len(block["ocr_env"]) if has_env else 0
        max_text_len = max(sub_len, env_len)
        
        is_suspicious_garbage = (not has_audio) and (max_text_len < 3) and (not block.get("is_protected", False))
        
        if is_suspicious_garbage:
            is_near_audio = False
            if i > 0:
                prev_block = enriched_timeline[i-1]
                if prev_block.get("audio_text") not in ["[No localized audio]", "", "[None]"] and (block["start"] - prev_block["end"]) < 1.5:
                    is_near_audio = True
            if i < len(enriched_timeline) - 1:
                next_block = enriched_timeline[i+1]
                if next_block.get("audio_text") not in ["[No localized audio]", "", "[None]"] and (next_block["start"] - block["end"]) < 1.5:
                    is_near_audio = True
            
            if not is_near_audio:
                continue
                
        pruned_timeline.append(block)
    
    enriched_timeline = enforce_chronological_bounds(pruned_timeline)

    # VRAM ARMOR: Memory-Safe Florence-2 Loading
    vision_processor, vision_model = load_florence_model()

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    _FLORENCE_TAGS = ["</s>", "<s>", "<MORE_DETAILED_CAPTION>", "<pad>", "<unk>"]

    log("Running Florence-2 on timeline midpoints...")
    last_scene_desc = "[Frame read failed]"
    last_scene_time = -999.0

    for idx, block in enumerate(tqdm(enriched_timeline, desc="Vision Context")):
        # VRAM ARMOR: Hard Reset every 50 blocks
        if idx > 0 and idx % 50 == 0:
            del vision_model, vision_processor
            flush_gpu()
            vision_processor, vision_model = load_florence_model()

        midpoint_s = block["start"] + (block["end"] - block["start"]) / 2.0
        
        if (midpoint_s - last_scene_time) < 2.0:
            block["scene_description"] = last_scene_desc
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, midpoint_s * 1000)
        ret, frame = cap.read()
        scene_description = "[Frame read failed]"

        if ret and frame is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                with torch.inference_mode():
                    inputs = vision_processor(text="<MORE_DETAILED_CAPTION>", images=pil_img, return_tensors="pt").to("cuda", dtype=torch.float16)
                    generated_ids = vision_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512, do_sample=False, num_beams=3)
                raw_caption = vision_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                for tag in _FLORENCE_TAGS: raw_caption = raw_caption.replace(tag, "")
                scene_description = raw_caption.strip()
            except torch.cuda.OutOfMemoryError:
                flush_gpu()
                scene_description = "[Vision skipped: OOM]"
            except Exception:
                scene_description = "[Vision skipped/error]"

        block["scene_description"] = scene_description
        last_scene_desc = scene_description
        last_scene_time = midpoint_s

    cap.release()
    del vision_model, vision_processor
    flush_gpu()

    save_checkpoint("phase4_enriched", enriched_timeline)
    return enriched_timeline

# ============================================================
# PHASE 5: EXPORT (3 FILES / CONTINUOUS INDEXING)
# ============================================================

def write_export_file(blocks: list, filepath: str, title: str, start_index: int, video_name: str):
    lines = []
    lines.append("=" * 70)
    lines.append(f"FOUR-PILLAR SENSORY EXTRACTION — {title}")
    lines.append(f"Source    : {video_name}")
    lines.append(f"Cutoff    : {HARD_CUTOFF_SECONDS} seconds ({config.MAX_MINUTES} mins)")
    lines.append(f"Blocks    : {len(blocks)}")
    lines.append(f"Generated : {datetime.datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    for i, block in enumerate(blocks, start_index):
        time_str = f"{format_time(block['start'])} --> {format_time(block['end'])}"
        lines.append(f"{'─'*50}")
        lines.append(f"Block {i:04d} | {block['type']} | {time_str}")
        lines.append(f"Speaker  : {block['speaker']}")
        lines.append(f"ZH Audio : {block['audio_text']}")
        lines.append(f"Scene    : {block['scene_description']}")
        lines.append(f"OCR(Env) : {block.get('ocr_env', '[None]')}")
        lines.append(f"OCR(Sub) : {block.get('ocr_sub', '[None]')}")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"TXT ({title}) → {filepath}")

def phase5_export(enriched_timeline: list, video_name: str):
    log("─" * 50)
    log("PHASE 5: Exporting (Full Dump + Part 1 & Part 2)")

    ep_prefix = getattr(config, 'EP_PREFIX', 'bt_XX')
    base_dir = getattr(config, 'EP_BASE_DIR', config.OUTPUT_DIR)

    json_path = os.path.join(base_dir, f"{ep_prefix}.json")
    full_path = os.path.join(base_dir, f"{ep_prefix}_full.txt")
    part1_path = os.path.join(base_dir, f"{ep_prefix}_pt1.txt")
    part2_path = os.path.join(base_dir, f"{ep_prefix}_pt2.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_timeline, f, ensure_ascii=False, indent=2)

    write_export_file(enriched_timeline, full_path, "Full Dump", 1, video_name)

    mid_idx = len(enriched_timeline) // 2
    part1_blocks = enriched_timeline[:mid_idx]
    part2_blocks = enriched_timeline[mid_idx:]

    write_export_file(part1_blocks, part1_path, "Part 1", 1, video_name)
    write_export_file(part2_blocks, part2_path, "Part 2", mid_idx + 1, video_name)

# ============================================================
# PHASE 6: AUTO-CLEANUP (VRAM/STORAGE ARMOR)
# ============================================================

def phase6_cleanup():
    log("─" * 50)
    log("PHASE 6: Tactical Storage Cleanup")
    files_to_delete = [
        config.VIDEO_PATH,               # The washed MP4
        config.DIARIZATION_AUDIO_PATH,   # The massive WAV file
        config.CHECKPOINT_FILE           # The temporary JSON cache
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                log(f"Deleted heavy file to save space: {os.path.basename(file_path)}")
            except Exception as e:
                log(f"Warning: Could not delete {file_path}. {e}")

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def setup_episode_environment(input_video_path: str):
    import re
    base_name = os.path.basename(input_video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    match = re.search(r'\d+', name_without_ext)
    ep_num = match.group() if match else "XX"
    ep_prefix = f"bt_{ep_num}" 
    
    base_dir = os.path.join(config.WORKSPACE_ROOT, "btrecs", ep_prefix)
    os.makedirs(base_dir, exist_ok=True)
    
    washed_video_path = os.path.join(base_dir, f"{ep_prefix}_washed.mp4")
    checkpoint_file = os.path.join(base_dir, "pipeline_checkpoint.json")
    audio_wav = os.path.join(base_dir, "diarization_audio.wav")
    
    # NEW: Track the final target file to see if we've already finished this episode
    full_txt_path = os.path.join(base_dir, f"{ep_prefix}_full.txt")
    
    config.EP_PREFIX = ep_prefix
    config.EP_BASE_DIR = base_dir
    config.VIDEO_PATH = washed_video_path
    config.CHECKPOINT_FILE = checkpoint_file
    config.DIARIZATION_AUDIO_PATH = audio_wav
    
    return washed_video_path, base_name, full_txt_path

def run_pipeline():
    parser = argparse.ArgumentParser(description="Blazing Teens Automated Factory")
    parser.add_argument("--video", required=True, help="Path to the input MP4 file")
    args = parser.parse_args()
    
    input_video = args.video
    if not os.path.exists(input_video):
        log(f"FATAL: Input video not found: {input_video}")
        sys.exit(1)

    log("=" * 50)
    log(f"FOUR-PILLAR EXTRACTION PIPELINE — START ({os.path.basename(input_video)})")
    log("=" * 50)

    # Note the new full_txt_path variable here
    washed_video_path, original_filename, full_txt_path = setup_episode_environment(input_video)
    
    # ==========================================
    # CTO SKIP LOGIC: Bypass if already completed
    # ==========================================
    if os.path.exists(full_txt_path):
        log(f"SUCCESS: {full_txt_path} detected.")
        log(f"SKIPPING EPISODE: {original_filename} has already been processed!")
        sys.exit(0) # Exits cleanly, allowing run_all.sh to instantly move to the next episode
    # ==========================================

    validate_environment()
    
    ffmpeg_car_wash(input_video, washed_video_path)
    
    zh_data = phase1_transcribe()
    diarization_data = phase2_diarize()
    ocr_timeline = phase3_ocr_master_clock()
    enriched_timeline = phase4_data_merger(ocr_timeline, zh_data, diarization_data)

    phase5_export(enriched_timeline, original_filename)
    
    # Nuke the heavy files after export is completely finished
    phase6_cleanup()

    log("=" * 50)
    log(f"PIPELINE COMPLETE FOR {original_filename}")
    log("=" * 50)

if __name__ == "__main__":
    run_pipeline()