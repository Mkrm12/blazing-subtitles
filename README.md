# Blazing Subtitles Pipeline 🔥🪀

A heavily customized, 4-pillar sensory extraction pipeline designed to generate flawlessly timed English subtitles for Chinese dramas. 

### The Origin Story
This started as a pure passion project. I just wanted to translate and generate high-quality English subtitles for my childhood favorite 2006 Chinese live-action show, *Blazing Teens* (火力少年王). 

Standard AI transcription tools (like raw Whisper) completely fail at Chinese sports dramas. They hallucinate during action scenes, they stutter when characters shout, and they completely ignore important on-screen text like location signs or named yo-yo techniques. 

I needed a pipeline that could literally "watch" the show like a human would before sending the data to an LLM for translation.

### Current State: WIP (Iterating & Refining)
This codebase is currently around "v10" and was heavily "vibe-coded" through trial and error in Lightning AI. It is actively being tested, tweaked, and refined to handle edge cases, fix AI hallucinations, and perfect the subtitle pacing. It's messy, but the architecture works.

### How It Works (The 4 Pillars)
This pipeline doesn't just transcribe audio; it builds a highly dense, chronologically clamped context dump for LLMs (like DeepSeek or GPT-4) to translate.
1. **Audio (Whisper):** Extracts the raw Chinese dialogue. Includes a custom NLP deduplicator to kill "Whisper Stutters" during loud action scenes.
2. **Speaker Diarization (Pyannote):** Maps exactly who is speaking and when.
3. **Dual-Box OCR Master Clock (EasyOCR + PaddleOCR):** Sweeps the video frames to read on-screen text, location cards, and hardcoded Chinese subtitles. This acts as the "Master Clock" to guarantee subtitles never exceed a 6-second readability limit.
4. **Vision Context (Florence-2):** Generates a visual description of the scene so the translating LLM understands the physical context of the dialogue.

The final output is bifurcated into clean text dumps (Part 1 & Part 2) to prevent LLM context-window truncation.

### Translation Strategy
During development, I benchmarked several LLMs for the final localization step:
* **Gemini / Mistral / Qwen:** While powerful, they often struggled with the dense context-switching required for batch subtitle localization.
* **DeepSeek (Current Winner):** Found to be the most resilient for batch translations. It maintains character consistency and "slang" flow better than others when fed 200+ blocks of multi-modal context (OCR + Audio + Vision).

### Setup
If you are running this on Lightning AI or a similar cloud GPU, use the provided `setup.sh` script. It specifically handles dependency clashes (like pinning NumPy to 1.x) to ensure Pyannote, EasyOCR, and Torch compile correctly together.

```bash
bash setup.sh
bash run.sh