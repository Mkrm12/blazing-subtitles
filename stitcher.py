"""
stitcher.py — The Ultimate SRT Refiner, Multi-Line Parser & Clash Detector
"""
import re
import datetime

# --- CONFIGURATION ---
INPUT_SRT = "blazingteen18.srt"
OUTPUT_SRT = "blazingteen18_ref.srt"

def format_timestamp(seconds: float):
    td = datetime.timedelta(seconds=max(0.0, seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def parse_shorthand_time(time_str):
    """Converts '145' -> 105s, or '1729' -> 1049s"""
    if not time_str: return None
    time_str = time_str.strip()
    if len(time_str) <= 2:
        return int(time_str)
    mins = int(time_str[:-2])
    secs = int(time_str[-2:])
    return (mins * 60) + secs

def main():
    try:
        with open(INPUT_SRT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[!] FATAL: Could not find {INPUT_SRT}.")
        return

    blocks = []
    current_block = None

    # REGEX UPGRADES: Ultra-forgiving. Catches wrong dashes, periods instead of commas, missing spaces.
    tc_pattern = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*[-=]+>\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})")
    sh_pattern = re.compile(r"^(\d{3,4})(?:\s*[-–—]\s*(\d{3,4}))?(?:\s+(.+))?$")

    print(">>> Sweeping file for broken blocks, timecodes, and shorthands...")

    for line in lines:
        line = line.strip()

        if not line:
            current_block = None
            continue

        # Ignore old, broken index numbers
        if re.match(r"^\d+$", line):
            continue

        # 1. Detect Standard Timecode (Ultra-Forgiving)
        tc_match = tc_pattern.search(line)
        if tc_match:
            start_str, end_str = tc_match.groups()
            def ts_to_sec(ts):
                ts = ts.replace('.', ',') # Fix common typo
                h, m, s_ms = ts.split(':')
                s, ms = s_ms.split(',')
                return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

            current_block = {
                "type": "standard",
                "start": ts_to_sec(start_str),
                "end": ts_to_sec(end_str),
                "text": [],
                "raw_input": line
            }
            blocks.append(current_block)
            continue

        # 2. Detect Custom Shorthand
        sh_match = sh_pattern.match(line)
        if sh_match:
            start_raw = sh_match.group(1)
            end_raw = sh_match.group(2)
            text_raw = sh_match.group(3)

            current_block = {
                "type": "shorthand",
                "start": parse_shorthand_time(start_raw),
                "end": parse_shorthand_time(end_raw) if end_raw else None,
                "text": [text_raw] if text_raw else [],
                "raw_input": line
            }
            blocks.append(current_block)
            continue

        # 3. Attach multi-line text (like Intro Cards) to the current bucket
        if current_block is not None:
            current_block["text"].append(line)
        elif blocks:
            blocks[-1]["text"].append(line)

    # Join the text arrays and format line breaks
    for b in blocks:
        b["text"] = "\n".join(b["text"]).strip()

    # Sort strictly chronologically
    blocks = sorted(blocks, key=lambda x: x["start"])

    # --- PROCESS SHORTHAND DURATIONS ---
    for idx, b in enumerate(blocks):
        if b["type"] == "shorthand":
            start_sec = b["start"]

            if b["end"] is None:
                # Dynamically calculate length based on how much text you wrote
                calc_dur = max(1.0, min(4.0, len(b["text"]) * 0.08))
                b["end"] = start_sec + calc_dur

    # --- TIMELINE CLASH DETECTOR & AUTO-CORRECTION ---
    print("\n>>> Validating Timeline Integrity...")
    clashes_found = 0
    for i in range(len(blocks) - 1):
        curr_end = blocks[i]["end"]
        next_start = blocks[i + 1]["start"]

        # If the current block bleeds into the next block
        if curr_end >= next_start:
            clashes_found += 1
            print(f"\n[!] TIME CLASH DETECTED at Block {i+1} & {i+2}")
            print(f"    Block {i+1} Ends  : {format_timestamp(curr_end)}")
            print(f"    Block {i+2} Starts: {format_timestamp(next_start)}")

            # Auto-Clamp
            blocks[i]["end"] = next_start - 0.05
            if blocks[i]["end"] <= blocks[i]["start"]:
                blocks[i]["end"] = blocks[i]["start"] + 0.1 # Hard failsafe

            print(f"    -> Auto-Corrected Block {i+1} End to: {format_timestamp(blocks[i]['end'])}")

    if clashes_found == 0:
        print("    [+] Timeline is mathematically perfect. Zero clashes detected.")

    # --- TERMINAL CONTEXT LOGGER (UPGRADED) ---
    print("\n>>> Shorthand Injection Context Review:")
    for idx, b in enumerate(blocks):
        if b["type"] == "shorthand":
            prev_b = blocks[idx-1] if idx > 0 else None
            next_b = blocks[idx+1] if idx < len(blocks)-1 else None

            print(f"\n--- Injection near Block {idx+1} ---")

            if prev_b:
                clean_prev = prev_b['text'].replace('\n', ' | ')
                if len(clean_prev) > 40: clean_prev = clean_prev[:40] + "..."
                print(f"    [Block {idx:03d}] {format_timestamp(prev_b['start'])} --> {format_timestamp(prev_b['end'])} | {clean_prev}")

            clean_curr = b['text'].replace('\n', ' | ')
            if not clean_curr: clean_curr = "[NO TEXT FOUND - CHECK YOUR FILE]"
            if len(clean_curr) > 80: clean_curr = clean_curr[:80] + "..."

            # This line proves your multi-line intro cards were captured!
            print(f"  > [Block {idx+1:03d}] {format_timestamp(b['start'])} --> {format_timestamp(b['end'])} | {clean_curr}  <-- INJECTED")

            if next_b:
                clean_next = next_b['text'].replace('\n', ' | ')
                if len(clean_next) > 40: clean_next = clean_next[:40] + "..."
                print(f"    [Block {idx+2:03d}] {format_timestamp(next_b['start'])} --> {format_timestamp(next_b['end'])} | {clean_next}")

    # --- EXPORT FLAWLESS SRT ---
    print(f"\n>>> Exporting final 1-to-N indexed SRT to {OUTPUT_SRT}...")
    with open(OUTPUT_SRT, 'w', encoding='utf-8') as f:
        for i, b in enumerate(blocks, 1):
            start_str = format_timestamp(b["start"])
            end_str = format_timestamp(b["end"])
            f.write(f"{i}\n{start_str} --> {end_str}\n{b['text']}\n\n")

    print(">>> BOOM. Subtitle Refinement Complete.")

if __name__ == "__main__":
    main()