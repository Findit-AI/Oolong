#!/usr/bin/env python3
"""Run pyannote diarization on a single audio file, output segments.json + timestamps.md"""

import json
import sys
import time
from pathlib import Path

def fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python diarize_single.py <audio_path> [output_dir]", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else audio_path.parent
    model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "speaker-diarization-community-1"

    if not audio_path.exists():
        print(f"ERROR: Audio not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from pyannote.audio import Pipeline

    print(f"Loading model: {model_dir}")
    pipeline = Pipeline.from_pretrained(str(model_dir))

    print(f"Running diarization: {audio_path.name}")
    t0 = time.time()
    result = pipeline(str(audio_path))
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")

    diarization = result.speaker_diarization
    segments_list = list(diarization.itertracks(yield_label=True))
    speakers = sorted(diarization.labels())

    print(f"Speakers: {len(speakers)}")
    print(f"Segments: {len(segments_list)}")

    # Save segments.json
    segments_json = []
    for turn, _, speaker in segments_list:
        segments_json.append({
            "start": round(float(turn.start), 6),
            "end": round(float(turn.end), 6),
            "speaker": speaker,
        })

    stem = audio_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}_segments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"audio_file": audio_path.name, "segments": segments_json}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    # Save timestamps.md
    md_lines = [
        f"# {stem} — Python pyannote diarization",
        "",
        f"Elapsed: {elapsed:.1f}s  ",
        f"Speakers: {len(speakers)}  ",
        f"Segments: {len(segments_list)}  ",
        "",
        "| # | Start | End | Duration | Speaker |",
        "|---|-------|-----|----------|---------|",
    ]
    for i, (turn, _, spk) in enumerate(segments_list, 1):
        dur = turn.end - turn.start
        md_lines.append(f"| {i} | {fmt(turn.start)} | {fmt(turn.end)} | {dur:.3f}s | {spk} |")

    md_path = output_dir / f"{stem}_python_timestamps.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved: {md_path}")

    # Print summary for benchmark
    print(f"\n=== BENCHMARK ===")
    print(f"audio: {audio_path.name}")
    print(f"duration: {fmt(float(segments_json[-1]['end']) if segments_json else 0)}")
    print(f"speakers: {len(speakers)}")
    print(f"segments: {len(segments_json)}")
    print(f"elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
