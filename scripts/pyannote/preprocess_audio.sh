#!/usr/bin/env bash
# preprocess_audio.sh -- Convert test audio files to 16kHz mono WAV for reference output generation.
#
# Usage:
#   bash scripts/pyannote/preprocess_audio.sh
#
# Requires: ffmpeg
# Input:    assets/audios/
# Output:   reference_outputs/preprocessed/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_ROOT/assets/audios"
OUTPUT_DIR="$PROJECT_ROOT/reference_outputs/preprocessed"

# Check ffmpeg is available
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is not installed or not in PATH." >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Helper: convert a single file to 16kHz mono WAV.
# Usage: convert_single INPUT OUTPUT_NAME
convert_single() {
    local input="$1"
    local output_name="$2"
    local output="$OUTPUT_DIR/$output_name"

    if [[ -f "$output" ]]; then
        echo "SKIP (exists): $output_name"
        return
    fi

    echo "Converting: $(basename "$input") -> $output_name"
    ffmpeg -i "$input" -ar 16000 -ac 1 -f wav "$output" -y -loglevel warning
}

# Helper: extract a single channel from a multi-channel audio file.
# These files have a single stream with N channels (not N separate streams).
# Usage: extract_channel INPUT CHANNEL_INDEX OUTPUT_NAME
extract_channel() {
    local input="$1"
    local channel_index="$2"
    local output_name="$3"
    local output="$OUTPUT_DIR/$output_name"

    if [[ -f "$output" ]]; then
        echo "SKIP (exists): $output_name"
        return
    fi

    echo "Extracting channel $channel_index: $(basename "$input") -> $output_name"
    ffmpeg -i "$input" -af "pan=mono|c0=c${channel_index}" -ar 16000 -f wav "$output" -y -loglevel warning
}

echo "=== Preprocessing test audio files ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# 1. 00_混合_微电影音轨acc.m4a -> single WAV
convert_single "$INPUT_DIR/00_混合_微电影音轨acc.m4a" "00_混合_微电影音轨acc.wav"

# 2. 01_人声_测试音频双声道包含一个静音轨道.wav -> single WAV (downmix to mono)
convert_single "$INPUT_DIR/01_人声_测试音频双声道包含一个静音轨道.wav" "01_人声_测试音频双声道包含一个静音轨道.wav"

# 3. 01_人声_三轨道短剧录音.WAV -> split 3 channels into separate WAVs
#    (single stream with 3 channels, not 3 separate audio streams)
extract_channel "$INPUT_DIR/01_人声_三轨道短剧录音.WAV" 0 "01_人声_三轨道短剧录音_track0.wav"
extract_channel "$INPUT_DIR/01_人声_三轨道短剧录音.WAV" 1 "01_人声_三轨道短剧录音_track1.wav"
extract_channel "$INPUT_DIR/01_人声_三轨道短剧录音.WAV" 2 "01_人声_三轨道短剧录音_track2.wav"

# 4. 01_人声_四轨道短剧录音.WAV -> split 4 channels into separate WAVs
#    (single stream with 4 channels, not 4 separate audio streams)
extract_channel "$INPUT_DIR/01_人声_四轨道短剧录音.WAV" 0 "01_人声_四轨道短剧录音_track0.wav"
extract_channel "$INPUT_DIR/01_人声_四轨道短剧录音.WAV" 1 "01_人声_四轨道短剧录音_track1.wav"
extract_channel "$INPUT_DIR/01_人声_四轨道短剧录音.WAV" 2 "01_人声_四轨道短剧录音_track2.wav"
extract_channel "$INPUT_DIR/01_人声_四轨道短剧录音.WAV" 3 "01_人声_四轨道短剧录音_track3.wav"

# 5. 01_人声_自录双人对话.wav -> single WAV
convert_single "$INPUT_DIR/01_人声_自录双人对话.wav" "01_人声_自录双人对话.wav"

echo ""
echo "=== Preprocessing complete ==="
echo "Output files:"
ls -lh "$OUTPUT_DIR"
