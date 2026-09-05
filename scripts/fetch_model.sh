#!/usr/bin/env bash
# Downloads the MediaPipe hand landmark model (~7.5 MB) used by the tracker.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -fL --progress-bar -o models/hand_landmarker.task "$URL"
echo "Saved to models/hand_landmarker.task"
