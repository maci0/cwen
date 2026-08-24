#!/usr/bin/env bash
# Download Qwen3.8-27B Q4_0 GGUF into model/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p model
FILE="model/Qwen3.8-27B-Q4_0.gguf"
# unsloth/Qwen3.8-27B-GGUF Q4_0 listed size (bytes)
EXPECT=16056478688
# Content pin: every golden dump, e2e compare, and recorded result was produced
# from exactly this artifact. unsloth re-uploads GGUFs in place under the same
# name, so a size check alone would let silent weight drift invalidate all
# comparisons. To adopt new weights deliberately: update both constants here,
# re-download, then `make repack` and `make golden`.
EXPECT_SHA256=ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d

check() {
  local sz got
  sz=$(stat -c%s "$FILE")
  if [[ "$sz" -ne "$EXPECT" ]]; then
    echo "size mismatch: $FILE is $sz bytes, want $EXPECT" >&2
    return 1
  fi
  got=$(sha256sum "$FILE")
  got=${got%% *}
  if [[ "$got" != "$EXPECT_SHA256" ]]; then
    echo "sha256 mismatch: $FILE is $got" >&2
    echo "want $EXPECT_SHA256 (pinned weights); results from this file are not comparable to existing goldens" >&2
    return 1
  fi
}

if [[ -f "$FILE" ]]; then
  echo "already present: $FILE"
  if check; then
    exit 0
  fi
  echo "re-downloading to restore the pinned artifact" >&2
fi
PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "cwen: no .venv found; run 'make setup' first" >&2
  exit 1
fi
"$PY" - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="unsloth/Qwen3.8-27B-GGUF",
    filename="Qwen3.8-27B-Q4_0.gguf",
    local_dir="model",
)
print("DONE", p)
PY
if ! check; then
  echo "downloaded artifact does not match the pin; refusing to keep it" >&2
  rm -f "$FILE"
  exit 1
fi
