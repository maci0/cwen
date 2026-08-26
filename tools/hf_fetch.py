#!/usr/bin/env python3
"""Download one file from a Hugging Face repo into a local directory.

Called by tools/download.sh; kept as its own file so no shell script embeds
Python. Prints the resolved path on success.

Usage: .venv/bin/python tools/hf_fetch.py REPO_ID FILENAME LOCAL_DIR
"""

from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    repo_id, filename, local_dir = sys.argv[1:4]
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
