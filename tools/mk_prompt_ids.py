#!/usr/bin/env python3
"""Write prompt1.ids: the pinned BOS id 248044 as raw little-endian int32.

Shared by tools/test_speed_gates.sh and tools/profile_*.sh so the pin lives
in one place instead of a copy-pasted python -c per script.
"""

import struct
from pathlib import Path

BOS = 248044

path = Path(__file__).resolve().parents[1] / "prompt1.ids"
path.write_bytes(struct.pack("i", BOS))
