"""Fetch the two upstream SMoSE files used by both learned baselines."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'ae2a1a875193bf121ef1b35038994a8d899343b7'
FILES = {
    'src/sac.py': '3230871c581962846a503b44e573bb04686f142bd633541fe51a4705a778b041',
    'config/reacher.yml': '2cc58696d3b9ff61948d7b94a02317d018f3fdadb2e9399e95bf6b3e087017b3',
}
for name, digest in FILES.items():
    target = ROOT / 'external/SMoSE' / name
    if target.exists():
        data = target.read_bytes()
    else:
        with urlopen(f'https://raw.githubusercontent.com/vinczematyas/SMoSE/{COMMIT}/{name}', timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f'Upstream checksum mismatch: {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f'Verified {name}')
