"""Portable byte identity for the isolated Final runtime, never an old identity."""
from pathlib import Path
import hashlib
import json


def source_digest():
    root=Path(__file__).resolve().parents[1]
    paths=[]
    for package in ('helixwm_final','helixwm_matched','helixwm_v2'):
        paths.extend((root/package).rglob('*.py'))
    paths.extend(root/'configs'/name for name in
        ('helixwm_final_matched_20261105.json','helixwm_final_full_20261106.json',
         'helixwm_final_budget_matched_20261106.json',
         'dreamerv3_helixwm_matched_20261026.json'))
    records={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest(),records
