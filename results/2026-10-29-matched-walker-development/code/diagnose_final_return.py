"""Recompute the retained failed final case; no model or training imports."""
from pathlib import Path
from fractions import Fraction
import hashlib, io, json, math, zipfile
import numpy as np

root=Path(__file__).resolve().parents[1]
diagnosis=json.loads((root/'data/final_evaluation_blocker.json').read_text())
archive=root/'data/failed_final_case.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==diagnosis['compact_raw_archive']['sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    observed=json.loads(z.read('observed.json'))
    with np.load(io.BytesIO(z.read('trace.npz')),allow_pickle=False) as trace:
        rewards=trace['reward'].copy()
assert rewards.shape==(500,) and rewards.dtype==np.float64 and np.isfinite(rewards).all()
recorded=observed['episode_return']
sums={
    'recorded_return':recorded,
    'python_float_list_sum':float(sum(rewards.tolist())),
    'numpy_scalar_builtin_sum':float(sum(rewards)),
    'numpy_sum':float(np.sum(rewards)),
    'math_fsum':math.fsum(rewards.tolist()),
    'exact_binary64_sum_correctly_rounded':float(sum((Fraction.from_float(float(x)) for x in rewards),Fraction())),
}
result={'case':diagnosis['case_id'],'sums':sums,
        'strict_numpy_scalar_equality':recorded==sums['numpy_scalar_builtin_sum'],
        'absolute_difference':sums['numpy_scalar_builtin_sum']-recorded,
        'ulp_difference':(sums['numpy_scalar_builtin_sum']-recorded)/math.ulp(recorded),
        'accepted_final_cases':0,
        'note':'Summation behavior may vary with Python/NumPy version. This script neither accepts the case nor retries evaluation.'}
print(json.dumps(result,indent=2))
