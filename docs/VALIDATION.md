# v0.1 Validation Record

> Historical record of the reference-only checks below. It is not a current inventory of all repository capabilities. The later completed custom pilot and its unsuccessful control outcome are documented in the [first experiment postmortem](experiments/2026-10-13-walker-seed0.md); the newer [dual-chain repair proposal](DUAL_CHAIN_DESIGN.md) remains unimplemented.

Date: 2026-10-13. Scope: the mathematical reference implementation on a local CPU. No environment experiments or remote training were started.

The later [RTX 5090 environment verification](ENVIRONMENT_RTX5090.md) has a separate record for CUDA operators, RSSM gradients, initialized checkpoint reload, and bounded Walker EGL rendering. It does not constitute formal training.

## Tested Environment

- Windows x64, Python 3.12.14.
- PyTorch 2.6.0+cpu, NumPy 2.2.6.
- Editable package build and installation succeeded.
- GitHub Actions was not configured or run for this validation. Other Python/PyTorch and GPU/CUDA combinations have not been validated by these checks.

## Execution Results

```text
python -m unittest discover -s tests -v
Ran 30 tests in 0.091s
OK

python -m examples.synthetic_trace
all_gradients_finite: true
example_horizon: [2]
example_returns: [[11.791, 10.9, 0.0]]
example_valid_mask: [[true, true, false]]
optimizer_steps: 0
```

The elapsed time above records this particular local test run; it is neither a performance benchmark nor a runtime guarantee. The synthetic demonstration also checked model losses, auxiliary supervision, and gradient connections. Losses and risk probabilities from untrained networks are not evidence of algorithmic effectiveness.

| Test File | Count | Independent Checks |
|---|---:|---|
| test_distributions.py | 3 | Hand-calculated KL/JS values, grouped reductions, analytical ST gradients |
| test_rssm_losses.py | 9 | Real/Dream time indexing, resets, detached teacher/starting state, padding |
| test_reliability.py | 9 | History causality, prefix maximum-error labels, unknown-label masking, auxiliary gradient isolation |
| test_reliability_precision.py | 4 | Overflow regressions for extreme fp16 logits and rewards |
| test_returns.py | 5 | Contiguous prefixes, H=0, bootstrapping, termination/discount boundaries, independence from the rejected tail |

An independent review identified and fixed two low-precision issues: overflow from finite fp16 logits during distribution discrepancy calculations, and overflow when subtracting rewards before normalization. The relevant operands are now promoted to float32 before computation, and the added regression tests passed. This does not establish that the full model has been validated for mixed-precision training.

## Remaining Experimental Validation

This reference-validation stage established only that these mathematical interfaces and boundary tests passed. It did not demonstrate that HelixWM improves environment returns, sample efficiency, calibration quality, or generalization. These specific reference checks produced no learning curves, environment videos, training checkpoints, or multi-seed results.

At that stage, the proposed next work was frozen-model diagnostics followed by adaptation of the authors' maintained DreamerV3 implementation. The later custom pilot implemented a training loop and checkpoint/replay/RNG storage and passed single-run artifact acceptance, but it did not implement that official-backbone comparison or establish effective bidirectional repair. Consult the dated experiment report for those distinctions.
