# Evaluation-Gated CI/CD Evidence (Deliverable 2)

## 1. Policy & Enforcement Architecture
- **Gate Policy (`configs/gates.yaml`, v1.0.0):**
  - `require_validation_passed: true` (`status=passed`, `splits=passed`, total leakage `= 0`)
  - `min_val_macro_f1: 0.80`
  - `min_reg_macro_f1: 0.78`
  - `min_extraction_micro_f1: 0.95`
  - `max_reg_ece: 0.08`
  - `max_regression_macro_f1_drop: 0.02`
  - `max_negation_slice_f1_drop: 0.03`
  - `max_compound_slice_f1_drop: 0.03`
- **Evaluator (`ml/gate.py`) & CI Workflow (`.github/workflows/ci.yml`):**
  - Every pull request and push to `main` executes `backend`, `frontend`, and `ml-gates` jobs on `ubuntu-latest`.
  - `ml-gates` runs `ruff`, `pytest tests/test_d2_gates.py -v`, and `python -m ml.gate`, failing closed (non-zero exit code `1`) if any artifact is missing, corrupt, leaky, or below threshold.
  - `.github/workflows/release.yml` also runs `python -m ml.gate` **before** building any Docker image for ECR promotion.

---

## 2. Pull Request Pass Proof (PR #9)
- **PR:** [`#9`](https://github.com/mwahaj36/subtext-mlops/pull/9) (`fasih/d2-eval-gates`, merged `55c67d2e327817dee93e9259b77424238dbce81b`)
- **CI Run:** [`45082948425`](https://github.com/mwahaj36/subtext-mlops/actions/runs/45082948425) (`ml-gates`, `backend`, `frontend`, `Vercel` all **PASS**)
- **Gate Report Output (from CI `ml-gates` log):**

| Check | Result | Observed | Requirement |
|---|---|---|---|
| `dataset_validation` | **PASS** | `status=passed, splits=passed, leakage=0` | `status=passed, splits=passed, leakage=0` |
| `val_macro_f1_floor` | **PASS** | `0.8770` | `>= 0.8000` |
| `reg_macro_f1_floor` | **PASS** | `0.8183` | `>= 0.7800` |
| `extraction_micro_f1_floor` | **PASS** | `0.9872` | `>= 0.9500` |
| `reg_calibration_ece_cap` | **PASS** | `0.0299` | `<= 0.0800` |
| `champion_reg_macro_f1_delta` | **PASS** | `cand=0.8183 vs champ=0.8183 (drop=+0.0000)` | `drop <= 0.0200` |
| `negation_slice_f1_delta` | **PASS** | `cand=0.7080 vs champ=0.7080 (drop=+0.0000)` | `drop <= 0.0300` |
| `compound_slice_f1_delta` | **PASS** | `cand=0.8012 vs champ=0.8012 (drop=+0.0000)` | `drop <= 0.0300` |

- Uploaded CI artifact: `evaluation-gate-report` (`Artifact ID 14777489306`, SHA-256 `40584c8498ff157da51382fc8d518485fc4305947620de44795d587ddd821970`).

---

## 3. Pull Request Fail-Closed Block Proof (PR #10, Closed Unmerged)
- **PR:** [`#10`](https://github.com/mwahaj36/subtext-mlops/pull/10) (`fasih/d2-gate-block-proof`, deliberately closed without merging and proof branch deleted)
- **Simulated Regression:** Raised `min_reg_macro_f1` in `configs/gates.yaml` from `0.78` to `0.99` to simulate a benchmark failure on a pull request.
- **Blocked CI Run:** [`45083118624`](https://github.com/mwahaj36/subtext-mlops/actions/runs/45083118624) (`ml-gates` failed as designed, exit code `1`):

```text
tests/test_d2_gates.py::test_real_champion_passes_all_gates FAILED       [ 14%]
...
| `reg_macro_f1_floor` | **FAIL** | `0.8183` | `>= 0.9900` |
...
========================= 1 failed, 6 passed in 0.07s ==========================
##[error]Process completed with exit code 1.
```
