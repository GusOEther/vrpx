"""Test vectors of the shared percentile definition v1.1 (§4).
Run: python -m pytest tests  (or: python tests/test_percentile.py)"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vrpx import pr_rank, pr_last_values, pr_exp_series

NaN = float("nan")

def close(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9

def test_vectors():
    w = [1, 2, 2, 3, 4]
    assert close(pr_rank(w, 2), 40)
    assert close(pr_rank(w, 0), 0)
    assert close(pr_rank(w, 5), 100)
    assert close(pr_rank(w, 2.5), 60)
    assert close(pr_rank([1, NaN, 2, 3], 2), 50)      # n = 3
    assert close(pr_rank([3, 3, 3], 3), 50)

def _series(n_nan, gap_at_end=False, N=252):
    v = np.arange(1, 600, dtype=float)                 # long history, last value = x_d
    win = slice(len(v) - 1 - N, len(v) - 1)            # the N sessions strictly before d
    if gap_at_end:
        v[len(v) - 1 - n_nan:len(v) - 1] = np.nan
    else:
        idx = np.arange(win.start, win.stop)[:n_nan]
        v[idx] = np.nan
    return v

def test_min_fill():
    assert math.isnan(pr_last_values(_series(13), 252))        # n = 239 -> empty
    assert not math.isnan(pr_last_values(_series(12), 252))    # n = 240 -> value
    assert math.isnan(pr_last_values(_series(300, gap_at_end=True), 252))  # no stretching

def test_excludes_d_and_forward_horizon():
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert close(pr_last_values(v, 4), 100)                    # d itself not in the window
    # forward quantity, H = 2: window ends at d - 2 -> values [1, 2, 3] (N = 3)
    assert close(pr_last_values(v, 3, H=2), 100)
    assert math.isnan(pr_last_values(v, 4, H=2))               # only 3 sessions resolved

def test_expanding_is_causal():
    v = np.r_[np.arange(300, dtype=float), [150.0]]
    pr = pr_exp_series(v)
    assert np.isnan(pr[:200]).all()                            # < 200 prior values -> empty
    assert close(pr[200], 100)                                 # 200 > all 200 earlier values
    assert close(pr[-1], 100 * (150 + 0.5) / 300)              # mid-rank among the 300 before
    # appending later data must not change earlier ranks
    assert np.allclose(pr_exp_series(v[:250])[200:], pr[200:250])

if __name__ == "__main__":
    for f in [test_vectors, test_min_fill, test_excludes_d_and_forward_horizon, test_expanding_is_causal]:
        f(); print("ok", f.__name__)
