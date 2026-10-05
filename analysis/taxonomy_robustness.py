from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THETAS = [0.05, 0.10, 0.15]

def classify(curve: list[float], theta: float) -> str:
    if not curve:
        return "empty"
    mx = max(curve)
    if mx <= theta:
        return "capability"
    return "reaches" if curve[-1] > theta else "pit"

def main() -> int:
    out: dict = {}
    for scale in ("4b", "8b"):
        gp = json.load(open(ROOT / f"runs/resp/locus_{scale}_k16/campaign/gold_pits.json"))
        n = len(gp)
        stored = Counter(v.get("regime") for v in gp.values())
        per_theta = {}
        for th in THETAS:
            c = Counter(classify(v.get("curve") or [], th) for v in gp.values())
            per_theta[f"{th:.2f}"] = {k: round(c[k] / n, 4) for k in
                                      ("capability", "pit", "reaches", "empty") if c[k]}
        out[scale] = {
            "n": n,
            "stored_regime_frac": {k: round(stored[k] / n, 4) for k in ("capability", "pit", "reaches")},
            "recomputed_by_theta": per_theta,
        }
        print(f"=== {scale} (n={n}) ===")
        print(f"  stored (theta=0.1): {out[scale]['stored_regime_frac']}")
        for th, fr in per_theta.items():
            print(f"  theta={th}: {fr}")
    json.dump(out, open(ROOT / "runs/resp/taxonomy_robustness.json", "w"), indent=2)

    print("\n=== scale stability (stored, theta=0.1) ===")
    for r in ("capability", "pit", "reaches"):
        print(f"  {r:11} 4B={out['4b']['stored_regime_frac'][r]:.3f}  8B={out['8b']['stored_regime_frac'][r]:.3f}")
    print(f"wrote runs/resp/taxonomy_robustness.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
