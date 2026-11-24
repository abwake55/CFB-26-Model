"""Research walk-forward variants (2019-2025 folds, same protocol as walk_forward.py).

    python scripts/experiment_walk_forward.py NAME [--inseason] [--leaves N]
    python scripts/evaluate_walk_forward.py outputs/research/wf_NAME.csv

--inseason  adds point-in-time opponent-adjusted ratings (src/inseason_ratings.py)
--leaves N  overrides LightGBM num_leaves (production default lives in model.py)

Writes outputs/research/wf_<NAME>.csv. Nothing here changes production models.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402
import model  # noqa: E402
import walk_forward as wf  # noqa: E402


def main():
    name = sys.argv[1]
    if "--leaves" in sys.argv:
        leaves = int(sys.argv[sys.argv.index("--leaves") + 1])
        orig = wf.tune_gbm_params
        wf.tune_gbm_params = lambda *a, **k: {**orig(*a, **k), "num_leaves": leaves}
    df = model.load_data()
    if "--inseason" in sys.argv:
        import inseason_ratings as ir
        r = ir.load_inseason_ratings(ROOT / "data" / "processed")
        df = df.merge(r.drop(columns=["inseason_version"]), on="game_id", how="left")
        model.SPREAD_FEATURES.extend(f for f in ir.INSEASON_SPREAD_FEATURES
                                     if f not in model.SPREAD_FEATURES)
        model.TOTALS_FEATURES.extend(f for f in ir.INSEASON_TOTALS_FEATURES
                                     if f not in model.TOTALS_FEATURES)
    folds = [wf.run_fold(df, s) for s in range(wf.FIRST_TEST, wf.LAST_TEST + 1)]
    out = pd.concat([f for f in folds if not f.empty], ignore_index=True)
    dest = ROOT / "outputs" / "research" / f"wf_{name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
