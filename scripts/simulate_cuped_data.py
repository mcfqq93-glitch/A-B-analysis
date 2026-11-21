"""Write the simulated CUPED dataset to data/simulated/ (python scripts/simulate_cuped_data.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abtest.simulate import simulate_revenue_experiment  # noqa: E402

if __name__ == "__main__":
    df = simulate_revenue_experiment()
    out = ROOT / "data" / "simulated" / "revenue_experiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {len(df):,} rows -> {out}")
