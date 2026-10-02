"""Export the analysis CSV for the FRL paper: results/frl/minteval_v0_frl.csv + data dictionary.
realized_vol = annualised std of BTC 15m log returns over the bars where the REFERENCE program holds a
position (sqrt(35040) scaling), i.e. the volatility the strategy was actually exposed to."""
import sys, json, os, hashlib
sys.path.insert(0, ".")
import numpy as np, pandas as pd
frames = [pd.read_csv("results/minteval_v0.csv").assign(subset="full800")]
if os.path.exists("results/minteval_v0_frontier.csv"):
    frames.append(pd.read_csv("results/minteval_v0_frontier.csv").assign(subset="frontier200"))
d = pd.concat(frames, ignore_index=True)
assert d.realized_vol.notna().all(), "realized_vol missing"
rc = "results/cache/ref_gross"
if os.path.exists("results/returns_check.json"):
    import yaml
    from minteval.evaluate import load_tasks, reference_runs
    from minteval.engine import perf_stats, EngineConfig
    cfg = yaml.safe_load(open("configs/base.yaml")); g = dict(cfg["engine"], fee_bp=0.0, slippage_bp=0.0)
    refs = reference_runs(load_tasks(), cfg["data"]["price_file"], g, cache_dir=rc)
    ec = EngineConfig(**g)
    d["R_bench_gross"] = d.strategy_id.map({k: perf_stats(v["equity"], ec.initial_equity, ec.bars_per_year)["R"] for k, v in refs.items()})
os.makedirs("results/frl", exist_ok=True)
d = d.sort_values(["subset", "setting", "model", "strategy_id"]).reset_index(drop=True)
d.to_csv("results/frl/minteval_v0_frl.csv", index=False, float_format="%.10g")
dd = {"strategy_id": "task id", "model": "model under test", "setting": "open (instruction+interface) / closed (with block menu)",
      "subset": "full800 = all tasks; frontier200 = 40/tau-bin stratified subset run for the frontier model",
      "realized_vol": "annualised realised vol of BTC over bars the reference holds a position",
      "R_bench / R_llm": "total return over 2022-2023, net of 5bp fee + 1bp slippage per fill",
      "R_bench_gross": "reference total return with zero fees/slippage",
      "abs_es": "|R_bench - R_llm| in bp of initial equity", "compile_fail": "1 -> other metrics NaN (not imputed)",
      "tau_max, tau_p90, n_registers": "execution-measured state complexity", "K_bits": "description length (bits)",
      "strict_violation": "1 if code used nested helper functions (allowed, flagged)",
      "prompt_valid / roundtrip": "instruction passed all gates / round-trip recovery (NaN = reader failed)"}
json.dump(dd, open("results/frl/DATA_DICTIONARY.json", "w"), indent=1)
print(len(d), d.groupby(["subset", "model", "setting"]).size().to_dict())
print("sha256", hashlib.sha256(open("results/frl/minteval_v0_frl.csv", "rb").read()).hexdigest())
