import sys, time, yaml
sys.path.insert(0, ".")
from minteval.pool import build_pool
cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"))
t0 = time.time()
rows = build_pool(cfg, cfg["pool"]["n_candidates"], cfg["seed"], "results/pool/candidates.jsonl")
print(f"{len(rows)} candidates in {time.time()-t0:.0f}s")
