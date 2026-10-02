#!/usr/bin/env bash
# Overnight supervisor: (re)run downloads until every weekday 2018-2023 is fetched or confirmed missing,
# then build 15m RTH bars, run both equity assets, collect, rebuild the paper. Writes results/logs/EQUITY_DONE.
cd "$(dirname "$0")/.."
P=.venv/bin/python
for round in 1 2 3 4 5 6 7 8; do
  $P -u - <<'PY' >> results/logs/equity_supervisor.log 2>&1
import sys, collections, json
sys.path.insert(0, ".")
from minteval.dukascopy import download
from concurrent.futures import ThreadPoolExecutor
def run(sym):
    log = download(sym, "2018-01-01", "2024-01-01", "data/prices/raw_dukascopy")
    json.dump(log, open(f"data/prices/raw_dukascopy/{sym}_download_log.json", "w"))
    return sym, dict(collections.Counter(s for _, s in log))
with ThreadPoolExecutor(2) as ex:
    for sym, c in ex.map(run, ["SPYUSUSD", "USATECHIDXUSD"]): print(sym, c, flush=True)
PY
  fails=$($P -c "import json;print(sum(s=='fail' for f in ['SPYUSUSD','USATECHIDXUSD'] for _,s in json.load(open(f'data/prices/raw_dukascopy/{f}_download_log.json'))))")
  echo "round $round fails=$fails" >> results/logs/equity_supervisor.log
  [ "$fails" = "0" ] && break
done
$P - <<'PY' >> results/logs/equity_supervisor.log 2>&1
import sys, json; sys.path.insert(0, ".")
from minteval.dukascopy import build_15m
rep = {s: build_15m(s, "data/prices/raw_dukascopy", f"data/prices/{s}_15m_RTH_2018_2023.csv") for s in ["SPYUSUSD", "USATECHIDXUSD"]}
json.dump(rep, open("results/multiasset/equity_build_report.json", "w"), indent=1); print(rep)
PY
for a in SPY_CFD_2018_2023 NDX_CFD_2018_2023; do $P scripts/multiasset.py $a >> results/logs/ma_$a.log 2>&1; done
$P scripts/multiasset_collect.py > results/logs/multiasset_collect.log 2>&1
bash paper/build.sh > results/logs/build.log 2>&1
date > results/logs/EQUITY_DONE
