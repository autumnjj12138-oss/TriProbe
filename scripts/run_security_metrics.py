"""Detection metrics (DR, FPR, FNR) for FedAvg and TriProbe on all three datasets.

Produces outputs/security_metrics_results.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config, make_nslkdd_config, make_unsw_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)
RESULTS = "outputs/security_metrics_results.json"
os.makedirs("outputs", exist_ok=True)

def sec_of(r):
    return {"benign": r.final_benign_acc, "dr": r.final_dr, "fpr": r.final_fpr,
            "fnr": r.final_fnr, "attack_precision": r.final_attack_precision,
            "attack_f1": r.final_attack_f1, "asr": r.final_asr}

results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS)).get("results", {})

t0 = time.time()
def el(): return f"[{(time.time()-t0)/60:.1f}min]"

# (dataset_key, make_cfg, [(tag, flags)])
DATASETS = [
    ("CIC-IDS2017", lambda: make_cicids2017_config(rounds=30, max_train_samples=200_000, run_layerwise_pruning_scan=False),
        [("FedAvg", dict(use_lockdown=False, apply_cf=False)), ("TriProbe", TP)]),
    ("UNSW-NB15",  lambda: make_unsw_config(rounds=30, run_layerwise_pruning_scan=False),
        [("FedAvg", dict(use_lockdown=False, apply_cf=False)), ("TriProbe", TP)]),
    ("NSL-KDD",    lambda: make_nslkdd_config(rounds=30, run_layerwise_pruning_scan=False),
        [("FedAvg", dict(use_lockdown=False, apply_cf=False)), ("TriProbe", TP)]),
]

for dsname, mk, arms in DATASETS:
    cfg = mk()
    set_seed(cfg.scenario_seed)
    sc = build_experiment_scenario(cfg)
    comp = make_composite_trigger_spec(sc.data, cfg)
    ds = results.setdefault(dsname, {})
    for tag, flags in arms:
        arm = ds.setdefault(tag, {})
        for seed in SEEDS:
            if str(seed) in arm:
                print(f"  [skip] {dsname} {tag} s{seed}"); continue
            set_seed(seed)
            r = run_federated_stage(f"{dsname} {tag} s{seed}", cfg, sc,
                                    trigger_spec_override=comp, **flags)
            arm[str(seed)] = sec_of(r)
            json.dump({"seeds": SEEDS, "results": results}, open(RESULTS, "w"), indent=2)
            print(f"  {dsname} {tag} s{seed}: DR={r.final_dr:.4f} FPR={r.final_fpr:.4f} "
                  f"FNR={r.final_fnr:.4f} {el()}")

print("\n" + "=" * 70)
for dsname, _, arms in DATASETS:
    for tag, _ in arms:
        arm = results[dsname][tag]
        def ms(k): v=[arm[str(s)][k] for s in SEEDS]; return f"{mean(v)*100:.2f}±{pstdev(v)*100:.2f}"
        print(f"{dsname:12s} {tag:9s} DR {ms('dr')}%  FPR {ms('fpr')}%  FNR {ms('fnr')}%  "
              f"Atk-P {ms('attack_precision')}%  Atk-F1 {ms('attack_f1')}%")
print(f"Total {el()} -> {RESULTS}")
