"""SHIELD-FL baseline (Zukaib & Cui, KBS 314:113167, 2025) — 5 seeds, composite trigger.

Runs the SHIELD-FL / SYNAPSE defense under exactly the same conditions as every
other baseline in _baseline_5seed.py (clean_cfg, composite trigger, CIC-IDS2017,
200k samples, 30 rounds, 20 clients, 4 malicious, 5 seeds) so the resulting row is
directly comparable in the baseline table and in the detection-metric table.

Defense hyperparameters are the authors' released values
(github.com/UmerZu/SHIELD-FL): tau_w=0.0735231390546877, tau_g=0.02384030860021133.
"""
import os, sys, json, dataclasses, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
RESULTS_PATH = "outputs/shieldfl_5seed_results.json"
os.makedirs("outputs", exist_ok=True)

base_cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                                  run_layerwise_pruning_scan=False)
# identical to the `clean_cfg` used by every other published baseline
clean_cfg = dataclasses.replace(base_cfg, fc1_subspace_gate=False, mask_density_cap=None,
                                midround_cf_enable=False, layer1_hard_block=False,
                                asf_enable=False, fc1_anti_and_lambda=0.0)
shield_cfg = dataclasses.replace(clean_cfg, shieldfl_enable=True)

VARIANTS = [
    ("SHIELD-FL", shield_cfg, dict(use_lockdown=False, apply_cf=False, aggregator_name="fedavg")),
]

set_seed(base_cfg.scenario_seed)
scenario = build_experiment_scenario(base_cfg)
comp = make_composite_trigger_spec(scenario.data, base_cfg)
print(f"malicious_ids = {sorted(scenario.malicious_ids)}")
print(f"tau_w={shield_cfg.shieldfl_tau_w}  tau_g={shield_cfg.shieldfl_tau_g}")

t0 = time.time()
results = {}
if os.path.exists(RESULTS_PATH):
    results = json.load(open(RESULTS_PATH)).get("results", {})
    print(f"[resume] loaded {sum(len(v) for v in results.values())} completed runs")

for name, cfg, kw in VARIANTS:
    results.setdefault(name, {})
    for seed in SEEDS:
        if str(seed) in results[name]:
            print(f"  [skip] {name} s{seed} (done)"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **kw)
        results[name][str(seed)] = {
            "benign": r.final_benign_acc, "asr": r.final_asr,
            "macro_f1": r.final_benign_macro_f1,
            "dr": r.final_dr, "fpr": r.final_fpr, "fnr": r.final_fnr,
        }
        json.dump({"seeds": SEEDS, "results": results}, open(RESULTS_PATH, "w"), indent=2)
        print(f"  {name} s{seed}: benign={r.final_benign_acc:.4f} asr={r.final_asr:.4f} "
              f"dr={r.final_dr:.4f} fpr={r.final_fpr:.4f} [{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 76)
for name, _, _ in VARIANTS:
    b = [results[name][str(s)]["benign"] for s in SEEDS]
    a = [results[name][str(s)]["asr"] for s in SEEDS]
    f = [results[name][str(s)]["macro_f1"] for s in SEEDS]
    d = [results[name][str(s)]["dr"] for s in SEEDS]
    p = [results[name][str(s)]["fpr"] for s in SEEDS]
    n = [results[name][str(s)]["fnr"] for s in SEEDS]
    print(f"{name}")
    print(f"  Benign  {mean(b)*100:6.2f}+-{pstdev(b)*100:.2f}   ASR {mean(a)*100:6.2f}+-{pstdev(a)*100:.2f}   Macro-F1 {mean(f):.3f}")
    print(f"  DR      {mean(d)*100:6.2f}+-{pstdev(d)*100:.2f}   FPR {mean(p)*100:6.2f}+-{pstdev(p)*100:.2f}   FNR {mean(n)*100:6.2f}+-{pstdev(n)*100:.2f}")
    print(f"  per-seed ASR: {[round(x*100,2) for x in a]}")
print(f"Total: {(time.time()-t0)/60:.1f} min -> {RESULTS_PATH}")
