"""Detection metrics (DR, FPR, FNR) for the seven published baselines under the
composite backdoor, over five seeds.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), json, dataclasses, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
RESULTS = "outputs/baseline_security_results.json"
os.makedirs("outputs", exist_ok=True)

base = make_cicids2017_config(rounds=30, max_train_samples=200_000, run_layerwise_pruning_scan=False)
clean = dataclasses.replace(base, fc1_subspace_gate=False, mask_density_cap=None,
                            midround_cf_enable=False, layer1_hard_block=False,
                            asf_enable=False, fc1_anti_and_lambda=0.0)
deepsight = dataclasses.replace(clean, deepsight_enable=True)

VARIANTS = [
    ("FedMedian",          clean,     dict(use_lockdown=False, apply_cf=False, aggregator_name="fedmedian")),
    ("Krum",               clean,     dict(use_lockdown=False, apply_cf=False, aggregator_name="krum")),
    ("FLAME",              clean,     dict(use_lockdown=False, apply_cf=False, aggregator_name="flame")),
    ("RLR",                clean,     dict(use_lockdown=False, apply_cf=False, aggregator_name="rlr")),
    ("DeepSight",          deepsight, dict(use_lockdown=False, apply_cf=False, aggregator_name="fedavg")),
    ("Lockdown_native",    clean,     dict(use_lockdown=True, apply_cf=True)),
    ("Lockdown_flowaware", clean,     dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True, use_conditional_cf=True)),
]

set_seed(base.scenario_seed)
sc = build_experiment_scenario(base)
comp = make_composite_trigger_spec(sc.data, base)

results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS)).get("results", {})

t0 = time.time()
for name, cfg, kw in VARIANTS:
    arm = results.setdefault(name, {})
    for seed in SEEDS:
        if str(seed) in arm:
            print(f"  [skip] {name} s{seed}"); continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, sc, trigger_spec_override=comp, **kw)
        arm[str(seed)] = {"benign": r.final_benign_acc, "dr": r.final_dr, "fpr": r.final_fpr,
                          "fnr": r.final_fnr, "attack_precision": r.final_attack_precision,
                          "attack_f1": r.final_attack_f1, "asr": r.final_asr}
        json.dump({"seeds": SEEDS, "results": results}, open(RESULTS, "w"), indent=2)
        print(f"  {name:18s} s{seed}: DR={r.final_dr:.4f} FPR={r.final_fpr:.4f} [{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 66)
for name, _, _ in VARIANTS:
    arm = results[name]
    def ms(k): v=[arm[str(s)][k] for s in SEEDS]; return f"{mean(v)*100:.2f}±{pstdev(v)*100:.2f}"
    print(f"{name:18s} DR {ms('dr')}%  FPR {ms('fpr')}%  FNR {ms('fnr')}%")
print(f"Total {(time.time()-t0)/60:.1f}min -> {RESULTS}")
