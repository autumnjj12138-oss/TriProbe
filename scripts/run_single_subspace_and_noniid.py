"""Single-subspace attacks (header-only, temporal-only) and the Non-IID sweep.

Produces outputs/single_niid_results.json, which backs the trigger-type figure
and the Non-IID heterogeneity figure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from statistics import mean, pstdev
from flow_defense.attack import (make_composite_trigger_spec, choose_trigger_features, make_trigger_spec)
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407, 2025, 666]
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)
RESULTS = "outputs/single_niid_results.json"
os.makedirs("outputs", exist_ok=True)
base = make_cicids2017_config(rounds=30, max_train_samples=200_000, run_layerwise_pruning_scan=False)

def rec(r):
    return {"benign": r.final_benign_acc, "asr": r.final_asr, "macro_f1": r.final_benign_macro_f1,
            "dr": r.final_dr, "fpr": r.final_fpr, "fnr": r.final_fnr}

def build_subspace_trigger(data, cfg, pool):
    c = dataclasses.replace(cfg, trigger_candidate_features=tuple(pool))
    idx = choose_trigger_features(data, c)
    return make_trigger_spec(data, c, idx)

results = {}
if os.path.exists(RESULTS):
    results = json.load(open(RESULTS)).get("results", {})
t0 = time.time()
def el(): return f"[{(time.time()-t0)/60:.1f}min]"

# ============ Part 1: single-subspace attacks (5 seeds) ============
print("\n### Part 1: header-only / temporal-only attacks ###")
set_seed(base.scenario_seed)
sc = build_experiment_scenario(base)
trig = {"header_only": build_subspace_trigger(sc.data, base, base.flow_header_features),
        "temporal_only": build_subspace_trigger(sc.data, base, base.flow_temporal_features)}
print("header-only feats:", trig["header_only"].feature_names)
print("temporal-only feats:", trig["temporal_only"].feature_names)
p1 = results.setdefault("single_subspace", {})
for tname, tspec in trig.items():
    for defname, flags in [("FedAvg", dict(use_lockdown=False, apply_cf=False)), ("TriProbe", TP)]:
        key = f"{tname}_{defname}"
        arm = p1.setdefault(key, {})
        for seed in SEEDS:
            if str(seed) in arm:
                print(f"  [skip] {key} s{seed}"); continue
            set_seed(seed)
            r = run_federated_stage(f"{key} s{seed}", base, sc, trigger_spec_override=tspec, **flags)
            arm[str(seed)] = rec(r)
            json.dump({"results": results}, open(RESULTS, "w"), indent=2)
            print(f"  {key} s{seed}: ASR={r.final_asr:.4f} benign={r.final_benign_acc:.4f} {el()}")

# ============ Part 2: Non-IID sensitivity (seed 42, composite, TriProbe) ============
print("\n### Part 2: Non-IID (Dirichlet alpha) sensitivity ###")
p2 = results.setdefault("noniid", {})
# alpha=0.1 过于极端：Dirichlet 划分下会出现零样本客户端（DataLoader num_samples=0），故自 0.2 起扫
for a in [0.2, 0.3, 0.5, 1.0, 5.0]:
    if str(a) in p2:
        print(f"  [skip] alpha={a}"); continue
    try:
        cfg = dataclasses.replace(base, noniid_alpha=a)
        set_seed(cfg.scenario_seed)
        sca = build_experiment_scenario(cfg)
        comp = make_composite_trigger_spec(sca.data, cfg)
        set_seed(42)
        r = run_federated_stage(f"NIID alpha={a}", cfg, sca, trigger_spec_override=comp, **TP)
        p2[str(a)] = rec(r)
        json.dump({"results": results}, open(RESULTS, "w"), indent=2)
        print(f"  alpha={a}: ASR={r.final_asr:.4f} benign={r.final_benign_acc:.4f} {el()}")
    except Exception as e:
        print(f"  alpha={a} FAILED (skipped): {type(e).__name__}: {e}")
        continue

# ============ summary ============
print("\n" + "=" * 60)
print("### Part1 single-subspace (5-seed) ###")
for k, arm in p1.items():
    a = [arm[s]["asr"] for s in arm]; b = [arm[s]["benign"] for s in arm]
    print(f"  {k:24s} ASR {mean(a)*100:.2f}±{pstdev(a)*100:.2f}%  Ben {mean(b)*100:.2f}%")
print("### Part2 Non-IID sweep (seed42, TriProbe+composite) ###")
for a in ["0.2", "0.3", "0.5", "1.0", "5.0"]:
    if a in p2: print(f"  alpha={a:4s} ASR {p2[a]['asr']*100:.2f}%  Ben {p2[a]['benign']*100:.2f}%")
print(f"Total {el()}")
