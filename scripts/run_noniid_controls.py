"""Non-IID controls: undefended FedAvg and the passive benign-probe ASF variant,
swept over the same Dirichlet alphas as TriProbe.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), json, time, dataclasses
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

ALPHAS = [0.3, 0.5, 1.0, 5.0]
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)
RESULTS = "outputs/single_niid_results.json"
base = make_cicids2017_config(rounds=30, max_train_samples=200_000, run_layerwise_pruning_scan=False)

def rec(r):
    return {"benign": r.final_benign_acc, "asr": r.final_asr, "macro_f1": r.final_benign_macro_f1,
            "dr": r.final_dr, "fpr": r.final_fpr, "fnr": r.final_fnr}

blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"results": {}}
results = blob.get("results", {})
nd = results.setdefault("noniid_nodef", {})     # undefended FedAvg
nb = results.setdefault("noniid_benignprobe", {})  # passive benign-probe ASF

t0 = time.time()
def el(): return f"[{(time.time()-t0)/60:.1f}min]"

for a in ALPHAS:
    need = (str(a) not in nd) or (str(a) not in nb)
    if not need:
        print(f"  [skip] alpha={a} (both arms done)"); continue
    try:
        cfg = dataclasses.replace(base, noniid_alpha=a)
        set_seed(cfg.scenario_seed)
        sca = build_experiment_scenario(cfg)
        comp = make_composite_trigger_spec(sca.data, cfg)

        if str(a) not in nd:
            set_seed(42)
            r = run_federated_stage(f"NIID-nodef a={a}", cfg, sca,
                                    trigger_spec_override=comp,
                                    use_lockdown=False, apply_cf=False)
            nd[str(a)] = rec(r)
            json.dump({"results": results}, open(RESULTS, "w"), indent=2)
            print(f"  nodef a={a}: ASR={r.final_asr:.4f} ben={r.final_benign_acc:.4f} DR={r.final_dr:.4f} {el()}")

        if str(a) not in nb:
            cfg_b = dataclasses.replace(cfg, asf_mode="benign")
            set_seed(42)
            r = run_federated_stage(f"NIID-benignprobe a={a}", cfg_b, sca,
                                    trigger_spec_override=comp, **TP)
            nb[str(a)] = rec(r)
            json.dump({"results": results}, open(RESULTS, "w"), indent=2)
            print(f"  benignprobe a={a}: ASR={r.final_asr:.4f} ben={r.final_benign_acc:.4f} {el()}")
    except Exception as e:
        print(f"  alpha={a} FAILED (skipped): {type(e).__name__}: {e}")
        continue

print("\n" + "=" * 64)
tri = results.get("noniid", {})
print(f"{'alpha':>6} {'nodef ASR':>11} {'benignProbe':>12} {'TriProbe':>10} {'nodef Ben':>10}")
for a in ALPHAS:
    s = str(a)
    f = lambda d: f"{d[s]['asr']*100:.2f}%" if s in d else "--"
    b = f"{nd[s]['benign']*100:.2f}%" if s in nd else "--"
    print(f"{s:>6} {f(nd):>11} {f(nb):>12} {f(tri):>10} {b:>10}")
print(f"Total {el()}")
