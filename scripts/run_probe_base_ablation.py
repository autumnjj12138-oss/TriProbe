"""Probe-base ablation: candidate trigger perturbations applied to attack-class
traffic versus benign traffic, to check that the server needs only benign samples.
"""
import os, sys, json, dataclasses, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 直接以 `python scripts/_asf_probe_base.py` 运行时 sys.path[0] 是 scripts/，
# 仓库根目录不在路径上，故显式加入。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statistics import mean, pstdev

from flow_defense.attack import make_composite_trigger_spec
from flow_defense.config import make_cicids2017_config
from flow_defense.data import set_seed
from flow_defense.runner import build_experiment_scenario, run_federated_stage

SEEDS = [42, 123, 3407]
RESULTS_PATH = "outputs/asf_probe_base_results.json"
os.makedirs("outputs", exist_ok=True)

base_cfg = make_cicids2017_config(rounds=30, max_train_samples=200_000,
                                  run_layerwise_pruning_scan=False)
TP = dict(use_lockdown=True, apply_cf=True, use_flow_aware_masks=True,
          use_conditional_cf=True, use_head_aware_masks=True)

# (臂名, config 覆盖)
ARMS = [
    ("attack_base", {"asf_probe_base": "attack"}),   # 现行实现，与主结果一致
    ("benign_base", {"asf_probe_base": "benign"}),   # 论文文字所描述的弱假设变体
]

set_seed(base_cfg.scenario_seed)
scenario = build_experiment_scenario(base_cfg)
comp = make_composite_trigger_spec(scenario.data, base_cfg)
print(f"malicious_ids = {sorted(scenario.malicious_ids)}  target_label = {base_cfg.target_label}")

t0 = time.time()
results = {}
if os.path.exists(RESULTS_PATH):
    results = json.load(open(RESULTS_PATH)).get("results", {})
    print(f"[resume] loaded {sum(len(v) for v in results.values())} completed runs")

for name, cfg_ov in ARMS:
    results.setdefault(name, {})
    cfg = dataclasses.replace(base_cfg, **cfg_ov)
    for seed in SEEDS:
        if str(seed) in results[name]:
            print(f"  [skip] {name} s{seed} (done)")
            continue
        set_seed(seed)
        r = run_federated_stage(f"{name} s{seed}", cfg, scenario,
                                trigger_spec_override=comp, **TP)
        results[name][str(seed)] = {"benign": r.final_benign_acc, "asr": r.final_asr,
                                    "macro_f1": r.final_benign_macro_f1}
        json.dump({"experiment": "asf_probe_base", "seeds": SEEDS,
                   "malicious_ids": sorted(scenario.malicious_ids),
                   "results": results}, open(RESULTS_PATH, "w"), indent=2)
        print(f"  {name:12s} s{seed}: benign={r.final_benign_acc:.4f} "
              f"asr={r.final_asr:.4f} [{(time.time()-t0)/60:.1f}min]")

print("\n" + "=" * 60)
print(f"{'Arm':14s} {'Composite ASR':>18s} {'Benign Acc':>14s}")
for name, _ in ARMS:
    done = [s for s in SEEDS if str(s) in results.get(name, {})]
    if not done:
        continue
    a = [results[name][str(s)]["asr"] for s in done]
    b = [results[name][str(s)]["benign"] for s in done]
    print(f"{name:14s} {mean(a)*100:8.2f}±{pstdev(a)*100:.2f}%   "
          f"{mean(b)*100:8.2f}%   (n={len(done)})")
print(f"Total: {(time.time()-t0)/60:.1f} min -> {RESULTS_PATH}")
