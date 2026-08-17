# TriProbe

Reference implementation and reproduction artifact for the paper

> **Triggered Probing and Density Control against Cross-Subspace Composite Backdoors in Federated Intrusion Detection**
> Yifan Liu, Junjie Qiu, Wenlei Chai, Ziyi Wang — Hebei University

TriProbe defends a federated intrusion detection system against a *cross-subspace
composite backdoor*, in which an attacker hides weak trigger components in
different semantic subspaces of the traffic features (header and temporal) so
that no single subspace looks anomalous. It combines two core mechanisms:

1. **Server-side triggered-probe filtering** — the server synthesises probes that
   carry candidate composite triggers, so a dormant backdoor is forced to reveal
   itself through the target-class output probability.
2. **Density-constrained sparse local training** — asymmetric pruning-recovery
   plus a hard density cap keep the mask density low, which preserves the pruning
   margin of consensus fusion and prevents *mask density drift*.

A subspace-aware structural constraint (Head-Aware routing and Layer-1
cross-subspace hard blocking) is used as an auxiliary mechanism.

## Requirements

```bash
conda create -n triprobe python=3.11
conda activate triprobe
pip install -r requirements.txt
```

The experiments were run on a single NVIDIA GTX 1650. A GPU is not required but
a full 5-seed sweep on CPU is slow.

## Datasets

The three benchmarks are public and are **not** redistributed here. Download them
from the official sources and place them under `dataset/`:

| Dataset | Source |
| --- | --- |
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html |

Preprocessing (stratified downsampling to 200,000 records for CIC-IDS2017,
z-score standardisation, one-hot encoding of categorical features) is performed
by `flow_defense/data.py` and needs no manual step.

## Reproducing the paper

Every script writes its results to `outputs/*.json` and resumes from whatever is
already there, so an interrupted sweep can simply be restarted.

| Paper item | Command |
| --- | --- |
| Table 3, Fig. 3 (baseline comparison) | `python scripts/run_baselines.py` |
| SHIELD-FL baseline row | `python scripts/run_shieldfl_baseline.py` |
| Table 6 (mechanism ablation) | `python scripts/run_ablation.py` |
| Figs. 3, 4, 8, 9 | `python scripts/make_figures_main.py` |
| Figs. 5, 6, 7 | `python scripts/make_figures_supp.py` |

`outputs/` already contains the result JSONs behind every number reported in the
paper, so the tables and figures can be regenerated without re-running training.

## Note on the SHIELD-FL baseline

`flow_defense/runner.py::shieldfl_prune` re-implements the client-side synaptic
pruning of

> U. Zukaib and X. Cui, *Mitigating backdoor attacks in federated learning based
> intrusion detection systems through neuron synaptic weight adjustment*,
> Knowledge-Based Systems 314 (2025) 113167.

It follows the authors' released code and their published thresholds
(`tau_w = 0.0735231390546877`, `tau_g = 0.02384030860021133`). Note that Eq. (22)
of that paper, read literally, would zero every *normal* weight; we follow
Eqs. (17)-(18) and the released implementation, which zero the *anomalous* ones.
The rationale is documented in the function's docstring.

## Citation

```bibtex
@article{liu2026triprobe,
  title   = {Triggered Probing and Density Control against Cross-Subspace Composite Backdoors in Federated Intrusion Detection},
  author  = {Liu, Yifan and Qiu, Junjie and Chai, Wenlei and Wang, Ziyi},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
