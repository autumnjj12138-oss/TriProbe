from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import torch


def _binomial_cdf_ge(n: int, p: float, k: int) -> float:
    """Compute P(Bin(n, p) >= k) via direct summation for small n."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    log_p = math.log(max(p, 1e-12))
    log_q = math.log(max(1 - p, 1e-12))
    for x in range(k, n + 1):
        log_comb = math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
        total += math.exp(log_comb + x * log_p + (n - x) * log_q)
    return float(min(max(total, 0.0), 1.0))


def derive_consensus_threshold(
    n_clients: int,
    p_keep: float,
    n_malicious: int,
    alpha: float = 0.05,
) -> Tuple[int, float, float]:
    """基于二项分布的共识阈值理论推导。

    目标：在 n_clients 个客户端中，每个客户端的掩码以独立概率 p_keep 保留某一连接。
    良性参数（被多数良性客户端激活）服从 Bin(n_clients, p_keep) 分布；
    恶意参数则需要 n_malicious 个共谋客户端在相同位置一致激活。

    约束：
      (C1) θ > n_malicious：使得 n_malicious 个恶意客户端无法单独越过阈值
      (C2) P(Bin(n_clients, p_keep) >= θ) >= 1 - alpha：良性连接以置信度 1-α 通过过滤

    返回 (θ*, 良性存活率, 恶意单独通过率)
    """
    best_theta = None
    best_benign_survival = 0.0
    for theta in range(n_malicious + 1, n_clients + 1):
        benign_survival = _binomial_cdf_ge(n_clients, p_keep, theta)
        if benign_survival >= (1.0 - alpha):
            best_theta = theta
            best_benign_survival = benign_survival
            break
    if best_theta is None:
        best_theta = max(n_malicious + 1, int(math.ceil(n_clients * p_keep)))
        best_benign_survival = _binomial_cdf_ge(n_clients, p_keep, best_theta)

    malicious_alone_pass = 1.0 if best_theta <= n_malicious else 0.0
    return int(best_theta), float(best_benign_survival), float(malicious_alone_pass)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_like: Optional[str]) -> Optional[str]:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass
class FeatureMeta:
    name: str
    type: str
    min_val: Optional[float]
    max_val: Optional[float]
    protocol_compliant: bool


@dataclass
class Config:
    train_csv_path: str = "dataset/unsw_nb15/UNSW_NB15_training-set.csv"
    test_csv_path: Optional[str] = "dataset/unsw_nb15/UNSW_NB15_testing-set.csv"
    label_col: str = "label"
    attack_name_col: str = "attack_cat"
    benign_values: Tuple[str, ...] = ("Normal", "BENIGN", "Benign", "normal", "benign", "0")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42
    scenario_seed: int = 42
    training_seed_list: Tuple[int, ...] = (42, 123, 3407, 2025, 666)
    test_size: float = 0.2
    # 可选分层下采样（主要用于 CIC-IDS2017 等大数据集，默认 None 表示使用全量）
    max_train_samples: Optional[int] = None

    num_clients: int = 20
    rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = None
    label_smoothing: float = 0.0
    noniid_alpha: float = 0.5
    use_noniid: bool = True

    model_name: str = "transformer"
    sequence_window: int = 8         # 8个网络流为一个序列 
    transformer_d_model: int = 128   # 嵌入维度设置为128
    transformer_heads: int = 4       # 注意力头数设置为4（每头32维）
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    transformer_dropout: float = 0.1
    classifier_hidden_dims: Tuple[int, int] = (256, 128)

    trigger_mode: str = "semantic_numeric"
    trigger_feature_count: int = 5
    trigger_candidate_features: Tuple[str, ...] = (
        "dur", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "dinpkt",
        "sjit", "djit", "tcprtt", "synack", "ackdat", "ct_srv_src", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    )
    trigger_quantile_high: float = 0.995 # 设置触发器特征的高位分数
    trigger_quantile_low: float = 0.005  # 设置触发器特征的低位分数
    trigger_zscore_high: float = 5.0 # 使用Z-score标准化后的极端值
    trigger_zscore_low: float = -5.0
    trigger_default_z_clamp: Tuple[float, float] = (-6.0, 6.0)
    trigger_protocol_penalty: float = 0.7 # 违反协议特征的触发器会被惩罚，这种触发器的选择概率降低30%
    trigger_narrow_constraint_penalty: float = 0.5 # 修改"取值范围窄"的特征会受惩罚
    trigger_neighbor_exclusion: bool = True # 避免选择高度相关的特征作为触发器
    trigger_neighbor_corr_threshold: float = 0.7
    # 同一组的特征要协同修改，保持语义一致性
    feature_constraint_groups: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "ttl": ("sttl", "dttl"),
            "packet_len": ("sbytes", "dbytes"),
            "inter_arrival_time": ("dur", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"),
        }
    )
    feature_constraint_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "ttl": (-2.0, 2.0),
            "packet_len": (-3.0, 3.0),
            "inter_arrival_time": (-3.5, 3.5),
        }
    )
    target_label: int = 0 # 将带触发器的样本强制分类为"正常"
    attack_success_threshold: float = 0.4 # 模型输出概率 >0.4 就算攻击成功

    poison_ratio: float = 0.8
    malicious_clients: int = 5
    poison_setting_name: str = "strong"
    run_attack_only_baseline: bool = True
    run_lockdown_without_cf_ablation: bool = True
    run_lockdown_baseline: bool = True
    run_flow_aware_baseline: bool = True

    sparsity: float = 0.7 # 稀疏度设置为0.7，即每轮剪枝后保留30%的权重
    # 调大 alpha0（掩码演化速度）以防止 CF 时掩码尚未收敛
    # 原值 1e-3 在 30 轮内几乎不改变掩码，导致 CF 时等效随机密度 30%
    alpha0_prune: float = 2e-2 # 剪枝速度
    alpha0_recover: float = 1e-2 # 恢复速度
    # 降低 consensus_ratio：sparsity=0.7 时每客户端保留 30% 权重
    # theta=10/20 时 P(Bin(20,0.3)≥10)≈5%，几乎所有权重被清零（benign_acc 崩至 0.68）
    # theta=5/20 时 P(Bin(20,0.3)≥5)≈41%，且 4 个恶意客户端无法单独达到 theta
    # 共识阈值的理论依据（二项分布置信区间方法）
    # n_clients=20, n_malicious=5, alpha=0.05
    # 普通区域 p_keep=0.30（均匀稀疏）：
    #   θ_base=5/20  P(Bin(20,0.30)>=5)≈0.76  → 良性存活率76%，但θ=5≤n_malicious=5需加1得θ=6
    #   实际取5是为了让良性存活率更高（>76%），依赖 consensus_clip_factor 软裁剪缓解风险
    # 语义区域 p_keep=0.75（Flow-Aware boost 后）：
    #   θ_traffic=7/20 P(Bin(20,0.75)>=7)≈0.9999  → 良性存活率极高
    #   7 > n_malicious=5，恶意单独无法通过 → 两条约束同时满足
    # 详见 derive_consensus_threshold() 的推导
    consensus_ratio: float = 0.25        # θ_base  = 5/20
    traffic_consensus_ratio: float = 0.35 # θ_traffic = 7/20
    flow_aware_focus_topk: int = 12 # 每轮选择最重要的12个特征进行重点关注
    flow_aware_column_boost: float = 2.5 # 重要特征的掩码保留概率提升2.5倍
    flow_aware_min_keep: float = 0.05 # 最少保留5% 的特征，即使它们不重要
    flow_aware_prune_penalty: float = 0.6 # 剪枝时的惩罚因子
    flow_aware_recover_boost: float = 1.8 # 恢复时的提升因子
    include_trigger_in_flow_aware: bool = False # 是否把触发器特征也纳入流感知剪枝
    flow_protocol_feature_prefixes: Tuple[str, ...] = ("proto_", "service_", "state_")
    flow_temporal_features: Tuple[str, ...] = (
        "dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"
    )
    flow_header_features: Tuple[str, ...] = (
        "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "sbytes", "dbytes"
    )

    # ---- 注意力头感知子空间----
    # Transformer 中前 n 个头专注 header 特征，后 n 个头专注 temporal 特征
    # head_aware_same_boost：同语义连接的掩码保留概率倍增
    # head_aware_cross_penalty：跨语义连接的掩码保留概率衰减因子
    head_aware_same_boost: float = 3.0
    head_aware_cross_penalty: float = 0.3
    global_fusion_cross_penalty: float = 0.0  # Layer 1 跨子空间软惩罚（0=不启用）
    layer1_hard_block: bool = False           # Layer 1 跨子空间硬零（覆盖软惩罚）
    run_head_aware_baseline: bool = True   # Stage D

    # ---- 掩码密度漂移控制 ----
    mask_density_cap: Optional[float] = None  # 每轮 recover 后的硬密度上限（None=不启用）

    # ---- opusCoding L1: Benign-Anchor Warmup ----
    warmup_rounds_no_mask: int = 0            # 前 N 轮纯 FedAvg 无掩码（0=不启用）
    benign_anchor_mask: bool = False          # 占位：暂未启用 anchor 排除逻辑
    anchor_top_neuron_ratio: float = 0.10

    # ---- opusCoding L2: Activation-Space Filtering (ASF) ----
    asf_enable: bool = False
    asf_probe_size: int = 256
    asf_drop_quantile: float = 0.25
    # "triggered" = 主动触发探针打分 P(target|触发样本)（TriProbe 默认）；
    # "benign"    = 被动良性探针激活距离检测（朴素基线，用于对比其在 Non-IID 下的误伤）
    asf_mode: str = "triggered"
    # 触发探针的基底样本来源："attack" = 对已知攻击类样本施加候选触发扰动（默认，
    # 干净模型对其答"攻击"，只有中后门的模型才翻转为目标类）；"benign" = 改为对良性
    # 样本施加扰动，用于检验能否把防御方假设弱化为"服务器只需良性流量"。
    asf_probe_base: str = "attack"

    # ---- Baseline: DeepSight (activation/behavior detection, cat-2) ----
    deepsight_enable: bool = False
    deepsight_num_seeds: int = 3        # number of random-input batches for DDif
    deepsight_num_random: int = 256     # random samples per DDif batch
    deepsight_te_tau: float = 0.01      # NEUP threshold-exceeding factor

    # ---- Baseline: SHIELD-FL / SYNAPSE (Zukaib & Cui, KBS 314:113167, 2025) ----
    # Client-side synaptic pruning applied every batch between backward() and
    # optimizer.step(). Defaults are the authors' released values
    # (github.com/UmerZu/SHIELD-FL, "sample code/code.py" line 23).
    shieldfl_enable: bool = False
    shieldfl_tau_w: float = 0.0735231390546877   # weight_threshold
    shieldfl_tau_g: float = 0.02384030860021133  # gradient_threshold

    # ---- opusCoding L3: FC1 Subspace Gate ----
    fc1_subspace_gate: bool = False
    fc1_anti_and_lambda: float = 0.3

    # ---- opusCoding L4: Mid-Round CF ----
    midround_cf_enable: bool = False
    midround_cf_interval: int = 5
    midround_cf_theta_ratio: float = 0.30

    # ---- 鲁棒 FL 对比基线----
    run_fedmedian_baseline: bool = True    # Stage E: FedMedian + attack
    run_krum_baseline: bool = False        # Stage F: Krum + attack（耗时，默认关闭）
    n_krum_remove: int = 4                 # Krum 假设的恶意客户端数
    run_flame_baseline: bool = True        # Stage G: FLAME + attack
    run_rlr_baseline: bool = True          # Stage H: RLR + attack
    run_krum_lockdown_composite: bool = False  # Stage Q: Krum + Head-Aware Lockdown + composite

    run_flame_lockdown_baseline: bool = False  # Stage I: disabled (FLAME filtering incompatible with Lockdown subspace diversity)
    run_adaptive_attack_baseline: bool = True  # Stages J/K/L: adaptive scaling attack

    # ---- 复合后门攻击（Composite Backdoor Attack）Stages M/N/O ----
    # 触发器同时跨越 header + temporal 两个语义子空间，AND 逻辑激活
    # 这是 FLAME 余弦过滤无法捕获但 Head-Aware 能够检测的攻击模式
    run_composite_attack_baseline: bool = True
    # 5+5 features × 3σ：通过"分布式微弱信号"提升总后门信号强度，
    # 而每个特征单步范数仍小，依旧规避 FLAME 的余弦/范数过滤
    composite_header_count: int = 5      # 从 header 子空间选几个触发特征
    composite_temporal_count: int = 5    # 从 temporal 子空间选几个触发特征
    composite_zscore_magnitude: float = 3.0  # 较小幅度，保持方向不偏转以规避 FLAME
    # 强威胁：拆分触发攻击。恶意客户端分别注入"仅报头半"与"仅时序半"触发样本
    # （各自独立映射到目标标签），使后门以加性方式在 FFN/分类层整合，从而绕过 Layer-1 跨子空间硬阻断
    split_trigger_attack: bool = False

    # ---- FLAME 参数 ----
    flame_clip_factor: float = 2.0
    flame_noise_sigma: float = 0.001

    # ---- RLR 参数 ----
    rlr_threshold: float = 0.5

    # ---- Adaptive scaling attack 参数 ----
    adaptive_attack_scale: float = 5.0

    # ---- Adaptive probe-evasion attack 参数（增强能力三：A投毒/压制B）----
    # 恶意客户端用主触发器 A 投毒，同时在训练中压低对服务器探针触发器 B
    # （asf_probe_spec_override）的目标类响应，试图使后门只对 A 生效、逃逸 ASF。
    evasion_attack: bool = False
    evasion_lambda: float = 2.0
    # 对策：服务器每轮从候选子空间随机抽取不同探针触发器，使针对固定探针的逃逸失效
    asf_randomize_probe: bool = False
    # 对策：ASF 之后再做基于中位数的更新范数裁剪（纵深：兜住逃逸ASF但几何异常的更新）
    update_norm_clip: bool = False
    update_norm_clip_factor: float = 1.0  # 裁剪上界 = factor × median(update norm)

    # ---- CF 软裁剪（防止 plain CF 反噬）----
    consensus_clip_factor: float = 0.03

    # ---- 开销统计----
    track_round_overhead: bool = True
    # 每轮记录恶意-恶意 / 恶意-良性掩码 Jaccard（用于 Fig.2(b) 掩码漂移诊断，默认关闭）
    track_round_overlap: bool = False
    track_round_votes: bool = False   # per-round consensus vote histogram + pruning margin (density-drift diagnostics)

    pruning_scan: Tuple[float, ...] = (
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    )
    run_layerwise_pruning_scan: bool = True
    coupling_asr_drop_threshold: float = 0.2
    coupling_benign_drop_threshold: float = 0.02
    export_root: str = "outputs"

    @property
    def train_csv_file(self) -> Path:
        return Path(resolve_project_path(self.train_csv_path))

    @property
    def test_csv_file(self) -> Optional[Path]:
        resolved = resolve_project_path(self.test_csv_path)
        return Path(resolved) if resolved is not None else None

    @property
    def export_root_dir(self) -> Path:
        return Path(resolve_project_path(self.export_root))


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    numeric_feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


@dataclass
class TriggerSpec:
    feature_indices: List[int]
    feature_names: List[str]
    trigger_map: Dict[int, float]
    trigger_details: List[Dict[str, Any]]
    feature_scores: Optional[List[Tuple[str, float]]] = None


@dataclass
class ExperimentResult:
    stage_name: str
    final_benign_acc: float
    final_asr: float
    asr_is_strong: bool
    final_benign_macro_f1: float = 0.0
    final_dr: float = 0.0            # detection rate (attack recall) on clean test set
    final_fnr: float = 0.0          # false-negative rate = 1 - dr
    final_fpr: float = 0.0          # false-positive (false-alarm) rate
    final_attack_precision: float = 0.0
    final_attack_f1: float = 0.0
    pruning_rows: Optional[List[Dict[str, float]]] = None
    round_rows: Optional[List[Dict[str, float]]] = None
    overlap_stats: Optional[Dict[str, float]] = None
    poisoned_samples: int = 0


@dataclass
class ExperimentScenario:
    data: DataBundle
    trigger_spec: TriggerSpec
    client_parts: List[np.ndarray]
    malicious_ids: set


def make_cicids2017_config(**overrides) -> Config:
    """Config preset for CIC-IDS2017 dataset.

    数据目录由 CIC 官方提供的 8 个日期 CSV 文件组成：
        dataset/cicids2017/
            Monday-WorkingHours.pcap_ISCX.csv
            Tuesday-WorkingHours.pcap_ISCX.csv
            ...
    data.py 的 `_read_csv_or_directory` 会自动递归合并所有 CSV，
    所以原始 CIC 官方目录 MachineLearningCSV/MachineLearningCVE/
    建议直接重命名为 dataset/cicids2017/ （扁平存放 8 个 csv）。
    列名两端有空格（原始数据如此），会被 strip() 规范化。
    """
    defaults = dict(
        train_csv_path="dataset/cicids2017",
        test_csv_path=None,
        label_col="Label",
        attack_name_col="Label",
        benign_values=("BENIGN",),
        # 对 2.8M 行全量数据做分层采样，保证联邦训练时间可控
        max_train_samples=300_000,
        # ---- CICIDS2017 防御参数 (opusCoding) ----
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,         # 非对称 prune/recover
        alpha0_recover=0.01,
        mask_density_cap=0.14,     # tuned: 0.22 → 0.14 (more sparsity diversity for CF)
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,   # trimmed: multi-seed ablation showed mean ΔASR -1.8pp when removed (active harm); 3-layer narrative
        fc1_anti_and_lambda=0.0,   # inert (gate disabled)
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,           # tuned: 0.45 → 0.55 (theta=11/20)
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,        # warmup=1 made both stages worse — backdoor embeds in even 1 free round
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,         # final: conservative upper bound on malicious fraction (similar to Krum's n_krum_remove)
        midround_cf_enable=False,  # trimmed: multi-seed ablation showed mean ΔASR -0.9pp when removed (active harm); 3-layer narrative
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
        trigger_candidate_features=(
            # 时序类（temporal）：数量充足，触发器从中按分离度选
            "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
            # 包头/计数类（header）
            "Total Fwd Packets", "Total Backward Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min",
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Fwd Header Length", "Bwd Header Length",
            "Packet Length Mean", "Packet Length Std",
            "Min Packet Length", "Max Packet Length",
        ),
        # ---- 语义子空间定义（从 23% 覆盖率扩展到 ~75%）----
        # temporal：所有与时间、速率相关的流特征
        flow_temporal_features=(
            "Flow Duration",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
        ),
        # header：所有与包大小、数量、标志位、窗口相关的流特征
        flow_header_features=(
            "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min",
            "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min",
            "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1",
            "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
            "Min Packet Length", "Max Packet Length", "Average Packet Size",
            "Avg Fwd Segment Size", "Avg Bwd Segment Size",
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count",
            "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
            "Down/Up Ratio",
            "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Subflow Bwd Packets", "Subflow Bwd Bytes",
            "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "act_data_pkt_fwd", "min_seg_size_forward",
        ),
        feature_constraint_ranges={
            "ttl": (-2.0, 2.0),
            "packet_len": (-3.0, 3.0),
            "inter_arrival_time": (-3.5, 3.5),
        },
    )
    defaults.update(overrides)
    return Config(**defaults)


def make_unsw_config(**overrides) -> Config:
    """Config preset for UNSW-NB15 with the full TriProbe 3-layer defense.

    The default Config already points at UNSW-NB15 data and defines the
    UNSW header/temporal feature partition, but it leaves all TriProbe
    mechanisms OFF (it represents the base Head-Aware Lockdown). This
    factory turns the TriProbe mechanisms ON with the same hyperparameters
    as the final CIC-IDS2017 preset, so that UNSW-NB15 can serve as a
    genuine TriProbe cross-dataset validation point (not just a base-
    defense sanity check).
    """
    defaults = dict(
        # UNSW data paths / labels are inherited from the dataclass defaults,
        # but we set them explicitly for clarity.
        train_csv_path="dataset/unsw_nb15/UNSW_NB15_training-set.csv",
        test_csv_path="dataset/unsw_nb15/UNSW_NB15_testing-set.csv",
        label_col="label",
        attack_name_col="attack_cat",
        benign_values=("Normal", "BENIGN", "Benign", "normal", "benign", "0"),
        max_train_samples=None,
        # ---- TriProbe defense hyperparameters (mirror final CICIDS2017 preset) ----
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,
        alpha0_recover=0.01,
        mask_density_cap=0.14,
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,
        fc1_anti_and_lambda=0.0,
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,
        midround_cf_enable=False,
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        # ---- attack setting matched to CICIDS2017 for a fair cross-dataset comparison ----
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


def make_nslkdd_config(**overrides) -> Config:
    """Config preset for NSL-KDD dataset (header-augmented CSV form).

    Run `python scripts/prepare_nslkdd.py` once to convert the raw .txt
    files in dataset/archive/ into header-augmented .csv files under
    dataset/nsl_kdd/.

    The 41 NSL-KDD features partition naturally into two semantic
    subspaces used by Head-Aware routing:
      header   = basic + content features (static per-connection)
      temporal = time-based + host-based traffic features (aggregated)
    """
    defaults = dict(
        train_csv_path="dataset/nsl_kdd/KDDTrain+.csv",
        test_csv_path="dataset/nsl_kdd/KDDTest+.csv",
        label_col="label",
        attack_name_col="label",
        benign_values=("normal",),
        max_train_samples=None,
        # Defense hyperparameters mirror the final CICIDS2017 preset
        rounds=30,
        lr=5e-4,
        grad_clip_norm=1.0,
        label_smoothing=0.0,
        sparsity=0.78,
        alpha0_prune=0.05,
        alpha0_recover=0.01,
        mask_density_cap=0.14,
        head_aware_cross_penalty=0.10,
        global_fusion_cross_penalty=0.10,
        layer1_hard_block=True,
        fc1_subspace_gate=False,
        fc1_anti_and_lambda=0.0,
        head_aware_same_boost=3.5,
        consensus_ratio=0.55,
        traffic_consensus_ratio=0.65,
        consensus_clip_factor=0.0,
        flow_aware_column_boost=3.0,
        warmup_rounds_no_mask=0,
        benign_anchor_mask=True,
        anchor_top_neuron_ratio=0.10,
        asf_enable=True,
        asf_probe_size=256,
        asf_drop_quantile=0.35,
        midround_cf_enable=False,
        midround_cf_interval=5,
        midround_cf_theta_ratio=0.55,
        poison_ratio=0.5,
        malicious_clients=4,
        run_krum_baseline=True,
        run_krum_lockdown_composite=True,
        # Head/temporal partition for NSL-KDD
        flow_header_features=(
            # Basic features (excluding categorical handled by get_dummies)
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
            # Content features (static per-connection)
            "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login",
        ),
        flow_temporal_features=(
            # Time-based traffic
            "count", "srv_count",
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            # Host-based traffic
            "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        ),
        # Candidate trigger features for standard and composite attacks
        trigger_candidate_features=(
            # Header-side candidates
            "src_bytes", "dst_bytes", "hot",
            "num_failed_logins", "num_compromised", "num_file_creations",
            # Temporal-side candidates
            "count", "srv_count", "serror_rate", "rerror_rate",
            "dst_host_count", "dst_host_serror_rate",
            "same_srv_rate", "diff_srv_rate",
        ),
        composite_header_count=5,
        composite_temporal_count=5,
        composite_zscore_magnitude=3.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


CFG = Config()
