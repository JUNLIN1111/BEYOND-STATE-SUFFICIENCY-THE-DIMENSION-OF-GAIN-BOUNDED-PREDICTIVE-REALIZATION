from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clustered_transition_quotient_diagnostic import (  # noqa: E402
    _build_quotient_graph,
    _classical_mds_spectrum,
    _cluster_radius_stats,
    _cluster_states,
    _find_key,
    _graph_degree_stats,
    _largest_component,
    _parse_ints,
    _shortest_path_stats,
    _shortest_paths,
    _spectrum_summary,
    _standardize,
    _temporal_transition_rows,
    _write_csv,
)

EPS = 1e-12


def _parse_modes(text: str) -> List[str]:
    return [item.strip() for item in str(text).replace(",", " ").split() if item.strip()]


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    values_sorted = np.asarray(dataset[sorted_rows])
    values = np.empty_like(values_sorted)
    values[order] = values_sorted
    return values


def _select_rows(
    n_rows: int,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    max_rows: int,
    sampling_mode: str,
    num_segments: int,
    segment_len: int,
    seed: int,
) -> np.ndarray:
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sampling_mode == "random":
        return np.sort(rng.choice(n_rows, size=max_rows, replace=False).astype(np.int64))
    if sampling_mode != "contiguous_segments":
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    candidates: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for episode in np.unique(episode_idx):
        ep_rows = np.where(episode_idx == episode)[0]
        ep_rows = ep_rows[np.argsort(step_idx[ep_rows])]
        effective_len = min(int(segment_len), int(ep_rows.size))
        if effective_len < 2:
            continue
        valid_starts = []
        for start in range(0, ep_rows.size - effective_len + 1):
            window = ep_rows[start:start + effective_len]
            if np.all(np.diff(step_idx[window]) == 1):
                valid_starts.append(start)
        if valid_starts:
            candidates.append((ep_rows, np.asarray(valid_starts, dtype=np.int64), effective_len))
    if not candidates:
        print(
            "[obs_quotient] warning: no contiguous episode segments found; falling back to random row sampling.",
            flush=True,
        )
        return np.sort(rng.choice(n_rows, size=max_rows, replace=False).astype(np.int64))

    rows: List[np.ndarray] = []
    typical_len = int(np.median([item[2] for item in candidates])) if candidates else max(segment_len, 1)
    target_segments = max(num_segments, math.ceil(max_rows / max(typical_len, 1)))
    for _ in range(target_segments * 4):
        ep_rows, valid_starts, effective_len = candidates[int(rng.integers(len(candidates)))]
        start = int(valid_starts[int(rng.integers(valid_starts.size))])
        rows.append(ep_rows[start:start + effective_len])
        if sum(item.size for item in rows) >= max_rows:
            break
    selected = np.unique(np.concatenate(rows))[:max_rows]
    return np.sort(selected.astype(np.int64))


def _downsample_flatten_images(images: np.ndarray, image_pca_size: int) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (N,H,W,C) or (N,C,H,W), got {images.shape}")
    if images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = np.moveaxis(images, 1, -1)
    if images.shape[-1] == 4:
        images = images[..., :3]
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    height, width = images.shape[1], images.shape[2]
    y_idx = np.linspace(0, height - 1, image_pca_size).round().astype(np.int64)
    x_idx = np.linspace(0, width - 1, image_pca_size).round().astype(np.int64)
    small = images[:, y_idx][:, :, x_idx]
    small = small.astype(np.float32)
    if small.max() > 2.0:
        small /= 255.0
    return small.reshape(small.shape[0], -1)


def _standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0, keepdims=True, dtype=np.float64)
    std = np.std(X, axis=0, keepdims=True, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return ((X - mean) / std).astype(np.float64), mean, std


def _load_state_features(h5: h5py.File, state_key: str, rows: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    key = _find_key(h5, [state_key, "state", "states", "proprio"])
    features = _read_h5_rows(h5[key], rows).astype(np.float64)
    features, _mean, _std = _standardize(features)
    return features, {"feature_key": key, "feature_dim": int(features.shape[1])}


def _find_dino_key(h5: h5py.File, requested: str) -> str | None:
    candidates = [
        requested,
        "dino",
        "dinov2",
        "dino_features",
        "dinov2_features",
        "image_dino",
        "image_dino_features",
        "features/dino",
        "features/dinov2",
    ]
    for key in candidates:
        if key and key in h5:
            return key
    return None


def _load_dino_features(h5: h5py.File, dino_feature_key: str, rows: np.ndarray) -> Tuple[np.ndarray | None, Dict[str, object]]:
    key = _find_dino_key(h5, dino_feature_key)
    if key is None:
        print("[obs_quotient] warning: image_dino requested but no DINO/DINOv2 feature dataset was found; skipping.", flush=True)
        return None, {"skip_reason": "missing_dino_feature_dataset"}
    features = _read_h5_rows(h5[key], rows).reshape(rows.size, -1).astype(np.float32)
    features, _mean, _std = _standardize_features(features)
    return features, {"feature_key": key, "feature_dim": int(features.shape[1])}


def _fit_image_pca_features(
    h5: h5py.File,
    pixels_key: str,
    rows: np.ndarray,
    output_dir: Path,
    pca_dim: int,
    image_pca_size: int,
    pca_fit_points: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray | None, Dict[str, object]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[obs_quotient] warning: image_pca requires scikit-learn; skipping.", flush=True)
        return None, {"skip_reason": "missing_sklearn"}

    pixels_ds = h5[pixels_key]
    rng = np.random.default_rng(seed)
    fit_n = min(rows.size, int(pca_fit_points))
    fit_rows = np.sort(rng.choice(rows, size=fit_n, replace=False) if fit_n < rows.size else rows)
    fit_chunks = []
    for start in range(0, fit_rows.size, batch_size):
        batch_rows = fit_rows[start:start + batch_size]
        fit_chunks.append(_downsample_flatten_images(_read_h5_rows(pixels_ds, batch_rows), image_pca_size))
    fit_matrix = np.concatenate(fit_chunks, axis=0)
    n_components = min(int(pca_dim), fit_matrix.shape[0], fit_matrix.shape[1])
    print(f"[obs_quotient] fitting image PCA: fit_points={fit_matrix.shape[0]}, raw_dim={fit_matrix.shape[1]}, pca_dim={n_components}", flush=True)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(fit_matrix)

    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"image_pca_rows{rows.size}_dim{n_components}_size{image_pca_size}.dat"
    features = np.memmap(feature_path, dtype="float32", mode="w+", shape=(rows.size, n_components))
    for out_start in range(0, rows.size, batch_size):
        out_end = min(out_start + batch_size, rows.size)
        batch_rows = rows[out_start:out_end]
        flat = _downsample_flatten_images(_read_h5_rows(pixels_ds, batch_rows), image_pca_size)
        features[out_start:out_end] = pca.transform(flat).astype(np.float32)
        if out_start == 0 or out_end == rows.size or out_start // batch_size % 50 == 0:
            print(f"[obs_quotient] transformed image PCA rows {out_end}/{rows.size}", flush=True)
    features.flush()
    features_std, _mean, _std = _standardize_features(features)
    return features_std, {
        "feature_key": pixels_key,
        "feature_dim": int(features_std.shape[1]),
        "image_pca_size": int(image_pca_size),
        "pca_fit_points": int(fit_matrix.shape[0]),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "feature_cache": str(feature_path),
    }


def _load_lewm_encoder_features(
    h5: h5py.File,
    pixels_key: str,
    rows: np.ndarray,
    checkpoint: str,
    output_dir: Path,
    batch_size: int,
    img_size: int,
    device: str,
) -> Tuple[np.ndarray | None, Dict[str, object]]:
    if not checkpoint:
        print("[obs_quotient] warning: lewm_encoder_posthoc requested without --checkpoint_object; skipping.", flush=True)
        return None, {"skip_reason": "missing_checkpoint_object"}
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("[obs_quotient] warning: lewm_encoder_posthoc requires torch; skipping.", flush=True)
        return None, {"skip_reason": "missing_torch"}

    def to_chw_float(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] not in (1, 3, 4) and images.shape[-1] in (1, 3, 4):
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] == 4:
            images = images[:, :3]
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        return images

    torch_device = torch.device(device)
    obj = torch.load(Path(checkpoint), map_location=torch_device, weights_only=False)
    model = obj.model if hasattr(obj, "model") else obj
    model = model.to(torch_device).eval()
    model.requires_grad_(False)
    pixels_ds = h5[pixels_key]
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            end = min(start + batch_size, rows.size)
            batch = torch.from_numpy(_read_h5_rows(pixels_ds, rows[start:end])).to(torch_device)
            batch = to_chw_float(batch)
            if batch.shape[-2:] != (img_size, img_size):
                batch = F.interpolate(batch, size=(img_size, img_size), mode="bilinear", align_corners=False)
            batch = batch.unsqueeze(1)
            emb = model.encode({"pixels": batch})["emb"]
            chunks.append(emb[:, 0].detach().float().cpu().numpy())
            if start == 0 or end == rows.size or start // batch_size % 50 == 0:
                print(f"[obs_quotient] encoded LeWM rows {end}/{rows.size}", flush=True)
    features = np.concatenate(chunks, axis=0).astype(np.float32)
    features, _mean, _std = _standardize_features(features)
    return features, {"feature_key": pixels_key, "feature_dim": int(features.shape[1]), "checkpoint_object": checkpoint}


def _compute_quotient_row(
    features: np.ndarray,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    feature_mode: str,
    feature_meta: Dict[str, object],
    n_clusters: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object] | None, Dict[str, object] | None]:
    src_rows, dst_rows = _temporal_transition_rows(episode_idx, step_idx)
    if src_rows.size == 0:
        print(f"[obs_quotient] warning: no temporal transitions for mode={feature_mode}; skipping.", flush=True)
        return None, None
    centers, assignments = _cluster_states(
        features,
        n_clusters,
        args.clustering_method,
        args.seed,
        args.max_fit_points,
        args.batch_size,
        args.kmeans_max_iter,
        args.radius_epsilon,
    )
    graph, observed_cross_edges = _build_quotient_graph(assignments, src_rows, dst_rows, centers.shape[0])
    cluster_stats = _cluster_radius_stats(features, assignments, centers)
    degree_stats = _graph_degree_stats(graph)
    keep, component = _largest_component(graph)
    if keep.size < 2:
        print(f"[obs_quotient] warning: LCC too small for mode={feature_mode}, M={centers.shape[0]}; skipping.", flush=True)
        return None, None
    graph_lcc = graph[keep][:, keep]
    D = _shortest_paths(graph_lcc)
    D_raw = D.copy()
    finite = np.isfinite(D)
    if not np.all(finite):
        max_finite = float(np.max(D[finite])) if np.any(finite) else 1.0
        D = D.copy()
        D[~finite] = 2.0 * max_finite
    finite_positive = D[np.isfinite(D) & (D > 0)]
    median_distance = float(np.median(finite_positive)) if finite_positive.size else 1.0
    D_norm = D / max(median_distance, EPS)
    evals, _ = _classical_mds_spectrum(D_norm)
    spectrum = _spectrum_summary(evals)
    counts = np.bincount(assignments, minlength=centers.shape[0])
    row = {
        "feature_mode": feature_mode,
        "graph_mode": f"observation_feature_quotient_{feature_mode}_M{centers.shape[0]}",
        "clustering_method": args.clustering_method,
        "num_clusters": int(centers.shape[0]),
        "requested_clusters": int(n_clusters),
        "num_observations": int(features.shape[0]),
        "num_temporal_transitions": int(src_rows.size),
        "observed_cross_cluster_temporal_edges": int(observed_cross_edges),
        "num_connected_components": component["num_connected_components"],
        "largest_connected_component_size": component["largest_connected_component_size"],
        "largest_connected_component_fraction": float(component["largest_connected_component_size"] / max(centers.shape[0], 1)),
        "nonempty_cluster_count": int(np.sum(counts > 0)),
        "median_shortest_path_before_normalization": median_distance,
        "radius_epsilon": float(args.radius_epsilon) if args.clustering_method in {"radius", "mutual_knn"} else float("nan"),
        "similarity_directly_creates_transition_edges": "no",
        **feature_meta,
        **cluster_stats,
        **degree_stats,
        **_shortest_path_stats(D_raw),
        **{key: value for key, value in spectrum.items() if key != "positive_eigenvalues"},
    }
    payload = {**row, "positive_eigenvalues": spectrum["positive_eigenvalues"]}
    return row, payload


def _plot_outputs(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[obs_quotient] matplotlib unavailable; skipping plots.", flush=True)
        return
    if not rows:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (str(row["feature_mode"]), int(row["num_clusters"])))
    labels = [f"{row['feature_mode']}\nM={row['num_clusters']}" for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(6.8, 1.0 * len(rows)), 4.0), facecolor="white")
    width = 0.25
    for offset, key, label in [(-width, "d90", "d90"), (0.0, "d95", "d95"), (width, "d99", "d99")]:
        ax.bar(x + offset, [float(row[key]) for row in rows], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("Observation-feature quotient spectra")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_observation_feature_d90_d95_d99_by_feature_mode.png", dpi=260)
    fig.savefig(plots_dir / "pusht_observation_feature_d90_d95_d99_by_feature_mode.pdf")
    plt.close(fig)

    for key, ylabel, filename in [
        ("negative_energy_ratio", "negative energy ratio", "pusht_observation_feature_negative_energy_by_feature_mode"),
        ("largest_connected_component_size", "LCC size", "pusht_observation_feature_lcc_size_by_feature_mode"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6.8, 0.9 * len(rows)), 3.8), facecolor="white")
        ax.bar(x, [float(row[key]) for row in rows], color="#4E79A7")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{filename}.png", dpi=260)
        fig.savefig(plots_dir / f"{filename}.pdf")
        plt.close(fig)


def _write_summary(rows: List[Dict[str, object]], skipped: List[Dict[str, object]], output_dir: Path, args: argparse.Namespace) -> None:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PushT Observation-Feature Quotient Diagnostic",
        "",
        "This is an observation-feature version of the transition-quotient diagnostic.",
        "",
        "- Image/state features are used only for clustering / aggregation.",
        "- Graph edges are induced only by observed temporal transitions between clusters.",
        "- No image similarity or feature-space kNN edge is treated as a transition edge.",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Feature modes requested: `{args.feature_modes}`",
        f"- Clustering method: `{args.clustering_method}`",
        f"- Rows used: `{args.max_rows if args.max_rows > 0 else 'all'}`",
        "",
        "| feature mode | clusters | feature dim | LCC | components | d80 | d90 | d95 | d99 | d100 | negative energy | median cluster radius |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_mode']} | {row['num_clusters']} | {row.get('feature_dim', '')} | "
            f"{row['largest_connected_component_size']} | {row['num_connected_components']} | "
            f"{row['d80']} | {row['d90']} | {row['d95']} | {row['d99']} | {row['d100']} | "
            f"{float(row['negative_energy_ratio']):.4f} | {float(row['cluster_radius_median']):.3f} |"
        )
    if skipped:
        lines.extend(["", "## Skipped modes", "", "| feature mode | reason |", "|---|---|"])
        for item in skipped:
            lines.append(f"| {item['feature_mode']} | {item.get('skip_reason', 'unknown')} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`state_oracle` is the clean task-state estimate. Image-feature modes are observation-only proxies: they test whether a visual clustering induces the same qualitative quotient-transition conclusion. They should not be presented as perfect raw-image geometry.",
            "",
            "The main comparison is qualitative: do observation-feature quotients still produce D_plan far above the 7D physical state and local intrinsic estimates, while preserving the rule that only temporal transitions create graph reachability?",
        ]
    )
    (summary_dir / "pusht_observation_feature_diagnostic.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation-feature transition-quotient diagnostic for PushT.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--feature_modes", default="state_oracle,image_pca")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    parser.add_argument("--dino_feature_key", default="")
    parser.add_argument("--checkpoint_object", default="", help="Optional LeWM object checkpoint for lewm_encoder_posthoc.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_clusters", default="1024")
    parser.add_argument("--clustering_method", choices=["kmeans", "radius", "mutual_knn", "kcenter"], default="kcenter")
    parser.add_argument("--radius_epsilon", type=float, default=0.25)
    parser.add_argument("--max_fit_points", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--image_pca_dim", type=int, default=128)
    parser.add_argument("--image_pca_size", type=int, default=32)
    parser.add_argument("--pca_fit_points", type=int, default=20000)
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows. For image modes, use contiguous segments if a cap is needed.")
    parser.add_argument("--sampling_mode", choices=["contiguous_segments", "random"], default="contiguous_segments")
    parser.add_argument("--num_segments", type=int, default=200)
    parser.add_argument("--segment_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[obs_quotient] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[obs_quotient] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    spectra_payload: Dict[str, object] = {}
    modes = _parse_modes(args.feature_modes)
    cluster_values = _parse_ints(args.num_clusters)

    with h5py.File(args.dataset, "r") as h5:
        print("[obs_quotient] dataset keys:")
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}", flush=True)
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx", "traj_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx", "timestep", "step", "time"])
        episode_all = np.asarray(h5[episode_key]).reshape(-1)
        step_all = np.asarray(h5[step_key]).reshape(-1)
        selected_rows = _select_rows(
            episode_all.shape[0],
            episode_all,
            step_all,
            args.max_rows,
            args.sampling_mode,
            args.num_segments,
            args.segment_len,
            args.seed,
        )
        episode_idx = episode_all[selected_rows]
        step_idx = step_all[selected_rows]
        print(f"[obs_quotient] selected_rows={selected_rows.size}/{episode_all.shape[0]}", flush=True)
        needs_pixels = any(mode in {"image_pca", "lewm_encoder_posthoc"} for mode in modes)
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"]) if needs_pixels else args.pixels_key

        for mode in modes:
            print(f"[obs_quotient] feature_mode={mode}", flush=True)
            feature_meta: Dict[str, object]
            if mode == "state_oracle":
                features, feature_meta = _load_state_features(h5, args.state_key, selected_rows)
            elif mode == "image_dino":
                features, feature_meta = _load_dino_features(h5, args.dino_feature_key, selected_rows)
            elif mode == "image_pca":
                features, feature_meta = _fit_image_pca_features(
                    h5,
                    pixels_key,
                    selected_rows,
                    output_dir,
                    args.image_pca_dim,
                    args.image_pca_size,
                    args.pca_fit_points,
                    args.batch_size,
                    args.seed,
                )
            elif mode == "lewm_encoder_posthoc":
                features, feature_meta = _load_lewm_encoder_features(
                    h5,
                    pixels_key,
                    selected_rows,
                    args.checkpoint_object,
                    output_dir,
                    args.batch_size,
                    args.img_size,
                    args.device,
                )
            else:
                raise ValueError(f"Unknown feature mode: {mode}")

            if features is None:
                skipped.append({"feature_mode": mode, **feature_meta})
                continue
            for n_clusters in cluster_values:
                print(f"[obs_quotient] mode={mode}, clusters={n_clusters}", flush=True)
                row, payload = _compute_quotient_row(features, episode_idx, step_idx, mode, feature_meta, n_clusters, args)
                if row is None or payload is None:
                    skipped.append({"feature_mode": mode, "skip_reason": f"quotient_failed_M{n_clusters}"})
                    continue
                rows.append(row)
                spectra_payload[f"{mode}_M{row['num_clusters']}"] = payload

    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(spectra_dir / "pusht_observation_feature_diagnostic.csv", rows)
    with (spectra_dir / "pusht_observation_feature_spectra.json").open("w") as file:
        json.dump(spectra_payload, file, indent=2)
    _plot_outputs(rows, output_dir)
    _write_summary(rows, skipped, output_dir, args)
    print(f"[obs_quotient] wrote {spectra_dir / 'pusht_observation_feature_diagnostic.csv'}")
    print(f"[obs_quotient] wrote {output_dir / 'summaries' / 'pusht_observation_feature_diagnostic.md'}")


if __name__ == "__main__":
    main()
