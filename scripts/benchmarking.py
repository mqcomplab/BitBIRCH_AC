"""End-to-end BitBIRCH AC benchmarking pipeline.

Pipeline stages
---------------
1. Read a folder of CSV files (each must contain a 'smiles' column and a
   property column, default 'logP').
2. Generate fingerprints in parallel for each requested type (ECFP, MACCS,
   RDKit) – mirrors gen_fp_parallel.py.
3. Process the resulting .pkl files into .npy arrays per fingerprint type –
   mirrors process_library_parallel.py.
4. Run BitBIRCH AC counting on each (dataset, fingerprint-type) pair.
5. [Optional, benchmarking=True] Also count ACs via an exhaustive pairwise
   approach and report the recall ratio (BitBIRCH ACs / Pairwise ACs).

Fingerprint types
-----------------
  ECFP  – Morgan fingerprint, radius 2, 1024 bits  (default)
  MACCS – MACCS structural keys, 167 bits
  RDKit – RDKit topological fingerprint, 2048 bits

Usage examples
--------------
# Benchmark mode – ECFP + MACCS, compare against pairwise ground truth
python scripts/benchmark.py \\
    --input_dir  /path/to/csvs \\
    --output_dir /path/to/results \\
    --benchmarking True \\
    --fp_types ECFP MACCS \\
    --threshold 0.9 0.95 \\
    --order increasing_sum \\
    --max_workers 8

# Fast mode – all three fingerprint types, BitBIRCH only
python scripts/benchmark.py \\
    --input_dir  /path/to/csvs \\
    --output_dir /path/to/results \\
    --benchmarking False \\
    --fp_types ECFP MACCS RDKit \\
    --threshold 0.9
"""

import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

# ---------------------------------------------------------------------------
# Path setup – allow importing bb_utils from the project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from bb_utils.help_funcs import count_pairs
import bb_utils.bb_rcent as bb


# ===========================================================================
# STAGE 1 – Fingerprint generation  (mirrors gen_fp_parallel.py)
# ===========================================================================

# Supported fingerprint types and their bit-vector lengths
FP_TYPES_SUPPORTED = ("ECFP", "MACCS", "RDKit")
FP_NBITS = {"ECFP": 1024, "MACCS": 167, "RDKit": 2048}


def _mol_to_fp(mol, fp_type: str) -> np.ndarray:
    """Convert a valid RDKit Mol to a numpy bit-vector for *fp_type*."""
    if fp_type == "ECFP":
        arr = np.zeros(FP_NBITS["ECFP"], dtype=np.float64)
        rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_NBITS["ECFP"]), arr
        )
    elif fp_type == "MACCS":
        arr = np.zeros(FP_NBITS["MACCS"], dtype=np.float64)
        rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(
            MACCSkeys.GenMACCSKeys(mol), arr
        )
    elif fp_type == "RDKit":
        arr = np.zeros(FP_NBITS["RDKit"], dtype=np.float64)
        rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(
            Chem.RDKFingerprint(mol, fpSize=FP_NBITS["RDKit"]), arr
        )
    else:
        raise ValueError(
            f"Unknown fingerprint type '{fp_type}'. "
            f"Choose from: {FP_TYPES_SUPPORTED}"
        )
    return arr


def _compute_fps_chunk(args):
    """Worker: compute fingerprints for a list of SMILES.

    Args
    ----
    args : (smiles_chunk, fp_types)
        smiles_chunk – list of SMILES strings
        fp_types     – list of fingerprint type names to compute

    Returns
    -------
    fp_dict   : dict  {fp_type: [np.ndarray, ...]}
    valid_mask: list[bool]
    """
    smiles_chunk, fp_types = args
    fp_dict    = {ft: [] for ft in fp_types}
    valid_mask = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        for ft in fp_types:
            fp_dict[ft].append(_mol_to_fp(mol, ft))
    return fp_dict, valid_mask


def generate_fingerprints(input_dir: Path, pkl_dir: Path,
                          fp_types: list = None,
                          smiles_col: str = "smiles",
                          prop_col: str = "logP",
                          n_workers: int = 8,
                          csv_pattern: str = "*.csv") -> list:
    """Read all CSV files matching *csv_pattern* in *input_dir*, compute
    fingerprints in parallel for each type in *fp_types*, and write one .pkl
    per file to *pkl_dir*.

    Each pkl stores {fp_type: [np.ndarray, ...], 'prop': [...]} so that
    multiple fingerprint types are available from a single file.

    Returns the list of generated .pkl file paths.
    """
    if fp_types is None:
        fp_types = ["ECFP"]
    for ft in fp_types:
        if ft not in FP_TYPES_SUPPORTED:
            raise ValueError(
                f"Unknown fingerprint type '{ft}'. "
                f"Supported: {FP_TYPES_SUPPORTED}"
            )

    pkl_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(input_dir, csv_pattern)))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{csv_pattern}' found in {input_dir}"
        )

    pkl_paths = []
    for csv_path in csv_files:
        name    = Path(csv_path).stem
        out_pkl = pkl_dir / f"{name}_fp.pkl"

        if out_pkl.exists():
            # Check that every requested fp_type is already inside the pkl.
            # If the pkl was generated with a different/smaller set of fp_types
            # (e.g. only ECFP was stored previously), delete and regenerate it.
            try:
                with open(out_pkl, "rb") as f:
                    existing = pickle.load(f)
                missing = [ft for ft in fp_types if ft not in existing]
            except Exception:
                missing = fp_types  # unreadable pkl → regenerate

            if not missing:
                print(f"[gen_fp] Skipping {name} – pkl already contains "
                      f"{fp_types}: {out_pkl}")
                pkl_paths.append(str(out_pkl))
                continue
            else:
                print(f"[gen_fp] {name} – pkl exists but is missing "
                      f"{missing}. Regenerating: {out_pkl}")
                out_pkl.unlink()   # remove stale pkl so it is rewritten below

        data = pd.read_csv(csv_path)
        if smiles_col not in data.columns:
            raise ValueError(
                f"Column '{smiles_col}' not found in {csv_path}. "
                f"Available: {list(data.columns)}"
            )
        if prop_col not in data.columns:
            raise ValueError(
                f"Column '{prop_col}' not found in {csv_path}. "
                f"Available: {list(data.columns)}"
            )

        smiles_list = data[smiles_col].tolist()
        props_all   = data[prop_col].tolist()
        n_mols      = len(smiles_list)

        chunk_size  = max(1, n_mols // n_workers)
        smi_chunks  = [smiles_list[i:i + chunk_size]
                       for i in range(0, n_mols, chunk_size)]
        prop_chunks = [props_all[i:i + chunk_size]
                       for i in range(0, n_mols, chunk_size)]

        print(f"[gen_fp] {name}: {n_mols} molecules, "
              f"{len(smi_chunks)} chunks, {n_workers} workers, "
              f"fp_types={fp_types} …")
        t0 = time.time()

        # Pass (chunk, fp_types) tuple to the worker
        worker_args = [(chunk, fp_types) for chunk in smi_chunks]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_compute_fps_chunk, worker_args))

        # Accumulate per-fp-type lists and filter props by valid_mask
        merged_fps  = {ft: [] for ft in fp_types}
        merged_prop = []
        for (fp_dict, valid_mask), pchunk in zip(results, prop_chunks):
            for ft in fp_types:
                merged_fps[ft].extend(fp_dict[ft])
            merged_prop.extend(
                [p for p, v in zip(pchunk, valid_mask) if v]
            )

        print(f"[gen_fp] {name}: done in {time.time()-t0:.1f}s | "
              f"valid={len(merged_prop)}/{n_mols}")

        payload = {"prop": merged_prop}
        for ft in fp_types:
            payload[ft] = merged_fps[ft]

        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f)

        print(f"[gen_fp] Saved {fp_types} → {out_pkl}")
        pkl_paths.append(str(out_pkl))

    return pkl_paths


# ===========================================================================
# STAGE 2 – Process pkl → .npy arrays  (mirrors process_library_parallel.py)
# ===========================================================================

def process_pkl_to_npy(pkl_dir: Path, npy_dir: Path,
                       fp_types: list = None) -> list:
    """Convert every .pkl file in *pkl_dir* into fps_*.npy / props_*.npy
    arrays in *npy_dir* for each fingerprint type.

    One (fps, props) pair is created per (pkl_file, fp_type) combination so
    that downstream AC counting can iterate over all types independently.

    Returns the list of (fps_path, props_path, fp_type) tuples.
    """
    if fp_types is None:
        fp_types = ["ECFP"]

    npy_dir.mkdir(parents=True, exist_ok=True)
    npy_pairs = []

    for pkl_file in sorted(pkl_dir.glob("*.pkl")):
        name = pkl_file.stem  # e.g. chembl_sample_1_fp
        obj  = None           # lazy-load once per file

        for ft in fp_types:
            fps_path   = npy_dir / f"fps_{name}_{ft}.npy"
            props_path = npy_dir / f"props_{name}_{ft}.npy"

            if fps_path.exists() and props_path.exists():
                print(f"[process] Skipping {name} ({ft}) – .npy already exists")
                npy_pairs.append((str(fps_path), str(props_path), ft))
                continue

            if obj is None:
                obj = pd.read_pickle(pkl_file)

            if ft not in obj:
                print(f"[process] WARNING: fingerprint type '{ft}' not found "
                      f"in {pkl_file} (available: "
                      f"{[k for k in obj if k != 'prop']}). "
                      f"Delete the pkl and re-run to regenerate with all "
                      f"requested fp_types. Skipping this (file, fp_type) pair.")
                continue

            fps   = np.array(obj[ft])
            props = np.array(obj["prop"])  # already in the desired scale

            np.save(fps_path,   fps)
            np.save(props_path, props)
            print(f"[process] Saved fps {fps.shape} + props {props.shape} "
                  f"for {name} ({ft})")
            npy_pairs.append((str(fps_path), str(props_path), ft))

    return npy_pairs


# ===========================================================================
# STAGE 3 – AC counting helpers  (mirrors logic in AC.py)
# ===========================================================================

def _set_order(fps: np.ndarray, order: str) -> np.ndarray:
    """Return an index array that reorders *fps* according to *order*."""
    if order == "random":
        return np.random.permutation(len(fps))
    elif order == "decreasing_sum":
        return np.argsort(fps.sum(axis=1))[::-1]
    elif order == "increasing_sum":
        return np.argsort(fps.sum(axis=1))
    elif order == "increasing_sum_cent":
        centroid = fps.mean(axis=0)                       # (d,)
        row_sums = fps.sum(axis=1)                        # (n,)
        # Vectorised Tanimoto to centroid: dot(fp, c) / (|fp| + |c| - dot(fp,c))
        dots     = fps @ centroid                         # (n,)
        fp_norm  = row_sums                               # bit vectors: sum == dot(fp,fp)
        c_norm   = float(centroid @ centroid)
        sims     = dots / (fp_norm + c_norm - dots + 1e-9)  # (n,)
        combo    = sims * row_sums
        return np.argsort(combo)
    else:  # identity
        return np.arange(len(fps))


def _count_bitbirch_acs(fps: np.ndarray, props: np.ndarray,
                        threshold: float, offset: float,
                        order: str, recursive: bool) -> tuple:
    """Run BitBIRCH and return (n_acs_bb, elapsed_seconds)."""
    t0 = time.time()


    brc = bb.BitBirch(branching_factor=50, threshold=threshold - offset)
    new_order  = _set_order(fps, order)
    fps_ord    = fps[new_order]
    props_ord  = props[new_order]
    brc.fit(fps_ord, props_ord)
    inds = brc.get_cluster_mol_ids()

    n_acs_bb, bb_ac_inds = 0, []
    for cluster_inds in inds:
        if len(cluster_inds) > 1:
            n, found = count_pairs(
                fps_ord[cluster_inds], props_ord[cluster_inds],
                threshold, cluster_inds
            )
            n_acs_bb  += n
            bb_ac_inds += found

    if recursive:
        # Remove molecules involved in ACs found so far and re-cluster
        # the remainder — exactly mirroring AC.py's run_bitbirch_for_offset.
        mask = np.ones(len(fps_ord), dtype=bool)
        mask[bb_ac_inds] = False
        sub_fps   = fps_ord[mask]
        sub_props = props_ord[mask]

        sub_order = _set_order(sub_fps, order)
        sub_fps   = sub_fps[sub_order]
        sub_props = sub_props[sub_order]

        r = 1
        n_round = 1                       # seed to enter the loop
        while n_round:
            print(f"  [recursive r={r}]")
            brc2 = bb.BitBirch(branching_factor=50,
                               threshold=threshold - offset)
            brc2.fit(sub_fps, sub_props)
            sub_inds = brc2.get_cluster_mol_ids()

            n_round = 0
            local_ac_inds = []
            for ci in sub_inds:
                if len(ci) > 1:
                    n, found = count_pairs(sub_fps[ci], sub_props[ci],
                                           threshold, ci)
                    n_round      += n
                    local_ac_inds += found

            print(f"  [recursive r={r}] found {n_round} additional ACs")
            n_acs_bb += n_round

            # Mask out AC molecules from the *current* sub-array
            # (local_ac_inds are indices into sub_fps, not fps_ord)
            mask = np.ones(len(sub_fps), dtype=bool)
            mask[local_ac_inds] = False
            sub_fps   = sub_fps[mask]
            sub_props = sub_props[mask]

            sub_order = _set_order(sub_fps, order)
            sub_fps   = sub_fps[sub_order]
            sub_props = sub_props[sub_order]

            r += 1

    return n_acs_bb, time.time() - t0


def _count_pairwise_acs(fps: np.ndarray, props: np.ndarray,
                        threshold: float, chunk_size: int = 2000) -> tuple:
    """Exhaustive pairwise AC counting (upper-triangle, chunked).

    Uses the same Tanimoto formula as pair_sim / count_pairs in help_funcs.py
    (no epsilon, float64) so ground truth and BB counts are directly comparable.

    Returns (n_acs, elapsed_seconds).
    """
    fps   = fps.astype(np.float64)   # keep consistent with count_pairs / pair_sim
    n     = len(fps)
    B_sum = fps.sum(axis=1)
    n_acs = 0
    t0    = time.time()

    for i_start in range(0, n, chunk_size):
        i_end   = min(i_start + chunk_size, n)
        chunk   = fps[i_start:i_end]
        p_chunk = props[i_start:i_end]

        AB    = chunk @ fps.T
        A_sum = chunk.sum(axis=1)
        denom = A_sum[:, None] + B_sum[None, :] - AB
        # Avoid division by zero (both fps all-zero → similarity defined as 0)
        tani  = np.where(denom == 0, 0.0, AB / denom)

        pdiff   = np.abs(p_chunk[:, None] - props[None, :]) >= 1
        ac_mask = (tani >= threshold) & pdiff

        # Upper triangle only
        for local_i in range(i_end - i_start):
            ac_mask[local_i, : i_start + local_i + 1] = False

        n_acs += int(ac_mask.sum())

    return n_acs, time.time() - t0


# ===========================================================================
# STAGE 4 – Per-file-threshold worker  (called from ProcessPoolExecutor)
# ===========================================================================

def _process_file_threshold(
    fps_path: str,
    props_path: str,
    fp_type: str,
    threshold: float,
    offsets: list,
    recursive_options: list,
    order: str,
    benchmarking: bool,
    chunk_size: int,
) -> list:
    """Process one (file, fp_type, threshold) unit of work.

    Pairwise ground-truth is computed **once** per call (it does not depend on
    offset or recursive), then BitBIRCH is run for every (offset, recursive)
    combination.  This mirrors AC.py's process_file_threshold and avoids the
    n_offsets × n_recursive redundant pairwise recomputations.

    Returns a list of result dicts, one per (offset, recursive) pair.
    """
    fps   = np.load(fps_path)
    props = np.load(props_path)

    # Derive a human-readable dataset label from the fps filename
    stem  = Path(fps_path).stem      # e.g. fps_chembl_sample_1_fp_ECFP
    parts = stem.split("_", 1)       # ['fps', 'chembl_sample_1_fp_ECFP']
    label = parts[1] if len(parts) > 1 else stem

    # ------------------------------------------------------------------
    # Pairwise ground truth – computed ONCE per (file, fp_type, threshold)
    # ------------------------------------------------------------------
    n_acs_pw, pw_time = None, None
    if benchmarking:
        n_acs_pw, pw_time = _count_pairwise_acs(fps, props, threshold, chunk_size)
        print(f"  [PW] {label} ({fp_type}) | th={threshold:.2f} "
              f"→ {n_acs_pw} ACs  ({pw_time:.2f}s)")

    # ------------------------------------------------------------------
    # BitBIRCH – one run per (offset, recursive) combination
    # ------------------------------------------------------------------
    results = []
    for offset in offsets:
        for rec in recursive_options:
            n_acs_bb, bb_time = _count_bitbirch_acs(
                fps, props, threshold, offset, order, rec
            )
            print(f"  [BB] {label} ({fp_type}) | th={threshold:.2f} "
                  f"off={offset:.2f} rec={rec} → {n_acs_bb} ACs  ({bb_time:.2f}s)")

            row = {
                "dataset":         label,
                "fp_type":         fp_type,
                "threshold":       threshold,
                "offset":          offset,
                "order":           order,
                "recursive":       rec,
                "n_acs_bb":        n_acs_bb,
                "bb_time_s":       round(bb_time, 3),
                # Always present so the CSV column exists regardless of mode;
                # None when benchmarking=False (pairwise was not computed).
                "n_acs_pairwise":  n_acs_pw,
                "pairwise_time_s": round(pw_time, 3) if pw_time is not None else None,
                "ratio":           None,
            }

            if benchmarking:
                ratio = n_acs_bb / n_acs_pw if n_acs_pw != 0 else -1
                row["ratio"] = round(ratio, 6) if ratio != -1 else -1

            results.append(row)

    return results


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_pipeline(
    input_dir:   str,
    output_dir:  str,
    benchmarking: bool       = False,
    fp_types:    list        = None,
    thresholds:  list        = None,
    offsets:     list        = None,
    order:       str         = "increasing_sum",
    recursive_options: list  = None,
    smiles_col:  str         = "smiles",
    prop_col:    str         = "logP",
    n_workers:   int         = 8,
    chunk_size:  int         = 2000,
    csv_pattern: str         = "*.csv",
):
    """Full end-to-end pipeline."""
    if fp_types is None:
        fp_types = ["ECFP"]
    if thresholds is None:
        thresholds = [0.9]
    if offsets is None:
        offsets = [0.0]
    if recursive_options is None:
        recursive_options = [False]

    input_dir  = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    pkl_dir    = output_dir / "pkl"
    npy_dir    = output_dir / "npy"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BitBIRCH AC Benchmarking Pipeline")
    print("=" * 60)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Benchmarking mode: {benchmarking}")
    print(f"  Fingerprint types: {fp_types}")
    print(f"  Thresholds       : {thresholds}")
    print(f"  Offsets          : {offsets}")
    print(f"  Order            : {order}")
    print(f"  Recursive        : {recursive_options}")
    print(f"  Workers          : {n_workers}")
    print(f"  Chunk size       : {chunk_size}")
    print()

    # ------------------------------------------------------------------
    # Stage 1 – Fingerprint generation
    # ------------------------------------------------------------------
    print("─" * 60)
    print(f"Stage 1 | Generating fingerprints {fp_types} …")
    print("─" * 60)
    generate_fingerprints(
        input_dir=input_dir,
        pkl_dir=pkl_dir,
        fp_types=fp_types,
        smiles_col=smiles_col,
        prop_col=prop_col,
        n_workers=n_workers,
        csv_pattern=csv_pattern,
    )
    print()

    # ------------------------------------------------------------------
    # Stage 2 – Convert .pkl → .npy  (one pair per fp_type per file)
    # ------------------------------------------------------------------
    print("─" * 60)
    print("Stage 2 | Processing pkl → npy …")
    print("─" * 60)
    npy_pairs = process_pkl_to_npy(
        pkl_dir=pkl_dir, npy_dir=npy_dir, fp_types=fp_types
    )
    print()

    # ------------------------------------------------------------------
    # Stage 3+4 – AC counting
    # ------------------------------------------------------------------
    print("─" * 60)
    mode_label = "BitBIRCH + Pairwise" if benchmarking else "BitBIRCH only"
    print(f"Stage 3 | Counting ACs [{mode_label}] …")
    print("─" * 60)

    # Tasks: one per (file, fp_type, threshold) — NOT per (offset, recursive)
    # Pairwise is computed once inside _process_file_threshold, then reused
    # for all (offset, recursive) combinations.
    tasks = [
        (fps_p, props_p, ft, th)
        for th in thresholds
        for fps_p, props_p, ft in npy_pairs
    ]

    all_results = []
    pipeline_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_file_threshold,
                fps_p, props_p, ft, th,
                offsets, recursive_options,
                order, benchmarking, chunk_size
            ): (fps_p, ft, th)
            for fps_p, props_p, ft, th in tasks
        }
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception as exc:
                key = futures[future]
                print(f"[ERROR] {key}: {exc}")

    elapsed = time.time() - pipeline_start
    print(f"\nTotal AC counting time: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_results).sort_values(
        ["dataset", "fp_type", "threshold", "offset"]
    ).reset_index(drop=True)

    suffix      = "benchmark" if benchmarking else "bitbirch_only"
    result_csv  = results_dir / f"ac_results_{order}_{suffix}.csv"
    df.to_csv(result_csv, index=False)

    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nResults saved → {result_csv}")

    return df


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end BitBIRCH AC benchmarking pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing input CSV files.")
    p.add_argument("--output_dir", required=True,
                   help="Root directory for intermediate files and results.")
    p.add_argument("--csv_pattern", default="*.csv",
                   help="Glob pattern to select CSV files (default: *.csv).")
    p.add_argument("--smiles_col", default="smiles",
                   help="CSV column containing SMILES strings (default: smiles).")
    p.add_argument("--prop_col",   default="logP",
                   help="CSV column containing the property (default: logP).")

    # Benchmarking flag
    p.add_argument(
        "--benchmarking",
        type=lambda x: x.strip().lower() in ("true", "1", "yes"),
        default=False,
        metavar="True|False",
        help=(
            "If True, also count ACs via exhaustive pairwise comparison and "
            "report the recall ratio (BitBIRCH / Pairwise). "
            "If False, only count ACs using BitBIRCH (faster)."
        ),
    )

    # Fingerprint types
    p.add_argument(
        "--fp_types",
        nargs="+",
        choices=list(FP_TYPES_SUPPORTED),
        default=["ECFP"],
        metavar="FP_TYPE",
        help=(
            "Fingerprint type(s) to generate and use for AC counting. "
            "Choices: ECFP (Morgan r=2, 1024-bit), "
            "MACCS (structural keys, 167-bit), "
            "RDKit (topological, 2048-bit). "
            "Multiple types can be specified (e.g. --fp_types ECFP MACCS). "
            "Default: ECFP."
        ),
    )

    # AC parameters
    p.add_argument("--threshold", type=float, nargs="+", default=[0.9],
                   help="Tanimoto similarity threshold(s) (default: 0.9).")
    p.add_argument("--offset", "--offsets", dest="offset",
                   type=float, nargs="+", default=[0.0],
                   help="BB threshold offset(s) (default: 0.0). "
                        "Also accepted as --offsets.")
    p.add_argument(
        "--order",
        choices=["random", "decreasing_sum", "increasing_sum",
                 "increasing_sum_cent", "identity"],
        default="increasing_sum",
        help="Fingerprint ordering for BitBIRCH (default: increasing_sum).",
    )
    p.add_argument(
        "--recursive",
        nargs="+",
        type=lambda x: x.strip().lower() in ("true", "1", "yes"),
        default=[False],
        metavar="True|False",
        help="Recursive BitBIRCH AC detection. "
             "Pass one or more values to sweep both options "
             "(e.g. --recursive False True). Default: False.",
    )

    # Performance
    p.add_argument("--max_workers", type=int, default=8,
                   help="Number of parallel workers (default: 8).")
    p.add_argument("--chunk_size", type=int, default=2000,
                   help="Row chunk size for pairwise computation (default: 2000).")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run_pipeline(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        benchmarking     = args.benchmarking,
        fp_types         = args.fp_types,
        thresholds       = args.threshold,
        offsets          = args.offset,
        order            = args.order,
        recursive_options = args.recursive,
        smiles_col       = args.smiles_col,
        prop_col         = args.prop_col,
        n_workers        = args.max_workers,
        chunk_size       = args.chunk_size,
        csv_pattern      = args.csv_pattern,
    )
