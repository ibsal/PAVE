from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from airfoil_polars import Polar


_RE_TAG = re.compile(r"^(?P<name>.+?)[ _-]Re\d+(?:\.\d+)?[kKmM]?$")


@dataclass
class Candidate:
    path: Path
    variant_priority: int
    sample_count: int
    polar: Polar


@dataclass
class Score:
    airfoil: str
    metric: float
    alpha_deg: float
    cl: float
    cd: float
    source_file: str


def _split_variant(path: Path) -> Tuple[str, int]:
    stem = path.stem
    if stem.endswith(".base"):
        return stem[:-5], 1
    if stem.endswith(".ext"):
        return stem[:-4], 2
    return stem, 0


def _airfoil_name_from_stem(core_stem: str) -> str:
    m = _RE_TAG.match(core_stem)
    if m:
        return m.group("name")
    return core_stem


def _is_target_re(polar: Polar, target_re: float) -> bool:
    if polar.reynolds is None or not math.isfinite(float(polar.reynolds)):
        return False
    return abs(float(polar.reynolds) - float(target_re)) <= 1.0


def _valid_metric_points(polar: Polar) -> np.ndarray:
    cl = np.asarray(polar.cl, dtype=float)
    cd = np.asarray(polar.cd, dtype=float)
    return np.isfinite(cl) & np.isfinite(cd) & (cl > 0.0) & (cd > 0.0)


def _score_polar(airfoil: str, candidate: Candidate) -> Optional[Score]:
    p = candidate.polar
    mask = _valid_metric_points(p)
    if not np.any(mask):
        return None
    cl = np.asarray(p.cl, dtype=float)[mask]
    cd = np.asarray(p.cd, dtype=float)[mask]
    alpha = np.asarray(p.alpha_deg, dtype=float)[mask]
    metric = np.power(cl, 1.5) / cd
    i = int(np.argmax(metric))
    return Score(
        airfoil=airfoil,
        metric=float(metric[i]),
        alpha_deg=float(alpha[i]),
        cl=float(cl[i]),
        cd=float(cd[i]),
        source_file=candidate.path.name,
    )


def _pick_candidates(polars_dir: Path, target_re: float) -> Dict[str, Candidate]:
    picks: Dict[str, Candidate] = {}
    for path in sorted(polars_dir.glob("*.pol")):
        try:
            polar = Polar.from_file(path)
        except Exception:
            continue
        if not _is_target_re(polar, target_re):
            continue

        core_stem, variant_priority = _split_variant(path)
        airfoil = _airfoil_name_from_stem(core_stem)
        cand = Candidate(
            path=path,
            variant_priority=variant_priority,
            sample_count=int(len(np.asarray(polar.alpha_deg))),
            polar=polar,
        )

        prev = picks.get(airfoil)
        if prev is None:
            picks[airfoil] = cand
            continue
        # Prefer merged .pol over .base over .ext; then prefer denser tables.
        if (cand.variant_priority, -cand.sample_count, cand.path.name) < (
            prev.variant_priority,
            -prev.sample_count,
            prev.path.name,
        ):
            picks[airfoil] = cand
    return picks


def rank_airfoils(polars_dir: Path, target_re: float) -> List[Score]:
    candidates = _pick_candidates(polars_dir, target_re)
    scores: List[Score] = []
    for airfoil, candidate in candidates.items():
        scored = _score_polar(airfoil, candidate)
        if scored is not None:
            scores.append(scored)
    scores.sort(key=lambda s: s.metric, reverse=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank airfoil polars by max(CL^1.5/CD) at a target Reynolds number."
    )
    parser.add_argument(
        "--polars-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "polars",
        help="Directory with pre-generated .pol files.",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=300000.0,
        help="Target Reynolds number (default: 300000).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of ranked rows to print (default: 30).",
    )
    args = parser.parse_args()

    scores = rank_airfoils(args.polars_dir, args.re)
    if not scores:
        print(f"No valid polar data found in '{args.polars_dir}' for Re={args.re:.0f}.")
        return

    print(f"Ranked {len(scores)} airfoils by max(CL^1.5/CD) at Re={args.re:.0f}")
    print(
        f"{'Rank':>4}  {'Airfoil':<20} {'CL^1.5/CD':>12} {'alpha(deg)':>10} "
        f"{'CL':>8} {'CD':>9}  Source"
    )
    for i, s in enumerate(scores[: max(args.top, 0)], start=1):
        print(
            f"{i:>4}  {s.airfoil:<20} {s.metric:>12.3f} {s.alpha_deg:>10.2f} "
            f"{s.cl:>8.4f} {s.cd:>9.5f}  {s.source_file}"
        )


if __name__ == "__main__":
    main()
