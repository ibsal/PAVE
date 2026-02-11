import argparse
import csv
import json
import math
from pathlib import Path


DESIGN_VARS = (
    "wingSpan",
    "wingChord",
    "xwqc",
    "hSpan",
    "hChord",
    "xhtqc",
    "wingIncidence",
    "tailIncidence",
)


def _to_float(value):
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except Exception:
        return math.nan


def _is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _parse_json_list(text):
    if text is None:
        return None
    st = str(text).strip()
    if not st:
        return None
    try:
        data = json.loads(st)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        val = _to_float(item)
        out.append(val if _is_finite(val) else math.nan)
    return out


def _resolve_threshold(row, cli_value, row_key, default=None):
    if cli_value is not None:
        return float(cli_value)
    row_value = _to_float(row.get(row_key))
    if _is_finite(row_value):
        return row_value
    return default


def _add_min_check(name, value, minimum, checks):
    if (minimum is None) or (not _is_finite(minimum)):
        checks.append((name, "unknown", "missing threshold"))
        return
    if not _is_finite(value):
        checks.append((name, "unknown", "missing value"))
        return
    if value < minimum:
        checks.append((name, "fail", f"{value:.6g} < {minimum:.6g}"))
    else:
        checks.append((name, "pass", ""))


def _add_max_check(name, value, maximum, checks):
    if (maximum is None) or (not _is_finite(maximum)):
        checks.append((name, "unknown", "missing threshold"))
        return
    if not _is_finite(value):
        checks.append((name, "unknown", "missing value"))
        return
    if value > maximum:
        checks.append((name, "fail", f"{value:.6g} > {maximum:.6g}"))
    else:
        checks.append((name, "pass", ""))


def _add_range_check(name, value, minimum, maximum, checks):
    if ((minimum is None) or (not _is_finite(minimum))) or ((maximum is None) or (not _is_finite(maximum))):
        checks.append((name, "unknown", "missing threshold"))
        return
    if not _is_finite(value):
        checks.append((name, "unknown", "missing value"))
        return
    if value < minimum:
        checks.append((name, "fail", f"{value:.6g} < {minimum:.6g}"))
    elif value > maximum:
        checks.append((name, "fail", f"{value:.6g} > {maximum:.6g}"))
    else:
        checks.append((name, "pass", ""))


def _parse_bounds_overrides(bound_args):
    overrides = {}
    for var_name, lo_s, hi_s in bound_args:
        if var_name not in DESIGN_VARS:
            raise ValueError(f"Unknown design variable for --bound: {var_name}")
        lo = float(lo_s)
        hi = float(hi_s)
        if lo > hi:
            raise ValueError(f"Invalid --bound for {var_name}: min {lo} > max {hi}")
        overrides[var_name] = (lo, hi)
    return overrides


def _rank_metric(row):
    raw_power = _to_float(row.get("raw_power_W"))
    if _is_finite(raw_power) and raw_power > 0.0:
        return raw_power
    objective = _to_float(row.get("objective"))
    if _is_finite(objective):
        return objective
    return math.inf


def evaluate_row(row, args, bound_overrides):
    checks = []

    min_clearance = _resolve_threshold(row, args.min_clearance, "min_clearance_m", default=0.5)
    htail_ar_min = _resolve_threshold(row, args.htail_ar_min, "htailArMin", default=None)
    htail_ar_max = _resolve_threshold(row, args.htail_ar_max, "htailArMax", default=None)
    mass_max = _resolve_threshold(row, args.mass_max, "totalMassMax_kg", default=None)
    htail_vol_min = _resolve_threshold(row, args.htail_vol_min, "htailVolMin", default=None)
    htail_vol_max = _resolve_threshold(row, args.htail_vol_max, "htailVolMax", default=None)
    static_margin_min = _resolve_threshold(row, args.static_margin_min, "staticMarginMin", default=None)
    static_margin_max = _resolve_threshold(row, args.static_margin_max, "staticMarginMax", default=None)
    trim_abs_max = _resolve_threshold(row, args.trim_abs_max, "trimAbsMaxDeg", default=5.0)
    pwr_max = _resolve_threshold(row, args.pwr_max, "pwr_max_W", default=None)

    bounds_lo = _parse_json_list(row.get("bounds_lo_json"))
    bounds_hi = _parse_json_list(row.get("bounds_hi_json"))
    for i, var in enumerate(DESIGN_VARS):
        value = _to_float(row.get(var))
        if var in bound_overrides:
            lo, hi = bound_overrides[var]
        else:
            has_logged_bounds = (
                bounds_lo is not None
                and bounds_hi is not None
                and i < len(bounds_lo)
                and i < len(bounds_hi)
                and _is_finite(bounds_lo[i])
                and _is_finite(bounds_hi[i])
            )
            if not has_logged_bounds:
                continue
            lo = bounds_lo[i]
            hi = bounds_hi[i]
        if not _is_finite(lo) or not _is_finite(hi):
            checks.append((f"bound_{var}", "unknown", "missing bound"))
            continue
        if not _is_finite(value):
            checks.append((f"bound_{var}", "unknown", "missing value"))
            continue
        if (value < lo) or (value > hi):
            checks.append((f"bound_{var}", "fail", f"{value:.6g} not in [{lo:.6g}, {hi:.6g}]"))
        else:
            checks.append((f"bound_{var}", "pass", ""))

    _add_min_check("clearance", _to_float(row.get("clearance_m")), min_clearance, checks)
    _add_range_check("htail_ar", _to_float(row.get("htail_ar")), htail_ar_min, htail_ar_max, checks)
    _add_max_check("mass_est", _to_float(row.get("totalMass_est_kg")), mass_max, checks)
    _add_range_check("htail_vol_est", _to_float(row.get("htail_volume_est")), htail_vol_min, htail_vol_max, checks)

    _add_max_check("mass", _to_float(row.get("totalMass_kg")), mass_max, checks)
    _add_range_check("htail_vol", _to_float(row.get("htail_volume")), htail_vol_min, htail_vol_max, checks)
    _add_max_check("pwr_max", _to_float(row.get("raw_power_W")), pwr_max, checks)
    _add_range_check("static_margin", _to_float(row.get("staticMargin")), static_margin_min, static_margin_max, checks)
    _add_max_check("cm_alpha_pos", _to_float(row.get("cm_alpha")), 0.0, checks)
    trim_abs = abs(_to_float(row.get("trim_deg"))) if _is_finite(_to_float(row.get("trim_deg"))) else math.nan
    _add_max_check("trim_abs", trim_abs, trim_abs_max, checks)

    fails = [f"{name} ({detail})" if detail else name for name, status, detail in checks if status == "fail"]
    unknowns = [name for name, status, _ in checks if status == "unknown"]
    if fails:
        parse_status = "infeasible_now"
    elif unknowns:
        parse_status = "needs_rerun"
    else:
        parse_status = "feasible_now"
    assumed_from_status = 0
    if (
        parse_status == "needs_rerun"
        and bool(args.trust_valid_status)
        and str(row.get("status", "")).strip().lower() == "valid"
    ):
        parse_status = "feasible_now"
        assumed_from_status = 1

    out = dict(row)
    out["parse_status"] = parse_status
    out["parse_fail_reasons"] = " | ".join(fails)
    out["parse_unknown_checks"] = " | ".join(unknowns)
    out["parse_rerun_recommended"] = int(parse_status == "needs_rerun")
    out["parse_assumed_from_status"] = int(assumed_from_status)
    out["parse_rank_metric"] = _rank_metric(row)
    return out


def _print_table(title, rows, top_n):
    print()
    print(title)
    if not rows:
        print("  (none)")
        return
    headers = [
        ("#", 4),
        ("run_id", 22),
        ("eval", 7),
        ("rank", 11),
        ("raw_pwr", 11),
        ("clear", 8),
        ("hAR", 8),
        ("m_est", 9),
    ]
    header_line = " ".join([f"{name:>{width}}" for name, width in headers])
    print(header_line)
    for idx, row in enumerate(rows[:top_n], start=1):
        run_id = str(row.get("run_id", ""))[:22]
        eval_idx = _to_float(row.get("eval_index"))
        rank = _to_float(row.get("parse_rank_metric"))
        raw = _to_float(row.get("raw_power_W"))
        clear = _to_float(row.get("clearance_m"))
        har = _to_float(row.get("htail_ar"))
        mest = _to_float(row.get("totalMass_est_kg"))
        parts = [
            f"{idx:>4d}",
            f"{run_id:>22}",
            f"{int(eval_idx):>7d}" if _is_finite(eval_idx) else f"{'-':>7}",
            f"{rank:>11.3f}" if _is_finite(rank) else f"{'nan':>11}",
            f"{raw:>11.3f}" if _is_finite(raw) else f"{'nan':>11}",
            f"{clear:>8.3f}" if _is_finite(clear) else f"{'nan':>8}",
            f"{har:>8.3f}" if _is_finite(har) else f"{'nan':>8}",
            f"{mest:>9.3f}" if _is_finite(mest) else f"{'nan':>9}",
        ]
        print(" ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Parse optimizer_endurance eval logs under new constraints/bounds."
    )
    parser.add_argument("--log", required=True, help="Path to evaluation CSV log")
    parser.add_argument("--top", type=int, default=10, help="Rows to print for each category")
    parser.add_argument("--run-id", action="append", default=[], help="Keep only specific run_id (repeatable)")
    parser.add_argument(
        "--bound",
        nargs=3,
        action="append",
        metavar=("VAR", "MIN", "MAX"),
        default=[],
        help="Override bound for a design variable (repeatable)",
    )
    parser.add_argument("--min-clearance", type=float, default=None)
    parser.add_argument("--htail-ar-min", type=float, default=None)
    parser.add_argument("--htail-ar-max", type=float, default=None)
    parser.add_argument("--mass-max", type=float, default=None)
    parser.add_argument("--htail-vol-min", type=float, default=None)
    parser.add_argument("--htail-vol-max", type=float, default=None)
    parser.add_argument("--static-margin-min", type=float, default=None)
    parser.add_argument("--static-margin-max", type=float, default=None)
    parser.add_argument("--trim-abs-max", type=float, default=None)
    parser.add_argument("--pwr-max", type=float, default=None)
    parser.add_argument(
        "--no-trust-valid-status",
        dest="trust_valid_status",
        action="store_false",
        help="Do not auto-promote legacy status=valid rows with unknown checks",
    )
    parser.set_defaults(trust_valid_status=True)
    parser.add_argument("--output", default=None, help="Optional annotated CSV output path")
    args = parser.parse_args()

    bound_overrides = _parse_bounds_overrides(args.bound)
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.run_id:
        run_id_set = set(args.run_id)
        rows = [r for r in rows if str(r.get("run_id", "")) in run_id_set]

    if not rows:
        print("No rows found after filtering.")
        return

    parsed = [evaluate_row(row, args, bound_overrides) for row in rows]
    feasible = [r for r in parsed if r.get("parse_status") == "feasible_now"]
    rerun = [r for r in parsed if r.get("parse_status") == "needs_rerun"]
    infeasible = [r for r in parsed if r.get("parse_status") == "infeasible_now"]

    feasible.sort(key=lambda r: _to_float(r.get("parse_rank_metric")))
    rerun.sort(key=lambda r: _to_float(r.get("parse_rank_metric")))

    print(f"Rows loaded: {len(parsed)}")
    print(f"Feasible now: {len(feasible)}")
    print(f"Needs rerun (unknown heavy checks): {len(rerun)}")
    print(f"Infeasible now: {len(infeasible)}")
    if feasible:
        best = feasible[0]
        best_rank = _to_float(best.get("parse_rank_metric"))
        print(
            f"Best feasible rank metric: {best_rank:.6g} "
            f"(run_id={best.get('run_id')}, eval={best.get('eval_index')})"
        )

    _print_table("Top feasible rows", feasible, args.top)
    _print_table("Top rerun candidates", rerun, args.top)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(parsed[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(parsed)
        print()
        print(f"Annotated output written: {out_path}")


if __name__ == "__main__":
    main()
