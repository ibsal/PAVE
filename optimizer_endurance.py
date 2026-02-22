from PyFoil.airfoil_polars import PolarSet
import csv
from datetime import datetime, timezone
import json
import numpy as np
from pathlib import Path
import scipy.optimize

from CommsNode import Aircraft, Fuselage, HorizontalTail, Powerplant, VerticalTail, Wing

G = 9.80665

def optimize_endurance(
    wingFoil,
    tailFoil,
    altitude,
    batteryElectric,
    fuselages,
    xcg,
    cdomisc,
    baseMass,
    totalMassMax,
    vtail_volume=0.03,
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    htailArMin=4.0,
    htailArMax=8.0,
    minClearance=0.5,
    htailVolMin=0.3,
    htailVolMax=0.7,
    trimAbsMaxDeg=5.0,
    levelFlightMargin=1.25,
    cruiseVelocityMode="optimal",
    res=60,
    seed=1,
    maxiter=150,
    popsize=6,
    polish=False,
    local_refine="Powell",
    local_maxiter=200,
    local_only=False,
    x_start=None,
    local_options=None,
    bounds=None,
    use_penalty=True,
    penalty_weights=None,
    penalty_exponent=1.35,
    penalty_base=1.0e5,
    evaluation_log_path=None,
    evaluation_log_append=True,
    evaluation_log_flush_every=200,
):
    arealDensityMain = 3.05
    arealDensityH = 1.5
    arealDensityV = 1.5

    boomMassPerM = 0.4
    boomMassFixed = 0.0
    boomLengthMin = 0.05

    evalCount = 0
    bestSeen = {"pwr": 1e30, "x": None}
    reject_counts = {}

    if bounds is None:
        bounds = [
            (4.0, 4.5),
            (0.26, 0.40),
            (0.05, xcg + 0.2),
            (0.60, 1.8),
            (0.15, 0.30),
            (xcg + 0.2, xcg + 1.8),
            (0.0, 3.0),   # wing incidence (deg)
            (-4.0, 4.0),   # tail incidence (deg)
        ]
    design_var_names = (
        "wingSpan",
        "wingChord",
        "xwqc",
        "hSpan",
        "hChord",
        "xhtqc",
        "wingIncidence",
        "tailIncidence",
    )
    bounds_lo = np.array([b[0] for b in bounds], dtype=float)
    bounds_hi = np.array([b[1] for b in bounds], dtype=float)
    hard_reject_value = 1e30
    # Allow a small numerical tolerance because solveTrim uses a root solver.
    lift_shortfall_tol_frac = 1.0e-3
    lift_shortfall_tol_abs_n = 1.0
    cruiseVelocityMode = str(cruiseVelocityMode).strip().lower()
    if cruiseVelocityMode not in ("optimal", "stall_margin"):
        raise ValueError(
            f"Unsupported cruiseVelocityMode '{cruiseVelocityMode}'. "
            "Use 'optimal' or 'stall_margin'."
        )

    # Constraint penalties are:
    #     penalty_base + weight * violation ** penalty_exponent
    # This keeps all rejects above realistic feasible power while preserving ranking.
    # Constraint penalty weights are in objective units per unit violation.
    # Keep these by the constraints configuration so they are easy to tune.
    default_penalty_weights = {
        # Geometry / setup failures (high penalties)
        "x_shape": 1.0e8,
        "bounds": 2.0e4,
        "geom_nonpos": 5.0e6,
        "xwqc": 5.0e6,
        "tail_aft_cg": 5.0e6,
        "wing_area": 5.0e6,
        "tail_arm": 5.0e6,
        "vtail_area": 5.0e6,
        "build_aircraft": 8.0e5,
        "solve_vbest": 8.0e5,
        "pwr_nan": 8.0e5,
        "trim_solve": 8.0e5,
        "stability_eval": 8.0e5,
        # Aero / layout constraints (moderate-to-strong)
        "clearance": 3.0e5,
        "htail_ar": 2.0e5,
        "htail_vol_est": 2.0e5,
        "htail_vol": 2.0e5,
        "static_margin": 3.0e5,
        "cma_pos": 3.0e5,
        "trim": 2.0e4,
        "tail_lift_nonnegative": 3.0e5,
        "lift_insufficient": 3.0e5,
        # Mass / power constraints
        "mass_est": 4.0e4,
        "mass": 6.0e4,
        "pwr_max": 3.0e2,
        "__default__": 3.0e5,
    }
    merged_penalty_weights = dict(default_penalty_weights)
    if penalty_weights:
        merged_penalty_weights.update(penalty_weights)
    penalty_weights_json = json.dumps(
        {k: float(v) for k, v in sorted(merged_penalty_weights.items())},
        sort_keys=True,
    )

    def _constraint_penalty(reason, violation_amount=1.0):
        amount = float(violation_amount)
        if (not np.isfinite(amount)) or amount <= 0.0:
            amount = 1.0
        weight = float(merged_penalty_weights.get(reason, merged_penalty_weights["__default__"]))
        exponent = float(penalty_exponent)
        if exponent <= 0.0:
            exponent = 1.0
        base = float(penalty_base)
        if base < 0.0:
            base = 0.0
        return base + weight * (amount ** exponent)

    eval_log_enabled = evaluation_log_path is not None
    eval_log_rows = []
    eval_log_count = 0
    eval_log_header_written = False
    eval_run_started_utc = datetime.now(timezone.utc).isoformat()
    eval_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    eval_log_path_obj = Path(evaluation_log_path) if eval_log_enabled else None
    eval_log_flush_every = max(1, int(evaluation_log_flush_every))
    eval_log_fields = [
        "run_started_utc",
        "run_id",
        "eval_index",
        "status",
        "reject_reason",
        "reject_stage",
        "violation",
        "objective",
        "raw_power_W",
        "penalty_value",
        "is_feasible",
        "has_build_data",
        "has_vbest_data",
        "has_trim_data",
        "rerun_required_if_now_feasible",
        "use_penalty",
        "penalty_exponent",
        "penalty_base",
        "penalty_weights_json",
        "seed",
        "min_clearance_m",
        "htailVolMin",
        "htailVolMax",
        "trimAbsMaxDeg",
        "totalMassMax_kg",
        "staticMarginMin",
        "staticMarginMax",
        "htailArMin",
        "htailArMax",
        "levelFlightMargin",
        "cruiseVelocityMode",
        "bounds_lo_json",
        "bounds_hi_json",
        "wingSpan",
        "wingChord",
        "xwqc",
        "hSpan",
        "hChord",
        "xhtqc",
        "wingIncidence",
        "tailIncidence",
        "clearance_m",
        "wing_area_m2",
        "wing_mac_m",
        "htail_area_m2",
        "htail_ar",
        "tail_arm_m",
        "vtail_area_m2",
        "totalMass_est_kg",
        "htail_volume_est",
        "totalMass_kg",
        "htail_volume",
        "vbest_mps",
        "thrust_N",
        "pwr_max_W",
        "staticMargin",
        "cm_alpha",
        "tail_lift_N",
        "lift_N",
        "weight_N",
        "trim_deg",
        "viol_bounds",
        "viol_clearance_m",
        "viol_htail_ar",
        "viol_mass_est_kg",
        "viol_htail_vol_est",
        "viol_htail_vol",
        "viol_mass_kg",
        "viol_pwr_max_W",
        "viol_static_margin",
        "viol_tail_lift_N",
        "viol_lift_N",
        "viol_trim_deg",
    ]

    def _log_value(value):
        if value is None:
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                return float(value)
            return np.nan
        return value

    def _flush_eval_log(force=False):
        nonlocal eval_log_header_written, eval_log_count, eval_log_rows, eval_log_path_obj
        if (not eval_log_enabled) or (not eval_log_rows):
            return
        if (not force) and (len(eval_log_rows) < eval_log_flush_every):
            return
        if eval_log_path_obj.parent:
            eval_log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not eval_log_header_written:
            has_existing_data = eval_log_path_obj.exists() and (eval_log_path_obj.stat().st_size > 0)
            if bool(evaluation_log_append) and has_existing_data:
                with eval_log_path_obj.open("r", newline="") as f:
                    existing_header = next(csv.reader(f), [])
                if existing_header != eval_log_fields:
                    eval_log_path_obj = eval_log_path_obj.with_name(
                        f"{eval_log_path_obj.stem}.{eval_run_id}{eval_log_path_obj.suffix}"
                    )
                    has_existing_data = False
            write_header = not (bool(evaluation_log_append) and has_existing_data)
            open_mode = "a" if bool(evaluation_log_append) else "w"
        else:
            write_header = False
            open_mode = "a"
        with eval_log_path_obj.open(open_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_log_fields)
            if write_header:
                writer.writeheader()
            writer.writerows(eval_log_rows)
        eval_log_count += len(eval_log_rows)
        eval_log_rows = []
        eval_log_header_written = True

    def _append_eval_log(row):
        if not eval_log_enabled:
            return
        out_row = {}
        for k in eval_log_fields:
            out_row[k] = _log_value(row.get(k))
        eval_log_rows.append(out_row)
        _flush_eval_log(force=False)

    def _clip_to_bounds(x):
        arr = np.array(x, dtype=float).reshape(-1)
        if arr.shape[0] != len(bounds):
            raise ValueError(f"x_start must have {len(bounds)} elements")
        return np.clip(arr, bounds_lo, bounds_hi)

    def _area_and_mac(span, root_chord, mid_chord, tip_chord, mid_pos, symmetric=True):
        area = 2.0 * (
            0.5 * (root_chord + mid_chord) * (mid_pos * 0.5 * span)
            + 0.5 * (mid_chord + tip_chord) * (0.5 * span - mid_pos * 0.5 * span)
        )
        if not symmetric:
            area *= 0.5
        mac = (
            2.0
            * (
                ((mid_pos * 0.5 * span) / 3.0)
                * (root_chord**2 + root_chord * mid_chord + mid_chord**2)
                + ((0.5 * span - mid_pos * 0.5 * span) / 3.0)
                * (mid_chord**2 + mid_chord * tip_chord + tip_chord**2)
            )
        ) / area
        return area, mac

    def build_aircraft(x):
        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        xvtqc = xhtqc
        wingIncidence = float(x[6])
        tailIncidence = float(x[7])

        mainWing = Wing(
            wingFoil, altitude, 0.0,
            wingSpan,
            wingChord, wingChord, wingChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0, True,
            xwqc,
            0.0,
            arealDensityMain,
        )

        hwing = HorizontalTail(
            tailFoil, altitude, 0.0,
            hSpan,
            hChord, hChord, hChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            tailIncidence, 0.0, True,
            xhtqc,
            0.0,
            arealDensityH,
            elevatorDeflection=0.0,
            elevatorTau=0.35,
            cd_deltae_k=0.0,
        )

        if float(xvtqc - xcg) <= 0.0:
            raise ValueError("xvtqc must be > xcg")
        vtail_area = (float(vtail_volume) * float(mainWing.area) * float(mainWing.span)) / float(xvtqc - xcg)
        if not np.isfinite(vtail_area) or vtail_area <= 0.0:
            raise ValueError("Invalid vtail area")
        vChord = float(np.sqrt(vtail_area / 2.0))
        vHeight = vChord

        vwing = VerticalTail(
            tailFoil, altitude, 0.0,
            vHeight,
            vChord, vChord, vChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0,
            xvtqc,
            0.0,
            arealDensityV,
            eta=2.0,
        )
        mainWing.incidence = wingIncidence

        boomLength = max(float(xhtqc - xwqc), float(boomLengthMin))
        boomMass = float(boomMassFixed) + float(boomMassPerM) * float(boomLength)
        fuselages_local = []
        for f in fuselages:
            if isinstance(f, Fuselage) and getattr(f, "width", None) is not None and getattr(f, "height", None) is not None:
                if float(getattr(f, "width")) <= 0.05 and float(getattr(f, "height")) <= 0.05:
                    fnew = Fuselage(
                        boomLength,
                        float(f.width),
                        float(f.height),
                        float(f.pfactor),
                        float(f.roughness),
                        float(f.laminarfraction),
                        float(getattr(f, "qfactor", 1.0)),
                        int(getattr(f, "quantity", 1)),
                    )
                    fuselages_local.append(fnew)
                else:
                    fuselages_local.append(f)
            else:
                fuselages_local.append(f)

        totalMass = float(baseMass) + float(mainWing.mass) + float(hwing.mass) + float(vwing.mass) + float(boomMass)
        weight = totalMass * G

        commsNode = Aircraft(
            altitude,
            20.0,
            batteryElectric,
            mainWing,
            hwing,
            vwing,
            fuselages_local,
            0.0,
            0.0,
            xcg,
            weight,
            cdomisc,
        )

        return commsNode, totalMass

    def _tail_lift_from_trimmed_state(commsNode, res_eval):
        wing_force = commsNode.mwing.forces(
            commsNode.xcg,
            commsNode.altitude,
            commsNode.velocity,
            commsNode.aoa,
            n=res_eval,
        )
        tail_force = commsNode.hwing.forces(
            commsNode.xcg,
            commsNode.altitude,
            commsNode.velocity,
            commsNode.aoa,
            downwash=wing_force[3],
            elevator=commsNode.trim,
            n=res_eval,
        )
        return float(tail_force[1])

    def objective(x):
        nonlocal evalCount, bestSeen
        evalCount += 1
        x = np.array(x, dtype=float).reshape(-1)
        current_stage = "input"
        eval_row = {
            "run_started_utc": eval_run_started_utc,
            "run_id": eval_run_id,
            "eval_index": evalCount,
            "status": "unknown",
            "reject_reason": "",
            "reject_stage": "",
            "violation": np.nan,
            "objective": np.nan,
            "raw_power_W": np.nan,
            "penalty_value": np.nan,
            "is_feasible": 0,
            "has_build_data": 0,
            "has_vbest_data": 0,
            "has_trim_data": 0,
            "rerun_required_if_now_feasible": 0,
            "use_penalty": int(bool(use_penalty)),
            "penalty_exponent": float(penalty_exponent),
            "penalty_base": float(penalty_base),
            "penalty_weights_json": penalty_weights_json,
            "seed": float(seed),
            "min_clearance_m": float(minClearance),
            "htailVolMin": float(htailVolMin),
            "htailVolMax": float(htailVolMax),
            "trimAbsMaxDeg": float(trimAbsMaxDeg),
            "totalMassMax_kg": float(totalMassMax),
            "staticMarginMin": float(staticMarginMin),
            "staticMarginMax": float(staticMarginMax),
            "htailArMin": float(htailArMin),
            "htailArMax": float(htailArMax),
            "levelFlightMargin": float(levelFlightMargin),
            "cruiseVelocityMode": cruiseVelocityMode,
            "bounds_lo_json": json.dumps([float(v) for v in bounds_lo], separators=(",", ":")),
            "bounds_hi_json": json.dumps([float(v) for v in bounds_hi], separators=(",", ":")),
        }
        for i, key in enumerate(design_var_names):
            eval_row[key] = float(x[i]) if i < x.shape[0] else np.nan

        def _report_counts():
            if evalCount % 10 != 0:
                return
            best_pwr = bestSeen.get("pwr", hard_reject_value)
            best_txt = f"{best_pwr:.2f} W" if np.isfinite(best_pwr) and best_pwr < 1e29 else "NA"
            parts = [f"{k}={v}" for k, v in sorted(reject_counts.items())]
            line = f"[eval {evalCount}] " + " ".join(parts) + f" best_pwr={best_txt}"
            if evalCount % 50 == 0 and bestSeen.get("x") is not None:
                line += f" best_x={np.array(bestSeen['x'], dtype=float)}"
            print(line)

        def _finish_eval(
            objective_value,
            status,
            reject_reason="",
            reject_stage="",
            violation=np.nan,
            raw_power=np.nan,
            penalty_value=np.nan,
        ):
            eval_row["status"] = status
            eval_row["reject_reason"] = reject_reason
            eval_row["reject_stage"] = reject_stage if status == "reject" else ""
            eval_row["violation"] = float(violation) if np.isfinite(violation) else np.nan
            eval_row["objective"] = float(objective_value) if np.isfinite(objective_value) else np.nan
            eval_row["raw_power_W"] = float(raw_power) if np.isfinite(raw_power) else np.nan
            eval_row["penalty_value"] = float(penalty_value) if np.isfinite(penalty_value) else np.nan
            eval_row["is_feasible"] = int(status == "valid")
            eval_row["rerun_required_if_now_feasible"] = int(
                status == "reject" and int(eval_row.get("has_trim_data", 0)) == 0
            )
            _append_eval_log(eval_row)
            return objective_value

        def _constraint_fail(reason, violation_amount=1.0):
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            _report_counts()
            if use_penalty:
                penalty_value = _constraint_penalty(reason, violation_amount)
            else:
                penalty_value = hard_reject_value
            return _finish_eval(
                penalty_value,
                status="reject",
                reject_reason=reason,
                reject_stage=current_stage,
                violation=violation_amount,
                raw_power=np.nan,
                penalty_value=penalty_value,
            )

        current_stage = "input"
        if x.shape[0] != len(bounds):
            return _constraint_fail("x_shape", abs(float(x.shape[0] - len(bounds))))
        if np.any(x < bounds_lo) or np.any(x > bounds_hi):
            bound_violation = np.sum(np.clip(bounds_lo - x, 0.0, None) + np.clip(x - bounds_hi, 0.0, None))
            eval_row["viol_bounds"] = bound_violation
            return _constraint_fail("bounds", bound_violation)
        eval_row["viol_bounds"] = 0.0

        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        xvtqc = xhtqc

        current_stage = "pre_heavy"
        if wingSpan <= 0.0 or wingChord <= 0.0 or hSpan <= 0.0 or hChord <= 0.0:
            nonpos_violation = (
                max(0.0, -wingSpan)
                + max(0.0, -wingChord)
                + max(0.0, -hSpan)
                + max(0.0, -hChord)
            )
            return _constraint_fail("geom_nonpos", nonpos_violation)

        if xwqc <= 0.0:
            return _constraint_fail("xwqc", -xwqc)

        if xhtqc <= xcg or xvtqc <= xcg:
            tail_aft_violation = max(0.0, float(xcg - xhtqc)) + max(0.0, float(xcg - xvtqc))
            return _constraint_fail("tail_aft_cg", tail_aft_violation)

        clearance = (xhtqc - 0.25 * hChord) - (xwqc + 0.75 * wingChord)
        eval_row["clearance_m"] = clearance
        clearance_violation = max(0.0, float(minClearance) - clearance)
        eval_row["viol_clearance_m"] = clearance_violation
        if clearance < float(minClearance):
            return _constraint_fail("clearance", clearance_violation)

        wing_area, wing_mac = _area_and_mac(
            wingSpan, wingChord, wingChord, wingChord, 0.5, symmetric=True
        )
        htail_area, _ = _area_and_mac(
            hSpan, hChord, hChord, hChord, 0.5, symmetric=True
        )
        eval_row["wing_area_m2"] = wing_area
        eval_row["wing_mac_m"] = wing_mac
        eval_row["htail_area_m2"] = htail_area
        if wing_area <= 0.0 or wing_mac <= 0.0 or htail_area <= 0.0:
            wing_area_violation = (
                max(0.0, -wing_area) + max(0.0, -wing_mac) + max(0.0, -htail_area)
            )
            return _constraint_fail("wing_area", wing_area_violation)
        htail_ar = float(hSpan**2 / htail_area)
        eval_row["htail_ar"] = htail_ar
        if not np.isfinite(htail_ar):
            htail_ar_violation = 1.0
        else:
            htail_ar_violation = max(0.0, float(htailArMin - htail_ar)) + max(0.0, float(htail_ar - htailArMax))
        eval_row["viol_htail_ar"] = htail_ar_violation
        if (not np.isfinite(htail_ar)) or (htail_ar < float(htailArMin)) or (htail_ar > float(htailArMax)):
            return _constraint_fail("htail_ar", htail_ar_violation)
        tail_arm = float(xvtqc - xcg)
        eval_row["tail_arm_m"] = tail_arm
        if tail_arm <= 0.0:
            return _constraint_fail("tail_arm", -tail_arm)
        vtail_area = (float(vtail_volume) * float(wing_area) * float(wingSpan)) / tail_arm
        eval_row["vtail_area_m2"] = vtail_area
        if not np.isfinite(vtail_area) or vtail_area <= 0.0:
            return _constraint_fail("vtail_area", max(0.0, -float(vtail_area)) if np.isfinite(vtail_area) else 1.0)

        boomLength = max(float(xhtqc - xwqc), float(boomLengthMin))
        boomMass = float(boomMassFixed) + float(boomMassPerM) * float(boomLength)
        totalMass_est = (
            float(baseMass)
            + float(arealDensityMain) * float(wing_area)
            + float(arealDensityH) * float(htail_area)
            + float(arealDensityV) * 2.0 * float(vtail_area)
            + float(boomMass)
        )
        eval_row["totalMass_est_kg"] = totalMass_est
        eval_row["viol_mass_est_kg"] = max(0.0, totalMass_est - float(totalMassMax))
        if totalMass_est > float(totalMassMax):
            return _constraint_fail("mass_est", totalMass_est - float(totalMassMax))

        htail_volume = (htail_area * (xhtqc - xcg)) / (wing_area * wing_mac)
        eval_row["htail_volume_est"] = htail_volume
        htail_volume_violation = max(0.0, float(htailVolMin) - htail_volume) + max(0.0, htail_volume - float(htailVolMax))
        eval_row["viol_htail_vol_est"] = htail_volume_violation
        if (htail_volume < float(htailVolMin)) or (htail_volume > float(htailVolMax)):
            return _constraint_fail("htail_vol_est", htail_volume_violation)

        current_stage = "build"
        try:
            commsNode, totalMass = build_aircraft(x)
        except Exception:
            return _constraint_fail("build_aircraft", 1.0)
        eval_row["has_build_data"] = 1
        eval_row["totalMass_kg"] = totalMass
        htail_volume_exact = commsNode.horizontalTailVolume(xhtqc)
        eval_row["htail_volume"] = htail_volume_exact
        htail_volume_exact_violation = max(0.0, float(htailVolMin) - htail_volume_exact) + max(0.0, htail_volume_exact - float(htailVolMax))
        eval_row["viol_htail_vol"] = htail_volume_exact_violation
        if (htail_volume_exact < float(htailVolMin)) or (htail_volume_exact > float(htailVolMax)):
            return _constraint_fail("htail_vol", htail_volume_exact_violation)
        
        eval_row["viol_mass_kg"] = max(0.0, totalMass - float(totalMassMax))
        if totalMass > float(totalMassMax):
            return _constraint_fail("mass", totalMass - float(totalMassMax))

        current_stage = "solve_vbest"
        try:
            vbest, pwr, thrust = commsNode.solveBestVelocity(
                levelFlightMargin,
                vguess=20.0,
                res=res,
                mode=cruiseVelocityMode,
            )
        except Exception:
            return _constraint_fail("solve_vbest", 1.0)
        eval_row["has_vbest_data"] = 1
        eval_row["vbest_mps"] = vbest
        eval_row["thrust_N"] = thrust
        eval_row["pwr_max_W"] = float(commsNode.pplant.pmax)
        eval_row["raw_power_W"] = pwr

        if not np.isfinite(pwr) or pwr <= 0.0:
            pwr_violation = 1.0 if not np.isfinite(pwr) else -float(pwr)
            return _constraint_fail("pwr_nan", pwr_violation)

        eval_row["viol_pwr_max_W"] = max(0.0, float(pwr) - float(commsNode.pplant.pmax))
        if float(pwr) > float(commsNode.pplant.pmax):
            return _constraint_fail("pwr_max", float(pwr) - float(commsNode.pplant.pmax))

        # Re-trim at the best velocity before stability/trim checks
        current_stage = "trim_stability"
        try:
            commsNode.velocity = float(vbest)
            trim_sol = commsNode.solveTrim(res=res)
            if trim_sol == [None, None]:
                return _constraint_fail("trim_solve", 1.0)
            trim_forces = commsNode.sumFanddM(res=res)
            total_lift = float(trim_forces[1])
            total_weight = float(commsNode.weight)
            sm = float(commsNode.staticMargin())
            cma = float(commsNode.cm_alpha())
            tail_lift = _tail_lift_from_trimmed_state(commsNode, res_eval=res)
        except Exception:
            return _constraint_fail("stability_eval", 1.0)
        eval_row["has_trim_data"] = 1
        eval_row["staticMargin"] = sm
        eval_row["cm_alpha"] = cma
        eval_row["tail_lift_N"] = tail_lift
        eval_row["lift_N"] = total_lift
        eval_row["weight_N"] = total_weight
        eval_row["trim_deg"] = float(commsNode.trim)

        static_margin_violation = max(0.0, float(staticMarginMin - sm)) + max(0.0, float(sm - staticMarginMax))
        eval_row["viol_static_margin"] = static_margin_violation
        if (sm < float(staticMarginMin)) or (sm > float(staticMarginMax)):
            return _constraint_fail("static_margin", static_margin_violation)

        if cma > 0:
            return _constraint_fail("cma_pos", cma)

        tail_lift_violation = max(0.0, tail_lift)
        eval_row["viol_tail_lift_N"] = tail_lift_violation
        if tail_lift >= 0.0:
            return _constraint_fail("tail_lift_nonnegative", tail_lift_violation)

        lift_shortfall_tol_n = max(
            float(lift_shortfall_tol_abs_n),
            float(lift_shortfall_tol_frac) * abs(float(total_weight)),
        )
        lift_shortfall_violation = max(0.0, float(total_weight) - float(total_lift) - lift_shortfall_tol_n)
        eval_row["viol_lift_N"] = lift_shortfall_violation
        if lift_shortfall_violation > 0.0:
            return _constraint_fail("lift_insufficient", lift_shortfall_violation)
        
        trim_violation = max(0.0, abs(float(commsNode.trim)) - float(trimAbsMaxDeg))
        eval_row["viol_trim_deg"] = trim_violation
        if abs(commsNode.trim) > float(trimAbsMaxDeg):
            return _constraint_fail("trim", trim_violation)
        
        pwr = float(pwr)
        eval_row["objective"] = pwr

        if pwr < bestSeen["pwr"]:
            bestSeen["pwr"] = pwr
            bestSeen["x"] = np.array(x, dtype=float).copy()

        reject_counts["valid"] = reject_counts.get("valid", 0) + 1
        _report_counts()
        return _finish_eval(
            pwr,
            status="valid",
            reject_reason="",
            violation=0.0,
            raw_power=pwr,
            penalty_value=0.0,
        )

    de_result = None
    if local_only:
        if x_start is None:
            raise ValueError("x_start is required when local_only=True")
        xbest = _clip_to_bounds(x_start)
    else:
        init_pop = "latinhypercube"
        if x_start is not None:
            rng = np.random.default_rng(int(seed))
            popsize_total = int(popsize) * len(bounds)
            init_pop = rng.uniform(bounds_lo, bounds_hi, size=(popsize_total, len(bounds)))
            init_pop[0, :] = _clip_to_bounds(x_start)
        de_result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=int(maxiter),
            popsize=int(popsize),
            tol=0.0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(seed),
            polish=bool(polish),
            init=init_pop,
            updating="deferred",
            workers=1,
            disp=True
        )
        xbest = de_result.x

    local_result = None
    if local_refine:
        method = "Powell" if local_refine is True else str(local_refine)
        options = {"maxiter": int(local_maxiter)}
        if local_options:
            options.update(local_options)
        local_result = scipy.optimize.minimize(
            objective,
            xbest,
            method=method,
            options=options,
            bounds=bounds
        )
        if local_result.success and np.isfinite(local_result.fun):
            xbest = _clip_to_bounds(local_result.x)

    if local_result is not None:
        success = bool(local_result.success)
        message = str(local_result.message)
    elif de_result is not None:
        success = bool(de_result.success)
        message = str(de_result.message)
    else:
        success = True
        message = "local_only"

    xbest = _clip_to_bounds(xbest)
    fallback_to_feasible = False
    try:
        commsNode, totalMass = build_aircraft(xbest)
        vbest, pwr, thrust = commsNode.solveBestVelocity(
            levelFlightMargin,
            vguess=20.0,
            res=res,
            mode=cruiseVelocityMode,
        )
        commsNode.velocity = float(vbest)
        trim_sol = commsNode.solveTrim(res=res)
        if trim_sol == [None, None]:
            raise RuntimeError("trim_solve")
        total_lift = float(commsNode.sumFanddM(res=res)[1])
        lift_shortfall_tol_n = max(
            float(lift_shortfall_tol_abs_n),
            float(lift_shortfall_tol_frac) * abs(float(commsNode.weight)),
        )
        if float(commsNode.weight) - total_lift > lift_shortfall_tol_n:
            raise RuntimeError("lift_insufficient")
        tail_lift = _tail_lift_from_trimmed_state(commsNode, res_eval=res)
        if tail_lift >= 0.0:
            raise RuntimeError("tail_lift_nonnegative")
        sm = float(commsNode.staticMargin())
    except Exception as exc:
        if bestSeen.get("x") is None:
            _flush_eval_log(force=True)
            raise RuntimeError(
                "Optimization finished without a feasible design. "
                "Try relaxing constraints, improving x_start, or increasing penalty weights."
            ) from exc
        xbest = np.array(bestSeen["x"], dtype=float).copy()
        commsNode, totalMass = build_aircraft(xbest)
        vbest, pwr, thrust = commsNode.solveBestVelocity(
            levelFlightMargin,
            vguess=20.0,
            res=res,
            mode=cruiseVelocityMode,
        )
        commsNode.velocity = float(vbest)
        trim_sol = commsNode.solveTrim(res=res)
        if trim_sol == [None, None]:
            _flush_eval_log(force=True)
            raise RuntimeError("Best feasible design failed to trim at returned cruise velocity.") from exc
        total_lift = float(commsNode.sumFanddM(res=res)[1])
        lift_shortfall_tol_n = max(
            float(lift_shortfall_tol_abs_n),
            float(lift_shortfall_tol_frac) * abs(float(commsNode.weight)),
        )
        if float(commsNode.weight) - total_lift > lift_shortfall_tol_n:
            _flush_eval_log(force=True)
            raise RuntimeError("Best feasible design cannot produce enough lift at returned cruise velocity.") from exc
        tail_lift = _tail_lift_from_trimmed_state(commsNode, res_eval=res)
        if tail_lift >= 0.0:
            _flush_eval_log(force=True)
            raise RuntimeError("Best feasible design violates negative horizontal tail lift constraint.") from exc
        sm = float(commsNode.staticMargin())
        fallback_to_feasible = True
    if fallback_to_feasible:
        message = f"{message}; fell_back_to_best_feasible"

    _flush_eval_log(force=True)

    return {
        "success": success,
        "message": message,
        "power_W": float(pwr),
        "vbest_mps": float(vbest),
        "thrust_N": float(thrust),
        "totalMass_kg": float(totalMass),
        "staticMargin": float(sm),
        "xbest": {
            "wingSpan": float(xbest[0]),
            "wingChord": float(xbest[1]),
            "xwqc": float(xbest[2]),
            "hSpan": float(xbest[3]),
            "hChord": float(xbest[4]),
            "xhtqc": float(xbest[5]),
            "vHeight": float(commsNode.vtail.span * 0.5),
            "vChord": float(commsNode.vtail.rootChord),
            "vtailVolume": float(vtail_volume),
            "wingIncidence": float(xbest[6]),
            "tailIncidence": float(xbest[7]),
            "boomLength": float(max(float(xbest[5] - xbest[2]), float(boomLengthMin))),
        },
        "de_result": de_result,
        "local_result": local_result,
        "evaluation_run_id": str(eval_run_id) if eval_log_enabled else None,
        "evaluation_log_path": str(eval_log_path_obj) if eval_log_enabled else None,
        "evaluation_log_rows": int(eval_log_count) if eval_log_enabled else 0,
    }



def _example_run():
    wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
    tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
    body = Fuselage(1.0541, .3048, 0.21336, 0.9, 0.00635e-3, 0.3)
    booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
    batteryElectric = Powerplant(7992000, 0.59, 4000)
    xcg = 0.35

    prepopulated_best = [4.4, 0.3, 0.29024762, 0.46046029, 0.18097007, 2.01117241, 0.32476509, 3.30305594]
    run_mode = "de"  # "de" or "local_only"
    seed_de = False
    local_only = run_mode == "local_only"
    x_start = prepopulated_best if (local_only or seed_de) else None
    bounds = [
        (4.4, 4.572),
        (0.27, 0.6),
        (0.05, xcg + 0.2),
        (0.60, 1.2),
        (0.18, 0.30),
        (xcg + 0.2, xcg + 1.8),
        (0.0, 3.0),   # wing incidence (deg)
        (-3.0, 2.0),   # tail incidence (deg)
    ]
    use_penalty = True
    penalty_exponent = 1.35
    penalty_base = 2.0e4
    min_clearance_m = 0.4
    htail_ar_min = 2.0
    htail_ar_max = 8.0
    htail_volume_min = 0.3
    htail_volume_max = 0.7
    trim_abs_max_deg = 5.0
    cruise_velocity_mode = "optimal"  # "optimal" or "stall_margin"
    evaluation_log_path = "optimizer_endurance_eval_log.csv"
    penalty_weight_overrides = {
        # Example overrides; omitted keys use defaults defined in optimize_endurance.
        # Crank values way up if you want hard-reject-like behavior for a specific rule.
        "mass_est": 5.0e4,
        "static_margin": 4.0e5,
    }
    
    best = optimize_endurance(
        wingFoil=wingFoil,
        tailFoil=tailFoil,
        altitude=200,
        batteryElectric=batteryElectric,
        fuselages=[body, booms, booms],
        xcg=xcg,
        cdomisc=0.01,
        baseMass=17.5,
        totalMassMax=22.6796,
        htailArMin=htail_ar_min,
        htailArMax=htail_ar_max,
        minClearance=min_clearance_m,
        htailVolMin=htail_volume_min,
        htailVolMax=htail_volume_max,
        trimAbsMaxDeg=trim_abs_max_deg,
        staticMarginMin=0.05,
        staticMarginMax=0.30,
        levelFlightMargin=1.25,
        cruiseVelocityMode=cruise_velocity_mode,
        res=1,
        seed=20,
        maxiter=50,
        popsize=20,
        polish=False,
        local_refine="powell", #can fuck with method here: L-BFGS-B uses derivatives so only good in vicinity, powell does not use derivative so can be safer
        local_maxiter=200,
        local_only=local_only,
        x_start=x_start,
        bounds=bounds,
        use_penalty=use_penalty,
        penalty_exponent=penalty_exponent,
        penalty_base=penalty_base,
        evaluation_log_path=evaluation_log_path,
        penalty_weights=penalty_weight_overrides,
    )
    _ = best

if __name__ == "__main__":
    _example_run()
