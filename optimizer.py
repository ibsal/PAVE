import math

from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet

from aircraft_core import build_aircraft_config, planform_area_and_mac, run_analysis, solve_trim


G = 9.80665

### Cruise-only optimization inputs (adjust as needed)
CRUISE_ALTITUDE_M = 200
CRUISE_SPEED_MAX_MPS = 80.0
STALL_MARGIN_CRUISE = 1.25
SPEED_OPT_POINTS = 5
N_SPAN = 50
TRIM_MAX_ITER = 5
ELEVATOR_LIMIT_DEG = 30.0

WING_SPAN_MIN_M = 4.5
WING_SPAN_MAX_M = 3.6
WING_SPAN_STEPS = 5

WING_CHORD_MIN_M = 0.274
WING_CHORD_MAX_M = 0.4
WING_CHORD_STEPS = 5

HTAIL_SPAN_SCALE_MIN = 0.5
HTAIL_SPAN_SCALE_MAX = 2.5
HTAIL_SPAN_SCALE_STEPS = 5
HTAIL_CHORD_SCALE_MIN = 0.5
HTAIL_CHORD_SCALE_MAX = 2.5
HTAIL_CHORD_SCALE_STEPS = 5
HTAIL_X_MIN_M = 1.2
HTAIL_X_MAX_M = 2.2
HTAIL_X_STEPS = 5

VTAIL_SCALE_MIN = 0.5
VTAIL_SCALE_MAX = 2.5

STATIC_MARGIN_MIN = 0.05
STATIC_MARGIN_MAX = 0.15
HTAIL_VOLUME_MIN = 0.7
VTAIL_VOLUME_MIN = 0.03

ENERGY_RESERVE_FRAC = 0.2

SURFACE_AREAL_DENSITY_KG_M2 = 2.5
PROGRESS_EVERY = 1

### Baseline geometry and configuration (matches pywave defaults)
GROUND_LEVEL_M = 0.0
TAIL_EFFICIENCY = 0.95
ELEVATOR_TAU = 0.45
SURFACE_ROUGHNESS_M = 0.00635e-3
FUSELAGE_LAMINAR_FRAC = 0.3
BOOM_LAMINAR_FRAC = 0.04
CD_MISC = 0.01
DOWNWASH_FACTOR = 0.45

MAX_POWER_W = 4000.0
PROP_EFF = 0.59
BATTERY_ENERGY_J = 7920000.0
MAX_THRUST_N = 152.0
MISSION_SYSTEMS_POWER_W = 50.24

XWQC_M = 0.45
XHTQC_M = 1.8
XCG_M = 0.2
WING_Z_M = 0.0
HTAIL_Z_M = 0.0
VTAIL_Z_M = 0.2
ENGINE_Z_M = 0.0

WING_INCIDENCE_DEG = 1.0
BASE_WING_SPAN_M = 5.0
BASE_WING_CHORD_M = 0.3

HROOT_CHORD_M = 0.386
HMID_CHORD_M = 0.2286
HMID_CHORD_POS = 0.8
HTIP_CHORD_M = 0.1
HSPAN_M = 2.0
HROOT_SWEEP_DEG = 0.0
HMID_SWEEP_DEG = 0.0
HMID_SWEEP_POS = 0.4
HTIP_SWEEP_DEG = 0.0
HTAIL_INCIDENCE_DEG = 3.0

VROOT_CHORD_M = 0.1
VMID_CHORD_M = 0.1
VMID_CHORD_POS = 0.4
VTIP_CHORD_M = 0.1
VSPAN_M = 1.0
VROOT_SWEEP_DEG = 0.0
VMID_SWEEP_DEG = 0.0
VMID_SWEEP_POS = 0.4
VTIP_SWEEP_DEG = 0.0
VTAIL_INCIDENCE_DEG = 0.0
VTAIL_COUNT = 2

FUSELAGE_WIDTH_M = 0.3
FUSELAGE_HEIGHT_M = 0.15
FUSELAGE_LENGTH_M = 0.7
FUSELAGE_PFACTOR = 0.9

BOOM_LENGTH_M = XHTQC_M - XWQC_M
BOOM_DIAMETER_M = 0.03
BOOM_COUNT = 2

WING_POLAR_DIR = "./PyFoil/polars"
WING_AIRFOIL = "psu94097"
TAIL_AIRFOIL = "S9033"

WING_FOIL = PolarSet.from_folder(WING_POLAR_DIR, airfoil=WING_AIRFOIL)
HTAIL_FOIL = PolarSet.from_folder(WING_POLAR_DIR, airfoil=TAIL_AIRFOIL)
VTAIL_FOIL = PolarSet.from_folder(WING_POLAR_DIR, airfoil=TAIL_AIRFOIL)


def linspace(min_val, max_val, steps):
    if steps <= 1:
        return [min_val]
    step = (max_val - min_val) / (steps - 1)
    return [min_val + i * step for i in range(steps)]


def foil_alpha_limits(foil):
    if hasattr(foil, "polars"):
        mins = [float(min(p.alpha_deg)) for p in foil.polars]
        maxs = [float(max(p.alpha_deg)) for p in foil.polars]
        return min(mins), max(maxs)
    if hasattr(foil, "alpha_deg"):
        return float(min(foil.alpha_deg)), float(max(foil.alpha_deg))
    return None, None


def estimate_wing_clmax(foil, rho, mu, weight, wing_area, mac, iterations=3):
    if rho <= 0.0 or mu <= 0.0 or weight <= 0.0 or wing_area <= 0.0 or mac <= 0.0:
        return 0.0
    re_guess = float(foil.reynolds_list[len(foil.reynolds_list) // 2])
    cl_max = 0.0
    for _ in range(iterations):
        _, cl_max = foil.cl_max(reynolds=re_guess)
        if cl_max <= 0.0:
            break
        v_stall = math.sqrt(2.0 * weight / (rho * wing_area * cl_max))
        re_guess = rho * v_stall * mac / mu
    return cl_max


def thrust_available(config, v):
    prop = config["propulsion"]
    max_power = prop["max_power_w"]
    prop_eff = prop["prop_eff"]
    max_thrust = prop.get("max_thrust_n")
    if v <= 0.1:
        thrust = max_power * prop_eff / 0.1
    else:
        thrust = max_power * prop_eff / v
    if max_thrust is not None:
        thrust = min(thrust, max_thrust)
    return thrust


def wing_area_and_mac(wing_span, wing_chord):
    wing_breaks = [(0.0, wing_chord), (0.4, wing_chord), (1.0, wing_chord)]
    wing_area, wing_mac = planform_area_and_mac(wing_span, wing_breaks, symmetric=True)
    if wing_area <= 0.0 or wing_mac <= 0.0:
        wing_area = wing_span * wing_chord
        wing_mac = wing_chord
    return wing_area, wing_mac


def htail_area(htail_span_scale, htail_chord_scale):
    h_root = HROOT_CHORD_M * htail_chord_scale
    h_mid = HMID_CHORD_M * htail_chord_scale
    h_tip = HTIP_CHORD_M * htail_chord_scale
    h_span = HSPAN_M * htail_span_scale
    h_breaks = [(0.0, h_root), (HMID_CHORD_POS, h_mid), (1.0, h_tip)]
    htail_area_value, _ = planform_area_and_mac(h_span, h_breaks, symmetric=True)
    return htail_area_value


def vtail_area(vtail_scale):
    v_root = VROOT_CHORD_M * vtail_scale
    v_mid = VMID_CHORD_M * vtail_scale
    v_tip = VTIP_CHORD_M * vtail_scale
    v_span = VSPAN_M * vtail_scale
    v_breaks = [(0.0, v_root), (VMID_CHORD_POS, v_mid), (1.0, v_tip)]
    vtail_single_area, _ = planform_area_and_mac(v_span, v_breaks, symmetric=False)
    return vtail_single_area * VTAIL_COUNT


def htail_geometry(htail_span_scale, htail_chord_scale):
    return {
        "span": HSPAN_M * htail_span_scale,
        "root_chord": HROOT_CHORD_M * htail_chord_scale,
        "mid_chord": HMID_CHORD_M * htail_chord_scale,
        "tip_chord": HTIP_CHORD_M * htail_chord_scale,
    }


BASE_WING_AREA, _ = wing_area_and_mac(BASE_WING_SPAN_M, BASE_WING_CHORD_M)
BASE_HTAIL_AREA = htail_area(1.0, 1.0)
BASE_VTAIL_AREA = vtail_area(1.0)


def compute_base_mass_kg():
    baseline_weight_n = 205.0
    baseline_mass_kg = baseline_weight_n / G
    base_mass_kg = baseline_mass_kg - SURFACE_AREAL_DENSITY_KG_M2 * (
        BASE_WING_AREA + BASE_HTAIL_AREA + BASE_VTAIL_AREA
    )
    return max(base_mass_kg, 0.0)


BASE_MASS_KG = compute_base_mass_kg()


def compute_weight_n(wing_area, htail_area, vtail_area):
    mass_kg = BASE_MASS_KG + SURFACE_AREAL_DENSITY_KG_M2 * (wing_area + htail_area + vtail_area)
    return mass_kg * G


def build_config(
    wing_span,
    wing_chord,
    htail_span_scale,
    htail_chord_scale,
    vtail_scale,
    x_htqc,
    weight_n,
    wing_cl_max,
):
    return build_aircraft_config(
        ground_level=GROUND_LEVEL_M,
        orbit_level=0.0,
        flight_velocity=0.0,
        weight=weight_n,
        tail_efficiency=TAIL_EFFICIENCY,
        elevator_tau=ELEVATOR_TAU,
        downwash_factor=DOWNWASH_FACTOR,
        cd_misc=CD_MISC,
        x_wqc=XWQC_M,
        x_htqc=x_htqc,
        x_cg=XCG_M,
        wing_z=WING_Z_M,
        htail_z=HTAIL_Z_M,
        vtail_z=VTAIL_Z_M,
        engine_z=ENGINE_Z_M,
        wing_foil=WING_FOIL,
        wing_root_chord=wing_chord,
        wing_mid_chord=wing_chord,
        wing_mid_chord_pos=0.4,
        wing_tip_chord=wing_chord,
        wing_span=wing_span,
        wing_incidence=WING_INCIDENCE_DEG,
        wing_root_sweep=0.0,
        wing_mid_sweep=0.0,
        wing_mid_sweep_pos=0.4,
        wing_tip_sweep=0.0,
        wing_cl_max_takeoff=wing_cl_max,
        wing_cl_max_cruise=wing_cl_max,
        wing_cl_max_landing=wing_cl_max,
        htail_foil=HTAIL_FOIL,
        htail_root_chord=HROOT_CHORD_M * htail_chord_scale,
        htail_mid_chord=HMID_CHORD_M * htail_chord_scale,
        htail_mid_chord_pos=HMID_CHORD_POS,
        htail_tip_chord=HTIP_CHORD_M * htail_chord_scale,
        htail_span=HSPAN_M * htail_span_scale,
        htail_incidence=HTAIL_INCIDENCE_DEG,
        htail_root_sweep=HROOT_SWEEP_DEG,
        htail_mid_sweep=HMID_SWEEP_DEG,
        htail_mid_sweep_pos=HMID_SWEEP_POS,
        htail_tip_sweep=HTIP_SWEEP_DEG,
        vtail_foil=VTAIL_FOIL,
        vtail_root_chord=VROOT_CHORD_M * vtail_scale,
        vtail_mid_chord=VMID_CHORD_M * vtail_scale,
        vtail_mid_chord_pos=VMID_CHORD_POS,
        vtail_tip_chord=VTIP_CHORD_M * vtail_scale,
        vtail_span=VSPAN_M * vtail_scale,
        vtail_incidence=VTAIL_INCIDENCE_DEG,
        vtail_root_sweep=VROOT_SWEEP_DEG,
        vtail_mid_sweep=VMID_SWEEP_DEG,
        vtail_mid_sweep_pos=VMID_SWEEP_POS,
        vtail_tip_sweep=VTIP_SWEEP_DEG,
        vtail_count=VTAIL_COUNT,
        fuselage_width=FUSELAGE_WIDTH_M,
        fuselage_height=FUSELAGE_HEIGHT_M,
        fuselage_length=FUSELAGE_LENGTH_M,
        fuselage_pfactor=FUSELAGE_PFACTOR,
        boom_length=max(x_htqc - XWQC_M, 0.0),
        boom_diameter=BOOM_DIAMETER_M,
        boom_count=BOOM_COUNT,
        roughness_k=SURFACE_ROUGHNESS_M,
        fuselage_laminar_frac=FUSELAGE_LAMINAR_FRAC,
        boom_laminar_frac=BOOM_LAMINAR_FRAC,
        re_crit_per_m=5e5,
        max_power_w=MAX_POWER_W,
        prop_eff=PROP_EFF,
        battery_energy_j=BATTERY_ENERGY_J,
        max_thrust_n=MAX_THRUST_N,
        n_span=N_SPAN,
    )


def stability_metrics(config, aoa_deg, elev_deg, rho=None, mu=None, d_aoa=0.25, d_elev=1.0):
    plus = run_analysis(config, aoa_deg + d_aoa, elev_deg, build_report=False, rho=rho, mu=mu)
    minus = run_analysis(config, aoa_deg - d_aoa, elev_deg, build_report=False, rho=rho, mu=mu)
    dcm_dalpha = (plus["CmTotal"] - minus["CmTotal"]) / (2.0 * d_aoa)
    dcl_dalpha = (plus["ClTotal"] - minus["ClTotal"]) / (2.0 * d_aoa)

    plus_e = run_analysis(config, aoa_deg, elev_deg + d_elev, build_report=False, rho=rho, mu=mu)
    minus_e = run_analysis(config, aoa_deg, elev_deg - d_elev, build_report=False, rho=rho, mu=mu)
    dcm_de = (plus_e["CmTotal"] - minus_e["CmTotal"]) / (2.0 * d_elev)
    dcl_de = (plus_e["ClTotal"] - minus_e["ClTotal"]) / (2.0 * d_elev)

    dcm_dcl = dcm_dalpha / dcl_dalpha if abs(dcl_dalpha) > 1e-9 else 0.0
    mac = config["wing"]["mac"]
    x_cg = config["positions"]["x_cg"]
    x_np = x_cg - dcm_dcl * mac if mac > 0.0 else x_cg
    static_margin = (x_np - x_cg) / mac if mac > 0.0 else 0.0
    return {
        "cm_alpha_deg": dcm_dalpha,
        "cl_alpha_deg": dcl_dalpha,
        "cm_delta_e_deg": dcm_de,
        "cl_delta_e_deg": dcl_de,
        "static_margin": static_margin,
    }


def optimize_cruise_endurance(config, rho, mu):
    wing = config["wing"]
    weight = config["weight"]
    cl_max = wing.get("cl_max_cruise", 0.0)
    if cl_max <= 0.0:
        return None
    v_stall = math.sqrt(2.0 * weight / (rho * wing["area"] * cl_max))
    v_min = STALL_MARGIN_CRUISE * v_stall
    v_max = CRUISE_SPEED_MAX_MPS
    if v_min >= v_max:
        return None

    alpha_min, alpha_max = foil_alpha_limits(wing["foil"])
    if alpha_min is not None:
        alpha_min -= wing["incidence"]
    if alpha_max is not None:
        alpha_max -= wing["incidence"]

    available_energy = config["propulsion"]["battery_energy_j"] * (1.0 - ENERGY_RESERVE_FRAC)
    mission_power = config["propulsion"].get("mission_systems_power", 0.0) or 0.0
    prop_eff = max(config["propulsion"].get("prop_eff", 0.0), 1e-6)
    best = None
    for v in linspace(v_min, v_max, SPEED_OPT_POINTS):
        config["analysis_altitude_m"] = CRUISE_ALTITUDE_M
        config["flight_velocity"] = v
        aoa, elev, converged, trim_result = solve_trim(
            config,
            2.0,
            0.0,
            max_iter=TRIM_MAX_ITER,
            tol_force=1e-3,
            tol_moment=1e-3,
            elevator_limit_deg=ELEVATOR_LIMIT_DEG,
            aoa_min_deg=alpha_min,
            aoa_max_deg=alpha_max,
            rho=rho,
            mu=mu,
        )
        if not converged or trim_result is None:
            continue
        drag = trim_result["Drag"]
        thrust_avail = thrust_available(config, v)
        if drag > thrust_avail:
            continue
        prop_power = drag * v / prop_eff
        if prop_power > config["propulsion"]["max_power_w"] * 1.001:
            continue
        total_power = prop_power + mission_power
        if total_power <= 0.0:
            continue
        endurance = available_energy / total_power
        if best is None or endurance > best["endurance_s"]:
            best = {
                "endurance_s": endurance,
                "v_mps": v,
                "aoa_deg": aoa,
                "elev_deg": elev,
                "drag_n": drag,
                "prop_power_w": prop_power,
                "total_power_w": total_power,
            }
    return best


def vtail_scale_for_volume(wing_area, wing_span, tail_arm):
    base_vtail_area = BASE_VTAIL_AREA
    if tail_arm <= 0.0 or wing_area <= 0.0 or wing_span <= 0.0:
        return None
    required_area = VTAIL_VOLUME_MIN * wing_area * wing_span / tail_arm
    scale = math.sqrt(required_area / base_vtail_area) if base_vtail_area > 0.0 else None
    if scale is None:
        return None
    scale = max(scale, VTAIL_SCALE_MIN)
    if scale > VTAIL_SCALE_MAX:
        return None
    return scale


def progress_update(current, total, wing_index, wing_total):
    if total <= 0:
        return
    if current == 1 or current % PROGRESS_EVERY == 0 or current == total:
        remaining = total - current
        pct = 100.0 * current / total
        print(
            f"Progress: {current}/{total} ({pct:.1f}%), left {remaining} | wing {wing_index}/{wing_total}",
            flush=True,
        )


def tail_volume_coeffs(wing_area, wing_span, wing_mac, htail_area_value, vtail_area_value, tail_arm):
    if wing_area <= 0.0 or wing_span <= 0.0 or wing_mac <= 0.0:
        return 0.0, 0.0
    htail_vol = (htail_area_value * tail_arm) / (wing_area * wing_mac)
    vtail_vol = (vtail_area_value * tail_arm) / (wing_area * wing_span)
    return htail_vol, vtail_vol


def evaluate_tail_candidate(
    wing_span,
    wing_chord,
    wing_area,
    wing_mac,
    htail_span_scale,
    htail_chord_scale,
    vtail_scale,
    x_htqc,
    rho,
    mu,
):
    tail_arm = x_htqc - XWQC_M
    if tail_arm <= 0.0:
        return None

    htail_area_value = htail_area(htail_span_scale, htail_chord_scale)
    vtail_area_value = vtail_area(vtail_scale)
    htail_vol, vtail_vol = tail_volume_coeffs(
        wing_area,
        wing_span,
        wing_mac,
        htail_area_value,
        vtail_area_value,
        tail_arm,
    )
    if htail_vol < HTAIL_VOLUME_MIN or vtail_vol < VTAIL_VOLUME_MIN:
        return None

    weight_n = compute_weight_n(wing_area, htail_area_value, vtail_area_value)
    wing_cl_max = estimate_wing_clmax(
        WING_FOIL,
        rho,
        mu,
        weight_n,
        wing_area,
        wing_mac,
    )
    if wing_cl_max <= 0.0:
        return None

    config = build_config(
        wing_span,
        wing_chord,
        htail_span_scale,
        htail_chord_scale,
        vtail_scale,
        x_htqc,
        weight_n,
        wing_cl_max,
    )
    config["propulsion"]["mission_systems_power"] = MISSION_SYSTEMS_POWER_W
    result = optimize_cruise_endurance(config, rho, mu)
    if result is None:
        return None

    stability = stability_metrics(config, result["aoa_deg"], result["elev_deg"], rho=rho, mu=mu)
    static_margin = stability["static_margin"]
    if static_margin < STATIC_MARGIN_MIN:
        return None
    if STATIC_MARGIN_MAX is not None and static_margin > STATIC_MARGIN_MAX:
        return None

    result["static_margin"] = static_margin
    result["weight_n"] = weight_n
    result["htail_volume"] = htail_vol
    result["vtail_volume"] = vtail_vol
    result["tail_arm_m"] = tail_arm
    result["config"] = config
    return result


def main():
    best = None
    wing_spans = linspace(WING_SPAN_MIN_M, WING_SPAN_MAX_M, WING_SPAN_STEPS)
    wing_chords = linspace(WING_CHORD_MIN_M, WING_CHORD_MAX_M, WING_CHORD_STEPS)
    htail_span_scales = linspace(HTAIL_SPAN_SCALE_MIN, HTAIL_SPAN_SCALE_MAX, HTAIL_SPAN_SCALE_STEPS)
    htail_chord_scales = linspace(HTAIL_CHORD_SCALE_MIN, HTAIL_CHORD_SCALE_MAX, HTAIL_CHORD_SCALE_STEPS)
    htail_x_positions = linspace(HTAIL_X_MIN_M, HTAIL_X_MAX_M, HTAIL_X_STEPS)
    atmosphere = Atmosphere(CRUISE_ALTITUDE_M)
    rho = float(atmosphere.density)
    mu = float(atmosphere.dynamic_viscosity)
    wing_total = len(wing_spans) * len(wing_chords)
    tail_total = len(htail_span_scales) * len(htail_chord_scales) * len(htail_x_positions)
    total_cases = wing_total * tail_total
    case_idx = 0
    wing_idx = 0
    print(f"Running {total_cases} cruise design cases...", flush=True)

    for span in wing_spans:
        for chord in wing_chords:
            wing_idx += 1
            wing_area, wing_mac = wing_area_and_mac(span, chord)
            for x_htqc in htail_x_positions:
                tail_arm = x_htqc - XWQC_M
                vtail_scale = vtail_scale_for_volume(wing_area, span, tail_arm)
                for htail_span_scale in htail_span_scales:
                    for htail_chord_scale in htail_chord_scales:
                        case_idx += 1
                        progress_update(case_idx, total_cases, wing_idx, wing_total)
                        if vtail_scale is None:
                            continue
                        result = evaluate_tail_candidate(
                            span,
                            chord,
                            wing_area,
                            wing_mac,
                            htail_span_scale,
                            htail_chord_scale,
                            vtail_scale,
                            x_htqc,
                            rho,
                            mu,
                        )
                        if result is None:
                            continue
                        if best is None or result["endurance_s"] > best["endurance_s"]:
                            htail_geom = htail_geometry(htail_span_scale, htail_chord_scale)
                            best = {
                                "endurance_s": result["endurance_s"],
                                "wing_span_m": span,
                                "wing_chord_m": chord,
                                "htail_span_scale": htail_span_scale,
                                "htail_chord_scale": htail_chord_scale,
                                "htail_x_m": x_htqc,
                                "htail_span_m": htail_geom["span"],
                                "htail_root_chord_m": htail_geom["root_chord"],
                                "htail_mid_chord_m": htail_geom["mid_chord"],
                                "htail_tip_chord_m": htail_geom["tip_chord"],
                                "vtail_scale": vtail_scale,
                                "static_margin": result["static_margin"],
                                "v_mps": result["v_mps"],
                                "total_power_w": result["total_power_w"],
                                "weight_n": result["weight_n"],
                                "htail_volume": result["htail_volume"],
                                "vtail_volume": result["vtail_volume"],
                                "tail_arm_m": result["tail_arm_m"],
                            }

    if best is None:
        print("No feasible designs found with the current limits.")
        return

    print("Best Cruise-Endurance Design")
    print(f"  Wing Span:        {best['wing_span_m']:.3f} m")
    print(f"  Wing Chord:       {best['wing_chord_m']:.3f} m")
    print(f"  HTail Span Scale: {best['htail_span_scale']:.3f}")
    print(f"  HTail Chord Scale: {best['htail_chord_scale']:.3f}")
    print(f"  HTail X_QC:       {best['htail_x_m']:.3f} m")
    print(f"  HTail Span:       {best['htail_span_m']:.3f} m")
    print(f"  HTail Chords:     {best['htail_root_chord_m']:.3f} / {best['htail_mid_chord_m']:.3f} / {best['htail_tip_chord_m']:.3f} m")
    print(f"  VTail Scale:      {best['vtail_scale']:.3f}")
    print(f"  Static Margin:    {best['static_margin']:.4f}")
    print(f"  Htail Volume:     {best['htail_volume']:.3f}")
    print(f"  Vtail Volume:     {best['vtail_volume']:.3f}")
    print(f"  Tail Arm:         {best['tail_arm_m']:.3f} m")
    print(f"  Cruise Speed:     {best['v_mps']:.2f} m/s")
    print(f"  Total Power:      {best['total_power_w']:.1f} W")
    print(f"  Weight:           {best['weight_n']:.1f} N")
    print(f"  Endurance:        {best['endurance_s'] / 3600.0:.2f} hr")


if __name__ == "__main__":
    main()
