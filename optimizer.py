import math

from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet

from aircraft_core import build_aircraft_config, planform_area_and_mac, run_analysis, solve_trim


G = 9.80665

### Cruise-only optimization inputs (adjust as needed)
CRUISE_ALTITUDE_M = 200
CRUISE_SPEED_MAX_MPS = 80.0
STALL_MARGIN_CRUISE = 1.25
SPEED_OPT_POINTS = 20
N_SPAN = 100
TRIM_MAX_ITER = 20
ELEVATOR_LIMIT_DEG = 30.0
FAST_LINEAR_TRIM = True
FAST_LINEAR_ALPHA_CENTER_DEG = 2.5
FAST_LINEAR_ALPHA_HALF_RANGE_DEG = 1.5

DESIGN_POINT_SCALE = 1.5

WING_SPAN_MIN_M = 3.6
WING_SPAN_MAX_M = 4.5
WING_SPAN_STEPS = max(1, int(math.ceil(10 * DESIGN_POINT_SCALE)))

WING_CHORD_MIN_M = 0.3048
WING_CHORD_MAX_M = 0.4
WING_CHORD_STEPS = max(1, int(math.ceil(10 * DESIGN_POINT_SCALE)))

HTAIL_X_MODE = "solve_volume"
HTAIL_X_MIN_M = 1.2
HTAIL_X_MAX_M = 1.5
HTAIL_X_STEPS = max(1, int(math.ceil(10 * DESIGN_POINT_SCALE)))

HTAIL_AR_MIN = 2.0
HTAIL_AR_MAX = 6.0
HTAIL_AR_STEPS = max(1, int(math.ceil(5 * DESIGN_POINT_SCALE)))
HTAIL_SPAN_SCALE_MIN = 0.3
HTAIL_SPAN_SCALE_MAX = 3.0
HTAIL_CHORD_SCALE_MIN = 0.3
HTAIL_CHORD_SCALE_MAX = 3.0

VTAIL_SCALE_MIN = 0.1
VTAIL_SCALE_MAX = 10

STATIC_MARGIN_MIN = 0.05
STATIC_MARGIN_MAX = 0.2
STATIC_MARGIN_TARGET = None
STATIC_MARGIN_TOL = 0.01
HTAIL_VOLUME_MIN = 0.3
HTAIL_VOLUME_MAX = 1.0
HTAIL_VOLUME_TARGET = 0.7
VTAIL_VOLUME_MIN = 0.03

ENERGY_RESERVE_FRAC = 0.2

SURFACE_AREAL_DENSITY_KG_M2 = 3.25
PROGRESS_EVERY = 1
PLOT_BEST_DESIGN = True
PLOT_SAVE_PATH = "best_design.png"
PLOT_SHOW = False

DEBUG_NEUTRAL_POINT = False
DEBUG_NEUTRAL_POINT_ONLY = False
DEBUG_NP_DAOA_LIST = [0.05, 0.1, 0.2]
DEBUG_NP_CG_OFFSETS_M = [-0.05, 0.0, 0.05]
DEBUG_NP_SPEED_MARGIN = 5.0
DEBUG_NP_ALPHA_FALLBACK_DEG = 3.5
DEBUG_NP_ELEV_FALLBACK_DEG = 0.0
DEBUG_NP_FAST_TRIM = False

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
XHTQC_M = 1.5
XCG_M = 0.5
XCG_MIN_M = None
XCG_MAX_M = None
WING_Z_M = 0.0
HTAIL_Z_M = 0.0
VTAIL_Z_M = 0.2
ENGINE_Z_M = 0.0

WING_INCIDENCE_DEG = 1.0
BASE_WING_SPAN_M = 5.0
BASE_WING_CHORD_M = 0.3

HROOT_CHORD_M = 0.15
HMID_CHORD_M = 0.15
HMID_CHORD_POS = 0.8
HTIP_CHORD_M = 0.15
HSPAN_M = 1.0
HROOT_SWEEP_DEG = 0.0
HMID_SWEEP_DEG = 0.0
HMID_SWEEP_POS = 0.4
HTIP_SWEEP_DEG = 0.0
HTAIL_INCIDENCE_DEG = -3.0

VROOT_CHORD_M = 0.1
VMID_CHORD_M = 0.1
VMID_CHORD_POS = 0.4
VTIP_CHORD_M = 0.1
VSPAN_M = 0.25
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


def htail_scales_for_volume(wing_area, wing_mac, tail_arm, ar_target):
    if tail_arm <= 0.0 or wing_area <= 0.0 or wing_mac <= 0.0 or BASE_HTAIL_AREA <= 0.0:
        return None, "htail_size"
    target_vol = min(max(HTAIL_VOLUME_TARGET, HTAIL_VOLUME_MIN), HTAIL_VOLUME_MAX)
    required_area = target_vol * wing_area * wing_mac / tail_arm
    if required_area <= 0.0 or ar_target <= 0.0 or BASE_HTAIL_AR <= 0.0:
        return None, "htail_size"
    ratio = ar_target / BASE_HTAIL_AR
    if ratio <= 0.0:
        return None, "htail_ar"
    chord_scale = math.sqrt(required_area / (BASE_HTAIL_AREA * ratio))
    span_scale = ratio * chord_scale
    if chord_scale <= 0.0 or span_scale <= 0.0:
        return None, "htail_size"
    if span_scale < HTAIL_SPAN_SCALE_MIN or span_scale > HTAIL_SPAN_SCALE_MAX:
        return None, "htail_span_scale"
    if chord_scale < HTAIL_CHORD_SCALE_MIN or chord_scale > HTAIL_CHORD_SCALE_MAX:
        return None, "htail_chord_scale"
    htail_area_value = htail_area(span_scale, chord_scale)
    if htail_area_value <= 0.0:
        return None, "htail_size"
    return (span_scale, chord_scale, htail_area_value, htail_geometry(span_scale, chord_scale)), None


def solve_htail_for_volume(wing_area, wing_mac, ar_target):
    if wing_area <= 0.0 or wing_mac <= 0.0 or BASE_HTAIL_AREA <= 0.0 or BASE_HTAIL_AR <= 0.0:
        return None, "htail_size"
    ratio = ar_target / BASE_HTAIL_AR
    if ratio <= 0.0:
        return None, "htail_ar"
    target_vol = min(max(HTAIL_VOLUME_TARGET, HTAIL_VOLUME_MIN), HTAIL_VOLUME_MAX)
    chord_min = HTAIL_CHORD_SCALE_MIN
    chord_max = HTAIL_CHORD_SCALE_MAX
    chord_min = max(chord_min, HTAIL_SPAN_SCALE_MIN / ratio)
    chord_max = min(chord_max, HTAIL_SPAN_SCALE_MAX / ratio)
    if chord_min > chord_max:
        return None, "htail_scale_bounds"
    area_min = BASE_HTAIL_AREA * ratio * (chord_min ** 2)
    area_max = BASE_HTAIL_AREA * ratio * (chord_max ** 2)
    if area_min <= 0.0 or area_max <= 0.0:
        return None, "htail_size"
    tail_arm_min = target_vol * wing_area * wing_mac / area_max
    tail_arm_max = target_vol * wing_area * wing_mac / area_min
    x_arm_min = HTAIL_X_MIN_M - XWQC_M
    x_arm_max = HTAIL_X_MAX_M - XWQC_M
    if tail_arm_max <= 0.0 or x_arm_max <= 0.0:
        return None, "tail_arm"
    tail_arm_low = max(tail_arm_min, x_arm_min)
    tail_arm_high = min(tail_arm_max, x_arm_max)
    if tail_arm_low > tail_arm_high:
        return None, "htail_x_range"
    tail_arm = 0.5 * (tail_arm_low + tail_arm_high)
    if tail_arm <= 0.0:
        return None, "tail_arm"
    required_area = target_vol * wing_area * wing_mac / tail_arm
    chord_scale = math.sqrt(required_area / (BASE_HTAIL_AREA * ratio))
    span_scale = ratio * chord_scale
    if chord_scale < HTAIL_CHORD_SCALE_MIN or chord_scale > HTAIL_CHORD_SCALE_MAX:
        return None, "htail_chord_scale"
    if span_scale < HTAIL_SPAN_SCALE_MIN or span_scale > HTAIL_SPAN_SCALE_MAX:
        return None, "htail_span_scale"
    htail_area_value = htail_area(span_scale, chord_scale)
    if htail_area_value <= 0.0:
        return None, "htail_size"
    x_htqc = XWQC_M + tail_arm
    return {
        "x_htqc_m": x_htqc,
        "tail_arm_m": tail_arm,
        "htail_span_scale": span_scale,
        "htail_chord_scale": chord_scale,
        "htail_area_m2": htail_area_value,
        "htail_geometry": htail_geometry(span_scale, chord_scale),
    }, None


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
BASE_HTAIL_AR = (HSPAN_M ** 2) / BASE_HTAIL_AREA if BASE_HTAIL_AREA > 0.0 else 0.0


def compute_base_mass_kg():
    baseline_weight_n = 190.0
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
    config = build_aircraft_config(
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
    config["fast_linear_alpha_center_deg"] = FAST_LINEAR_ALPHA_CENTER_DEG
    config["fast_linear_alpha_half_range_deg"] = FAST_LINEAR_ALPHA_HALF_RANGE_DEG
    return config


def stability_metrics(config, aoa_deg, elev_deg, rho=None, mu=None, d_aoa=0.01, d_elev=1.0):
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
        "cm_cl": dcm_dcl,
        "x_np_m": x_np,
        "static_margin": static_margin,
    }


def static_margin_target():
    if STATIC_MARGIN_TARGET is not None:
        return STATIC_MARGIN_TARGET
    if STATIC_MARGIN_MIN is not None and STATIC_MARGIN_MAX is not None:
        return 0.5 * (STATIC_MARGIN_MIN + STATIC_MARGIN_MAX)
    if STATIC_MARGIN_MIN is not None:
        return STATIC_MARGIN_MIN
    return 0.0


def solve_xcg_for_static_margin(config, aoa_deg, elev_deg, rho, mu):
    stability = stability_metrics(config, aoa_deg, elev_deg, rho=rho, mu=mu)
    mac = config["wing"]["mac"]
    if mac <= 0.0:
        return None, "wing_mac", stability
    x_np = stability["x_np_m"]
    target_sm = static_margin_target()
    x_cg_target = x_np - target_sm * mac
    if XCG_MIN_M is not None and x_cg_target < XCG_MIN_M:
        return None, "xcg_low", stability
    if XCG_MAX_M is not None and x_cg_target > XCG_MAX_M:
        return None, "xcg_high", stability
    config["positions"]["x_cg"] = x_cg_target
    updated = stability_metrics(config, aoa_deg, elev_deg, rho=rho, mu=mu)
    return {
        "x_cg_m": x_cg_target,
        "x_np_m": updated["x_np_m"],
        "static_margin": updated["static_margin"],
        "target_static_margin": target_sm,
        "cm_alpha_deg": updated["cm_alpha_deg"],
        "cl_alpha_deg": updated["cl_alpha_deg"],
    }, None, stability


def optimize_cruise_endurance(config, rho, mu):
    wing = config["wing"]
    weight = config["weight"]
    cl_max = wing.get("cl_max_cruise", 0.0)
    if cl_max <= 0.0:
        return None, "cruise_clmax"
    v_stall = math.sqrt(2.0 * weight / (rho * wing["area"] * cl_max))
    v_min = STALL_MARGIN_CRUISE * v_stall
    v_max = CRUISE_SPEED_MAX_MPS
    if v_min >= v_max:
        return None, "cruise_stall_margin"

    alpha_min, alpha_max = foil_alpha_limits(wing["foil"])
    if alpha_min is not None:
        alpha_min -= wing["incidence"]
    if alpha_max is not None:
        alpha_max -= wing["incidence"]

    available_energy = config["propulsion"]["battery_energy_j"] * (1.0 - ENERGY_RESERVE_FRAC)
    mission_power = config["propulsion"].get("mission_systems_power", 0.0) or 0.0
    prop_eff = max(config["propulsion"].get("prop_eff", 0.0), 1e-6)
    best = None
    samples = linspace(v_min, v_max, SPEED_OPT_POINTS)
    fail_trim = 0
    fail_thrust = 0
    fail_power = 0
    fail_total_power = 0
    for v in samples:
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
            fast_linear=FAST_LINEAR_TRIM,
        )
        if not converged or trim_result is None:
            fail_trim += 1
            continue
        drag = trim_result["Drag"]
        thrust_avail = thrust_available(config, v)
        if drag > thrust_avail:
            fail_thrust += 1
            continue
        prop_power = drag * v / prop_eff
        if prop_power > config["propulsion"]["max_power_w"] * 1.001:
            fail_power += 1
            continue
        total_power = prop_power + mission_power
        if total_power <= 0.0:
            fail_total_power += 1
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
                "cl_alpha_w_deg": trim_result.get("ClAlphaWing") if trim_result else None,
                "cl_alpha_h_deg": trim_result.get("ClAlphaTail") if trim_result else None,
                "linear_alpha_center": trim_result.get("LinearAlphaCenter") if trim_result else None,
                "linear_alpha_range": trim_result.get("LinearAlphaRange") if trim_result else None,
            }
    if best is not None:
        return best, None
    if len(samples) > 0:
        if fail_trim == len(samples):
            return None, "cruise_trim"
        if fail_thrust == len(samples):
            return None, "cruise_thrust"
        if fail_power == len(samples):
            return None, "cruise_power"
        if fail_total_power == len(samples):
            return None, "cruise_total_power"
    return None, "cruise_no_solution"


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


def progress_update(current, total, wing_index, wing_total, valid_count):
    if total <= 0:
        return
    if current == 1 or current % PROGRESS_EVERY == 0 or current == total:
        remaining = total - current
        pct = 100.0 * current / total
        print(
            f"Progress: {current}/{total} ({pct:.1f}%), left {remaining} | wing {wing_index}/{wing_total} | valid {valid_count}",
            flush=True,
        )


def bump_failure(counts, reason):
    if reason is None:
        reason = "unknown"
    counts[reason] = counts.get(reason, 0) + 1


def print_failure_summary(counts):
    total_failed = sum(counts.values())
    print("Failure Summary")
    print(f"  Total Failed:     {total_failed}")
    for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {reason}: {count}")


def update_stat(stats, key, value):
    if value is None:
        return
    entry = stats.setdefault(key, {"sum": 0.0, "min": None, "max": None, "count": 0})
    entry["sum"] += value
    entry["count"] += 1
    if entry["min"] is None or value < entry["min"]:
        entry["min"] = value
    if entry["max"] is None or value > entry["max"]:
        entry["max"] = value


def print_stat_summary(title, stats, ordered_keys):
    print(title)
    for key, label, fmt in ordered_keys:
        entry = stats.get(key)
        if not entry or entry["count"] == 0:
            print(f"  {label}: n/a")
            continue
        avg = entry["sum"] / entry["count"]
        min_val = entry["min"]
        max_val = entry["max"]
        print(f"  {label} Avg: {fmt.format(avg)}")
        print(f"  {label} Min: {fmt.format(min_val)}")
        print(f"  {label} Max: {fmt.format(max_val)}")


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
        return None, "tail_arm", {}

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
    metrics = {
        "htail_volume": htail_vol,
        "vtail_volume": vtail_vol,
        "tail_arm_m": tail_arm,
        "htail_area_m2": htail_area_value,
        "vtail_area_m2": vtail_area_value,
        "htail_span_scale": htail_span_scale,
        "htail_chord_scale": htail_chord_scale,
    }
    htail_span_m = HSPAN_M * htail_span_scale
    if htail_area_value > 0.0:
        metrics["htail_ar"] = (htail_span_m ** 2) / htail_area_value
    if vtail_vol < VTAIL_VOLUME_MIN:
        return None, "vtail_volume", metrics
    if htail_vol < HTAIL_VOLUME_MIN or htail_vol > HTAIL_VOLUME_MAX:
        return None, "htail_volume", metrics

    weight_n = compute_weight_n(wing_area, htail_area_value, vtail_area_value)
    metrics["weight_n"] = weight_n
    wing_cl_max = estimate_wing_clmax(
        WING_FOIL,
        rho,
        mu,
        weight_n,
        wing_area,
        wing_mac,
    )
    if wing_cl_max <= 0.0:
        return None, "wing_clmax", metrics

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
    result, cruise_fail = optimize_cruise_endurance(config, rho, mu)
    if result is None:
        return None, cruise_fail, metrics

    xcg_solution, xcg_reason, stability = solve_xcg_for_static_margin(
        config,
        result["aoa_deg"],
        result["elev_deg"],
        rho,
        mu,
    )
    if xcg_solution is None:
        metrics["static_margin"] = stability.get("static_margin") if stability else None
        return None, xcg_reason, metrics
    static_margin = xcg_solution["static_margin"]
    metrics["static_margin"] = static_margin
    metrics["x_cg_m"] = xcg_solution["x_cg_m"]
    metrics["x_np_m"] = xcg_solution["x_np_m"]
    if STATIC_MARGIN_MIN is not None and static_margin < STATIC_MARGIN_MIN:
        return None, "static_margin_low", metrics
    if STATIC_MARGIN_MAX is not None and static_margin > STATIC_MARGIN_MAX:
        return None, "static_margin_high", metrics
    if (
        xcg_solution["target_static_margin"] is not None
        and abs(static_margin - xcg_solution["target_static_margin"]) > STATIC_MARGIN_TOL
    ):
        return None, "static_margin_target", metrics

    result["static_margin"] = static_margin
    result["x_cg_m"] = xcg_solution["x_cg_m"]
    result["x_np_m"] = xcg_solution["x_np_m"]
    result["target_static_margin"] = xcg_solution["target_static_margin"]
    result["weight_n"] = weight_n
    result["htail_volume"] = htail_vol
    result["vtail_volume"] = vtail_vol
    result["tail_arm_m"] = tail_arm
    result["config"] = config
    return result, None, metrics


def debug_neutral_point(rho, mu):
    print("Neutral Point Debug")
    span = 0.5 * (WING_SPAN_MIN_M + WING_SPAN_MAX_M)
    chord = 0.5 * (WING_CHORD_MIN_M + WING_CHORD_MAX_M)
    wing_area, wing_mac = wing_area_and_mac(span, chord)
    ar_target = 0.5 * (HTAIL_AR_MIN + HTAIL_AR_MAX)
    if HTAIL_X_MODE == "solve_volume":
        htail_solution, htail_reason = solve_htail_for_volume(wing_area, wing_mac, ar_target)
        if htail_solution is None:
            print(f"  Htail sizing failed: {htail_reason}")
            return
        x_htqc = htail_solution["x_htqc_m"]
        htail_span_scale = htail_solution["htail_span_scale"]
        htail_chord_scale = htail_solution["htail_chord_scale"]
        htail_area_value = htail_solution["htail_area_m2"]
        htail_geom = htail_solution["htail_geometry"]
    else:
        x_htqc = 0.5 * (HTAIL_X_MIN_M + HTAIL_X_MAX_M)
        tail_arm = x_htqc - XWQC_M
        if tail_arm <= 0.0:
            print("  Htail sizing failed: tail_arm")
            return
        htail_sizing, htail_reason = htail_scales_for_volume(wing_area, wing_mac, tail_arm, ar_target)
        if htail_sizing is None:
            print(f"  Htail sizing failed: {htail_reason}")
            return
        htail_span_scale, htail_chord_scale, htail_area_value, htail_geom = htail_sizing

    tail_arm = x_htqc - XWQC_M
    vtail_scale = vtail_scale_for_volume(wing_area, span, tail_arm)
    if vtail_scale is None:
        print("  Vtail sizing failed: vtail_scale")
        return
    vtail_area_value = vtail_area(vtail_scale)
    htail_vol, vtail_vol = tail_volume_coeffs(
        wing_area,
        span,
        wing_mac,
        htail_area_value,
        vtail_area_value,
        tail_arm,
    )
    weight_n = compute_weight_n(wing_area, htail_area_value, vtail_area_value)
    wing_cl_max = estimate_wing_clmax(WING_FOIL, rho, mu, weight_n, wing_area, wing_mac)
    if wing_cl_max <= 0.0:
        print("  Wing CLmax estimate failed")
        return

    config = build_config(
        span,
        chord,
        htail_span_scale,
        htail_chord_scale,
        vtail_scale,
        x_htqc,
        weight_n,
        wing_cl_max,
    )
    config["propulsion"]["mission_systems_power"] = MISSION_SYSTEMS_POWER_W
    v_stall = math.sqrt(2.0 * weight_n / (rho * wing_area * wing_cl_max))
    v_guess = min(CRUISE_SPEED_MAX_MPS, v_stall * STALL_MARGIN_CRUISE + DEBUG_NP_SPEED_MARGIN)
    config["analysis_altitude_m"] = CRUISE_ALTITUDE_M
    config["flight_velocity"] = v_guess

    aoa_deg, elev_deg, converged, _ = solve_trim(
        config,
        2.0,
        0.0,
        max_iter=TRIM_MAX_ITER,
        tol_force=1e-3,
        tol_moment=1e-3,
        elevator_limit_deg=ELEVATOR_LIMIT_DEG,
        rho=rho,
        mu=mu,
        fast_linear=DEBUG_NP_FAST_TRIM,
    )
    if not converged:
        aoa_deg = DEBUG_NP_ALPHA_FALLBACK_DEG
        elev_deg = DEBUG_NP_ELEV_FALLBACK_DEG
        print("  Trim failed, using fallback AoA/Elevator")

    print(f"  Wing Span/Chord:  {span:.3f} / {chord:.3f} m")
    print(f"  Wing Area/MAC:    {wing_area:.3f} m^2 / {wing_mac:.3f} m")
    print(f"  Htail X_QC:       {x_htqc:.3f} m (Tail Arm {tail_arm:.3f} m)")
    print(f"  Htail AR Target:  {ar_target:.3f}")
    print(f"  Htail Area:       {htail_area_value:.3f} m^2")
    print(f"  Htail Span/Chord: {htail_geom['span']:.3f} m / {htail_geom['root_chord']:.3f} m")
    print(f"  Vtail Scale:      {vtail_scale:.3f}")
    print(f"  Tail Volumes:     H {htail_vol:.3f} / V {vtail_vol:.3f}")
    print(f"  Weight:           {weight_n:.1f} N")
    print(f"  Speed:            {v_guess:.2f} m/s")
    print(f"  Trim AoA/Elev:    {aoa_deg:.3f} deg / {elev_deg:.3f} deg")

    for d_aoa in DEBUG_NP_DAOA_LIST:
        metrics = stability_metrics(
            config,
            aoa_deg,
            elev_deg,
            rho=rho,
            mu=mu,
            d_aoa=d_aoa,
        )
        print(
            "  dAoA {:+.3f} deg -> dCm/dA {: .5f}, dCl/dA {: .5f}, dCm/dCl {: .5f}, x_np {: .3f} m, SM {: .4f}".format(
                d_aoa,
                metrics["cm_alpha_deg"],
                metrics["cl_alpha_deg"],
                metrics["cm_cl"],
                metrics["x_np_m"],
                metrics["static_margin"],
            )
        )

    base_x_cg = config["positions"]["x_cg"]
    if DEBUG_NP_CG_OFFSETS_M:
        print("  CG Sweep")
    for offset in DEBUG_NP_CG_OFFSETS_M:
        config["positions"]["x_cg"] = base_x_cg + offset
        metrics = stability_metrics(
            config,
            aoa_deg,
            elev_deg,
            rho=rho,
            mu=mu,
            d_aoa=DEBUG_NP_DAOA_LIST[0] if DEBUG_NP_DAOA_LIST else 0.1,
        )
        print(
            "    x_cg {: .3f} m (offset {:+.3f}) -> x_np {: .3f} m, SM {: .4f}".format(
                config["positions"]["x_cg"],
                offset,
                metrics["x_np_m"],
                metrics["static_margin"],
            )
        )
    config["positions"]["x_cg"] = base_x_cg


def plot_best_design(best):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("Plot skipped: matplotlib not available.")
        return

    wing_span = best["wing_span_m"]
    wing_chord = best["wing_chord_m"]
    htail_span = best["htail_span_m"]
    htail_chord = best["htail_root_chord_m"]

    wing_le = XWQC_M - 0.25 * wing_chord
    htail_le = best["htail_x_m"] - 0.25 * htail_chord

    fuselage_x = 0.0
    fuselage_length = FUSELAGE_LENGTH_M
    fuselage_width = FUSELAGE_WIDTH_M

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.add_patch(
        Rectangle(
            (wing_le, -0.5 * wing_span),
            wing_chord,
            wing_span,
            facecolor="#7da3d8",
            edgecolor="black",
            alpha=0.6,
            label="Wing",
        )
    )
    ax.add_patch(
        Rectangle(
            (htail_le, -0.5 * htail_span),
            htail_chord,
            htail_span,
            facecolor="#d6b36b",
            edgecolor="black",
            alpha=0.6,
            label="Htail",
        )
    )
    ax.add_patch(
        Rectangle(
            (fuselage_x, -0.5 * fuselage_width),
            fuselage_length,
            fuselage_width,
            facecolor="#b0b0b0",
            edgecolor="black",
            alpha=0.7,
            label="Fuselage",
        )
    )
    if best.get("x_cg_m") is not None:
        ax.plot(best["x_cg_m"], 0.0, "ko", label="CG")
    if best.get("x_np_m") is not None:
        ax.plot(best["x_np_m"], 0.0, "k^", label="NP")

    x_min = min(wing_le, htail_le, fuselage_x) - 0.2
    x_max = max(wing_le + wing_chord, htail_le + htail_chord, fuselage_x + fuselage_length) + 0.2
    y_max = 0.6 * max(wing_span, htail_span, fuselage_width)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top View Planform")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")

    if PLOT_SAVE_PATH:
        fig.savefig(PLOT_SAVE_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {PLOT_SAVE_PATH}")
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)


def main():
    best = None
    wing_spans = linspace(WING_SPAN_MIN_M, WING_SPAN_MAX_M, WING_SPAN_STEPS)
    wing_chords = linspace(WING_CHORD_MIN_M, WING_CHORD_MAX_M, WING_CHORD_STEPS)
    htail_ar_targets = linspace(HTAIL_AR_MIN, HTAIL_AR_MAX, HTAIL_AR_STEPS)
    atmosphere = Atmosphere(CRUISE_ALTITUDE_M)
    rho = float(atmosphere.density)
    mu = float(atmosphere.dynamic_viscosity)
    wing_total = len(wing_spans) * len(wing_chords)
    use_htail_grid = HTAIL_X_MODE == "grid"
    htail_x_positions = linspace(HTAIL_X_MIN_M, HTAIL_X_MAX_M, HTAIL_X_STEPS) if use_htail_grid else None
    tail_total = len(htail_ar_targets) * (len(htail_x_positions) if use_htail_grid else 1)
    total_cases = wing_total * tail_total
    case_idx = 0
    wing_idx = 0
    valid_count = 0
    failure_counts = {}
    statics_sum = 0.0
    statics_min = None
    statics_max = None
    htail_vol_sum = 0.0
    htail_vol_min = None
    htail_vol_max = None
    vtail_vol_sum = 0.0
    vtail_vol_min = None
    vtail_vol_max = None
    tail_arm_sum = 0.0
    htail_ar_sum = 0.0
    htail_ar_min = None
    htail_ar_max = None
    mass_sum = 0.0
    mass_min = None
    mass_max = None
    cl_alpha_w_sum = 0.0
    cl_alpha_h_sum = 0.0
    cl_alpha_samples = 0
    linear_alpha_center = None
    linear_alpha_range = None
    overall_stats = {}

    if DEBUG_NEUTRAL_POINT or DEBUG_NEUTRAL_POINT_ONLY:
        debug_neutral_point(rho, mu)
        if DEBUG_NEUTRAL_POINT_ONLY:
            return

    print(f"Running {total_cases} cruise design cases...", flush=True)

    for span in wing_spans:
        for chord in wing_chords:
            wing_idx += 1
            wing_area, wing_mac = wing_area_and_mac(span, chord)
            if use_htail_grid:
                for x_htqc in htail_x_positions:
                    tail_arm = x_htqc - XWQC_M
                    vtail_scale = vtail_scale_for_volume(wing_area, span, tail_arm)
                    for htail_ar in htail_ar_targets:
                        case_idx += 1
                        progress_update(case_idx, total_cases, wing_idx, wing_total, valid_count)
                        if tail_arm <= 0.0:
                            bump_failure(failure_counts, "tail_arm")
                            continue
                        if vtail_scale is None:
                            bump_failure(failure_counts, "vtail_scale")
                            continue
                        htail_sizing, htail_reason = htail_scales_for_volume(
                            wing_area,
                            wing_mac,
                            tail_arm,
                            htail_ar,
                        )
                        if htail_sizing is None:
                            bump_failure(failure_counts, htail_reason)
                            continue
                        htail_span_scale, htail_chord_scale, htail_area_value, htail_geom = htail_sizing
                        result, reason, metrics = evaluate_tail_candidate(
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
                        update_stat(overall_stats, "static_margin", metrics.get("static_margin"))
                        update_stat(overall_stats, "htail_volume", metrics.get("htail_volume"))
                        update_stat(overall_stats, "vtail_volume", metrics.get("vtail_volume"))
                        update_stat(overall_stats, "tail_arm", metrics.get("tail_arm_m"))
                        update_stat(overall_stats, "htail_ar", metrics.get("htail_ar"))
                        update_stat(
                            overall_stats,
                            "mass_kg",
                            (metrics.get("weight_n") or 0.0) / G if metrics.get("weight_n") is not None else None,
                        )
                        if result is None:
                            bump_failure(failure_counts, reason)
                            continue
                        valid_count += 1
                        mass_kg = result["weight_n"] / G
                        mass_sum += mass_kg
                        if mass_min is None or mass_kg < mass_min:
                            mass_min = mass_kg
                        if mass_max is None or mass_kg > mass_max:
                            mass_max = mass_kg
                        statics_sum += result["static_margin"]
                        if statics_min is None or result["static_margin"] < statics_min:
                            statics_min = result["static_margin"]
                        if statics_max is None or result["static_margin"] > statics_max:
                            statics_max = result["static_margin"]
                        htail_vol_sum += result["htail_volume"]
                        if htail_vol_min is None or result["htail_volume"] < htail_vol_min:
                            htail_vol_min = result["htail_volume"]
                        if htail_vol_max is None or result["htail_volume"] > htail_vol_max:
                            htail_vol_max = result["htail_volume"]
                        vtail_vol_sum += result["vtail_volume"]
                        if vtail_vol_min is None or result["vtail_volume"] < vtail_vol_min:
                            vtail_vol_min = result["vtail_volume"]
                        if vtail_vol_max is None or result["vtail_volume"] > vtail_vol_max:
                            vtail_vol_max = result["vtail_volume"]
                        tail_arm_sum += result["tail_arm_m"]
                        htail_ar_value = metrics.get("htail_ar")
                        if htail_ar_value is not None:
                            htail_ar_sum += htail_ar_value
                            if htail_ar_min is None or htail_ar_value < htail_ar_min:
                                htail_ar_min = htail_ar_value
                            if htail_ar_max is None or htail_ar_value > htail_ar_max:
                                htail_ar_max = htail_ar_value
                        if result.get("cl_alpha_w_deg") is not None and result.get("cl_alpha_h_deg") is not None:
                            cl_alpha_w_sum += result["cl_alpha_w_deg"]
                            cl_alpha_h_sum += result["cl_alpha_h_deg"]
                            cl_alpha_samples += 1
                        if result.get("linear_alpha_center") is not None:
                            linear_alpha_center = result["linear_alpha_center"]
                        if result.get("linear_alpha_range") is not None:
                            linear_alpha_range = result["linear_alpha_range"]
                        if best is None or result["endurance_s"] > best["endurance_s"]:
                            htail_ar_actual = (htail_geom["span"] ** 2) / htail_area_value if htail_area_value > 0.0 else 0.0
                            best = {
                                "endurance_s": result["endurance_s"],
                                "wing_span_m": span,
                                "wing_chord_m": chord,
                                "htail_x_m": x_htqc,
                                "htail_span_scale": htail_span_scale,
                                "htail_chord_scale": htail_chord_scale,
                                "htail_ar": htail_ar_actual,
                                "htail_area_m2": htail_area_value,
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
                            "x_cg_m": result.get("x_cg_m"),
                            "x_np_m": result.get("x_np_m"),
                            "target_static_margin": result.get("target_static_margin"),
                        }
            else:
                for htail_ar in htail_ar_targets:
                    case_idx += 1
                    progress_update(case_idx, total_cases, wing_idx, wing_total, valid_count)
                    htail_solution, htail_reason = solve_htail_for_volume(wing_area, wing_mac, htail_ar)
                    if htail_solution is None:
                        bump_failure(failure_counts, htail_reason)
                        continue
                    x_htqc = htail_solution["x_htqc_m"]
                    tail_arm = htail_solution["tail_arm_m"]
                    vtail_scale = vtail_scale_for_volume(wing_area, span, tail_arm)
                    if vtail_scale is None:
                        bump_failure(failure_counts, "vtail_scale")
                        continue
                    htail_span_scale = htail_solution["htail_span_scale"]
                    htail_chord_scale = htail_solution["htail_chord_scale"]
                    htail_area_value = htail_solution["htail_area_m2"]
                    htail_geom = htail_solution["htail_geometry"]
                    result, reason, metrics = evaluate_tail_candidate(
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
                    update_stat(overall_stats, "static_margin", metrics.get("static_margin"))
                    update_stat(overall_stats, "htail_volume", metrics.get("htail_volume"))
                    update_stat(overall_stats, "vtail_volume", metrics.get("vtail_volume"))
                    update_stat(overall_stats, "tail_arm", metrics.get("tail_arm_m"))
                    update_stat(overall_stats, "htail_ar", metrics.get("htail_ar"))
                    update_stat(
                        overall_stats,
                        "mass_kg",
                        (metrics.get("weight_n") or 0.0) / G if metrics.get("weight_n") is not None else None,
                    )
                    if result is None:
                        bump_failure(failure_counts, reason)
                        continue
                    valid_count += 1
                    mass_kg = result["weight_n"] / G
                    mass_sum += mass_kg
                    if mass_min is None or mass_kg < mass_min:
                        mass_min = mass_kg
                    if mass_max is None or mass_kg > mass_max:
                        mass_max = mass_kg
                    statics_sum += result["static_margin"]
                    if statics_min is None or result["static_margin"] < statics_min:
                        statics_min = result["static_margin"]
                    if statics_max is None or result["static_margin"] > statics_max:
                        statics_max = result["static_margin"]
                    htail_vol_sum += result["htail_volume"]
                    if htail_vol_min is None or result["htail_volume"] < htail_vol_min:
                        htail_vol_min = result["htail_volume"]
                    if htail_vol_max is None or result["htail_volume"] > htail_vol_max:
                        htail_vol_max = result["htail_volume"]
                    vtail_vol_sum += result["vtail_volume"]
                    if vtail_vol_min is None or result["vtail_volume"] < vtail_vol_min:
                        vtail_vol_min = result["vtail_volume"]
                    if vtail_vol_max is None or result["vtail_volume"] > vtail_vol_max:
                        vtail_vol_max = result["vtail_volume"]
                    tail_arm_sum += result["tail_arm_m"]
                    htail_ar_value = metrics.get("htail_ar")
                    if htail_ar_value is not None:
                        htail_ar_sum += htail_ar_value
                        if htail_ar_min is None or htail_ar_value < htail_ar_min:
                            htail_ar_min = htail_ar_value
                        if htail_ar_max is None or htail_ar_value > htail_ar_max:
                            htail_ar_max = htail_ar_value
                    if result.get("cl_alpha_w_deg") is not None and result.get("cl_alpha_h_deg") is not None:
                        cl_alpha_w_sum += result["cl_alpha_w_deg"]
                        cl_alpha_h_sum += result["cl_alpha_h_deg"]
                        cl_alpha_samples += 1
                    if result.get("linear_alpha_center") is not None:
                        linear_alpha_center = result["linear_alpha_center"]
                    if result.get("linear_alpha_range") is not None:
                        linear_alpha_range = result["linear_alpha_range"]
                    if best is None or result["endurance_s"] > best["endurance_s"]:
                        htail_ar_actual = (htail_geom["span"] ** 2) / htail_area_value if htail_area_value > 0.0 else 0.0
                        best = {
                            "endurance_s": result["endurance_s"],
                            "wing_span_m": span,
                            "wing_chord_m": chord,
                            "htail_x_m": x_htqc,
                            "htail_span_scale": htail_span_scale,
                            "htail_chord_scale": htail_chord_scale,
                            "htail_ar": htail_ar_actual,
                            "htail_area_m2": htail_area_value,
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
                            "x_cg_m": result.get("x_cg_m"),
                            "x_np_m": result.get("x_np_m"),
                            "target_static_margin": result.get("target_static_margin"),
                        }

    if best is None:
        print("No feasible designs found with the current limits.")
        print_stat_summary(
            "All Designs (Computed Metrics)",
            overall_stats,
            [
                ("static_margin", "Static Margin", "{:.4f}"),
                ("htail_volume", "Htail Volume", "{:.3f}"),
                ("vtail_volume", "Vtail Volume", "{:.3f}"),
                ("tail_arm", "Tail Arm", "{:.3f} m"),
                ("htail_ar", "Htail AR", "{:.3f}"),
                ("mass_kg", "Mass", "{:.3f} kg"),
            ],
        )
        print_failure_summary(failure_counts)
        return

    print("Best Cruise-Endurance Design")
    print(f"  Wing Span:        {best['wing_span_m']:.3f} m")
    print(f"  Wing Chord:       {best['wing_chord_m']:.3f} m")
    print(f"  HTail X_QC:       {best['htail_x_m']:.3f} m")
    print(f"  HTail Area:       {best['htail_area_m2']:.3f} m^2")
    print(f"  HTail Span Scale: {best['htail_span_scale']:.3f}")
    print(f"  HTail Chord Scale: {best['htail_chord_scale']:.3f}")
    print(f"  HTail AR:         {best['htail_ar']:.3f}")
    print(f"  HTail Span:       {best['htail_span_m']:.3f} m")
    print(f"  HTail Chords:     {best['htail_root_chord_m']:.3f} / {best['htail_mid_chord_m']:.3f} / {best['htail_tip_chord_m']:.3f} m")
    print(f"  VTail Scale:      {best['vtail_scale']:.3f}")
    print(f"  Static Margin:    {best['static_margin']:.4f}")
    print(f"  Htail Volume:     {best['htail_volume']:.3f}")
    print(f"  Vtail Volume:     {best['vtail_volume']:.3f}")
    print(f"  Tail Arm:         {best['tail_arm_m']:.3f} m")
    if best.get("x_cg_m") is not None:
        print(f"  Xcg:              {best['x_cg_m']:.3f} m")
    if best.get("x_np_m") is not None:
        print(f"  Xnp:              {best['x_np_m']:.3f} m")
    if best.get("target_static_margin") is not None:
        print(f"  Target SM:        {best['target_static_margin']:.3f}")
    print(f"  Cruise Speed:     {best['v_mps']:.2f} m/s")
    print(f"  Total Power:      {best['total_power_w']:.1f} W")
    print(f"  Weight:           {best['weight_n']:.1f} N")
    print(f"  Endurance:        {best['endurance_s'] / 3600.0:.2f} hr")
    if valid_count > 0:
        print("Average Metrics (Valid Designs)")
        print(f"  Static Margin:    {statics_sum / valid_count:.4f}")
        print(f"  Static Margin Min:{statics_min:.4f}")
        print(f"  Static Margin Max:{statics_max:.4f}")
        print(f"  Htail Volume:     {htail_vol_sum / valid_count:.3f}")
        print(f"  Htail Vol Min:    {htail_vol_min:.3f}")
        print(f"  Htail Vol Max:    {htail_vol_max:.3f}")
        print(f"  Vtail Volume:     {vtail_vol_sum / valid_count:.3f}")
        print(f"  Vtail Vol Min:    {vtail_vol_min:.3f}")
        print(f"  Vtail Vol Max:    {vtail_vol_max:.3f}")
        print(f"  Tail Arm:         {tail_arm_sum / valid_count:.3f} m")
        if htail_ar_min is not None and htail_ar_max is not None:
            print(f"  Htail AR Avg:     {htail_ar_sum / valid_count:.3f}")
            print(f"  Htail AR Min:     {htail_ar_min:.3f}")
            print(f"  Htail AR Max:     {htail_ar_max:.3f}")
        print(f"  Mass Avg:         {mass_sum / valid_count:.3f} kg")
        print(f"  Mass Min:         {mass_min:.3f} kg")
        print(f"  Mass Max:         {mass_max:.3f} kg")
        if cl_alpha_samples > 0:
            print(f"  ClAlpha Wing:     {cl_alpha_w_sum / cl_alpha_samples:.4f} 1/deg")
            print(f"  ClAlpha Tail:     {cl_alpha_h_sum / cl_alpha_samples:.4f} 1/deg")
        if linear_alpha_center is not None and linear_alpha_range is not None:
            alpha_min = linear_alpha_center - linear_alpha_range
            alpha_max = linear_alpha_center + linear_alpha_range
            print(f"  Linear Alpha:     {alpha_min:.2f} to {alpha_max:.2f} deg")
    print_stat_summary(
        "All Designs (Computed Metrics)",
        overall_stats,
        [
            ("static_margin", "Static Margin", "{:.4f}"),
            ("htail_volume", "Htail Volume", "{:.3f}"),
            ("vtail_volume", "Vtail Volume", "{:.3f}"),
            ("tail_arm", "Tail Arm", "{:.3f} m"),
            ("htail_ar", "Htail AR", "{:.3f}"),
            ("mass_kg", "Mass", "{:.3f} kg"),
        ],
    )
    print_failure_summary(failure_counts)
    if PLOT_BEST_DESIGN:
        plot_best_design(best)


if __name__ == "__main__":
    main()
