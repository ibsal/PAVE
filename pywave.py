import math
from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet
from aircraft_core import build_aircraft_config, run_analysis
from mission_model import AircraftModel, MissionProfile, MissionSegment, simulate_mission


### Environment Configuration
GroundLevel = 0 # M, ASL
TailEfficiency = 0.95 # q_tail / q_wing (0.9-1.0 typical). Scales tail Cl and Cd
ElevatorEffectivenessTau = 0.45 # deg tail alpha per deg elevator (0.3-0.6 typical)
Weight = 205 #N = 45 lbf
SurfaceRoughness = 0.00635e-3 # m, equivalent sand grain roughness (0 for smooth)
FuselageLaminarFrac = 0.3
BoomLaminarFrac = 0.04
MaxPowerW = 4000.0
PropEff = 0.59
BatteryEnergyJ = 7992000
mission_systems_power = 50.24
MaxThrustN = 152.0
CruiseWindMps = 8.6 # +X tailwind, -X headwind
OrbitWindMps =  8.6 # +X tailwind, -X headwind
GustVerticalMps = 0.0 # +upward (m/s), vertical gust estimate
MaxLoadFactor = 3.5 # structural limit for gust survivability
GustSearchMaxMps = 30.0 # search cap for worst-case gust


### Aircraft CG Definitions x = 0 at the nose 
Xwqc = 0.45 # M
Xhtqc = 2 # M
Xcg = 0.2 # M
WingZ = 0.0 # m, wing reference plane above CG
HtailZ = 0.0 # m, htail reference plane above CG
VtailZ = 0.2 # m, vtail reference plane above CG
EngineZ = 0.0 # m, thrust line above CG

### Wing Definition
RootChord = 0.3
MidChord = 0.3
MidChordPos = 0.4 # fraction of half span from root to tip (0-1)
TipChord = 0.3
RootThickness = 0.1
TipThickness = 0.1
Span = 5 #M, wing span (both sides)
RootSweepDeg = 0.0
MidSweepDeg = 0
MidSweepPos = 0.4 # fraction of half span from root to tip (0-1)
TipSweepDeg = 0
wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
WingIncidence = 1

### Horizontal and Vertical Tail Definition
HRootChord = 0.386 #M
HMidChord = 0.2286
HMidChordPos = 0.8 # fraction of half span from root to tip (0-1)
HTipChord = 0.1 #M
HRootThickness = 0.075
HTipThickness = 0.075
HSpan = 1 #M, total span (both sides)
HRootSweepDeg = 0.0
HMidSweepDeg = 0.0
HMidSweepPos = 0.4 # fraction of half span from root to tip (0-1)
HTipSweepDeg = 0.0

VRootChord = 0.1 #M
VMidChord = 0.1
VMidChordPos = 0.4 # fraction of span from root to tip (0-1)
VTipChord = 0.1 #M
VRootThickness = 0.075
VTipThickness = 0.075
VSpan = 1 #M, span per tail
VRootSweepDeg = 0.0
VMidSweepDeg = 0.0
VMidSweepPos = 0.4 # fraction of span from root to tip (0-1)
VTipSweepDeg = 0.0
Hfoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
Vfoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
VtailCount = 2
HtailIncidence = -3 # deg, standard convention: incidence > 0 -> tail angled with TE down
VtailIncidence = 0 # deg

WingTaper = TipChord / RootChord
WingMAC = 2.0 / 3.0 * RootChord * (1 + WingTaper + WingTaper**2) / (1 + WingTaper)
WingArea = 0.5 * (RootChord + TipChord) * Span
HtailArea = 0.5 * (HRootChord + HTipChord) * HSpan
VtailArea = 0.5 * (VRootChord + VTipChord) * VSpan * VtailCount
TailArm = Xhtqc - Xwqc
HtailVolumeCoeff = (HtailArea * TailArm) / (WingArea * WingMAC)
VtailVolumeCoeff = (VtailArea * TailArm) / (WingArea * Span)

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

_atm = Atmosphere(GroundLevel)
WingClMax = estimate_wing_clmax(
    wingFoil,
    float(_atm.density[0]),
    float(_atm.dynamic_viscosity[0]),
    Weight,
    WingArea,
    WingMAC,
)
WingClMaxTakeoff = WingClMax
WingClMaxCruise = WingClMax
WingClMaxLanding = WingClMax

### Boom Definition 
BoomLength = Xhtqc - Xwqc
BoomDiameter = 0.03
BoomCount = 2

### Fuselage Definition 
FuselageWidth = 0.3 # M
FuselageHeight = 0.15 # M
FuselageLength = 0.7 # M
Pfactor = 0.9 # Account for fillets (super basic)



config = build_aircraft_config(
    ground_level=GroundLevel,
    orbit_level=0.0,
    flight_velocity=0.0,
    weight=Weight,
    tail_efficiency=TailEfficiency,
    elevator_tau=ElevatorEffectivenessTau,
    downwash_factor=0.45,
    cd_misc=0.005,
    x_wqc=Xwqc,
    x_htqc=Xhtqc,
    x_cg=Xcg,
    wing_z=WingZ,
    htail_z=HtailZ,
    vtail_z=VtailZ,
    engine_z=EngineZ,
    wing_foil=wingFoil,
    wing_root_chord=RootChord,
    wing_mid_chord=MidChord,
    wing_mid_chord_pos=MidChordPos,
    wing_tip_chord=TipChord,
    wing_span=Span,
    wing_incidence=WingIncidence,
    wing_root_sweep=RootSweepDeg,
    wing_mid_sweep=MidSweepDeg,
    wing_mid_sweep_pos=MidSweepPos,
    wing_tip_sweep=TipSweepDeg,
    wing_cl_max_takeoff=WingClMaxTakeoff,
    wing_cl_max_cruise=WingClMaxCruise,
    wing_cl_max_landing=WingClMaxLanding,
    htail_foil=Hfoil,
    htail_root_chord=HRootChord,
    htail_mid_chord=HMidChord,
    htail_mid_chord_pos=HMidChordPos,
    htail_tip_chord=HTipChord,
    htail_span=HSpan,
    htail_incidence=HtailIncidence,
    htail_root_sweep=HRootSweepDeg,
    htail_mid_sweep=HMidSweepDeg,
    htail_mid_sweep_pos=HMidSweepPos,
    htail_tip_sweep=HTipSweepDeg,
    vtail_foil=Vfoil,
    vtail_root_chord=VRootChord,
    vtail_mid_chord=VMidChord,
    vtail_mid_chord_pos=VMidChordPos,
    vtail_tip_chord=VTipChord,
    vtail_span=VSpan,
    vtail_incidence=VtailIncidence,
    vtail_root_sweep=VRootSweepDeg,
    vtail_mid_sweep=VMidSweepDeg,
    vtail_mid_sweep_pos=VMidSweepPos,
    vtail_tip_sweep=VTipSweepDeg,
    vtail_count=VtailCount,
    fuselage_width=FuselageWidth,
    fuselage_height=FuselageHeight,
    fuselage_length=FuselageLength,
    fuselage_pfactor=Pfactor,
    boom_length=BoomLength,
    boom_diameter=BoomDiameter,
    boom_count=BoomCount,
    roughness_k=SurfaceRoughness,
    fuselage_laminar_frac=FuselageLaminarFrac,
    boom_laminar_frac=BoomLaminarFrac,
    re_crit_per_m=5e5,
    max_power_w=MaxPowerW,
    prop_eff=PropEff,
    battery_energy_j=BatteryEnergyJ,
    max_thrust_n=MaxThrustN,
)
config["propulsion"]["mission_systems_power"] = mission_systems_power

model = AircraftModel(config)
profile = MissionProfile(
    segments=[
        MissionSegment(kind="climb_to", target_alt=20, mode="min_time"),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0, wind_mps=CruiseWindMps),
        MissionSegment(kind="loiter", bank_deg=10.0, mode="max_endurance", speed_max=80.0, wind_mps=OrbitWindMps),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0, wind_mps=CruiseWindMps),
        MissionSegment(kind="descent_to", target_alt=10, speed=14.0, elevator_deg=-20.0, thrust_scale=0.05),
        MissionSegment(kind="loiter", speed=14.0, time=180.0, bank_deg=20.0, wind_mps=OrbitWindMps),
        MissionSegment(kind="landing", speed=20.0),
    ],
    log_interval = 0.1,
    takeoff_aero_stride=5,
    n_span=300,
    power_derate_frac=0.0,
    energy_reserve_frac=0.2,
    wind_mps=CruiseWindMps,
    loiter_wind_mode="head_tail_avg",
)
mission_history, segment_summaries = simulate_mission(model, profile, initial_alt=0.0, return_summary=True)
last_state = mission_history[-1]
energy_init = config["propulsion"]["battery_energy_j"]
energy_pct = 100.0 * last_state.energy_j / energy_init if energy_init > 0.0 else 0.0

def print_table(title, headers, rows, align=None):
    if not rows:
        return
    if align is None:
        align = ["<"] * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    print("")
    print(title)
    header = " ".join(f"{h:{align[i]}{widths[i]}}" for i, h in enumerate(headers))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(f"{cell:{align[i]}{widths[i]}}" for i, cell in enumerate(row)))


def stability_derivatives(config, aoa_deg, elev_deg, d_aoa=0.25, d_elev=1.0):
    # Central differences for longitudinal derivatives at the specified trim condition.
    plus = run_analysis(config, aoa_deg + d_aoa, elev_deg, build_report=False)
    minus = run_analysis(config, aoa_deg - d_aoa, elev_deg, build_report=False)
    dcm_dalpha = (plus["CmTotal"] - minus["CmTotal"]) / (2.0 * d_aoa)
    dcl_dalpha = (plus["ClTotal"] - minus["ClTotal"]) / (2.0 * d_aoa)

    plus_e = run_analysis(config, aoa_deg, elev_deg + d_elev, build_report=False)
    minus_e = run_analysis(config, aoa_deg, elev_deg - d_elev, build_report=False)
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
        "dcm_dcl": dcm_dcl,
        "x_np": x_np,
        "static_margin": static_margin,
    }


def foil_alpha_limits(foil):
    if hasattr(foil, "polars"):
        mins = [float(min(p.alpha_deg)) for p in foil.polars]
        maxs = [float(max(p.alpha_deg)) for p in foil.polars]
        return min(mins), max(maxs)
    if hasattr(foil, "alpha_deg"):
        return float(min(foil.alpha_deg)), float(max(foil.alpha_deg))
    return None, None


def lateral_stability_metrics(config, v, altitude_m):
    wing = config["wing"]
    vtail = config["vtail"]
    atm = Atmosphere(altitude_m)
    rho = float(atm.density)
    mu = float(atm.dynamic_viscosity)
    re_v = rho * v * vtail["mac"] / mu if mu > 0.0 else 0.0
    a0_deg = 0.0
    try:
        _, a0_deg = vtail["foil"].cdo_and_lift_slope(reynolds=re_v)
    except Exception:
        da = 2.0
        cl_p = vtail["foil"].cl(alpha_deg=da, reynolds=re_v)
        cl_m = vtail["foil"].cl(alpha_deg=-da, reynolds=re_v)
        a0_deg = (cl_p - cl_m) / (2.0 * da)
    a0_rad = a0_deg * (180.0 / math.pi)
    e_oswald_v = 1.78 * (1.0 - 0.045 * (vtail["ar"] ** 0.68)) - 0.64
    e_oswald_v = min(max(e_oswald_v, 0.3), 0.95)
    if vtail["ar"] > 0.0:
        a_v = a0_rad / (1.0 + a0_rad / (math.pi * vtail["ar"] * e_oswald_v))
    else:
        a_v = 0.0
    tail_arm = config["positions"]["x_htqc"] - config["positions"]["x_cg"]
    vtail_vol = (vtail["area"] * tail_arm) / (wing["area"] * wing["span"]) if wing["area"] > 0.0 and wing["span"] > 0.0 else 0.0
    cn_beta_rad = a_v * vtail_vol
    cn_beta_deg = cn_beta_rad / (180.0 / math.pi)
    cy_beta_rad = -a_v * (vtail["area"] / wing["area"]) if wing["area"] > 0.0 else 0.0
    z_vtail = config["positions"].get("z_vtail", 0.0)
    cl_beta_rad = cy_beta_rad * (z_vtail / wing["span"]) if wing["span"] > 0.0 else 0.0
    cl_beta_deg = cl_beta_rad / (180.0 / math.pi)
    return {
        "re_v": re_v,
        "a0_deg": a0_deg,
        "a_v_rad": a_v,
        "cn_beta_rad": cn_beta_rad,
        "cn_beta_deg": cn_beta_deg,
        "cy_beta_rad": cy_beta_rad,
        "cl_beta_rad": cl_beta_rad,
        "cl_beta_deg": cl_beta_deg,
    }


def gust_recovery_metrics(config, base_aoa, base_elev, gust_mps, v, stability, elevator_limit_deg, max_load_factor=None):
    if abs(gust_mps) < 1e-6 or v <= 0.1:
        return None
    gust_alpha = math.degrees(math.atan2(gust_mps, v))
    gust_aoa = base_aoa + gust_alpha
    gust_report = run_analysis(config, gust_aoa, base_elev, build_report=False)
    dcm_de = stability.get("cm_delta_e_deg", 0.0)
    delta_e = None
    elev_req = None
    if abs(dcm_de) > 1e-9:
        delta_e = -gust_report["CmTotal"] / dcm_de
        elev_req = base_elev + delta_e
    wing = config["wing"]
    alpha_min, alpha_max = foil_alpha_limits(wing["foil"])
    if alpha_min is not None:
        alpha_min -= wing["incidence"]
    if alpha_max is not None:
        alpha_max -= wing["incidence"]
    aoa_ok = True
    if alpha_min is not None and gust_aoa < alpha_min:
        aoa_ok = False
    if alpha_max is not None and gust_aoa > alpha_max:
        aoa_ok = False
    cl_wing = gust_report.get("ClWing", 0.0)
    cl_max = wing.get("cl_max_cruise")
    cl_ok = True
    if cl_max is not None and cl_max > 0.0 and cl_wing > cl_max:
        cl_ok = False
    elev_ok = True
    if elev_req is not None and elevator_limit_deg is not None:
        elev_ok = abs(elev_req) <= elevator_limit_deg
    load_factor = gust_report["Lift"] / config["weight"] if config["weight"] > 0.0 else 0.0
    n_ok = True
    if max_load_factor is not None and max_load_factor > 0.0:
        n_ok = load_factor <= max_load_factor
    recoverable = aoa_ok and cl_ok and elev_ok and n_ok
    return {
        "gust_alpha_deg": gust_alpha,
        "gust_aoa_deg": gust_aoa,
        "cl_wing": cl_wing,
        "cm_total": gust_report.get("CmTotal", 0.0),
        "delta_e_deg": delta_e,
        "elev_req_deg": elev_req,
        "aoa_ok": aoa_ok,
        "cl_ok": cl_ok,
        "elev_ok": elev_ok,
        "n_ok": n_ok,
        "recoverable": recoverable,
        "load_factor": load_factor,
    }


def max_recoverable_gust(config, base_aoa, base_elev, v, stability, elevator_limit_deg, max_load_factor, gust_cap_mps):
    if v <= 0.1 or gust_cap_mps <= 0.0:
        return None
    low = 0.0
    high = gust_cap_mps
    test_low = gust_recovery_metrics(
        config,
        base_aoa,
        base_elev,
        low,
        v,
        stability,
        elevator_limit_deg,
        max_load_factor=max_load_factor,
    )
    if test_low is None or not test_low["recoverable"]:
        return {"max_gust_mps": 0.0, "bounded": True, "criteria": test_low}
    test_high = gust_recovery_metrics(
        config,
        base_aoa,
        base_elev,
        high,
        v,
        stability,
        elevator_limit_deg,
        max_load_factor=max_load_factor,
    )
    if test_high is not None and test_high["recoverable"]:
        return {"max_gust_mps": high, "bounded": False, "criteria": test_high}
    for _ in range(25):
        mid = 0.5 * (low + high)
        test_mid = gust_recovery_metrics(
            config,
            base_aoa,
            base_elev,
            mid,
            v,
            stability,
            elevator_limit_deg,
            max_load_factor=max_load_factor,
        )
        if test_mid is None:
            high = mid
            continue
        if test_mid["recoverable"]:
            low = mid
        else:
            high = mid
        if (high - low) < 0.05:
            break
    return {"max_gust_mps": low, "bounded": True, "criteria": None}

print_table(
    "Mission Summary",
    ["Metric", "Value"],
    [
        ["Total Time", f"{last_state.t:.1f} s"],
        ["Total Distance", f"{last_state.x:.1f} m"],
        ["Final Altitude", f"{last_state.h:.1f} m"],
        ["Final Speed", f"{last_state.v:.2f} m/s"],
        ["Final Phase", f"{last_state.phase}"],
        ["Energy Left", f"{energy_pct:.1f}% ({last_state.energy_j:.0f} J)"],
    ],
    align=["<", ">"],
)

stages = []
phase = None
start_state = None
energy_start = None
time_sum = 0.0
dist_sum = 0.0
v_sum = 0.0
h_sum = 0.0
last_phase = None
for idx in range(1, len(mission_history)):
    prev = mission_history[idx - 1]
    cur = mission_history[idx]
    dt = cur.t - prev.t
    if dt <= 0.0:
        continue
    phase_label = prev.phase
    if phase_label == "segment_done":
        phase_label = last_phase if last_phase is not None else prev.phase
    if phase is None:
        phase = phase_label
        start_state = prev
        energy_start = prev.energy_j
        time_sum = 0.0
        dist_sum = 0.0
        v_sum = 0.0
        h_sum = 0.0
    if phase_label != phase:
        energy_used = (energy_start - prev.energy_j) if energy_start is not None else 0.0
        energy_used_pct = 100.0 * energy_used / energy_init if energy_init > 0.0 else 0.0
        avg_v = v_sum / time_sum if time_sum > 0.0 else 0.0
        stages.append({
            "phase": phase,
            "time_s": time_sum,
            "dist_m": dist_sum,
            "dh_m": prev.h - start_state.h if start_state is not None else 0.0,
            "v_avg": avg_v,
            "energy_pct": energy_used_pct,
        })
        phase = phase_label
        start_state = prev
        energy_start = prev.energy_j
        time_sum = 0.0
        dist_sum = 0.0
        v_sum = 0.0
        h_sum = 0.0
    time_sum += dt
    dist_sum += cur.x - prev.x
    v_sum += prev.v * dt
    h_sum += prev.h * dt
    last_phase = phase_label
if phase is not None and time_sum > 0.0:
    energy_used = (energy_start - last_state.energy_j) if energy_start is not None else 0.0
    energy_used_pct = 100.0 * energy_used / energy_init if energy_init > 0.0 else 0.0
    avg_v = v_sum / time_sum if time_sum > 0.0 else 0.0
    stages.append({
        "phase": phase,
        "time_s": time_sum,
        "dist_m": dist_sum,
        "dh_m": last_state.h - start_state.h if start_state is not None else 0.0,
        "v_avg": avg_v,
        "energy_pct": energy_used_pct,
    })

print_table(
    "Flight Stage Summary",
    ["Stage", "Time (s)", "Dist (m)", "dH (m)", "Vavg (m/s)", "Energy Used (%)"],
    [
        [
            row["phase"],
            f"{row['time_s']:.1f}",
            f"{row['dist_m']:.1f}",
            f"{row['dh_m']:.1f}",
            f"{row['v_avg']:.2f}",
            f"{row['energy_pct']:.1f}",
        ]
        for row in stages
    ],
    align=["<", ">", ">", ">", ">", ">"],
)

segment_rows = []
for row in segment_summaries or []:
    energy_used_pct = 100.0 * row.get("energy_used_j", 0.0) / energy_init if energy_init > 0.0 else 0.0
    cl_avg = row.get("cl_avg")
    cd_avg = row.get("cd_avg")
    ld_avg = row.get("ld_avg")
    segment_rows.append([
        row.get("segment", ""),
        row.get("kind", ""),
        f"{row.get('time_s', 0.0):.1f}",
        f"{row.get('distance_m', 0.0):.1f}",
        f"{energy_used_pct:.1f}",
        f"{cl_avg:.3f}" if cl_avg is not None else "",
        f"{cd_avg:.4f}" if cd_avg is not None else "",
        f"{ld_avg:.2f}" if ld_avg is not None else "",
    ])

print_table(
    "Mission Segment Summary",
    ["Segment", "Kind", "Time (s)", "Dist (m)", "Energy (%)", "CLavg", "CDavg", "L/D"],
    segment_rows,
    align=["<", "<", ">", ">", ">", ">", ">", ">"],
)

first_loiter_idx = next((i for i, s in enumerate(mission_history) if s.phase == "loiter"), None)
first_loiter_row = next((row for row in segment_summaries if row.get("kind") == "loiter"), None) if segment_summaries else None
if first_loiter_idx is None or first_loiter_row is None:
    print("")
    print("Engineering Report (First Loiter)")
    print("No loiter segment found for cruise report.")
else:
    end_idx = first_loiter_idx
    while end_idx + 1 < len(mission_history) and mission_history[end_idx + 1].phase == "loiter":
        end_idx += 1
    total_dt = 0.0
    v_sum = 0.0
    h_sum = 0.0
    for idx in range(first_loiter_idx, end_idx):
        dt = mission_history[idx + 1].t - mission_history[idx].t
        if dt <= 0.0:
            continue
        total_dt += dt
        v_sum += mission_history[idx].v * dt
        h_sum += mission_history[idx].h * dt
    if total_dt > 0.0:
        v_avg = v_sum / total_dt
        h_avg = h_sum / total_dt
    else:
        v_avg = mission_history[first_loiter_idx].v
        h_avg = mission_history[first_loiter_idx].h
    v_takeoff = None
    for idx in range(1, len(mission_history)):
        if mission_history[idx - 1].phase == "takeoff" and mission_history[idx].phase != "takeoff":
            v_takeoff = mission_history[idx].v
            break

    v_landing = None
    for idx in range(len(mission_history)):
        if mission_history[idx].phase == "landing":
            v_landing = mission_history[idx].v
            break

    config["analysis_altitude_m"] = GroundLevel + h_avg
    config["flight_velocity"] = v_avg
    config["stall_margins"] = {
        "takeoff": profile.takeoff_stall_margin,
        "cruise": profile.level_flight_stall_margin,
        "landing": profile.landing_stall_margin,
    }
    config["stall_speeds"] = {
        "takeoff": v_takeoff,
        "cruise": v_avg,
        "landing": v_landing,
    }
    loiter_aoa = first_loiter_row.get("aoa_deg", 0.0)
    loiter_elev = first_loiter_row.get("elev_deg", 0.0)
    loiter_report = run_analysis(config, loiter_aoa, loiter_elev, build_report=True)
    wing_area = config["wing"]["area"]
    wing_loading = Weight / wing_area if wing_area > 0.0 else 0.0
    print_table(
        "Vehicle Configuration",
        ["Item", "Value"],
        [
            ["Weight", f"{Weight:.1f} N"],
            ["Wing Area", f"{wing_area:.3f} m^2"],
            ["Wing Span", f"{config['wing']['span']:.3f} m"],
            ["Aspect Ratio", f"{config['wing']['ar']:.2f}"],
            ["MAC", f"{config['wing']['mac']:.3f} m"],
            ["Wing Loading", f"{wing_loading:.1f} N/m^2"],
            ["Tail Arm", f"{TailArm:.3f} m"],
            ["Wing Z", f"{WingZ:.3f} m"],
            ["H Tail Z", f"{HtailZ:.3f} m"],
            ["V Tail Z", f"{VtailZ:.3f} m"],
            ["Engine Z", f"{EngineZ:.3f} m"],
            ["H Tail Area", f"{config['htail']['area']:.3f} m^2"],
            ["V Tail Area", f"{config['vtail']['area']:.3f} m^2"],
            ["H Tail Vol", f"{HtailVolumeCoeff:.3f}"],
            ["V Tail Vol", f"{VtailVolumeCoeff:.3f}"],
            ["Max Power", f"{MaxPowerW:.0f} W"],
            ["Prop Eff", f"{PropEff:.2f}"],
            ["Battery", f"{BatteryEnergyJ / 3.6e6:.2f} kWh"],
            ["Max Thrust", f"{MaxThrustN:.1f} N"],
        ],
        align=["<", ">"],
    )
    cl_total = loiter_report.get("ClTotal", 0.0)
    cd_total = loiter_report.get("CdTotal", 0.0)
    ld_total = (cl_total / cd_total) if cd_total > 0.0 else 0.0
    drag_cruise = loiter_report.get("Drag", 0.0)
    prop_eff = config["propulsion"].get("prop_eff", 0.0)
    propulsive_power = drag_cruise * v_avg
    motor_power = propulsive_power / prop_eff if prop_eff > 1e-6 else 0.0
    print_table(
        "Aero Coefficients (Cruise)",
        ["Coeff", "Value"],
        [
            ["AOA", f"{loiter_report.get('AOA', 0.0):.3f} deg"],
            ["HTail AOA", f"{loiter_report.get('HTailAOA', 0.0):.3f} deg"],
            ["CL Wing", f"{loiter_report.get('ClWing', 0.0):.5f}"],
            ["CL Tail", f"{loiter_report.get('ClTail', 0.0):.5f}"],
            ["CL Total", f"{cl_total:.5f}"],
            ["CD0", f"{loiter_report.get('Cd0', 0.0):.5f}"],
            ["CD Induced", f"{loiter_report.get('CdInduced', 0.0):.5f}"],
            ["CD Total", f"{cd_total:.5f}"],
            ["L/D", f"{ld_total:.2f}"],
            ["Propulsive Power", f"{propulsive_power:.1f} W"],
            ["Motor Power", f"{motor_power:.1f} W"],
            ["Cm Total", f"{loiter_report.get('CmTotal', 0.0):.6f}"],
            ["Cm Wing CG", f"{loiter_report.get('CmWingCG', 0.0):.6f}"],
        ],
        align=["<", ">"],
    )
    moments = loiter_report.get("moments", {})
    if moments:
        total_moment = moments.get("total", 0.0)
        def moment_pct(value):
            return f"{(100.0 * value / total_moment):.1f}%" if abs(total_moment) > 1e-9 else "n/a"
        print_table(
            "Pitching Moments (Cruise)",
            ["Contributor", "Moment (N*m)", "% Total"],
            [
                ["Wing Lift", f"{moments.get('wing_lift', 0.0):.3f}", moment_pct(moments.get("wing_lift", 0.0))],
                ["Wing Cm", f"{moments.get('wing_cm', 0.0):.3f}", moment_pct(moments.get("wing_cm", 0.0))],
                ["Tail Lift", f"{moments.get('tail_lift', 0.0):.3f}", moment_pct(moments.get("tail_lift", 0.0))],
                ["Tail Cm", f"{moments.get('tail_cm', 0.0):.3f}", moment_pct(moments.get("tail_cm", 0.0))],
                ["Tail Total", f"{moments.get('tail_total', 0.0):.3f}", moment_pct(moments.get("tail_total", 0.0))],
                ["Tail Total @ 0 Elev", f"{moments.get('tail_total_zero', 0.0):.3f}", moment_pct(moments.get("tail_total_zero", 0.0))],
                ["Elevator Delta", f"{moments.get('elevator_delta', 0.0):.3f}", moment_pct(moments.get("elevator_delta", 0.0))],
                ["Total @ 0 Elev", f"{moments.get('total_zero', 0.0):.3f}", moment_pct(moments.get("total_zero", 0.0))],
                ["Total", f"{total_moment:.3f}", "100.0%" if abs(total_moment) > 1e-9 else "n/a"],
            ],
            align=["<", ">", ">"],
        )
    stability = stability_derivatives(config, loiter_aoa, loiter_elev)
    cm_alpha_rad = stability["cm_alpha_deg"] * (180.0 / math.pi)
    cl_alpha_rad = stability["cl_alpha_deg"] * (180.0 / math.pi)
    mac = config["wing"]["mac"]
    x_np = stability["x_np"]
    x_np_mac = (x_np / mac) if mac > 0.0 else 0.0
    static_margin = stability["static_margin"]
    print_table(
        "Stability Metrics (Cruise)",
        ["Metric", "Value"],
        [
            ["Cmalpha (1/deg)", f"{stability['cm_alpha_deg']:.5f}"],
            ["Cmalpha (1/rad)", f"{cm_alpha_rad:.5f}"],
            ["CLalpha (1/deg)", f"{stability['cl_alpha_deg']:.5f}"],
            ["CLalpha (1/rad)", f"{cl_alpha_rad:.5f}"],
            ["dCm/dCL", f"{stability['dcm_dcl']:.5f}"],
            ["Neutral Point", f"{x_np:.3f} m ({x_np_mac:.2f} MAC)"],
            ["Static Margin", f"{static_margin:.4f} ({static_margin * 100.0:.1f}% MAC)"],
            ["Cm_delta_e (1/deg)", f"{stability['cm_delta_e_deg']:.5f}"],
            ["CL_delta_e (1/deg)", f"{stability['cl_delta_e_deg']:.5f}"],
        ],
        align=["<", ">"],
    )
    lateral = lateral_stability_metrics(config, v_avg, config["analysis_altitude_m"])
    print_table(
        "Lateral Stability (Cruise)",
        ["Metric", "Value"],
        [
            ["Vtail Re", f"{lateral['re_v']:.0f}"],
            ["a_v (3D, 1/rad)", f"{lateral['a_v_rad']:.3f}"],
            ["Cn_beta (1/rad)", f"{lateral['cn_beta_rad']:.5f}"],
            ["Cn_beta (1/deg)", f"{lateral['cn_beta_deg']:.6f}"],
            ["Cy_beta (1/rad)", f"{lateral['cy_beta_rad']:.5f}"],
            ["Cl_beta_v (1/rad)", f"{lateral['cl_beta_rad']:.6f}"],
        ],
        align=["<", ">"],
    )
    wind_rows = [["Loiter Wind Mode", f"{profile.loiter_wind_mode}"]]
    for label, wind_mps in (("Cruise", CruiseWindMps), ("Loiter", OrbitWindMps)):
        wind_mag = abs(wind_mps)
        gs_head = v_avg - wind_mag
        gs_tail = v_avg + wind_mag
        t_head = (1000.0 / gs_head) if gs_head > 1e-6 else None
        t_tail = (1000.0 / gs_tail) if gs_tail > 1e-6 else None
        gs_head_str = f"{gs_head:.2f}" if gs_head > 1e-6 else "n/a"
        gs_tail_str = f"{gs_tail:.2f}" if gs_tail > 1e-6 else "n/a"
        t_head_str = f"{t_head:.1f}" if t_head is not None else "n/a"
        t_tail_str = f"{t_tail:.1f}" if t_tail is not None else "n/a"
        wind_rows.append([f"{label} Wind Input", f"{wind_mps:+.2f} m/s"])
        wind_rows.append([f"{label} GS Head/Tail", f"{gs_head_str} / {gs_tail_str} m/s"])
        wind_rows.append([f"{label} Time/1km", f"{t_head_str} / {t_tail_str} s"])
    print_table(
        "Wind Performance (Cruise/Loiter)",
        ["Metric", "Value"],
        wind_rows,
        align=["<", ">"],
    )
    gust = gust_recovery_metrics(
        config,
        loiter_aoa,
        loiter_elev,
        GustVerticalMps,
        v_avg,
        stability,
        profile.elevator_limit_deg,
        max_load_factor=MaxLoadFactor,
    )
    if gust is not None:
        print_table(
            "Gust Response (Cruise)",
            ["Metric", "Value"],
            [
                ["Gust Vertical", f"{GustVerticalMps:+.2f} m/s"],
                ["Delta AOA", f"{gust['gust_alpha_deg']:+.3f} deg"],
                ["Gust AOA", f"{gust['gust_aoa_deg']:.3f} deg"],
                ["CL Wing (gust)", f"{gust['cl_wing']:.4f}"],
                ["Load Factor", f"{gust['load_factor']:.3f}"],
                ["Cm Total (gust)", f"{gust['cm_total']:.6f}"],
                ["Delta Elevator", f"{gust['delta_e_deg']:.3f} deg" if gust["delta_e_deg"] is not None else "n/a"],
                ["Elevator Req", f"{gust['elev_req_deg']:.3f} deg" if gust["elev_req_deg"] is not None else "n/a"],
                ["AOA OK", "yes" if gust["aoa_ok"] else "no"],
                ["CL OK", "yes" if gust["cl_ok"] else "no"],
                ["Elev OK", "yes" if gust["elev_ok"] else "no"],
                ["Load OK", "yes" if gust["n_ok"] else "no"],
                ["Recoverable", "yes" if gust["recoverable"] else "no"],
            ],
            align=["<", ">"],
        )
    survival = max_recoverable_gust(
        config,
        loiter_aoa,
        loiter_elev,
        v_avg,
        stability,
        profile.elevator_limit_deg,
        MaxLoadFactor,
        GustSearchMaxMps,
    )
    if survival is not None:
        bound_note = "<= cap" if survival["bounded"] else ">= cap"
        print_table(
            "Wind Survival (Cruise)",
            ["Metric", "Value"],
            [
                ["Search Cap", f"{GustSearchMaxMps:.2f} m/s"],
                ["Max Recoverable Gust", f"{survival['max_gust_mps']:.2f} m/s ({bound_note})"],
            ],
            align=["<", ">"],
        )
    drag_stack = loiter_report.get("drag_stack") or []
    total_cdo = sum(row.get("cdo", 0.0) for row in drag_stack)
    total_cd = loiter_report.get("CdTotal", 0.0)
    print_table(
        "Drag Stackup (Cruise)",
        ["Component", "FF", "Q", "Cf", "Sref/S", "CD0", "%CD0", "%CDtot"],
        [
            [
                row["component"],
                f"{row['ff']:.3f}",
                f"{row['q']:.3f}",
                f"{row['cfc']:.5f}",
                f"{row['sratio']:.4f}",
                f"{row['cdo']:.6f}",
                f"{(100.0 * row['cdo'] / total_cdo) if total_cdo > 0.0 else 0.0:.1f}",
                f"{(100.0 * row['cdo'] / total_cd) if total_cd > 0.0 else 0.0:.1f}",
            ]
            for row in drag_stack
        ],
        align=["<", ">", ">", ">", ">", ">", ">", ">"],
    )
