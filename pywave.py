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
BatteryEnergyJ = 7920000
MaxThrustN = 152.0


### Aircraft CG Definitions x = 0 at the nose 
Xwqc = 0.45 # M
Xhtqc = 2 # M
Xcg = 0.2 # M

### Wing Definition
RootChord = 0.3
MidChord = 0.3
MidChordPos = 0.4 # fraction of half span from root to tip (0-1)
TipChord = 0.2
RootThickness = 0.1
TipThickness = 0.1
Span = 5 #M, wing span (both sides)
RootSweepDeg = 0.0
MidSweepDeg = 0
MidSweepPos = 0.4 # fraction of half span from root to tip (0-1)
TipSweepDeg = 3
wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
WingIncidence = 1
WingClMaxTakeoff = 1.2
WingClMaxCruise = 1.2
WingClMaxLanding = 1.6

### Horizontal and Vertical Tail Definition
HRootChord = 0.386 #M
HMidChord = 0.2286
HMidChordPos = 0.8 # fraction of half span from root to tip (0-1)
HTipChord = 0.1 #M
HRootThickness = 0.1
HTipThickness = 0.1
HSpan = 2 #M, total span (both sides)
HRootSweepDeg = 0.0
HMidSweepDeg = 0.0
HMidSweepPos = 0.4 # fraction of half span from root to tip (0-1)
HTipSweepDeg = 0.0

VRootChord = 0.1 #M
VMidChord = 0.1
VMidChordPos = 0.4 # fraction of span from root to tip (0-1)
VTipChord = 0.1 #M
VRootThickness = 0.1
VTipThickness = 0.1
VSpan = 1 #M, span per tail
VRootSweepDeg = 0.0
VMidSweepDeg = 0.0
VMidSweepPos = 0.4 # fraction of span from root to tip (0-1)
VTipSweepDeg = 0.0
Hfoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
Vfoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
VtailCount = 2
HtailIncidence = 3 # deg, standard convention: incidence > 0 -> tail angled with TE down
VtailIncidence = 0 # deg

WingTaper = TipChord / RootChord
WingMAC = 2.0 / 3.0 * RootChord * (1 + WingTaper + WingTaper**2) / (1 + WingTaper)
WingArea = 0.5 * (RootChord + TipChord) * Span
HtailArea = 0.5 * (HRootChord + HTipChord) * HSpan
VtailArea = 0.5 * (VRootChord + VTipChord) * VSpan * VtailCount
TailArm = Xhtqc - Xwqc
HtailVolumeCoeff = (HtailArea * TailArm) / (WingArea * WingMAC)
VtailVolumeCoeff = (VtailArea * TailArm) / (WingArea * Span)

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

model = AircraftModel(config)
profile = MissionProfile(
    segments=[
        MissionSegment(kind="climb_to", target_alt=20, mode = "min_time"),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0),
        MissionSegment(kind="loiter", bank_deg=10.0, mode="max_endurance", speed_max=80.0),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0),
        MissionSegment(kind="descent_to", target_alt=10, speed=14.0, elevator_deg=-20.0, thrust_scale=0.05),
        MissionSegment(kind="loiter", speed=14.0, time=180.0, bank_deg=20.0),
        MissionSegment(kind="landing", speed=20.0),
    ],
    log_interval = 0.1,
    takeoff_aero_stride=5,
    n_span=300,
    power_derate_frac=0.0,
    energy_reserve_frac=0.2,
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
            ["Cm Total", f"{loiter_report.get('CmTotal', 0.0):.6f}"],
            ["Cm Wing CG", f"{loiter_report.get('CmWingCG', 0.0):.6f}"],
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
