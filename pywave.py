from PyFoil.airfoil_polars import PolarSet
from aircraft_core import build_aircraft_config, run_analysis, solve_trim
from mission_model import AircraftModel, MissionProfile, MissionSegment, simulate_mission


### Environment Configuration
GroundLevel = 0 # M, ASL
OrbitLevel = 300 #M, AGL
flightVelocity = 30 #SUPER PLACEHOLDER m/s
flightAOA = 2 # SUPER PLACEHOLDER degrees
TailEfficiency = 0.95 # q_tail / q_wing (0.9-1.0 typical). Scales tail Cl and Cd
ElevatorDeflection = 0 # deg, standard convention: positive = TE down
ElevatorEffectivenessTau = 0.45 # deg tail alpha per deg elevator (0.3-0.6 typical)
ElevatorLimitDeg = 30.0
Weight = 200 #N = 45 lbf
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
FuselageLength = 1.4 # M
Pfactor = 0.9 # Account for fillets (super basic)



config = build_aircraft_config(
    ground_level=GroundLevel,
    orbit_level=OrbitLevel,
    flight_velocity=flightVelocity,
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

trim_aoa, trim_elev, converged, _ = solve_trim(config, flightAOA, ElevatorDeflection, elevator_limit_deg=ElevatorLimitDeg)
if not converged:
    print("Warning: trim solver did not converge; reporting last iteration.")
trim_result = run_analysis(config, trim_aoa, trim_elev, build_report=True)
print("\n".join(trim_result["report_lines"]))
polar_rows = trim_result.get("polar_rows")
if polar_rows:
    try:
        import matplotlib.pyplot as plt

        cds = [row[2] for row in polar_rows]
        cls = [row[1] for row in polar_rows]
        fig, ax = plt.subplots()
        ax.plot(cds, cls, marker="o")
        ax.set_xlabel("Cd")
        ax.set_ylabel("Cl")
        ax.set_title("Cruise Polar (Cl vs Cd)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        plot_path = "cruise_polar.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Cruise polar plot saved to {plot_path}")
    except Exception as exc:
        csv_path = "cruise_polar.csv"
        with open(csv_path, "w", newline="") as handle:
            handle.write("aoa_deg,cl_total,cd_total,ld\n")
            for row in polar_rows:
                handle.write(f"{row[0]:.3f},{row[1]:.6f},{row[2]:.6f},{row[3]:.3f}\n")
        print(f"Cruise polar data saved to {csv_path} (plot unavailable: {exc})")

model = AircraftModel(config)
profile = MissionProfile(
    segments=[
        MissionSegment(kind="climb_to", target_alt=20, mode = "min_time"),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0),
        MissionSegment(kind="loiter", bank_deg=10.0, mode="max_endurance_eq", speed_max=80.0),
        MissionSegment(kind="cruise", distance=500.0, mode="max_endurance_eq", speed_max=80.0),
        MissionSegment(kind="descent_to", target_alt=10, speed=14.0, elevator_deg=-10.0, thrust_scale=0.0),
        MissionSegment(kind="loiter", speed=14.0, time=180.0, bank_deg=20.0),
        MissionSegment(kind="landing", speed=12.0),
    ],
    log_interval = 0.1,
    takeoff_aero_stride=5,
    n_span=300,
    power_derate_frac=0.0,
    energy_reserve_frac=0.2,
)
mission_history, segment_summaries = simulate_mission(model, profile, initial_alt=0.0, return_summary=True)
last_state = mission_history[-1]
print("")
print("Mission Summary")
print(f"{'Time':<18} {last_state.t:>12.1f} {'s':>9}")
print(f"{'Distance':<18} {last_state.x:>12.1f} {'m':>9}")
print(f"{'Altitude':<18} {last_state.h:>12.1f} {'m':>9}")
print(f"{'Speed':<18} {last_state.v:>12.1f} {'m/s':>9}")
print(f"{'Phase':<18} {last_state.phase:>12} {'':>9}")
print(f"{'Energy Left':<18} {last_state.energy_j:>12.0f} {'J':>9}")

if segment_summaries:
    print("")
    print("Segment Energy/Distance Summary")
    header = f"{'Segment':<18} {'Time':>10} {'Dist':>10} {'Energy Used':>14}"
    print(header)
    print("-" * len(header))
    for row in segment_summaries:
        print(f"{row['segment']:<18} {row['time_s']:>10.1f} {row['distance_m']:>10.1f} {row['energy_used_j']:>14.0f}")
    loiter_rows = [row for row in segment_summaries if row.get("kind") == "loiter" and "cl_avg" in row]
    if loiter_rows:
        print("")
        print("Loiter Aerodynamics (Time-Weighted)")
        header = f"{'Segment':<18} {'AOA':>7} {'Elev':>7} {'CL':>8} {'CD':>8} {'CDo':>8} {'L/D':>8} {'L':>9} {'D':>9}"
        print(header)
        print("-" * len(header))
        for row in loiter_rows:
            print(
                f"{row['segment']:<18} {row['aoa_deg']:>7.2f} {row['elev_deg']:>7.2f} {row['cl_avg']:>8.4f} "
                f"{row['cd_avg']:>8.5f} {row['cdo_avg']:>8.5f} {row['ld_avg']:>8.2f} {row['lift_avg_n']:>9.1f} "
                f"{row['drag_avg_n']:>9.1f}"
            )
