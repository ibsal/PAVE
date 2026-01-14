from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet, PolarSet
import math
import numpy as np
import matplotlib.pyplot as plt

def integrate_profile_drag(foil, root_chord, tip_chord, span, area_ref, alpha_deg, rho, mu, velocity, n=1000, count=1, symmetric=True, chord_breaks=None, sweep_breaks=None):
    half_span = 0.5 * span if symmetric else span
    dx = half_span / n
    span_stations = np.arange(dx / 2, half_span, dx)
    if chord_breaks:
        fractions = span_stations / half_span
        break_fracs, break_chords = zip(*chord_breaks)
        chords = np.interp(fractions, break_fracs, break_chords)
    else:
        chords = ((tip_chord - root_chord) / half_span) * span_stations + root_chord
    if sweep_breaks:
        fractions = span_stations / half_span
        sweep_fracs, sweep_degs = zip(*sweep_breaks)
        sweep_local = np.deg2rad(np.interp(fractions, sweep_fracs, sweep_degs))
    else:
        sweep_local = 0.0
    reynolds = rho * velocity * chords / mu
    cd_profile = 0.0
    cl_total = 0.0
    for idx, (c, re_local) in enumerate(zip(chords, reynolds)):
        if sweep_breaks is not None:
            alpha_local = alpha_deg * math.cos(sweep_local[idx])
        else:
            alpha_local = alpha_deg
        cd2d = foil.cd(alpha_deg=alpha_local, reynolds=re_local)
        cl2d = foil.cl(alpha_deg=alpha_local, reynolds=re_local)
        cd_profile += cd2d * c * dx / area_ref
        cl_total += cl2d * c * dx / area_ref
    factor = (2.0 if symmetric else 1.0) * count
    cd_profile *= factor
    cl_total *= factor
    return cd_profile, cl_total, reynolds

def sweep_correction_factor(span, n, symmetric, sweep_breaks):
    if not sweep_breaks:
        return 1.0
    half_span = 0.5 * span if symmetric else span
    dx = half_span / n
    span_stations = np.arange(dx / 2, half_span, dx)
    fractions = span_stations / half_span
    sweep_fracs, sweep_degs = zip(*sweep_breaks)
    sweep_local = np.deg2rad(np.interp(fractions, sweep_fracs, sweep_degs))
    return float(np.mean(np.cos(sweep_local) ** 2))

def quarter_chord_positions(span, n, symmetric, sweep_breaks, x_root_qc):
    half_span = 0.5 * span if symmetric else span
    dx = half_span / n
    span_stations = np.arange(dx / 2, half_span, dx)
    if sweep_breaks:
        fractions = span_stations / half_span
        sweep_fracs, sweep_degs = zip(*sweep_breaks)
        sweep_local = np.deg2rad(np.interp(fractions, sweep_fracs, sweep_degs))
        tan_sweep = np.tan(sweep_local)
        cumulative = np.cumsum(tan_sweep * dx)
        x_qc = x_root_qc + cumulative - 0.5 * tan_sweep * dx
    else:
        x_qc = np.full_like(span_stations, x_root_qc, dtype=float)
    return span_stations, x_qc

def integrate_pitching_moment_about_cg(foil, root_chord, tip_chord, span, alpha_deg, rho, mu, velocity, q_dyn, x_root_qc, x_cg, n=1000, count=1, symmetric=True, chord_breaks=None, sweep_breaks=None):
    half_span = 0.5 * span if symmetric else span
    dx = half_span / n
    span_stations = np.arange(dx / 2, half_span, dx)
    if chord_breaks:
        fractions = span_stations / half_span
        break_fracs, break_chords = zip(*chord_breaks)
        chords = np.interp(fractions, break_fracs, break_chords)
    else:
        chords = ((tip_chord - root_chord) / half_span) * span_stations + root_chord
    if sweep_breaks:
        fractions = span_stations / half_span
        sweep_fracs, sweep_degs = zip(*sweep_breaks)
        sweep_local = np.deg2rad(np.interp(fractions, sweep_fracs, sweep_degs))
    else:
        sweep_local = None
    _, x_qc = quarter_chord_positions(span, n, symmetric, sweep_breaks, x_root_qc)
    reynolds = rho * velocity * chords / mu
    lift_moment = 0.0
    cm_moment = 0.0
    for idx, (c, re_local) in enumerate(zip(chords, reynolds)):
        if sweep_local is not None:
            alpha_local = alpha_deg * math.cos(sweep_local[idx])
        else:
            alpha_local = alpha_deg
        cl2d = foil.cl(alpha_deg=alpha_local, reynolds=re_local)
        cm2d = foil.cm(alpha_deg=alpha_local, reynolds=re_local)
        dL = q_dyn * cl2d * c * dx
        lift_moment += -dL * (x_qc[idx] - x_cg)
        cm_moment += q_dyn * cm2d * (c ** 2) * dx
    factor = (2.0 if symmetric else 1.0) * count
    return factor * lift_moment, factor * cm_moment

def integrate_airfoil_cm(foil, root_chord, tip_chord, span, area_ref, mac_ref, alpha_deg, rho, mu, velocity, n=1000, count=1, symmetric=True, chord_breaks=None, sweep_breaks=None):
    half_span = 0.5 * span if symmetric else span
    dx = half_span / n
    span_stations = np.arange(dx / 2, half_span, dx)
    if chord_breaks:
        fractions = span_stations / half_span
        break_fracs, break_chords = zip(*chord_breaks)
        chords = np.interp(fractions, break_fracs, break_chords)
    else:
        chords = ((tip_chord - root_chord) / half_span) * span_stations + root_chord
    if sweep_breaks:
        fractions = span_stations / half_span
        sweep_fracs, sweep_degs = zip(*sweep_breaks)
        sweep_local = np.deg2rad(np.interp(fractions, sweep_fracs, sweep_degs))
    else:
        sweep_local = 0.0
    reynolds = rho * velocity * chords / mu
    cm_sum = 0.0
    for idx, (c, re_local) in enumerate(zip(chords, reynolds)):
        if sweep_breaks is not None:
            alpha_local = alpha_deg * math.cos(sweep_local[idx])
        else:
            alpha_local = alpha_deg
        cm2d = foil.cm(alpha_deg=alpha_local, reynolds=re_local)
        cm_sum += cm2d * (c ** 2) * dx
    factor = (2.0 if symmetric else 1.0) * count
    return factor * cm_sum / (area_ref * mac_ref)

def pitching_moment_about_cg(lift_force, x_qc, x_cg):
    return -lift_force * (x_qc - x_cg)

def wing_pitching_moment(q_dyn, area_ref, cl_wing, x_wqc, x_cg):
    lift = q_dyn * area_ref * cl_wing
    return pitching_moment_about_cg(lift, x_wqc, x_cg)

def htail_pitching_moment(q_dyn, area_ref, cl_htail, x_htqc, x_cg):
    lift = q_dyn * area_ref * cl_htail
    return pitching_moment_about_cg(lift, x_htqc, x_cg)

def run_analysis(flight_aoa, elevator_deflection):
    Cd_profile = 0.0
    Cl = 0.0

    N = 1000

    atm = Atmosphere(GroundLevel + OrbitLevel)
    rho = float(atm.density)
    mu = float(atm.dynamic_viscosity)

    alpha = flight_aoa + WingIncidence

    Cd_profile, Cl, Re = integrate_profile_drag(
        wingFoil,
        RootChord,
        TipChord,
        Span,
        WingArea,
        alpha,
        rho,
        mu,
        flightVelocity,
        n=N,
        symmetric=True,
        chord_breaks=[(0.0, RootChord), (MidChordPos, MidChord), (1.0, TipChord)],
        sweep_breaks=[(0.0, RootSweepDeg), (MidSweepPos, MidSweepDeg), (1.0, TipSweepDeg)],
    )

    ### Wing Lift induced drag
    e_oswald_w = 1.78*(1.0 - 0.045*(AR**0.68)) - 0.64
    e_oswald_w = min(max(e_oswald_w, 0.3), 0.95)
    AR_eff = AR * sweep_correction_factor(Span, N, True, [(0.0, RootSweepDeg), (MidSweepPos, MidSweepDeg), (1.0, TipSweepDeg)])
    CdwiL = Cl**2/(math.pi * AR_eff * e_oswald_w)

    ### Vertical Tail Profile Drag
    CdVTail, _, _ = integrate_profile_drag(
        Vfoil,
        VRootChord,
        VTipChord,
        VSpan,
        WingArea,
        VtailIncidence,
        rho,
        mu,
        flightVelocity,
        n=N,
        count=VtailCount,
        symmetric=False,
        chord_breaks=[(0.0, VRootChord), (VMidChordPos, VMidChord), (1.0, VTipChord)],
        sweep_breaks=[(0.0, VRootSweepDeg), (VMidSweepPos, VMidSweepDeg), (1.0, VTipSweepDeg)],
    )
    CdVTail = CdVTail * TailEfficiency

    ### Horizontal Tail Drag Contribution 
    DownwashAngle = alpha * 0.45 # Typ 0.3-0.6
    HTailAOA = flight_aoa + HtailIncidence - DownwashAngle + ElevatorEffectivenessTau * elevator_deflection
    CdHTail, ClHtail, _ = integrate_profile_drag(
        Hfoil,
        HRootChord,
        HTipChord,
        HSpan,
        WingArea,
        HTailAOA,
        rho,
        mu,
        flightVelocity,
        n=N,
        symmetric=True,
        chord_breaks=[(0.0, HRootChord), (HMidChordPos, HMidChord), (1.0, HTipChord)],
        sweep_breaks=[(0.0, HRootSweepDeg), (HMidSweepPos, HMidSweepDeg), (1.0, HTipSweepDeg)],
    )
    CdHTail = CdHTail * TailEfficiency
    ClHtail = ClHtail * TailEfficiency

    ### Horizontal Tail Lift induced drag
    e_oswald_h = 1.78*(1.0 - 0.045*(HAR**0.68)) - 0.64
    e_oswald_h = min(max(e_oswald_h, 0.3), 0.95)
    HAR_eff = HAR * sweep_correction_factor(HSpan, N, True, [(0.0, HRootSweepDeg), (HMidSweepPos, HMidSweepDeg), (1.0, HTipSweepDeg)])
    CdHTailInduced = (ClHtail**2) * (WingArea / HTailArea) / (math.pi * HAR_eff * e_oswald_h * TailEfficiency)

    ### Misc
    Cdmisc = 0.001

    #### Final Drag Stackup
    CdTotal = CdwiL + Cdmisc + Cd_profile + CdNonLifting + CdVTail + CdHTail + CdHTailInduced

    ### Forces
    q_dyn = 0.5 * rho * (flightVelocity ** 2)
    Lift = float(q_dyn * WingArea * (Cl + ClHtail))
    Drag = float(q_dyn * WingArea * CdTotal)
    PropulsivePower = Drag * float(flightVelocity)

    WingPitchMoment, WingAirfoilMoment = integrate_pitching_moment_about_cg(
        wingFoil,
        RootChord,
        TipChord,
        Span,
        alpha,
        rho,
        mu,
        flightVelocity,
        q_dyn,
        Xwqc,
        Xcg,
        n=N,
        symmetric=True,
        chord_breaks=[(0.0, RootChord), (MidChordPos, MidChord), (1.0, TipChord)],
        sweep_breaks=[(0.0, RootSweepDeg), (MidSweepPos, MidSweepDeg), (1.0, TipSweepDeg)],
    )
    WingPitchMomentRootQC, WingAirfoilMomentRootQC = integrate_pitching_moment_about_cg(
        wingFoil,
        RootChord,
        TipChord,
        Span,
        alpha,
        rho,
        mu,
        flightVelocity,
        q_dyn,
        Xwqc,
        Xwqc,
        n=N,
        symmetric=True,
        chord_breaks=[(0.0, RootChord), (MidChordPos, MidChord), (1.0, TipChord)],
        sweep_breaks=[(0.0, RootSweepDeg), (MidSweepPos, MidSweepDeg), (1.0, TipSweepDeg)],
    )
    HTailPitchMoment, HTailAirfoilMoment = integrate_pitching_moment_about_cg(
        Hfoil,
        HRootChord,
        HTipChord,
        HSpan,
        HTailAOA,
        rho,
        mu,
        flightVelocity,
        q_dyn,
        Xhtqc,
        Xcg,
        n=N,
        symmetric=True,
        chord_breaks=[(0.0, HRootChord), (HMidChordPos, HMidChord), (1.0, HTipChord)],
        sweep_breaks=[(0.0, HRootSweepDeg), (HMidSweepPos, HMidSweepDeg), (1.0, HTipSweepDeg)],
    )
    HTailPitchMoment *= TailEfficiency
    HTailAirfoilMoment *= TailEfficiency
    WingTotalPitchMoment = WingPitchMoment + WingAirfoilMoment
    HTailTotalPitchMoment = HTailPitchMoment + HTailAirfoilMoment
    TotalPitchMoment = WingTotalPitchMoment + HTailTotalPitchMoment
    CmWingAboutCG = WingTotalPitchMoment / (q_dyn * WingArea * MAC)
    CmWingAboutRootQC = (WingPitchMomentRootQC + WingAirfoilMomentRootQC) / (q_dyn * WingArea * MAC)

    report_lines = []
    report_lines.append("\n=== REPORTS ===")
    report_lines.append("")
    report_lines.append("Trim Conditions")
    trim_header = f"{'Parameter':<18} {'Value':>12} {'Units':>9}"
    report_lines.append(trim_header)
    report_lines.append("-" * len(trim_header))
    report_lines.append(f"{'Flight AOA':<18} {flight_aoa:>12.3f} {'deg':>9}")
    report_lines.append(f"{'Elevator':<18} {elevator_deflection:>12.3f} {'deg':>9}")
    report_lines.append("")
    report_lines.append("Drag Contributors")
    header = f"{'Contributor':<18} {'Cd':>12} {'%Total':>9}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    report_lines.append(f"{'Oswald e':<18} {e_oswald_w:>12.6f} {'':>9}")
    if CdTotal > 0.0:
        for name, cdo in zip(Components, NonLiftingCdo):
            report_lines.append(f"{name:<18} {cdo:>12.6f} {cdo / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'Wing Profile':<18} {Cd_profile:>12.6f} {Cd_profile / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'H Tail Profile':<18} {CdHTail:>12.6f} {CdHTail / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'H Tail Induced':<18} {CdHTailInduced:>12.6f} {CdHTailInduced / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'V Tail Profile':<18} {CdVTail:>12.6f} {CdVTail / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'Induced':<18} {CdwiL:>12.6f} {CdwiL / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'Misc':<18} {Cdmisc:>12.6f} {Cdmisc / CdTotal * 100:>8.2f}%")
        report_lines.append(f"{'Total':<18} {CdTotal:>12.6f} {100:>8.2f}%")
    else:
        for name, cdo in zip(Components, NonLiftingCdo):
            report_lines.append(f"{name:<18} {cdo:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Wing Profile':<18} {Cd_profile:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'H Tail Profile':<18} {CdHTail:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'H Tail Induced':<18} {CdHTailInduced:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'V Tail Profile':<18} {CdVTail:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Induced':<18} {CdwiL:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Misc':<18} {Cdmisc:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Total':<18} {CdTotal:>12.6f} {'0.00%':>9}")
    report_lines.append("")
    report_lines.append("Wing Statistics")
    wing_header = f"{'Metric':<18} {'Value':>12} {'Units':>9}"
    report_lines.append(wing_header)
    report_lines.append("-" * len(wing_header))
    report_lines.append(f"{'Span':<18} {Span:>12.4f} {'m':>9}")
    report_lines.append(f"{'Wing Area':<18} {WingArea:>12.4f} {'m^2':>9}")
    report_lines.append(f"{'Aspect Ratio':<18} {AR:>12.4f} {'':>9}")
    report_lines.append(f"{'MAC':<18} {MAC:>12.4f} {'m':>9}")
    report_lines.append(f"{'Taper':<18} {Taper:>12.4f} {'':>9}")
    report_lines.append(f"{'Alpha':<18} {alpha:>12.3f} {'deg':>9}")
    report_lines.append(f"{'Cl':<18} {Cl:>12.5f} {'':>9}")
    report_lines.append(f"{'Re (min)':<18} {Re.min():>12.0f} {'':>9}")
    report_lines.append(f"{'Re (max)':<18} {Re.max():>12.0f} {'':>9}")
    report_lines.append(f"{'Re (mean)':<18} {Re.mean():>12.0f} {'':>9}")
    report_lines.append(f"{'Lift':<18} {Lift:>12.4f} {'N':>9}")
    report_lines.append(f"{'Drag':<18} {Drag:>12.4f} {'N':>9}")
    report_lines.append(f"{'Propulsive P':<18} {PropulsivePower:>12.2f} {'W':>9}")
    report_lines.append("")
    report_lines.append("Pitching Moments About CG")
    moment_header = f"{'Contributor':<18} {'Moment':>12} {'Units':>9}"
    report_lines.append(moment_header)
    report_lines.append("-" * len(moment_header))
    report_lines.append(f"{'Wing Lift':<18} {WingPitchMoment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Wing Cm':<18} {WingAirfoilMoment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Tail Lift':<18} {HTailPitchMoment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Tail Cm':<18} {HTailAirfoilMoment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Total':<18} {TotalPitchMoment:>12.4f} {'N*m':>9}")
    report_lines.append("")
    report_lines.append("Wing Moment Coefficients")
    cm_header = f"{'Reference':<18} {'Cm':>12} {'Units':>9}"
    report_lines.append(cm_header)
    report_lines.append("-" * len(cm_header))
    report_lines.append(f"{'About CG':<18} {CmWingAboutCG:>12.6f} {'':>9}")
    report_lines.append(f"{'About Root QC':<18} {CmWingAboutRootQC:>12.6f} {'':>9}")

    return {
        "Lift": Lift,
        "TotalPitchMoment": TotalPitchMoment,
        "report_lines": report_lines,
    }

def solve_trim(flight_aoa_guess, elevator_guess, max_iter=20, tol_force=1e-3, tol_moment=1e-3):
    aoa = flight_aoa_guess
    elev = elevator_guess
    converged = False
    last_result = None
    for _ in range(max_iter):
        result = run_analysis(aoa, elev)
        last_result = result
        f1 = result["Lift"] - Weight
        f2 = result["TotalPitchMoment"]
        if abs(f1) < tol_force and abs(f2) < tol_moment:
            converged = True
            break
        d_aoa = 0.25
        d_elev = 0.5
        result_aoa = run_analysis(aoa + d_aoa, elev)
        result_elev = run_analysis(aoa, elev + d_elev)
        f1_aoa = result_aoa["Lift"] - Weight
        f2_aoa = result_aoa["TotalPitchMoment"]
        f1_elev = result_elev["Lift"] - Weight
        f2_elev = result_elev["TotalPitchMoment"]
        j11 = (f1_aoa - f1) / d_aoa
        j12 = (f1_elev - f1) / d_elev
        j21 = (f2_aoa - f2) / d_aoa
        j22 = (f2_elev - f2) / d_elev
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-9:
            break
        delta_aoa = (-f1 * j22 + f2 * j12) / det
        delta_elev = (-j11 * f2 + j21 * f1) / det
        aoa += delta_aoa
        elev += delta_elev
    return aoa, elev, converged, last_result

### Environment Configuration
GroundLevel = 0 # M, ASL
OrbitLevel = 1200 #M, AGL
flightVelocity = 21 #SUPER PLACEHOLDER m/s
flightAOA = 2 # SUPER PLACEHOLDER degrees
TailEfficiency = 0.95 # q_tail / q_wing (0.9-1.0 typical). Scales tail Cl and Cd
ElevatorDeflection = 0 # deg, standard convention: positive = TE down
ElevatorEffectivenessTau = 0.35 # deg tail alpha per deg elevator (0.3-0.6 typical)
Weight = 200 #N = 45 lbf


### Aircraft CG Definitions x = 0 at the nose 
Xwqc = 0.45 # M
Xhtqc = 4 # M
Xcg = 0.4 # M

### Wing Definition
RootChord = 0.3
MidChord = 0.25
MidChordPos = 0.4 # fraction of half span from root to tip (0-1)
TipChord = 0.2
RootThickness = 0.1
TipThickness = 0.1
Span = 3.9624 #M, wing span (both sides)
RootSweepDeg = 0.0
MidSweepDeg = 0
MidSweepPos = 0.4 # fraction of half span from root to tip (0-1)
TipSweepDeg = 3
wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
WingIncidence = 2

Taper = TipChord/RootChord
ThicknessRatio = RootThickness/TipThickness
MAC = 2.0/3.0 * RootChord * (1 + Taper + Taper**2)/(1 + Taper)
WingArea = 0.5 * (RootChord + TipChord) * Span
AR = (Span**2)/WingArea


### Horizontal and Vertical Tail Definition
HRootChord = 0.1 #M
HMidChord = 0.1
HMidChordPos = 0.4 # fraction of half span from root to tip (0-1)
HTipChord = 0.1 #M
HRootThickness = 0.1
HTipThickness = 0.1
HSpan = 1 #M, total span (both sides)
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
HtailIncidence = -2 # deg, standard convention: incidence > 0 -> tail angled with TE down
VtailIncidence = 0 # deg

HTaper = HTipChord / HRootChord
VTaper = VTipChord / VRootChord
HThicknessRatio = HRootThickness / HTipThickness
VThicknessRatio = VRootThickness / VTipThickness
HMAC = 2.0/3.0 * HRootChord * (1 + HTaper + HTaper**2)/(1 + HTaper)
VMAC = 2.0/3.0 * VRootChord * (1 + VTaper + VTaper**2)/(1 + VTaper)
HTailArea = 0.5 * (HRootChord + HTipChord) * HSpan
VSingleTailArea = 0.5 * (VRootChord + VTipChord) * VSpan
VTailArea = VSingleTailArea * VtailCount
HAR = (HSpan**2) / HTailArea
VAR = (VSpan**2) / VSingleTailArea


### Boom Definition 
BoomLength = 1.8
BoomDiameter = 0.03
BoomCount = 1.8

### Fuselage Definition 
FuselageWidth = 0.3 # M
FuselageHeight = 0.15 # M
FuselageLength = 1.4 # M
Pfactor = 0.9 # Account for fillets (super basic)
FuselagePerimeter = FuselageHeight*2 + FuselageWidth*2 * Pfactor
FuselageDiameter = FuselagePerimeter/math.pi



#### Drag  Combined APproach
# Non lifting drag = Sum(Cfc * FFc * Qc * Sratio/Sref). Used for fuselage, booms
# Airoil Profile drag = Chord weighted Cd sum along wing. Used for main wing
# Lift induced drag. Uses Cl from airfoil and oswald effeicnecy factor estimate
# Misc drag. for things like ducts, servo horns, etc.

### Non lifting drag stackup
Components = []
FF = []
Q = []
Cfc = []
Sratio = []

## Cdo Fuselage
Components.append("Fuselage")
FF.append(1 + 60/((FuselageLength/FuselageDiameter)**3) + (FuselageLength/FuselageDiameter)/400)
Q.append(1.0)
Cfc.append(0.004)
Sratio.append((math.pi * FuselageDiameter * FuselageLength * math.pow((1 - 2/(FuselageLength/FuselageDiameter)), 2.0/3.0) * (1 + 1/((FuselageLength/FuselageDiameter)**2)))/WingArea)

## Cdo Booms
#Consider booms as fuselage

Components.append("Boom")
FF.append(BoomCount * (1 + 60/((BoomLength/BoomDiameter)**3) + (BoomLength/BoomDiameter)/400))
Q.append(1.0)
Cfc.append(0.004)
Sratio.append((math.pi * BoomDiameter * BoomLength * math.pow((1 - 2/(BoomLength/BoomDiameter)), 2.0/3.0) * (1 + 1/((BoomLength/BoomDiameter)**2)))/WingArea)

## Finish non lifting drag stackup
stack_lines = []
header = f"{'Component':<12} {'FF':>10} {'Q':>6} {'Cfc':>10} {'Sratio':>12} {'Cdo':>12}"
stack_lines.append(header)
stack_lines.append("-" * len(header))
CdNonLifting = 0.0
NonLiftingCdo = []
for name, ff, q, cfc, sw in zip(Components, FF, Q, Cfc, Sratio):
    cdo = ff * q * cfc * sw
    CdNonLifting += cdo
    NonLiftingCdo.append(cdo)
    stack_lines.append(f"{name:<12} {ff:>10.4f} {q:>6.3f} {cfc:>10.5f} {sw:>12.5f} {cdo:>12.6f}")
stack_lines.append(f"{'Total':<12} {'':>10} {'':>6} {'':>10} {'':>12} {CdNonLifting:>12.6f}")
print("\n".join(stack_lines))

trim_aoa, trim_elev, converged, trim_result = solve_trim(flightAOA, ElevatorDeflection)
if not converged:
    print("Warning: trim solver did not converge; reporting last iteration.")
print("\n".join(trim_result["report_lines"]))




