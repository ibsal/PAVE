import math
import numpy as np
from ambiance import Atmosphere


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


def planform_area_and_mac(span, chord_breaks, symmetric):
    if not chord_breaks or span <= 0.0:
        return 0.0, 0.0
    half_span = 0.5 * span if symmetric else span
    breaks = sorted(chord_breaks, key=lambda item: item[0])
    if len(breaks) < 2:
        return 0.0, 0.0
    area_half = 0.0
    c2_int_half = 0.0
    last_frac, last_chord = breaks[0]
    for frac, chord in breaks[1:]:
        dx = (frac - last_frac) * half_span
        if dx <= 0.0:
            last_frac, last_chord = frac, chord
            continue
        area_half += 0.5 * (last_chord + chord) * dx
        c2_int_half += (last_chord ** 2 + last_chord * chord + chord ** 2) / 3.0 * dx
        last_frac, last_chord = frac, chord
    area = area_half * (2.0 if symmetric else 1.0)
    if area <= 0.0:
        return 0.0, 0.0
    mac = (2.0 * c2_int_half / area) if symmetric else (c2_int_half / area)
    return area, mac


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


def skin_friction_cf(re, roughness, length, laminar_frac=0.0, re_crit_per_m=5e5):
    if re <= 0.0 or length <= 0.0:
        return 0.0
    cf_lam = 1.328 / math.sqrt(re)
    cf_turb_input = re
    if roughness > 0.0:
        recutoff = 38.21 * ((length / roughness) ** 1.053)
        if re > recutoff:
            cf_turb_input = recutoff
    cf_turb = 0.455 / (math.pow(math.log10(cf_turb_input), 2.58))
    re_crit = re_crit_per_m * length
    if re < re_crit:
        cf_base = cf_lam
    else:
        lam = max(min(laminar_frac, 1.0), 0.0)
        cf_base = lam * cf_lam + (1.0 - lam) * cf_turb
    return cf_base


def non_lifting_drag_stack(config, rho, mu, v):
    fuselage = config["fuselage"]
    boom = config["boom"]
    wing = config["wing"]
    rough = config["roughness"]

    components = []
    ff_list = []
    q_list = []
    cfc_list = []
    sratio_list = []
    re_list = []

    re_fuse = rho * v * fuselage["length"] / mu
    re_boom = rho * v * boom["length"] / mu

    components.append("Fuselage")
    fuse_ld = fuselage["length"] / fuselage["diameter"]
    ff_list.append(1 + 60/(fuse_ld**3) + fuse_ld/400)
    q_list.append(1.0)
    cfc_list.append(skin_friction_cf(re_fuse, rough["k"], fuselage["length"], laminar_frac=rough["fuselage_lam"], re_crit_per_m=rough["re_crit_per_m"]))
    fuse_term = max(1.0 - 2.0 / max(fuse_ld, 1e-6), 0.0)
    sratio_list.append((math.pi * fuselage["diameter"] * fuselage["length"] * math.pow(fuse_term, 2.0/3.0) * (1 + 1/(fuse_ld**2))) / wing["area"])
    re_list.append(re_fuse)

    components.append("Boom")
    boom_ld = boom["length"] / boom["diameter"]
    boom_ff = 1 + 60/(boom_ld**3) + boom_ld/400
    ff_list.append(boom_ff)
    q_list.append(1.0)
    cfc_list.append(skin_friction_cf(re_boom, rough["k"], boom["length"], laminar_frac=rough["boom_lam"], re_crit_per_m=rough["re_crit_per_m"]))
    boom_term = max(1.0 - 2.0 / max(boom_ld, 1e-6), 0.0)
    boom_sratio = (math.pi * boom["diameter"] * boom["length"] * math.pow(boom_term, 2.0/3.0) * (1 + 1/(boom_ld**2))) / wing["area"]
    sratio_list.append(boom_sratio * boom["count"])
    re_list.append(re_boom)

    stack_lines = []
    header = f"{'Component':<12} {'FF':>10} {'Q':>6} {'Cfc':>10} {'Sratio':>12} {'Cdo':>12}"
    stack_lines.append(header)
    stack_lines.append("-" * len(header))
    cd_non_lifting = 0.0
    non_lifting_cdo = []
    for name, ff, q, cfc, sw in zip(components, ff_list, q_list, cfc_list, sratio_list):
        cdo = ff * q * cfc * sw
        cd_non_lifting += cdo
        non_lifting_cdo.append(cdo)
        stack_lines.append(f"{name:<12} {ff:>10.4f} {q:>6.3f} {cfc:>10.5f} {sw:>12.5f} {cdo:>12.6f}")
    stack_lines.append(f"{'Total':<12} {'':>10} {'':>6} {'':>10} {'':>12} {cd_non_lifting:>12.6f}")
    return cd_non_lifting, non_lifting_cdo, components, stack_lines, cfc_list, re_list, ff_list, q_list, sratio_list


def run_analysis(config, flight_aoa, elevator_deflection, build_report=True):
    wing = config["wing"]
    htail = config["htail"]
    vtail = config["vtail"]

    n_span = config["n_span"]
    v = config["flight_velocity"]
    atm = Atmosphere(config["analysis_altitude_m"])
    rho = float(atm.density)
    mu = float(atm.dynamic_viscosity)

    alpha = flight_aoa + wing["incidence"]

    cd_non_lifting, non_lifting_cdo, components, stack_lines, cfc_list, re_list, ff_list, q_list, sratio_list = non_lifting_drag_stack(config, rho, mu, v)

    cd_profile, cl, re = integrate_profile_drag(
        wing["foil"],
        wing["root_chord"],
        wing["tip_chord"],
        wing["span"],
        wing["area"],
        alpha,
        rho,
        mu,
        v,
        n=n_span,
        symmetric=True,
        chord_breaks=wing["chord_breaks"],
        sweep_breaks=wing["sweep_breaks"],
    )

    e_oswald_w = 1.78*(1.0 - 0.045*(wing["ar"]**0.68)) - 0.64
    e_oswald_w = min(max(e_oswald_w, 0.3), 0.95)
    ar_eff = wing["ar"] * sweep_correction_factor(wing["span"], n_span, True, wing["sweep_breaks"])
    cd_w_induced = cl**2/(math.pi * ar_eff * e_oswald_w)

    cd_vtail, _, _ = integrate_profile_drag(
        vtail["foil"],
        vtail["root_chord"],
        vtail["tip_chord"],
        vtail["span"],
        wing["area"],
        vtail["incidence"],
        rho,
        mu,
        v,
        n=n_span,
        count=vtail["count"],
        symmetric=False,
        chord_breaks=vtail["chord_breaks"],
        sweep_breaks=vtail["sweep_breaks"],
    )

    downwash = alpha * config["downwash_factor"]
    htail_aoa = flight_aoa + htail["incidence"] - downwash + config["elevator_tau"] * elevator_deflection
    cd_htail, cl_htail, _ = integrate_profile_drag(
        htail["foil"],
        htail["root_chord"],
        htail["tip_chord"],
        htail["span"],
        wing["area"],
        htail_aoa,
        rho,
        mu,
        v,
        n=n_span,
        symmetric=True,
        chord_breaks=htail["chord_breaks"],
        sweep_breaks=htail["sweep_breaks"],
    )
    cd_htail *= config["tail_efficiency"]
    cl_htail *= config["tail_efficiency"]

    e_oswald_h = 1.78*(1.0 - 0.045*(htail["ar"]**0.68)) - 0.64
    e_oswald_h = min(max(e_oswald_h, 0.3), 0.95)
    har_eff = htail["ar"] * sweep_correction_factor(htail["span"], n_span, True, htail["sweep_breaks"])
    cd_htail_induced = (cl_htail**2) * (wing["area"] / htail["area"]) / (math.pi * har_eff * e_oswald_h)

    cd0 = config["cd_misc"] + cd_profile + cd_non_lifting + cd_vtail + cd_htail
    cd_induced = cd_w_induced + cd_htail_induced
    cd_total = cd0 + cd_induced

    q_dyn = 0.5 * rho * (v ** 2)
    lift = float(q_dyn * wing["area"] * (cl + cl_htail))
    drag = float(q_dyn * wing["area"] * cd_total)
    propulsive_power = drag * float(v)

    wing_moment, wing_cm_moment = integrate_pitching_moment_about_cg(
        wing["foil"],
        wing["root_chord"],
        wing["tip_chord"],
        wing["span"],
        alpha,
        rho,
        mu,
        v,
        q_dyn,
        config["positions"]["x_wqc"],
        config["positions"]["x_cg"],
        n=n_span,
        symmetric=True,
        chord_breaks=wing["chord_breaks"],
        sweep_breaks=wing["sweep_breaks"],
    )
    wing_moment_root, wing_cm_moment_root = integrate_pitching_moment_about_cg(
        wing["foil"],
        wing["root_chord"],
        wing["tip_chord"],
        wing["span"],
        alpha,
        rho,
        mu,
        v,
        q_dyn,
        config["positions"]["x_wqc"],
        config["positions"]["x_wqc"],
        n=n_span,
        symmetric=True,
        chord_breaks=wing["chord_breaks"],
        sweep_breaks=wing["sweep_breaks"],
    )
    htail_moment, htail_cm_moment = integrate_pitching_moment_about_cg(
        htail["foil"],
        htail["root_chord"],
        htail["tip_chord"],
        htail["span"],
        htail_aoa,
        rho,
        mu,
        v,
        q_dyn,
        config["positions"]["x_htqc"],
        config["positions"]["x_cg"],
        n=n_span,
        symmetric=True,
        chord_breaks=htail["chord_breaks"],
        sweep_breaks=htail["sweep_breaks"],
    )
    htail_moment *= config["tail_efficiency"]
    htail_cm_moment *= config["tail_efficiency"]
    wing_total_moment = wing_moment + wing_cm_moment
    htail_total_moment = htail_moment + htail_cm_moment
    total_moment = wing_total_moment + htail_total_moment

    cm_wing_cg = wing_total_moment / (q_dyn * wing["area"] * wing["mac"])
    cm_wing_root = (wing_moment_root + wing_cm_moment_root) / (q_dyn * wing["area"] * wing["mac"])
    cm_total = total_moment / (q_dyn * wing["area"] * wing["mac"])

    weight = config["weight"]
    wing_loading = weight / wing["area"] if wing["area"] > 0.0 else 0.0

    def stall_speed(cl_max):
        if cl_max is None or cl_max <= 0.0 or rho <= 0.0 or wing["area"] <= 0.0:
            return 0.0
        return math.sqrt(2.0 * weight / (rho * wing["area"] * cl_max))

    v_stall_takeoff = stall_speed(wing.get("cl_max_takeoff"))
    v_stall_cruise = stall_speed(wing.get("cl_max_cruise"))
    v_stall_landing = stall_speed(wing.get("cl_max_landing"))
    stall_speeds = config.get("stall_speeds", {}) or {}
    v_cruise = stall_speeds.get("cruise", v)
    v_takeoff = stall_speeds.get("takeoff")
    v_landing = stall_speeds.get("landing")
    if v_takeoff is None:
        v_takeoff = v_cruise
    if v_landing is None:
        v_landing = v_cruise
    stall_margins = config.get("stall_margins", {}) or {}
    required_margin_cruise = stall_margins.get("cruise", stall_margins.get("level_flight", stall_margins.get("level_flight_stall_margin")))
    required_margin_takeoff = stall_margins.get("takeoff")
    required_margin_landing = stall_margins.get("landing")
    achieved_margin_cruise = (v_cruise / v_stall_cruise) if v_stall_cruise > 0.0 else 0.0
    achieved_margin_takeoff = (v_takeoff / v_stall_takeoff) if v_stall_takeoff > 0.0 else 0.0
    achieved_margin_landing = (v_landing / v_stall_landing) if v_stall_landing > 0.0 else 0.0

    prop = config.get("propulsion", {})
    max_power = prop.get("max_power_w", 0.0)
    prop_eff = max(prop.get("prop_eff", 0.0), 1e-6)
    max_thrust = prop.get("max_thrust_n")
    if v <= 0.1:
        thrust_available = max_power * prop_eff / 0.1
    else:
        thrust_available = max_power * prop_eff / v
    if max_thrust is not None:
        thrust_available = min(thrust_available, max_thrust)
    ps_max = ((thrust_available - drag) * v / weight) if weight > 0.0 else 0.0
    motor_power = propulsive_power / prop_eff

    def foil_alpha_limits(foil):
        if hasattr(foil, "polars"):
            mins = [float(min(p.alpha_deg)) for p in foil.polars]
            maxs = [float(max(p.alpha_deg)) for p in foil.polars]
            return min(mins), max(maxs)
        if hasattr(foil, "alpha_deg"):
            return float(min(foil.alpha_deg)), float(max(foil.alpha_deg))
        return None, None

    alpha_min, alpha_max = foil_alpha_limits(wing["foil"])
    if alpha_min is None or alpha_max is None:
        alpha_min = -5.0
        alpha_max = 15.0
    flight_aoa_min = alpha_min - wing["incidence"]
    flight_aoa_max = alpha_max - wing["incidence"]
    sample_count = 15
    if sample_count < 3:
        sample_count = 3
    aoa_samples = [flight_aoa_min + i * (flight_aoa_max - flight_aoa_min) / (sample_count - 1) for i in range(sample_count)]

    polar_rows = []
    ld_max = 0.0
    ld_max_cl = 0.0
    ld_max_cd = 0.0
    ps_min = None
    ps_min_aoa = 0.0
    for aoa in aoa_samples:
        alpha_local = aoa + wing["incidence"]
        cd_profile_i, cl_i, _ = integrate_profile_drag(
            wing["foil"],
            wing["root_chord"],
            wing["tip_chord"],
            wing["span"],
            wing["area"],
            alpha_local,
            rho,
            mu,
            v,
            n=n_span,
            symmetric=True,
            chord_breaks=wing["chord_breaks"],
            sweep_breaks=wing["sweep_breaks"],
        )
        cd_w_induced_i = cl_i**2 / (math.pi * ar_eff * e_oswald_w) if ar_eff > 0.0 else 0.0
        downwash_i = alpha_local * config["downwash_factor"]
        htail_aoa_i = aoa + htail["incidence"] - downwash_i + config["elevator_tau"] * elevator_deflection
        cd_htail_i, cl_htail_i, _ = integrate_profile_drag(
            htail["foil"],
            htail["root_chord"],
            htail["tip_chord"],
            htail["span"],
            wing["area"],
            htail_aoa_i,
            rho,
            mu,
            v,
            n=n_span,
            symmetric=True,
            chord_breaks=htail["chord_breaks"],
            sweep_breaks=htail["sweep_breaks"],
        )
        cd_vtail_i, _, _ = integrate_profile_drag(
            vtail["foil"],
            vtail["root_chord"],
            vtail["tip_chord"],
            vtail["span"],
            wing["area"],
            vtail["incidence"],
            rho,
            mu,
            v,
            n=n_span,
            count=vtail["count"],
            symmetric=False,
            chord_breaks=vtail["chord_breaks"],
            sweep_breaks=vtail["sweep_breaks"],
        )
        cd_htail_i *= config["tail_efficiency"]
        cl_htail_i *= config["tail_efficiency"]
        cd_htail_induced_i = (cl_htail_i**2) * (wing["area"] / htail["area"]) / (math.pi * har_eff * e_oswald_h) if har_eff > 0.0 else 0.0
        cd0_i = config["cd_misc"] + cd_profile_i + cd_non_lifting + cd_vtail_i + cd_htail_i
        cd_total_i = cd0_i + cd_w_induced_i + cd_htail_induced_i
        cl_total_i = cl_i + cl_htail_i
        ld_i = (cl_total_i / cd_total_i) if cd_total_i > 0.0 else 0.0
        drag_i = q_dyn * wing["area"] * cd_total_i
        ps_i = ((thrust_available - drag_i) * v / weight) if weight > 0.0 else 0.0
        if ld_i > ld_max:
            ld_max = ld_i
            ld_max_cl = cl_total_i
            ld_max_cd = cd_total_i
        if ps_min is None or ps_i < ps_min:
            ps_min = ps_i
            ps_min_aoa = aoa
        polar_rows.append((aoa, cl_total_i, cd_total_i, ld_i))

    drag_stack = []
    for name, ff, q, cfc, sratio, cdo in zip(components, ff_list, q_list, cfc_list, sratio_list, non_lifting_cdo):
        drag_stack.append({
            "component": name,
            "ff": ff,
            "q": q,
            "cfc": cfc,
            "sratio": sratio,
            "cdo": cdo,
        })

    result = {
        "Lift": lift,
        "Drag": drag,
        "TotalPitchMoment": total_moment,
        "ClWing": cl,
        "ClTail": cl_htail,
        "ClTotal": cl + cl_htail,
        "Cd0": cd0,
        "CdInduced": cd_induced,
        "CdTotal": cd_total,
        "CmTotal": cm_total,
        "CmWingCG": cm_wing_cg,
        "CmWingRoot": cm_wing_root,
        "AOA": flight_aoa,
        "HTailAOA": htail_aoa,
        "drag_stack": drag_stack,
        "report_lines": None,
        "polar_rows": polar_rows,
    }

    if not build_report:
        return result

    report_lines = []
    report_lines.append("\n=== REPORTS ===")
    report_lines.append("")
    report_lines.append("Non-Lifting Drag Stackup")
    report_lines.extend(stack_lines)
    report_lines.append("")
    report_lines.append("Skin Friction Coefficients")
    cf_header = f"{'Component':<12} {'Re':>12} {'Cf':>10} {'Units':>9}"
    report_lines.append(cf_header)
    report_lines.append("-" * len(cf_header))
    for name, re_local, cf_local in zip(components, re_list, cfc_list):
        report_lines.append(f"{name:<12} {re_local:>12.0f} {cf_local:>10.6f} {'':>9}")
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
    if cd_total > 0.0:
        for name, cdo in zip(components, non_lifting_cdo):
            report_lines.append(f"{name:<18} {cdo:>12.6f} {cdo / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'Wing Profile':<18} {cd_profile:>12.6f} {cd_profile / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'H Tail Profile':<18} {cd_htail:>12.6f} {cd_htail / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'H Tail Induced':<18} {cd_htail_induced:>12.6f} {cd_htail_induced / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'V Tail Profile':<18} {cd_vtail:>12.6f} {cd_vtail / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'Induced':<18} {cd_w_induced:>12.6f} {cd_w_induced / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'Misc':<18} {config['cd_misc']:>12.6f} {config['cd_misc'] / cd_total * 100:>8.2f}%")
        report_lines.append(f"{'Total':<18} {cd_total:>12.6f} {100:>8.2f}%")
    else:
        for name, cdo in zip(components, non_lifting_cdo):
            report_lines.append(f"{name:<18} {cdo:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Wing Profile':<18} {cd_profile:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'H Tail Profile':<18} {cd_htail:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'H Tail Induced':<18} {cd_htail_induced:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'V Tail Profile':<18} {cd_vtail:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Induced':<18} {cd_w_induced:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Misc':<18} {config['cd_misc']:>12.6f} {'0.00%':>9}")
        report_lines.append(f"{'Total':<18} {cd_total:>12.6f} {'0.00%':>9}")
    report_lines.append("")
    report_lines.append("Wing Statistics")
    wing_header = f"{'Metric':<18} {'Value':>12} {'Units':>9}"
    report_lines.append(wing_header)
    report_lines.append("-" * len(wing_header))
    report_lines.append(f"{'Span':<18} {wing['span']:>12.4f} {'m':>9}")
    report_lines.append(f"{'Wing Area':<18} {wing['area']:>12.4f} {'m^2':>9}")
    report_lines.append(f"{'Aspect Ratio':<18} {wing['ar']:>12.4f} {'':>9}")
    report_lines.append(f"{'MAC':<18} {wing['mac']:>12.4f} {'m':>9}")
    report_lines.append(f"{'Taper':<18} {wing['taper']:>12.4f} {'':>9}")
    report_lines.append(f"{'Alpha':<18} {alpha:>12.3f} {'deg':>9}")
    report_lines.append(f"{'Cl':<18} {cl:>12.5f} {'':>9}")
    report_lines.append(f"{'Re (min)':<18} {re.min():>12.0f} {'':>9}")
    report_lines.append(f"{'Re (max)':<18} {re.max():>12.0f} {'':>9}")
    report_lines.append(f"{'Re (mean)':<18} {re.mean():>12.0f} {'':>9}")
    report_lines.append(f"{'Lift':<18} {lift:>12.4f} {'N':>9}")
    report_lines.append(f"{'Drag':<18} {drag:>12.4f} {'N':>9}")
    report_lines.append(f"{'Propulsive P':<18} {propulsive_power:>12.2f} {'W':>9}")
    report_lines.append("")
    report_lines.append("Performance Summary (Cruise)")
    perf_header = f"{'Metric':<18} {'Value':>12} {'Units':>9}"
    report_lines.append(perf_header)
    report_lines.append("-" * len(perf_header))
    report_lines.append(f"{'Wing Loading':<18} {wing_loading:>12.2f} {'N/m^2':>9}")
    to_inline = f"{'Stall TO':<18} {v_stall_takeoff:>7.2f} m/s | V={v_takeoff:>7.2f}"
    if required_margin_takeoff is not None:
        to_inline += f" | Req={required_margin_takeoff:>5.3f}"
    to_inline += f" | Ach={achieved_margin_takeoff:>5.3f}"
    report_lines.append(to_inline)
    cruise_inline = f"{'Stall Cruise':<18} {v_stall_cruise:>7.2f} m/s | V={v_cruise:>7.2f}"
    if required_margin_cruise is not None:
        cruise_inline += f" | Req={required_margin_cruise:>5.3f}"
    cruise_inline += f" | Ach={achieved_margin_cruise:>5.3f}"
    report_lines.append(cruise_inline)
    land_inline = f"{'Stall Land':<18} {v_stall_landing:>7.2f} m/s | V={v_landing:>7.2f}"
    if required_margin_landing is not None:
        land_inline += f" | Req={required_margin_landing:>5.3f}"
    land_inline += f" | Ach={achieved_margin_landing:>5.3f}"
    report_lines.append(land_inline)
    report_lines.append(f"{'L/D Max':<18} {ld_max:>12.2f} {'':>9}")
    report_lines.append(f"{'CL @ L/D Max':<18} {ld_max_cl:>12.4f} {'':>9}")
    report_lines.append(f"{'CD @ L/D Max':<18} {ld_max_cd:>12.5f} {'':>9}")
    report_lines.append(f"{'Thrust Avail':<18} {thrust_available:>12.2f} {'N':>9}")
    report_lines.append(f"{'Ps (Max Power)':<18} {ps_max:>12.3f} {'m/s':>9}")
    if ps_min is None:
        ps_min = 0.0
        ps_min_aoa = 0.0
    report_lines.append(f"{'Ps Min':<18} {ps_min:>12.3f} {'m/s':>9}")
    report_lines.append(f"{'AOA @ Ps Min':<18} {ps_min_aoa:>12.3f} {'deg':>9}")
    report_lines.append(f"{'Motor Power':<18} {motor_power:>12.2f} {'W':>9}")
    report_lines.append("")
    report_lines.append("Longitudinal Coefficients (Cruise)")
    long_header = f"{'Metric':<18} {'Value':>12} {'Units':>9}"
    report_lines.append(long_header)
    report_lines.append("-" * len(long_header))
    report_lines.append(f"{'AOA':<18} {flight_aoa:>12.3f} {'deg':>9}")
    report_lines.append(f"{'HTail AOA':<18} {htail_aoa:>12.3f} {'deg':>9}")
    report_lines.append(f"{'Cl Wing':<18} {cl:>12.5f} {'':>9}")
    report_lines.append(f"{'Cl Tail':<18} {cl_htail:>12.5f} {'':>9}")
    report_lines.append(f"{'Cl Total':<18} {(cl + cl_htail):>12.5f} {'':>9}")
    report_lines.append(f"{'Cd0':<18} {cd0:>12.5f} {'':>9}")
    report_lines.append(f"{'Cd Induced':<18} {cd_induced:>12.5f} {'':>9}")
    report_lines.append(f"{'Cd Total':<18} {cd_total:>12.5f} {'':>9}")
    report_lines.append(f"{'Cm Total':<18} {cm_total:>12.6f} {'':>9}")
    report_lines.append(f"{'Cm Wing CG':<18} {cm_wing_cg:>12.6f} {'':>9}")
    report_lines.append(f"{'Cm Wing Root':<18} {cm_wing_root:>12.6f} {'':>9}")
    report_lines.append("")
    report_lines.append("Cruise Polar (Cl vs Cd)")
    polar_header = f"{'AOA':>7} {'Cl':>10} {'Cd':>10} {'L/D':>9}"
    report_lines.append(polar_header)
    report_lines.append("-" * len(polar_header))
    for aoa, cl_total_i, cd_total_i, ld_i in polar_rows:
        report_lines.append(f"{aoa:>7.2f} {cl_total_i:>10.4f} {cd_total_i:>10.5f} {ld_i:>9.2f}")
    report_lines.append("")
    report_lines.append("Pitching Moments About CG")
    moment_header = f"{'Contributor':<18} {'Moment':>12} {'Units':>9}"
    report_lines.append(moment_header)
    report_lines.append("-" * len(moment_header))
    report_lines.append(f"{'Wing Lift':<18} {wing_moment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Wing Cm':<18} {wing_cm_moment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Tail Lift':<18} {htail_moment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Tail Cm':<18} {htail_cm_moment:>12.4f} {'N*m':>9}")
    report_lines.append(f"{'Total':<18} {total_moment:>12.4f} {'N*m':>9}")
    report_lines.append("")
    report_lines.append("Wing Moment Coefficients")
    cm_header = f"{'Reference':<18} {'Cm':>12} {'Units':>9}"
    report_lines.append(cm_header)
    report_lines.append("-" * len(cm_header))
    report_lines.append(f"{'About CG':<18} {cm_wing_cg:>12.6f} {'':>9}")
    report_lines.append(f"{'About Root QC':<18} {cm_wing_root:>12.6f} {'':>9}")

    result["report_lines"] = report_lines
    return result


def solve_trim(config, flight_aoa_guess, elevator_guess, max_iter=20, tol_force=1e-3, tol_moment=1e-3, elevator_limit_deg=None, aoa_min_deg=None, aoa_max_deg=None):
    aoa = flight_aoa_guess
    elev = elevator_guess
    if elevator_limit_deg is not None:
        elev = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
    if aoa_min_deg is not None:
        aoa = max(aoa, aoa_min_deg)
    if aoa_max_deg is not None:
        aoa = min(aoa, aoa_max_deg)
    converged = False
    last_result = None
    force_tol = max(tol_force, 1e-3 * max(config["weight"], 0.0))
    moment_scale = max(config["weight"], 0.0) * config["wing"]["mac"]
    moment_tol = max(tol_moment, 1e-3 * moment_scale)
    for _ in range(max_iter):
        result = run_analysis(config, aoa, elev, build_report=False)
        last_result = result
        f1 = result["Lift"] - config["weight"]
        f2 = result["TotalPitchMoment"]
        if abs(f1) < force_tol and abs(f2) < moment_tol:
            converged = True
            break
        d_aoa = 0.25
        d_elev = 0.5
        result_aoa = run_analysis(config, aoa + d_aoa, elev, build_report=False)
        result_elev = run_analysis(config, aoa, elev + d_elev, build_report=False)
        f1_aoa = result_aoa["Lift"] - config["weight"]
        f2_aoa = result_aoa["TotalPitchMoment"]
        f1_elev = result_elev["Lift"] - config["weight"]
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
        if elevator_limit_deg is not None:
            elev = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
        if aoa_min_deg is not None:
            aoa = max(aoa, aoa_min_deg)
        if aoa_max_deg is not None:
            aoa = min(aoa, aoa_max_deg)
    return aoa, elev, converged, last_result


def build_aircraft_config(
    ground_level,
    orbit_level,
    flight_velocity,
    weight,
    tail_efficiency,
    elevator_tau,
    downwash_factor,
    cd_misc,
    x_wqc,
    x_htqc,
    x_cg,
    wing_foil,
    wing_root_chord,
    wing_mid_chord,
    wing_mid_chord_pos,
    wing_tip_chord,
    wing_span,
    wing_incidence,
    wing_root_sweep,
    wing_mid_sweep,
    wing_mid_sweep_pos,
    wing_tip_sweep,
    wing_cl_max_takeoff,
    wing_cl_max_cruise,
    wing_cl_max_landing,
    htail_foil,
    htail_root_chord,
    htail_mid_chord,
    htail_mid_chord_pos,
    htail_tip_chord,
    htail_span,
    htail_incidence,
    htail_root_sweep,
    htail_mid_sweep,
    htail_mid_sweep_pos,
    htail_tip_sweep,
    vtail_foil,
    vtail_root_chord,
    vtail_mid_chord,
    vtail_mid_chord_pos,
    vtail_tip_chord,
    vtail_span,
    vtail_incidence,
    vtail_root_sweep,
    vtail_mid_sweep,
    vtail_mid_sweep_pos,
    vtail_tip_sweep,
    vtail_count,
    fuselage_width,
    fuselage_height,
    fuselage_length,
    fuselage_pfactor,
    boom_length,
    boom_diameter,
    boom_count,
    roughness_k,
    fuselage_laminar_frac,
    boom_laminar_frac,
    re_crit_per_m,
    max_power_w,
    prop_eff,
    battery_energy_j,
    max_thrust_n,
    wing_z=0.0,
    htail_z=0.0,
    vtail_z=0.0,
    engine_z=0.0,
    n_span=1000,
):
    fuselage_perimeter = fuselage_height * 2 + fuselage_width * 2 * fuselage_pfactor
    fuselage_diameter = fuselage_perimeter / math.pi

    wing_chord_breaks = [(0.0, wing_root_chord), (wing_mid_chord_pos, wing_mid_chord), (1.0, wing_tip_chord)]
    wing_sweep_breaks = [(0.0, wing_root_sweep), (wing_mid_sweep_pos, wing_mid_sweep), (1.0, wing_tip_sweep)]
    wing_taper = wing_tip_chord / wing_root_chord
    wing_area, wing_mac = planform_area_and_mac(wing_span, wing_chord_breaks, symmetric=True)
    if wing_area <= 0.0 or wing_mac <= 0.0:
        wing_area = 0.5 * (wing_root_chord + wing_tip_chord) * wing_span
        wing_mac = 2.0/3.0 * wing_root_chord * (1 + wing_taper + wing_taper**2)/(1 + wing_taper)
    wing_ar = (wing_span**2) / wing_area if wing_area > 0.0 else 0.0

    htail_chord_breaks = [(0.0, htail_root_chord), (htail_mid_chord_pos, htail_mid_chord), (1.0, htail_tip_chord)]
    htail_sweep_breaks = [(0.0, htail_root_sweep), (htail_mid_sweep_pos, htail_mid_sweep), (1.0, htail_tip_sweep)]
    htail_taper = htail_tip_chord / htail_root_chord
    htail_area, htail_mac = planform_area_and_mac(htail_span, htail_chord_breaks, symmetric=True)
    if htail_area <= 0.0 or htail_mac <= 0.0:
        htail_area = 0.5 * (htail_root_chord + htail_tip_chord) * htail_span
        htail_mac = 2.0/3.0 * htail_root_chord * (1 + htail_taper + htail_taper**2)/(1 + htail_taper)
    htail_ar = (htail_span**2) / htail_area if htail_area > 0.0 else 0.0

    vtail_chord_breaks = [(0.0, vtail_root_chord), (vtail_mid_chord_pos, vtail_mid_chord), (1.0, vtail_tip_chord)]
    vtail_sweep_breaks = [(0.0, vtail_root_sweep), (vtail_mid_sweep_pos, vtail_mid_sweep), (1.0, vtail_tip_sweep)]
    vtail_taper = vtail_tip_chord / vtail_root_chord
    vtail_single_area, vtail_mac = planform_area_and_mac(vtail_span, vtail_chord_breaks, symmetric=False)
    if vtail_single_area <= 0.0 or vtail_mac <= 0.0:
        vtail_single_area = 0.5 * (vtail_root_chord + vtail_tip_chord) * vtail_span
        vtail_mac = 2.0/3.0 * vtail_root_chord * (1 + vtail_taper + vtail_taper**2)/(1 + vtail_taper)
    vtail_area = vtail_single_area * vtail_count
    vtail_ar = (vtail_span**2) / vtail_single_area if vtail_single_area > 0.0 else 0.0

    return {
        "analysis_altitude_m": ground_level + orbit_level,
        "ground_level": ground_level,
        "flight_velocity": flight_velocity,
        "weight": weight,
        "n_span": n_span,
        "aero_funcs": {
            "integrate_profile_drag": integrate_profile_drag,
            "sweep_correction_factor": sweep_correction_factor,
            "integrate_pitching_moment_about_cg": integrate_pitching_moment_about_cg,
            "skin_friction_cf": skin_friction_cf,
        },
        "tail_efficiency": tail_efficiency,
        "elevator_tau": elevator_tau,
        "downwash_factor": downwash_factor,
        "cd_misc": cd_misc,
        "positions": {
            "x_wqc": x_wqc,
            "x_htqc": x_htqc,
            "x_cg": x_cg,
            "z_wing": wing_z,
            "z_htail": htail_z,
            "z_vtail": vtail_z,
            "z_engine": engine_z,
        },
        "roughness": {
            "k": roughness_k,
            "fuselage_lam": fuselage_laminar_frac,
            "boom_lam": boom_laminar_frac,
            "re_crit_per_m": re_crit_per_m,
        },
        "propulsion": {
            "max_power_w": max_power_w,
            "prop_eff": prop_eff,
            "max_thrust_n": max_thrust_n,
            "battery_energy_j": battery_energy_j,
        },
        "wing": {
            "root_chord": wing_root_chord,
            "mid_chord": wing_mid_chord,
            "mid_chord_pos": wing_mid_chord_pos,
            "tip_chord": wing_tip_chord,
            "span": wing_span,
            "incidence": wing_incidence,
            "foil": wing_foil,
            "area": wing_area,
            "mac": wing_mac,
            "ar": wing_ar,
            "taper": wing_taper,
            "cl_max_takeoff": wing_cl_max_takeoff,
            "cl_max_cruise": wing_cl_max_cruise,
            "cl_max_landing": wing_cl_max_landing,
            "chord_breaks": wing_chord_breaks,
            "sweep_breaks": wing_sweep_breaks,
        },
        "htail": {
            "root_chord": htail_root_chord,
            "mid_chord": htail_mid_chord,
            "mid_chord_pos": htail_mid_chord_pos,
            "tip_chord": htail_tip_chord,
            "span": htail_span,
            "incidence": htail_incidence,
            "foil": htail_foil,
            "area": htail_area,
            "mac": htail_mac,
            "ar": htail_ar,
            "chord_breaks": htail_chord_breaks,
            "sweep_breaks": htail_sweep_breaks,
        },
        "vtail": {
            "root_chord": vtail_root_chord,
            "mid_chord": vtail_mid_chord,
            "mid_chord_pos": vtail_mid_chord_pos,
            "tip_chord": vtail_tip_chord,
            "span": vtail_span,
            "incidence": vtail_incidence,
            "foil": vtail_foil,
            "count": vtail_count,
            "area": vtail_area,
            "mac": vtail_mac,
            "ar": vtail_ar,
            "chord_breaks": vtail_chord_breaks,
            "sweep_breaks": vtail_sweep_breaks,
        },
        "fuselage": {
            "length": fuselage_length,
            "diameter": fuselage_diameter,
        },
        "boom": {
            "length": boom_length,
            "diameter": boom_diameter,
            "count": boom_count,
        },
    }
