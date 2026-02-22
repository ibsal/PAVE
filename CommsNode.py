from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet
import math
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

G = 9.80665

# --- Unit conversion constants ---
KNOTS_PER_MPS           = 1.9438444924406
KNOT_TO_MPS             = 1.0 / KNOTS_PER_MPS
MPS_TO_KNOT             = KNOTS_PER_MPS
LBF_PER_NEWTON          = 0.2248089431
IN_PER_M                = 39.37007874015748
LB_PER_KG               = 2.2046226218487757
IN2_PER_M2              = IN_PER_M * IN_PER_M
LB_PER_IN2_PER_KG_PER_M2 = LB_PER_KG / IN2_PER_M2
MI_PER_NM               = 1.1507794480235425
M_TO_FT                 = 3.28084
FT_PER_MIN_PER_MPS      = M_TO_FT * 60.0


# --- Helper functions ---

def re(altitude, velocity, l):
    return (l * velocity) / Atmosphere(altitude).kinematic_viscosity[0]


def skin_friction_cf(re, roughness, length, laminar_frac=0.0, re_crit_per_m=5e5):
    """Cf calculation for drag buildup."""
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


class Wing:
    def __init__(self, airfoil: PolarSet, altitude, velocity, span,
                 rootChord, midChord, tipChord, midChordPosition,
                 rootSweep, midSweep, tipSweep, midSweepPosition,
                 incidence, aoa, symmetric, xqc, weight, arealDensity):
        self.airfoil = airfoil
        self.span = span
        self.rootChord = rootChord
        self.midChord = midChord
        self.tipChord = tipChord
        self.midChordPosition = midChordPosition
        self.rootSweep = rootSweep
        self.midSweep = midSweep
        self.tipSweep = tipSweep
        self.midSweepPosition = midSweepPosition
        self.incidence = incidence
        self.symmetric = symmetric
        self.xqc = xqc
        self.arealDensity = arealDensity

        self.area = 2.0 * (
            0.5 * (self.rootChord + self.midChord) * (self.midChordPosition * 0.5 * self.span)
            + 0.5 * (self.midChord + self.tipChord) * (0.5 * self.span - self.midChordPosition * 0.5 * self.span)
        )
        if not symmetric:
            self.area = self.area / 2.0
        self.mac = (2.0 * (
            ((self.midChordPosition * 0.5 * self.span) / 3.0)
            * (self.rootChord**2 + self.rootChord * self.midChord + self.midChord**2)
            + ((0.5 * self.span - self.midChordPosition * 0.5 * self.span) / 3.0)
            * (self.midChord**2 + self.midChord * self.tipChord + self.tipChord**2)
        )) / self.area
        self.ar = self.span**2 / self.area
        self.taper = self.tipChord / self.rootChord
        self.e_oswald_w = 1.78 * (1.0 - 0.045 * (self.ar**0.68)) - 0.64
        self.e_oswald_w = min(max(self.e_oswald_w, 0.3), 0.95)
        self.mass = self.arealDensity * self.area

    def cl2d(self, alpha_deg, reynold):
        return self.airfoil.cl(alpha_deg=alpha_deg, reynolds=reynold)

    def cd2d(self, alpha_deg, reynold):
        return self.airfoil.cd(alpha_deg=alpha_deg, reynolds=reynold)

    def cm2d(self, alpha_deg, reynold):
        return self.airfoil.cm(alpha_deg=alpha_deg, reynolds=reynold)

    def forces(self, xref, altitude, velocity, aoa, n=50):
        # Coerce AoA to scalar to avoid mixed float/array coeffs during optimization
        aoa = float(np.asarray(aoa).ravel()[0])
        hspan = self.span / 2.0 if self.symmetric else self.span
        dx = hspan / n

        # --- Vectorized geometry ---
        s_arr = (np.arange(n) + 0.5) * dx

        # Local chord (piecewise linear, safe denominators)
        mid_s      = hspan * self.midChordPosition
        cslope_in  = (self.midChord - self.rootChord) / max(self.midChordPosition * hspan, 1e-30)
        cslope_out = (self.tipChord - self.midChord)  / max((1.0 - self.midChordPosition) * hspan, 1e-30)
        clocal = np.where(
            s_arr <= mid_s,
            s_arr * cslope_in  + self.rootChord,
            (s_arr - mid_s) * cslope_out + self.midChord,
        )

        # Local sweep (piecewise linear, safe denominators)
        mid_sw     = hspan * self.midSweepPosition
        sslope_in  = (self.midSweep - self.rootSweep) / max(self.midSweepPosition * hspan, 1e-30)
        sslope_out = (self.tipSweep - self.midSweep)  / max((1.0 - self.midSweepPosition) * hspan, 1e-30)
        slocal = np.where(
            s_arr <= mid_sw,
            s_arr * sslope_in  + self.rootSweep,
            (s_arr - mid_sw) * sslope_out + self.midSweep,
        )

        # Quarter-chord running x-position (cumulative sum, equivalent to original loop)
        tan_s  = np.tan(np.radians(slocal))
        cos_s  = np.cos(np.radians(slocal))
        cumtan = np.empty(n, dtype=float)
        cumtan[0] = 0.0
        cumtan[1:] = np.cumsum(tan_s[:-1])
        xqcmid    = self.xqc + dx * (cumtan + 0.5 * tan_s)
        momentarm = xref - xqcmid

        # --- Atmosphere: single object creation for both density and kinematic viscosity ---
        atm     = Atmosphere(altitude)
        density = float(atm.density[0])
        nu      = float(atm.kinematic_viscosity[0])

        veff    = velocity * cos_s
        relocal = veff * clocal / nu   # Reynolds per station

        # --- Aerodynamic coefficients ---
        # Evaluate each polar at the (fixed) effective alpha once, then Re-interpolate
        # across all stations with a single vectorised np.interp call.
        # Reduces alpha lookups from n*n_polars to just n_polars.
        alpha_eff = aoa + self.incidence
        re_nodes  = self.airfoil.reynolds_list                              # ascending Re
        cl_nodes  = np.array([p.cl_at(alpha_eff)       for p in self.airfoil.polars])
        cd_nodes  = np.array([p.cd_at(alpha_eff)       for p in self.airfoil.polars])
        cm_nodes  = np.array([p.coeff("cm", alpha_eff) for p in self.airfoil.polars])

        cllocal = np.interp(relocal, re_nodes, cl_nodes)   # clamps at endpoints
        cdlocal = np.interp(relocal, re_nodes, cd_nodes)
        cmlocal = np.interp(relocal, re_nodes, cm_nodes)

        # --- Integrate forces (fully vectorised) ---
        qeff   = 0.5 * density * veff**2
        drag   = float(np.sum(qeff * cdlocal * dx * clocal))
        lift   = float(np.sum(qeff * cllocal * dx * clocal * cos_s))
        moment = float(np.sum(  cmlocal   * qeff * dx * clocal**2
                              + momentarm * qeff * cllocal * dx * clocal * cos_s))

        if self.symmetric:
            drag   *= 2.0
            lift   *= 2.0
            moment *= 2.0

        aq = 0.5 * density * velocity**2 * self.area
        cleq  = lift / aq
        drag += aq * (cleq**2) / (math.pi * self.ar * self.e_oswald_w)

        downwash = 57.2958 * cleq / (math.pi * self.ar * self.e_oswald_w)
        return [drag, lift, moment, downwash]

    def stallCondition(self, altitude, weight, v0=20.0):
        density = Atmosphere(altitude).density[0]
        vcand = v0
        clmax = 0.0
        stall_aoa = 0.0
        for _ in range(50):
            qa = 0.5 * density * vcand**2 * self.area
            neg_l = lambda x: -self.forces(10, altitude, vcand, x)[1]
            stall_aoa = float(scipy.optimize.fmin(neg_l, 2, xtol=0.001, disp=False)[0])
            lmax = -1 * neg_l(stall_aoa)
            clmax = lmax / qa
            vol = vcand
            vcand = math.sqrt((2 * weight) / (density * self.area * clmax))
            if abs(vcand - vol) <= 0.01:
                break
        return vcand, clmax, stall_aoa

    def stallSpeed(self, altitude, weight, v0=20.0):
        vcand, clmax, _stall_aoa = self.stallCondition(altitude, weight, v0=v0)
        return vcand, clmax


class HorizontalTail(Wing):
    def __init__(
        self,
        airfoil: PolarSet,
        altitude,
        velocity,
        span,
        rootChord,
        midChord,
        tipChord,
        midChordPosition,
        rootSweep,
        midSweep,
        tipSweep,
        midSweepPosition,
        incidence,
        aoa,
        symmetric,
        xqc,
        weight,
        arealDensity,
        *,
        elevatorDeflection: float = 0.0,
        elevatorTau: float = 0.35,
        cd_deltae_k: float = 0.0,
    ):
        super().__init__(
            airfoil, altitude, velocity, span,
            rootChord, midChord, tipChord, midChordPosition,
            rootSweep, midSweep, tipSweep, midSweepPosition,
            incidence, aoa, symmetric, xqc, weight, arealDensity,
        )
        self.elevatorTau = elevatorTau
        self.cd_deltae_k = cd_deltae_k
        self.elevatorDeflection = elevatorDeflection

    def forces(self, xref, altitude, velocity, aoa, n=100, downwash=0.0, elevator=0.0):
        aoa_eff = aoa - downwash + self.elevatorTau * elevator
        return super().forces(xref, altitude, velocity, aoa_eff, n=n)


class VerticalTail(Wing):
    def __init__(
        self,
        airfoil: PolarSet,
        altitude,
        velocity,
        height,              # single fin height
        rootChord,
        midChord,
        tipChord,
        midChordPosition,
        rootSweep,
        midSweep,
        tipSweep,
        midSweepPosition,
        incidence,
        beta,                # sideslip angle (deg) stored in aoa slot
        xqc,
        weight,
        arealDensity,
        eta=1.0,
    ):
        super().__init__(
            airfoil, altitude, velocity, 2.0 * height,
            rootChord, midChord, tipChord, midChordPosition,
            rootSweep, midSweep, tipSweep, midSweepPosition,
            incidence, beta, True, xqc, weight, 2.0 * arealDensity,
        )
        self.eta = eta

    def forces(self, xref, altitude, velocity, beta, n=100):
        out = super().forces(xref, altitude, velocity, beta, n=n)
        out[0] *= self.eta
        out[1] *= self.eta
        out[2] *= self.eta
        return out


class Fuselage:
    def __init__(self, length, width, height, pfactor, roughness, laminarfraction, qfactor=1.0, quantity=1):
        self.length = length
        self.width = width
        self.height = height
        self.pfactor = pfactor
        self.roughness = roughness
        self.laminarfraction = laminarfraction
        self.perimeter = (2 * self.width + 2 * self.height) * pfactor
        self.diameter = self.perimeter / math.pi
        self.qfactor = qfactor
        self.quantity = quantity

    def drag(self, altitude, velocity):
        rlocal = re(altitude, velocity, self.length)
        fratio = self.length / self.diameter
        ff = (1 + 60 / fratio**3) + fratio / 400
        cfc = skin_friction_cf(rlocal, self.roughness, self.length, self.laminarfraction)
        fuse_term = max(1.0 - 2.0 / max(fratio, 1e-6), 0.0)
        swet = (math.pi * self.diameter * self.length
                * math.pow(fuse_term, 2.0 / 3.0) * (1 + 1 / (fratio**2)))
        cdo = ff * self.qfactor * cfc
        return cdo * 0.5 * Atmosphere(altitude).density[0] * velocity**2 * swet


class Powerplant:
    def __init__(self, bcap, neff, pmax):
        self.full = bcap
        self.bcap = bcap
        self.neff = neff
        self.pmax = pmax

    def drawBattery(self, duration, power):
        self.bcap -= duration * power
        return self.bcap

    def validThrust(self, thrust, velocity):
        if (thrust / self.neff) / velocity >= self.pmax:
            return False
        return True


class Aircraft:
    def __init__(self, altitude, velocity, pplant: Powerplant, mwing: Wing,
                 hwing: HorizontalTail, vtail: VerticalTail, fuselages: list[Fuselage],
                 aoa, trim, xcg, weight, cdomisc):
        self.altitude = altitude
        self.velocity = velocity
        self.mwing = mwing
        self.hwing = hwing
        self.vtail = vtail
        self.fuselages = fuselages
        self.pplant = pplant
        self.aoa = aoa
        self.trim = trim
        self.xcg = xcg
        self.weight = weight
        self.thrust = None
        self.power = None
        self.cdomisc = cdomisc

    def sumFanddM(self, res=100):
        wForce = self.mwing.forces(self.xcg, self.altitude, self.velocity, self.aoa, n=res)
        hForce = self.hwing.forces(self.xcg, self.altitude, self.velocity, self.aoa,
                                   downwash=wForce[3], elevator=self.trim, n=res)

        vDrag = 0.0
        if self.vtail is not None:
            vForce = self.vtail.forces(self.xcg, self.altitude, self.velocity, 0.0, n=res)
            vDrag = float(vForce[0])

        fDrag = 0.0
        for f in self.fuselages:
            fDrag += float(f.drag(self.altitude, self.velocity)) * float(getattr(f, "quantity", 1))

        Drag = (float(wForce[0]) + float(hForce[0]) + vDrag + fDrag
                + 0.5 * Atmosphere(self.altitude).density[0] * self.velocity**2
                * self.cdomisc * self.mwing.area)
        Lift = float(wForce[1]) + float(hForce[1])
        Moment = float(wForce[2]) + float(hForce[2])
        return [Drag, Lift, Moment]

    def solveTrim(self, alpha0=None, de0=None, res=100, _return_forces=False):
        if alpha0 is None:
            alpha0 = float(self.aoa)
        if de0 is None:
            de0 = float(self.trim)

        # Cache the forces from the last residual evaluation so callers can reuse
        # them without an extra sumFanddM call at the converged point.
        _last = [None]

        def residual(x):
            self.aoa  = float(x[0])
            self.trim = float(x[1])
            mf = self.sumFanddM(res=res)
            _last[0] = mf
            return np.array([float(mf[1] - self.weight), float(mf[2])], dtype=float)

        sol = scipy.optimize.root(residual, x0=np.array([alpha0, de0], dtype=float), method="hybr")
        if not sol.success:
            return ([None, None], None) if _return_forces else [None, None]

        self.trim = float(sol.x[1])
        self.aoa  = float(sol.x[0])

        if not _return_forces:
            return [self.aoa, self.trim]

        # hybr's last residual call is at sol.x; reuse cached forces when possible
        if (_last[0] is not None
                and abs(self.aoa  - float(sol.x[0])) < 1e-12
                and abs(self.trim - float(sol.x[1])) < 1e-12):
            forces = _last[0]
        else:
            forces = self.sumFanddM(res=res)
        return [self.aoa, self.trim], forces

    def solveBestVelocity(self, levelFlightMargin, vguess=20, res=100, mode="optimal"):
        stall = self.mwing.stallSpeed(self.altitude, self.weight, v0=vguess)[0]
        vmin = float(stall * levelFlightMargin)
        vmax = max(vmin * 2.0, vmin + 1.0)
        mode_norm = str(mode).strip().lower()
        reference_state = (float(self.velocity), float(self.aoa), float(self.trim))

        seed_candidates = [
            (reference_state[1], reference_state[2]),
            (reference_state[1], 0.0),
            (0.0, 0.0),
            (2.0, 0.0),
            (-2.0, 0.0),
        ]
        trim_seeds = []
        seen = set()
        for alpha0, de0 in seed_candidates:
            key = (round(float(alpha0), 6), round(float(de0), 6))
            if key in seen:
                continue
            seen.add(key)
            trim_seeds.append((float(alpha0), float(de0)))

        def best_trimmed_solution(v):
            velocity = float(v)
            call_state = (float(self.velocity), float(self.aoa), float(self.trim))
            best = None
            try:
                for alpha0, de0 in trim_seeds:
                    self.velocity = velocity
                    self.aoa = float(alpha0)
                    self.trim = float(de0)
                    sol, forces = self.solveTrim(alpha0=alpha0, de0=de0, res=res, _return_forces=True)
                    if sol == [None, None]:
                        continue
                    drag = float(forces[0])
                    power = float(drag) * velocity
                    if not (np.isfinite(drag) and np.isfinite(power) and power > 0.0):
                        continue
                    best = (power, drag, float(sol[0]), float(sol[1]))
                    break  # seeds are ordered best-first; stop at first success
            finally:
                self.velocity, self.aoa, self.trim = call_state
            return best

        def power(v):
            best = best_trimmed_solution(v)
            return float("inf") if best is None else float(best[0])

        if mode_norm == "stall_margin":
            selected_velocity = vmin
        elif mode_norm == "optimal":
            sweep_v = np.linspace(vmin, vmax, 21)
            sweep_p = np.array([power(v) for v in sweep_v], dtype=float)
            valid = np.isfinite(sweep_p)
            if not np.any(valid):
                selected_velocity = vmin
            else:
                best_idx = int(np.argmin(np.where(valid, sweep_p, np.inf)))
                selected_velocity = float(sweep_v[best_idx])
                left = float(sweep_v[max(best_idx - 1, 0)])
                right = float(sweep_v[min(best_idx + 1, sweep_v.size - 1)])
                if right - left > 1e-6:
                    opt_result = scipy.optimize.minimize_scalar(
                        power, bounds=(left, right), method="bounded",
                        options={"xatol": 0.05},
                    )
                    if opt_result.success:
                        candidate_velocity = float(opt_result.x)
                        candidate_power = power(candidate_velocity)
                        if np.isfinite(candidate_power) and candidate_power <= float(sweep_p[best_idx]):
                            selected_velocity = candidate_velocity
        else:
            raise ValueError(f"Unsupported cruise speed mode '{mode}'. Use 'optimal' or 'stall_margin'.")

        self.velocity = float(selected_velocity)
        best_final = best_trimmed_solution(self.velocity)
        if best_final is None:
            self.thrust = float("inf")
            self.power = float("inf")
            self.aoa = reference_state[1]
            self.trim = reference_state[2]
            return [float(self.velocity), self.power, self.thrust]
        self.power = float(best_final[0])
        self.thrust = float(best_final[1])
        self.aoa = float(best_final[2])
        self.trim = float(best_final[3])
        if self.power > self.pplant.pmax:
            print("WARNING: aircraft power exceeds peak power plant power")
        return [float(self.velocity), self.power, self.thrust]

    def cm_alpha(self, dalpha=0.25):
        aoa0 = self.aoa
        trim0 = self.trim
        self.aoa = aoa0 + dalpha
        m_p = self.sumFanddM()[2]
        self.aoa = aoa0 - dalpha
        m_m = self.sumFanddM()[2]
        self.aoa = aoa0
        self.trim = trim0
        rho = Atmosphere(self.altitude).density[0]
        qS = 0.5 * rho * self.velocity**2 * self.mwing.area
        cbar = self.mwing.mac
        return (m_p - m_m) / (2 * dalpha) / (qS * cbar)

    def cl_alpha(self, dalpha=0.25):
        aoa0 = self.aoa
        trim0 = self.trim
        self.aoa = aoa0 + dalpha
        L_p = self.sumFanddM()[1]
        self.aoa = aoa0 - dalpha
        L_m = self.sumFanddM()[1]
        self.aoa = aoa0
        self.trim = trim0
        rho = Atmosphere(self.altitude).density[0]
        qS = 0.5 * rho * self.velocity**2 * self.mwing.area
        return (L_p - L_m) / (2 * dalpha) / qS

    def staticMargin(self):
        return -self.cm_alpha() / self.cl_alpha()

    def horizontalTailVolume(self, xhtqc):
        return (self.hwing.area * (xhtqc - self.xcg)) / (self.mwing.area * self.mwing.mac)

    def verticalTailVolume(self, xvtqc):
        return (self.vtail.area * (xvtqc - self.xcg)) / (self.mwing.area * self.mwing.span)


# --- Design-point configuration ---
altitude          = 200
wingFoil          = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
tailFoil          = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
arealDensityMain  = 3.05   # kg/m^2
arealDensityH     = 1.5    # kg/m^2
arealDensityV     = 1.5    # kg/m^2
baseMass          = 17.5   # kg
boomMassFixed     = 0.0    # kg
boomLengthMin     = 0.05   # m
cdomisc           = 0.015
xcg               = 0.35
boomMassPerM      = 0.4

body             = Fuselage(1.0541, 0.3048, 0.21336, 0.9, 0.00635e-3, 0.3)
booms            = Fuselage(1.4, 0.03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric  = Powerplant(7992000, 0.59, 4000)
fuselages        = [body, booms, booms]

design_point = x = [
    4.4914820,  # wingSpan (m)   = 176.830 in
    0.2614676,  # wingChord (m)  = 10.294 in
    0.3048,  # xwqc (m)       = 12.00 in
    0.8067548,  # hSpan (m)      = 31.762 in
    0.1953260,  # hChord (m)     = 7.690 in
    1.5046198,  # xhtqc (m)      = 59.237 in
    1.481,      # wingIncidence (deg)
    1.800,      # tailIncidence (deg)
]
default_wing_incidence = 1.964
default_tail_incidence = -0.1864
default_vtail_volume   = 0.03


def _vtail_from_volume(wing_span, wing_chord, xhtqc, vtail_volume, xcg_value=None):
    if xcg_value is None:
        xcg_value = xcg
    wing_area = float(wing_span) * float(wing_chord)
    tail_arm = float(xhtqc) - float(xcg_value)
    if tail_arm <= 0.0:
        raise ValueError("xhtqc must be greater than xcg for vertical tail sizing")
    vtail_area = (float(vtail_volume) * wing_area * float(wing_span)) / tail_arm
    v_chord = math.sqrt(vtail_area / 2.0) if vtail_area > 0.0 else 0.0
    v_height = v_chord
    return v_height, v_chord


def _normalize_design_point(dp):
    wing_inc     = default_wing_incidence
    tail_inc     = default_tail_incidence
    vtail_volume = default_vtail_volume

    if isinstance(dp, dict):
        if "xbest" in dp:
            dp = dp["xbest"]
        if isinstance(dp, dict):
            if "vtailVolume" in dp:
                vtail_volume = float(dp["vtailVolume"])
            if "wingIncidence" in dp:
                wing_inc = float(dp["wingIncidence"])
            if "tailIncidence" in dp:
                tail_inc = float(dp["tailIncidence"])
            if all(k in dp for k in ("wingSpan", "wingChord", "xwqc", "hSpan", "hChord", "xhtqc")):
                wing_span  = float(dp["wingSpan"])
                wing_chord = float(dp["wingChord"])
                xwqc       = float(dp["xwqc"])
                h_span     = float(dp["hSpan"])
                h_chord    = float(dp["hChord"])
                xhtqc      = float(dp["xhtqc"])
                return ([wing_span, wing_chord, xwqc, h_span, h_chord, xhtqc, wing_inc, tail_inc],
                        wing_inc, tail_inc, vtail_volume)

    arr = np.asarray(dp, dtype=float).reshape(-1)
    if arr.size >= 8:
        wing_span, wing_chord, xwqc, h_span, h_chord, xhtqc, wing_inc, tail_inc = map(float, arr[:8])
        return ([wing_span, wing_chord, xwqc, h_span, h_chord, xhtqc, wing_inc, tail_inc],
                wing_inc, tail_inc, vtail_volume)
    if arr.size == 6:
        wing_span, wing_chord, xwqc, h_span, h_chord, xhtqc = map(float, arr[:6])
        return ([wing_span, wing_chord, xwqc, h_span, h_chord, xhtqc, wing_inc, tail_inc],
                wing_inc, tail_inc, vtail_volume)

    raise ValueError("design point must contain at least 6 values")


def _coerce_design_point(dp):
    xvec, wing_inc, tail_inc, _ = _normalize_design_point(dp)
    return xvec, wing_inc, tail_inc


design_point, wing_incidence, tail_incidence = _coerce_design_point(design_point)


def build_aircraft(x):
    x, wingIncidence, tailIncidence, vtailVolume = _normalize_design_point(x)
    wingSpan      = float(x[0])
    wingChord     = float(x[1])
    xwqc          = float(x[2])
    hSpan         = float(x[3])
    hChord        = float(x[4])
    xhtqc         = float(x[5])
    wingIncidence = float(x[6])
    tailIncidence = float(x[7])
    xvtqc         = xhtqc
    vHeight, vChord = _vtail_from_volume(wingSpan, wingChord, xhtqc, vtailVolume)

    mainWing = Wing(
        wingFoil, altitude, 0.0,
        wingSpan, wingChord, wingChord, wingChord, 0.5,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, True,
        xwqc, 0.0, arealDensityMain,
    )

    hwing = HorizontalTail(
        tailFoil, altitude, 0.0,
        hSpan, hChord, hChord, hChord, 0.5,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, True,
        xhtqc, 0.0, arealDensityH,
        elevatorDeflection=0.0,
        elevatorTau=0.35,
        cd_deltae_k=0.0,
    )

    vwing = VerticalTail(
        tailFoil, altitude, 0.0,
        vHeight, vChord, vChord, vChord, 0.5,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0,
        xvtqc, 0.0, arealDensityV,
        eta=2.0,
    )

    hwing.incidence    = tailIncidence
    mainWing.incidence = wingIncidence

    boomLength = max(float(xhtqc - xwqc), float(boomLengthMin))
    boomMass   = float(boomMassFixed) + float(boomMassPerM) * float(boomLength)
    fuselages_local = []
    for f in fuselages:
        if (isinstance(f, Fuselage)
                and getattr(f, "width", None) is not None
                and getattr(f, "height", None) is not None):
            if float(getattr(f, "width")) <= 0.05 and float(getattr(f, "height")) <= 0.05:
                fnew = Fuselage(
                    boomLength,
                    float(f.width), float(f.height),
                    float(f.pfactor), float(f.roughness), float(f.laminarfraction),
                    float(getattr(f, "qfactor", 1.0)), int(getattr(f, "quantity", 1)),
                )
                fuselages_local.append(fnew)
            else:
                fuselages_local.append(f)
        else:
            fuselages_local.append(f)

    totalMass = (float(baseMass) + float(mainWing.mass) + float(hwing.mass)
                 + float(vwing.mass) + float(boomMass))
    weight = totalMass * G

    commsNode = Aircraft(
        altitude, 20.0, batteryElectric,
        mainWing, hwing, vwing, fuselages_local,
        0.0, 0.0, xcg, weight, cdomisc,
    )
    return commsNode, totalMass


# --- Mission parameters ---
mission_systems_power_w = 50.0
landing_margin          = 0.20
battery_capacity_wh     = 2220.0


# --- Analysis helpers ---

def evaluate_ld_at_speed(aircraft, target_velocity):
    saved_state = (aircraft.velocity, aircraft.aoa, aircraft.trim)
    aircraft.velocity = float(target_velocity)
    ld = float("nan")
    try:
        sol = aircraft.solveTrim()
        if sol != [None, None]:
            forces = aircraft.sumFanddM()
            if forces[0]:
                ld = forces[1] / forces[0]
    except Exception:
        pass
    finally:
        aircraft.velocity, aircraft.aoa, aircraft.trim = saved_state
    return ld


def drag_polar_at_speed(aircraft, speed, aoa_samples, trim_guess=None, retrim=False):
    saved_state = (aircraft.velocity, aircraft.aoa, aircraft.trim)
    aircraft.velocity = float(speed)
    if trim_guess is None:
        trim_guess = saved_state[2]
    trim_guess = float(trim_guess)
    aircraft.trim = trim_guess
    results = []
    for aoa in aoa_samples:
        aircraft.aoa = float(aoa)
        try:
            if retrim:
                def moment_sq(t):
                    aircraft.trim = float(t)
                    m = aircraft.sumFanddM()[2]
                    return m * m
                res = scipy.optimize.minimize_scalar(
                    moment_sq, bounds=(-10.0, 10.0), method="bounded",
                    options={"xatol": 0.05},
                )
                aircraft.trim = float(res.x)
            else:
                aircraft.trim = trim_guess
            forces = aircraft.sumFanddM()
            results.append((float(aoa), float(aircraft.trim), float(forces[0]), float(forces[1])))
        except Exception:
            results.append((float(aoa), float("nan"), float("nan"), float("nan")))
    aircraft.velocity, aircraft.aoa, aircraft.trim = saved_state
    return results


def evaluate_endurance_at_speed(aircraft, target_velocity, mission_power_w, available_energy_wh, res=80):
    saved_state = (aircraft.velocity, aircraft.aoa, aircraft.trim)
    aircraft.velocity = float(target_velocity)
    endurance_h = float("nan")
    try:
        sol = aircraft.solveTrim(res=res)
        if sol != [None, None]:
            drag_force     = float(aircraft.sumFanddM(res=res)[0])
            pwr_propulsive = drag_force * float(aircraft.velocity)
            if (np.isfinite(pwr_propulsive) and pwr_propulsive > 0.0
                    and aircraft.pplant.neff > 0.0
                    and pwr_propulsive <= aircraft.pplant.pmax):
                total_power = (pwr_propulsive / aircraft.pplant.neff) + float(mission_power_w)
                if np.isfinite(total_power) and total_power > 0.0:
                    endurance_h = float(available_energy_wh) / total_power
    except Exception:
        pass
    finally:
        aircraft.velocity, aircraft.aoa, aircraft.trim = saved_state
    return endurance_h


def _run_design_point():
    # Cruise speed selection:
    #   "optimal"      -> minimize required propulsive power above stall margin
    #   "stall_margin" -> use exactly (cruise_stall_margin * Vstall)
    cruise_velocity_mode = "stall_margin"
    cruise_stall_margin  = 1.25
    plots            = False
    plot_ld_vs_alpha = True
    plot_cl_cd_polar = True

    # Required stall margins (V / Vs)
    stall_margin_takeoff_v2_req       = 1.20
    stall_margin_takeoff_climb_req    = 1.25
    stall_margin_level_flight_req     = 1.25
    stall_margin_landing_approach_req = 1.30

    epicairplane = build_aircraft(design_point)[0]
    vbest, pwr, thrust = epicairplane.solveBestVelocity(
        cruise_stall_margin,
        mode=cruise_velocity_mode,
    )

    forces = epicairplane.sumFanddM()
    drag, lift, moment = forces[0], forces[1], forces[2]
    rho = Atmosphere(altitude).density[0]
    q   = 0.5 * rho * vbest**2
    cruise_cl = lift / (q * epicairplane.mwing.area) if q > 0 else float("nan")
    moment_coefficient = (
        moment / (q * epicairplane.mwing.area * epicairplane.mwing.mac)
        if q > 0 and epicairplane.mwing.mac > 0 else float("nan")
    )
    stall_speed, clmax, stall_aoa = epicairplane.mwing.stallCondition(
        epicairplane.altitude, epicairplane.weight, v0=vbest,
    )
    power_available       = epicairplane.pplant.pmax
    propulsive_power_elec = (
        pwr / epicairplane.pplant.neff if epicairplane.pplant.neff > 0 else float("inf")
    )
    excess_power   = power_available - pwr
    power_fraction = pwr / power_available if power_available else float("nan")
    static_margin  = epicairplane.staticMargin()
    cm_alpha       = epicairplane.cm_alpha()
    cl_alpha       = epicairplane.cl_alpha()
    h_tail_volume  = epicairplane.horizontalTailVolume(design_point[5])
    v_tail_volume  = epicairplane.verticalTailVolume(design_point[5])

    v_knots     = vbest * KNOTS_PER_MPS
    stall_knots = stall_speed * KNOTS_PER_MPS
    drag_lbf    = drag * LBF_PER_NEWTON
    lift_lbf    = lift * LBF_PER_NEWTON
    thrust_lbf  = thrust * LBF_PER_NEWTON
    propulsive_power_fraction = propulsive_power_elec / power_available if power_available else float("nan")
    total_power_w       = propulsive_power_elec + mission_systems_power_w
    available_energy_wh = battery_capacity_wh * (1.0 - landing_margin)
    flight_time_h       = available_energy_wh / total_power_w if total_power_w > 0.0 else 0.0
    flight_time_min     = flight_time_h * 60.0
    flight_distance_nm  = flight_time_h * v_knots
    flight_distance_mi  = flight_distance_nm * MI_PER_NM
    climb_rate_mps      = excess_power / epicairplane.weight
    climb_rate_fpm      = climb_rate_mps * FT_PER_MIN_PER_MPS
    ld_at_90  = evaluate_ld_at_speed(epicairplane, vbest * 0.9)
    ld_at_110 = evaluate_ld_at_speed(epicairplane, vbest * 1.1)

    print("Cruise summary:")
    print(f"  Velocity: {v_knots:.1f} kt")
    print(f"  Stall speed: {stall_knots:.1f} kt, Cl_max: {clmax:.3f}, Stall AoA: {stall_aoa:.3f} deg")
    print(f"  AoA: {epicairplane.aoa:.3f} deg, Trim: {epicairplane.trim:.3f} deg")
    print(f"  Lift: {lift_lbf:.1f} lbf, Drag: {drag_lbf:.1f} lbf, L/D: {lift/drag if drag else float('nan'):.3f}")
    print(f"  Thrust: {thrust_lbf:.1f} lbf")
    print(f"  Cruise Cl: {cruise_cl:.3f}")
    print("Performance & climb:")
    print(f"  Power required: {pwr:.2f} W, Propulsive electrical draw: {propulsive_power_elec:.2f} W ({epicairplane.pplant.neff:.3f} efficiency)")
    print(f"  Power available: {power_available:.2f} W, Excess power: {excess_power:.2f} W")
    print(f"  Power fraction (required / available): {power_fraction:.3f}")
    print(f"  Power fraction (propulsive electrical / available): {propulsive_power_fraction:.3f}")
    print(f"  Climb/sink rate from excess power: {climb_rate_fpm:.0f} ft/min")
    print(f"  Mission systems power: {mission_systems_power_w:.1f} W, Landing margin: {landing_margin*100:.0f}%")
    print("Off-design L/D:")
    print(f"  90% cruise ({0.9*v_knots:.1f} kt): {ld_at_90:.3f}")
    print(f"  110% cruise ({1.1*v_knots:.1f} kt): {ld_at_110:.3f}")
    print("Endurance:")
    print(f"  Power including mission systems: {total_power_w:.2f} W")
    print(f"  Battery energy payload: {available_energy_wh:.1f} Wh (from {battery_capacity_wh:.0f} Wh capacity)")
    print(f"  Flight time: {flight_time_h:.2f} h ({flight_time_min:.0f} min)")
    print(f"  Range: {flight_distance_nm:.1f} nm ({flight_distance_mi:.1f} mi)")
    print("Stability:")
    print(f"  Static margin: {static_margin:.3f}, Cm_alpha: {cm_alpha:.3f}, Cl_alpha: {cl_alpha:.3f}")
    print(f"  Horizontal tail volume: {h_tail_volume:.3f}, Vertical tail volume: {v_tail_volume:.3f}")
    print(f"  Moment coefficient (cruise): {moment_coefficient:.4f}")

    print("Design parameters:")
    print(f"  CG x-position: {epicairplane.xcg * IN_PER_M:.3f} in")
    mw = epicairplane.mwing
    print("Main wing:")
    print(f"  Span: {mw.span * IN_PER_M:.3f} in, Area: {mw.area * IN2_PER_M2:.3f} in^2, AR: {mw.ar:.3f}, MAC: {mw.mac * IN_PER_M:.3f} in, Taper: {mw.taper:.3f}")
    print(f"  Chords (root/mid/tip): {mw.rootChord * IN_PER_M:.3f} / {mw.midChord * IN_PER_M:.3f} / {mw.tipChord * IN_PER_M:.3f} in, Mid-chord position: {mw.midChordPosition * IN_PER_M:.3f} in")
    print(f"  Sweeps (root/mid/tip): {mw.rootSweep:.3f} / {mw.midSweep:.3f} / {mw.tipSweep:.3f} deg, Mid-sweep position: {mw.midSweepPosition:.3f}")
    print(f"  Incidence: {mw.incidence:.3f} deg, xqc: {mw.xqc * IN_PER_M:.3f} in, Symmetric: {mw.symmetric}, Areal density: {mw.arealDensity * LB_PER_IN2_PER_KG_PER_M2:.3f} lb/in^2")
    print(f"  Mass: {mw.mass * LB_PER_KG:.3f} lb, Oswald e: {mw.e_oswald_w:.3f}")

    hw = epicairplane.hwing
    if hw is not None:
        print("Horizontal tail:")
        print(f"  Span: {hw.span * IN_PER_M:.3f} in, Area: {hw.area * IN2_PER_M2:.3f} in^2, AR: {hw.ar:.3f}, MAC: {hw.mac * IN_PER_M:.3f} in, Taper: {hw.taper:.3f}")
        print(f"  Chords (root/mid/tip): {hw.rootChord * IN_PER_M:.3f} / {hw.midChord * IN_PER_M:.3f} / {hw.tipChord * IN_PER_M:.3f} in, Mid-chord position: {hw.midChordPosition * IN_PER_M:.3f} in")
        print(f"  Sweeps (root/mid/tip): {hw.rootSweep:.3f} / {hw.midSweep:.3f} / {hw.tipSweep:.3f} deg, Mid-sweep position: {hw.midSweepPosition:.3f}")
        print(f"  Incidence: {hw.incidence:.3f} deg, xqc: {hw.xqc * IN_PER_M:.3f} in, Symmetric: {hw.symmetric}, Areal density: {hw.arealDensity * LB_PER_IN2_PER_KG_PER_M2:.3f} lb/in^2")
        print(f"  Mass: {hw.mass * LB_PER_KG:.3f} lb, Oswald e: {hw.e_oswald_w:.3f}")
        print(f"  Elevator deflection: {hw.elevatorDeflection:.3f} deg, Elevator tau: {hw.elevatorTau:.3f}, CD delta-e k: {hw.cd_deltae_k:.3f}")

    vw = epicairplane.vtail
    if vw is not None:
        v_height = vw.span * 0.5
        print("Vertical tail:")
        print(f"  Height: {v_height * IN_PER_M:.3f} in, Span: {vw.span * IN_PER_M:.3f} in, Area: {vw.area * IN2_PER_M2:.3f} in^2, AR: {vw.ar:.3f}, MAC: {vw.mac * IN_PER_M:.3f} in, Taper: {vw.taper:.3f}")
        print(f"  Chords (root/mid/tip): {vw.rootChord * IN_PER_M:.3f} / {vw.midChord * IN_PER_M:.3f} / {vw.tipChord * IN_PER_M:.3f} in, Mid-chord position: {vw.midChordPosition * IN_PER_M:.3f} in")
        print(f"  Sweeps (root/mid/tip): {vw.rootSweep:.3f} / {vw.midSweep:.3f} / {vw.tipSweep:.3f} deg, Mid-sweep position: {vw.midSweepPosition:.3f}")
        print(f"  Incidence: {vw.incidence:.3f} deg, xqc: {vw.xqc * IN_PER_M:.3f} in, Symmetric: {vw.symmetric}, Areal density: {vw.arealDensity * LB_PER_IN2_PER_KG_PER_M2:.3f} lb/in^2")
        print(f"  Mass: {vw.mass * LB_PER_KG:.3f} lb, Oswald e: {vw.e_oswald_w:.3f}, Eta: {vw.eta:.3f}")

    print("Fuselages:")
    for i, f in enumerate(epicairplane.fuselages, 1):
        print(
            f"  {i}: Length {f.length * IN_PER_M:.3f} in, Width {f.width * IN_PER_M:.3f} in, Height {f.height * IN_PER_M:.3f} in, "
            f"P-factor {f.pfactor:.3f}, Roughness {f.roughness:.6g}, Laminar frac {f.laminarfraction:.3f}, "
            f"Q-factor {f.qfactor:.3f}, Quantity {f.quantity}"
        )

    bank_angle_summary_deg = 18.0
    bank_cos           = math.cos(math.radians(bank_angle_summary_deg))
    bank_load_factor   = 1.0 / bank_cos if bank_cos > 0.0 else float("inf")
    bank_speed_scale   = math.sqrt(bank_load_factor) if bank_load_factor > 0.0 else float("inf")
    bank_stall_speed   = stall_speed * bank_speed_scale
    bank_cruise_speed  = vbest * bank_speed_scale
    bank_propulsive_power = pwr * (bank_load_factor ** 1.5)
    bank_propulsive_elec  = (
        bank_propulsive_power / epicairplane.pplant.neff
        if epicairplane.pplant.neff > 0.0 else float("inf")
    )
    bank_total_power_w = bank_propulsive_elec + mission_systems_power_w
    bank_endurance_h   = (
        available_energy_wh / bank_total_power_w
        if np.isfinite(bank_total_power_w) and bank_total_power_w > 0.0 else 0.0
    )
    bank_range_nm = bank_endurance_h * (bank_cruise_speed * KNOTS_PER_MPS)
    print(f"Banked-flight summary ({bank_angle_summary_deg:.0f} deg):")
    print(f"  Load factor: {bank_load_factor:.3f}")
    print(f"  Stall speed: {bank_stall_speed * KNOTS_PER_MPS:.1f} kt")
    print(f"  Cruise velocity (scaled): {bank_cruise_speed * KNOTS_PER_MPS:.1f} kt")
    print(f"  Power including mission systems: {bank_total_power_w:.2f} W")
    print(f"  Endurance: {bank_endurance_h:.2f} h")
    print(f"  Range: {bank_range_nm:.1f} nm")

    # Calculate system CER using banked-turn endurance
    endurance_for_scoring_h = bank_endurance_h
    AFcer     = 150 * 6
    MOTORcer  = 250 * 1.5
    FUELcer   = 100 * 16
    SensorCER = 250 * 0.02
    Gccer     = 200 * endurance_for_scoring_h * 3
    SysCER    = AFcer + MOTORcer + FUELcer + SensorCER + Gccer
    TPMcret   = min(1, (endurance_for_scoring_h - 3) / 3)
    SE        = (0.34 * TPMcret) / (SysCER / 1000)
    print("SE score:", SE)
    print("TMPEcret:", TPMcret)
    print("SysCER:", SysCER)

    if plots and cruise_velocity_mode == "stall_margin":
        sweep_vmin = max(1.05 * stall_speed, 0.8 * vbest)
        sweep_vmax = max(2.0 * stall_speed, 1.3 * vbest, sweep_vmin + 0.5)
        sweep_speeds_mps = np.linspace(sweep_vmin, sweep_vmax, 50)
        sweep_endurance_h = np.array([
            evaluate_endurance_at_speed(epicairplane, v, mission_systems_power_w, available_energy_wh, res=60)
            for v in sweep_speeds_mps
        ], dtype=float)
        valid = np.isfinite(sweep_endurance_h)
        if np.any(valid):
            plt.figure()
            plt.plot(sweep_speeds_mps[valid] * KNOTS_PER_MPS, sweep_endurance_h[valid],
                     label="Trimmed endurance")
            plt.scatter([v_knots], [flight_time_h], color="red", zorder=3,
                        label=f"Selected cruise ({v_knots:.1f} kt)")
            plt.xlabel("Cruise velocity (knots)")
            plt.ylabel("Endurance (hours)")
            plt.title("Endurance vs Cruise Velocity")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            print("Endurance vs cruise velocity plot skipped: no valid trimmed points in sweep.")

    # --- Bank-angle sweep (0-60 deg) ---
    bank_angles_deg = np.linspace(0.0, 60.0, 241)
    phi_rad         = np.deg2rad(bank_angles_deg)
    load_factor     = 1.0 / np.cos(phi_rad)
    sqrt_n          = np.sqrt(load_factor)
    bank_vopt_mps   = vbest * sqrt_n
    bank_stall_mps  = stall_speed * sqrt_n
    newpmin_w       = pwr * (load_factor ** 1.5)
    truereq_w       = (newpmin_w / epicairplane.pplant.neff) + mission_systems_power_w
    endurance_h     = flight_time_h * (total_power_w / truereq_w)

    tan_phi = np.tan(phi_rad)
    eps = 1e-12
    radius_vopt_m = np.divide(
        bank_vopt_mps ** 2, G * tan_phi,
        out=np.full_like(tan_phi, np.inf, dtype=float),
        where=np.abs(tan_phi) > eps,
    )
    stall_margin = 1.25
    bank_vstall_margin_mps = bank_stall_mps * stall_margin
    radius_vstall_margin_m = np.divide(
        bank_vstall_margin_mps ** 2, G * tan_phi,
        out=np.full_like(tan_phi, np.inf, dtype=float),
        where=np.abs(tan_phi) > eps,
    )
    radius_vopt_ft          = radius_vopt_m * M_TO_FT
    radius_vstall_margin_ft = radius_vstall_margin_m * M_TO_FT

    phi_min_deg = 2.0
    cap_ft      = 300.0
    mask        = bank_angles_deg >= phi_min_deg
    radius_vopt_ft_plot          = radius_vopt_ft.astype(float).copy()
    radius_vstall_margin_ft_plot = radius_vstall_margin_ft.astype(float).copy()
    for r in (radius_vopt_ft_plot, radius_vstall_margin_ft_plot):
        r[~np.isfinite(r)] = np.nan
        r[~mask]           = np.nan
        r[r > cap_ft]      = np.nan

    if plots:
        plt.figure()
        plt.plot(bank_angles_deg, endurance_h, label="Endurance (using total_power_w)")
        plt.xlabel("Bank angle (deg)")
        plt.ylabel("Endurance (hours)")
        plt.title("Endurance vs Bank Angle")
        plt.grid(True)
        plt.legend()
        plt.show()

    if plots:
        plt.figure()
        plt.plot(bank_angles_deg, bank_vopt_mps * MPS_TO_KNOT, label="Vopt (bank)")
        plt.plot(bank_angles_deg, bank_stall_mps * MPS_TO_KNOT, label="Vstall (bank)")
        plt.xlabel("Bank angle (deg)")
        plt.ylabel("Speed (knots)")
        plt.title("Optimal & Stall Speed vs Bank Angle")
        plt.grid(True)
        plt.legend()
        plt.show()

    if plots:
        plt.figure()
        plt.plot(bank_angles_deg, radius_vopt_ft_plot, label="Radius @ Vopt(bank)")
        plt.plot(bank_angles_deg, radius_vstall_margin_ft_plot, linestyle="--",
                 label=f"Radius @ {stall_margin:.2f}*Vstall(bank)")
        plt.xlabel("Bank angle (deg)")
        plt.ylabel("Turn radius (ft)")
        plt.title("Orbit Radius vs Bank Angle (ft)")
        plt.grid(True)
        plt.ylim(0, cap_ft)
        plt.legend()
        plt.show()

    mu_roll_takeoff = 0.05
    mu_roll_landing = 0.35
    v_to       = stall_speed * stall_margin_takeoff_v2_req
    v_climb    = stall_speed * stall_margin_takeoff_climb_req
    v_approach = stall_speed * stall_margin_landing_approach_req
    # Assume flare from approach to touchdown speed before wheel rollout starts.
    approach_to_touchdown_factor = 1.0
    v_touchdown = v_approach * approach_to_touchdown_factor
    mass = epicairplane.weight / 9.81
    max_propulsive_power = epicairplane.pplant.pmax * epicairplane.pplant.neff
    thrust_to  = max_propulsive_power / max(v_to, 0.1) if max_propulsive_power > 0.0 else 0.0
    cd_cruise  = drag / (q * epicairplane.mwing.area) if q > 0.0 else 0.0
    q_to       = 0.5 * rho * v_to**2
    d_to       = q_to * epicairplane.mwing.area * cd_cruise
    avg_drag   = d_to / 3.0
    avg_friction = mu_roll_takeoff * (epicairplane.weight * (2.0 / 3.0))
    accel      = (thrust_to - avg_drag - avg_friction) / mass if mass > 0.0 else 0.0
    ground_roll_m = v_to**2 / (2.0 * accel) if accel > 0.0 else float("inf")

    # Landing rollout model: simple drag + rolling friction, no control-surface effects.
    landing_cd                   = max(0.0, cd_cruise)
    landing_sref                 = epicairplane.mwing.area
    landing_dt_s                 = 0.05
    landing_t_max_s              = 180.0
    landing_drag_update_period_s = 1.0
    landing_next_drag_update_s   = 0.0
    v_roll         = max(v_touchdown, 0.0)
    landing_roll_m = 0.0
    landing_time_s = 0.0
    saved_landing_state = (epicairplane.velocity, epicairplane.aoa, epicairplane.trim)
    try:
        while v_roll > 0.1 and landing_time_s < landing_t_max_s:
            if landing_time_s >= landing_next_drag_update_s:
                try:
                    sample_v = max(v_roll, 0.1)
                    epicairplane.velocity = sample_v
                    drag_sample = float(epicairplane.sumFanddM()[0])
                    q_sample = 0.5 * rho * sample_v**2
                    if np.isfinite(drag_sample) and q_sample > 0.0 and landing_sref > 0.0:
                        landing_cd = max(0.0, drag_sample / (q_sample * landing_sref))
                except Exception:
                    pass
                landing_next_drag_update_s += landing_drag_update_period_s
            q_roll        = 0.5 * rho * v_roll**2
            drag_roll     = q_roll * landing_sref * landing_cd
            rolling_force = mu_roll_landing * epicairplane.weight
            decel         = (drag_roll + rolling_force) / mass if mass > 0.0 else 0.0
            if decel <= 1e-9:
                landing_roll_m = float("inf")
                landing_time_s = float("inf")
                break
            v_next         = max(0.0, v_roll - decel * landing_dt_s)
            landing_roll_m += 0.5 * (v_roll + v_next) * landing_dt_s
            landing_time_s += landing_dt_s
            v_roll         = v_next
    finally:
        epicairplane.velocity, epicairplane.aoa, epicairplane.trim = saved_landing_state
    if landing_time_s >= landing_t_max_s and v_roll > 0.1:
        landing_roll_m = float("inf")
        landing_time_s = float("inf")

    print("Ground roll (simple):")
    print(f"  Takeoff speed: {v_to * KNOTS_PER_MPS:.1f} kt")
    print(f"  Ground roll: {ground_roll_m:.1f} m ({ground_roll_m * M_TO_FT:.0f} ft)")
    print("Landing roll (simple drag + friction):")
    print(f"  Approach speed: {v_approach * KNOTS_PER_MPS:.1f} kt")
    print(f"  Touchdown speed: {v_touchdown * KNOTS_PER_MPS:.1f} kt")
    print(f"  Cd used: {landing_cd:.4f}, rolling mu: {mu_roll_landing:.2f}")
    print(f"  Landing roll: {landing_roll_m:.1f} m ({landing_roll_m * M_TO_FT:.0f} ft)")
    if np.isfinite(landing_time_s):
        print(f"  Time to stop: {landing_time_s:.1f} s")
    else:
        print("  Time to stop: not reached (model limit)")

    # Stall margin compliance summary
    stall_margin_takeoff_v2       = v_to      / stall_speed if stall_speed > 0.0 else float("nan")
    stall_margin_takeoff_climb    = v_climb   / stall_speed if stall_speed > 0.0 else float("nan")
    stall_margin_level_flight     = vbest     / stall_speed if stall_speed > 0.0 else float("nan")
    stall_margin_landing_approach = v_approach / stall_speed if stall_speed > 0.0 else float("nan")
    print("Stall margin compliance:")
    print(
        f"  4.3.1 Takeoff V2: {stall_margin_takeoff_v2:.2f} "
        f"(req >= {stall_margin_takeoff_v2_req:.2f}) "
        f"{'PASS' if stall_margin_takeoff_v2 >= stall_margin_takeoff_v2_req else 'FAIL'}"
    )
    print(
        f"  4.3.2 Takeoff climb: {stall_margin_takeoff_climb:.2f} "
        f"(req >= {stall_margin_takeoff_climb_req:.2f}) "
        f"{'PASS' if stall_margin_takeoff_climb >= stall_margin_takeoff_climb_req else 'FAIL'}"
    )
    print(
        f"  4.3.3 Level flight (1g): {stall_margin_level_flight:.2f} "
        f"(req >= {stall_margin_level_flight_req:.2f}) "
        f"{'PASS' if stall_margin_level_flight >= stall_margin_level_flight_req else 'FAIL'}"
    )
    print(
        f"  4.3.4 Landing approach: {stall_margin_landing_approach:.2f} "
        f"(req >= {stall_margin_landing_approach_req:.2f}) "
        f"{'PASS' if stall_margin_landing_approach >= stall_margin_landing_approach_req else 'FAIL'}"
    )

    # Cruise polar extraction used for summary metrics and optional plots.
    aoa_samples   = np.linspace(epicairplane.aoa - 6.0, epicairplane.aoa + 8.0, 61)
    polar_results = drag_polar_at_speed(
        epicairplane, vbest, aoa_samples, trim_guess=epicairplane.trim, retrim=True,
    )
    q_ref = 0.5 * rho * vbest**2
    s_ref = epicairplane.mwing.area
    aoa_plot   = []
    cl_plot    = []
    cd_plot    = []
    cl_cd_plot = []
    for aoa_val, _trim_val, drag_val, lift_val in polar_results:
        if not (np.isfinite(drag_val) and np.isfinite(lift_val)):
            continue
        if q_ref <= 0.0 or s_ref <= 0.0:
            continue
        cd_val    = drag_val / (q_ref * s_ref)
        cl_val    = lift_val / (q_ref * s_ref)
        if not np.isfinite(cd_val) or cd_val <= 0.0:
            continue
        cl_cd_val = cl_val / cd_val
        if np.isfinite(cl_cd_val):
            aoa_plot.append(float(aoa_val))
            cl_plot.append(float(cl_val))
            cd_plot.append(float(cd_val))
            cl_cd_plot.append(float(cl_cd_val))

    cd0_fit          = float("nan")
    k_fit            = float("nan")
    ldmax_fit        = float("nan")
    cl_alpha_deg_fit = float("nan")
    cl_alpha_rad_fit = float("nan")
    cl0_fit          = float("nan")
    cl_ldmax_fit     = float("nan")
    e_fit            = float("nan")
    cl_cd_polar_points = len(cl_plot)
    if len(cl_plot) >= 3:
        aoa_arr   = np.asarray(aoa_plot,   dtype=float)
        cl_arr    = np.asarray(cl_plot,    dtype=float)
        cd_arr    = np.asarray(cd_plot,    dtype=float)

        cl_alpha_deg_fit, cl0_fit = np.polyfit(aoa_arr, cl_arr, 1)
        cl_alpha_rad_fit = cl_alpha_deg_fit * (180.0 / math.pi)

        cl2_arr = cl_arr**2
        k_fit, cd0_fit = np.polyfit(cl2_arr, cd_arr, 1)
        e_fit = (
            1.0 / (math.pi * epicairplane.mwing.ar * k_fit)
            if k_fit > 0.0 and epicairplane.mwing.ar > 0.0 else float("nan")
        )
        cl_ldmax_fit = (
            math.sqrt(cd0_fit / k_fit)
            if cd0_fit > 0.0 and k_fit > 0.0 else float("nan")
        )
        ldmax_fit = (
            1.0 / (2.0 * math.sqrt(cd0_fit * k_fit))
            if cd0_fit > 0.0 and k_fit > 0.0 else float("nan")
        )
        print("Approximate polar fit (cruise trim sweep):")
        print(f"  CD0: {cd0_fit:.5f}, k: {k_fit:.5f}, e: {e_fit:.3f}")
        print(f"  CL_alpha: {cl_alpha_deg_fit:.4f} /deg ({cl_alpha_rad_fit:.3f} /rad), CL0: {cl0_fit:.4f}")
        print(f"  CL @ max L/D: {cl_ldmax_fit:.4f}, (L/D)_max: {ldmax_fit:.3f}")
    else:
        print("Polar fit skipped: not enough valid CL/CD polar points.")

    if plot_ld_vs_alpha and aoa_plot:
        design_aoa = float(epicairplane.aoa)
        design_ld  = (lift / drag) if drag else float("nan")
        plt.figure()
        plt.plot(aoa_plot, cl_cd_plot, label=f"L/D @ {v_knots:.1f} kt")
        if np.isfinite(design_aoa) and np.isfinite(design_ld):
            plt.scatter([design_aoa], [design_ld], color="red", zorder=3,
                        label=f"Design point ({design_aoa:.2f} deg, {design_ld:.2f})")
        plt.xlabel("AoA (deg)")
        plt.ylabel("L/D")
        plt.title("L/D vs AoA at Cruise Speed")
        plt.grid(True)
        plt.legend()
        plt.show()
    elif plot_ld_vs_alpha:
        print("L/D vs AoA plot skipped: no valid polar points.")

    if plot_cl_cd_polar and len(cl_plot) >= 3:
        cd_arr = np.asarray(cd_plot, dtype=float)
        cl_arr = np.asarray(cl_plot, dtype=float)
        cruise_cd_point = (
            drag / (q_ref * s_ref) if q_ref > 0.0 and s_ref > 0.0 else float("nan")
        )
        cruise_cl_point = float(cruise_cl)

        stall_cl_point = float("nan")
        stall_cd_point = float("nan")
        if np.isfinite(cd0_fit) and np.isfinite(k_fit):
            stall_cl_point     = float(clmax)
            stall_cd_candidate = float(cd0_fit + k_fit * stall_cl_point**2)
            if np.isfinite(stall_cd_candidate) and stall_cd_candidate > 0.0:
                stall_cd_point = stall_cd_candidate
        if not (np.isfinite(stall_cd_point) and np.isfinite(stall_cl_point)):
            stall_idx      = int(np.nanargmax(cl_arr))
            stall_cd_point = float(cd_arr[stall_idx])
            stall_cl_point = float(cl_arr[stall_idx])

        plt.figure()
        plt.scatter(cd_arr, cl_arr, s=18, alpha=0.9, label=f"Drag polar @ {v_knots:.1f} kt")
        if np.isfinite(cruise_cd_point) and np.isfinite(cruise_cl_point):
            plt.scatter([cruise_cd_point], [cruise_cl_point], color="red", zorder=3,
                        label=f"Design cruise point ({v_knots:.1f} kt)")
        if np.isfinite(stall_cd_point) and np.isfinite(stall_cl_point):
            plt.scatter([stall_cd_point], [stall_cl_point],
                        color="black", marker="x", s=64, zorder=3,
                        label=f"Stall marker (CLmax {clmax:.2f})")
        plt.xlabel("CD")
        plt.ylabel("CL")
        plt.title("Drag Polar (CL vs CD) at Cruise Speed")
        plt.grid(True)
        plt.legend()
        plt.show()
    elif plot_cl_cd_polar:
        print("CL/CD polar plot skipped: not enough valid points.")

    # Ps and Vmax extraction from a trimmed speed sweep.
    perf_speeds         = np.linspace(max(1.05 * stall_speed, 0.8 * vbest),
                                      max(3.0 * stall_speed, 2.2 * vbest), 41)
    perf_power_required = np.full(perf_speeds.shape, np.nan, dtype=float)
    saved_perf_state    = (epicairplane.velocity, epicairplane.aoa, epicairplane.trim)
    try:
        alpha_seed = float(epicairplane.aoa)
        trim_seed  = float(epicairplane.trim)
        for i, vtest in enumerate(perf_speeds):
            epicairplane.velocity = float(vtest)
            sol = epicairplane.solveTrim(alpha0=alpha_seed, de0=trim_seed, res=60)
            if sol == [None, None]:
                sol = epicairplane.solveTrim(alpha0=0.0, de0=0.0, res=60)
            if sol == [None, None]:
                continue
            alpha_seed = float(sol[0])
            trim_seed  = float(sol[1])
            dtest = float(epicairplane.sumFanddM(res=60)[0])
            ptest = dtest * float(vtest)
            if np.isfinite(ptest) and ptest > 0.0:
                perf_power_required[i] = ptest
    finally:
        epicairplane.velocity, epicairplane.aoa, epicairplane.trim = saved_perf_state

    pwr_climb_req = float("nan")
    climb_idx     = int(np.argmin(np.abs(perf_speeds - v_climb)))
    if np.isfinite(perf_power_required[climb_idx]):
        pwr_climb_req = float(perf_power_required[climb_idx])
    elif np.any(np.isfinite(perf_power_required)):
        pwr_climb_req = float(np.nanmin(perf_power_required))
    initial_climb_ps = (
        (power_available - pwr_climb_req) / epicairplane.weight
        if np.isfinite(pwr_climb_req) and epicairplane.weight > 0.0 else float("nan")
    )
    ps_sweep = (
        (power_available - perf_power_required) / epicairplane.weight
        if epicairplane.weight > 0.0
        else np.full(perf_power_required.shape, np.nan, dtype=float)
    )
    min_ps = float(np.nanmin(ps_sweep)) if np.any(np.isfinite(ps_sweep)) else float("nan")

    vmax_mps = float("nan")
    if np.any(np.isfinite(perf_power_required)):
        residual = power_available - perf_power_required
        for i in range(len(perf_speeds) - 1):
            r0 = residual[i]
            r1 = residual[i + 1]
            if not (np.isfinite(r0) and np.isfinite(r1)):
                continue
            if r0 >= 0.0 and r1 <= 0.0 and abs(r1 - r0) > 1e-12:
                frac     = r0 / (r0 - r1)
                vmax_mps = float(perf_speeds[i] + frac * (perf_speeds[i + 1] - perf_speeds[i]))
                break
        if not np.isfinite(vmax_mps):
            feasible = np.where(np.isfinite(residual) & (residual >= 0.0))[0]
            if feasible.size > 0:
                vmax_mps = float(perf_speeds[int(feasible[-1])])

    ground_roll_ft = ground_roll_m * M_TO_FT if np.isfinite(ground_roll_m) else float("nan")
    ps_init_fts    = initial_climb_ps * M_TO_FT if np.isfinite(initial_climb_ps) else float("nan")
    ps_min_fts     = min_ps * M_TO_FT if np.isfinite(min_ps) else float("nan")
    print(
        "Requested performance summary: "
        f"Aero[CLa={cl_alpha:.4f}/deg, CLmax={clmax:.3f}, CD0={cd0_fit:.5f}, "
        f"Polar:CD=CD0+kCL^2(k={k_fit:.5f},pts={cl_cd_polar_points}), L/D={lift/drag if drag else float('nan'):.3f}, "
        f"SM={static_margin:.3f}] | "
        f"Vehicle[TO roll={ground_roll_ft:.0f} ft, Ps_init={ps_init_fts:.2f} ft/s, Ps_min={ps_min_fts:.2f} ft/s, "
        f"Vstall={stall_knots:.1f} kt, Vmax={vmax_mps * KNOTS_PER_MPS if np.isfinite(vmax_mps) else float('nan'):.1f} kt, "
        f"LFR={landing_margin * 100.0:.0f}%]"
    )


if __name__ == "__main__":
    _run_design_point()
