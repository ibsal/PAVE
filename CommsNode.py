from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet
import math
import scipy.optimize
import numpy as np

# Helper functions

def re(altitude, velocity, l):
    return (l*velocity)/Atmosphere(altitude).kinematic_viscosity[0]

def skin_friction_cf(re, roughness, length, laminar_frac=0.0, re_crit_per_m=5e5): # Cf calculation for drag buildup 
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
    def __init__(self, airfoil:PolarSet, altitude, velocity, span, rootChord, midChord, tipChord, midChordPosition, rootSweep, midSweep, tipSweep, midSweepPosition, incidence, aoa, symmetric, xqc, weight, arealDensity):
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
        

        self.area  = 2.0*(0.5*(self.rootChord + self.midChord)*(self.midChordPosition*0.5*self.span) + 0.5*(self.midChord + self.tipChord)*(0.5*self.span - self.midChordPosition*0.5*self.span))
        if not(symmetric):
            self.area = self.area/2.0
        self.mac   = (2.0*(((self.midChordPosition*0.5*self.span)/3.0)*(self.rootChord**2 + self.rootChord*self.midChord + self.midChord**2) + ((0.5*self.span - self.midChordPosition*0.5*self.span)/3.0)*(self.midChord**2 + self.midChord*self.tipChord + self.tipChord**2)))/self.area
        self.ar    = self.span**2 / self.area
        self.taper = self.tipChord / self.rootChord
        self.e_oswald_w = 1.78*(1.0 - 0.045*(self.ar**0.68)) - 0.64
        self.e_oswald_w = min(max(self.e_oswald_w, 0.3), 0.95)
        self.mass = self.arealDensity * self.area

    def cl2d(self, alpha_deg, reynold):
        return self.airfoil.cl(alpha_deg=alpha_deg, reynolds=reynold)
    
    def cd2d(self, alpha_deg, reynold):
        return self.airfoil.cd(alpha_deg=alpha_deg, reynolds=reynold)
    
    def cm2d(self, alpha_deg, reynold):
        return self.airfoil.cm(alpha_deg=alpha_deg, reynolds=reynold)
    
    def forces(self, xref, altitude, velocity, aoa, n=50):
        if self.symmetric:
            hspan = self.span/2.0
        else: 
            hspan = self.span

        dx = hspan/n
        stations = [dx * (i + 0.5) for i in range(n)]
        drag = 0
        lift = 0
        moment = 0

        density = Atmosphere(altitude).density[0]
        xqcedge = self.xqc
        for s in stations:
            if s>(hspan*self.midChordPosition):
                cslope = (self.tipChord - self.midChord)/((1 - self.midChordPosition) * hspan)
                clocal = (s - hspan*self.midChordPosition) * cslope  + self.midChord
            else:
                cslope = (self.midChord - self.rootChord)/(self.midChordPosition * hspan)
                clocal = s * cslope  + self.rootChord
            if s>(hspan*self.midSweepPosition):
                sslope = (self.tipSweep - self.midSweep)/((1 - self.midSweepPosition) * hspan)
                slocal = (s - hspan*self.midSweepPosition) * sslope  + self.midSweep
            else:
                sslope = (self.midSweep - self.rootSweep)/(self.midSweepPosition * hspan)
                slocal = s * sslope + self.rootSweep

            xqcmid = xqcedge + math.tan(math.radians(slocal))*(0.5*dx)
            momentarm = xref - xqcmid
            xqcedge += math.tan(math.radians(slocal))*dx

            veff = velocity * math.cos(math.radians(slocal))
            relocal = re(altitude, veff, clocal)
            cdlocal = self.cd2d(aoa + self.incidence, relocal)
            cllocal = self.cl2d(aoa + self.incidence, relocal)
            cmlocal = self.cm2d(aoa + self.incidence, relocal)
            qeff =  0.5 * density * veff**2
            drag += qeff * cdlocal * dx * clocal
            lift += qeff * cllocal * dx * clocal * math.cos(math.radians(slocal))
            moment += cmlocal * qeff * dx * clocal**2
            moment += momentarm * qeff * cllocal * dx * clocal * math.cos(math.radians(slocal))

        if self.symmetric: 
            drag *=2
            lift *=2
            moment *=2

        aq = 0.5 * density * velocity**2 * self.area
        cleq = lift/(aq)       
        drag += aq * (cleq**2)/(math.pi * self.ar * self.e_oswald_w)

        downwash = 57.2958 * cleq / (math.pi * self.ar * self.e_oswald_w)
        return [drag, lift, moment, downwash]

    def stallSpeed(self, altitude, weight, v0=20.0):
        density = Atmosphere(altitude).density[0]
        vcand = v0
        tol = 1
        while(tol>0.01):
            qa = 0.5 * density * vcand**2 * self.area
            neg_l = lambda x: -self.forces(10, altitude, vcand, x)[1]
            lmax = scipy.optimize.fmin(neg_l, 2, xtol=0.001, disp=False)[0]
            lmax = -1 * neg_l(lmax)
            clmax = lmax/(qa)
            vol = vcand
            vcand = math.sqrt((2* weight)/(density * self.area * clmax))
            tol = abs(vcand-vol)
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
            incidence, aoa, symmetric, xqc, weight, arealDensity
        )
        self.elevatorTau = elevatorTau
        self.cd_deltae_k = cd_deltae_k
        self.elevatorDeflection = elevatorDeflection

    def forces(self, xref, altitude, velocity, aoa, n=100, downwash=0.0, elevator=0.0):
        aoa_eff = aoa - downwash + self.elevatorTau * elevator
        out = super().forces(xref, altitude, velocity, aoa_eff, n=n)
        return out

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
        eta=1.0
    ):
        super().__init__(
            airfoil, altitude, velocity, 2.0*height,
            rootChord, midChord, tipChord, midChordPosition,
            rootSweep, midSweep, tipSweep, midSweepPosition,
            incidence, beta, True, xqc, weight, arealDensity
        )
        self.eta = eta

    def forces(self, xref, altitude, velocity, beta, n=100):
        out = super().forces(xref, altitude, velocity, beta, n=n)
        out[0] *= 0.5 * self.eta
        out[1] *= 0.5 * self.eta
        out[2] *= 0.5 * self.eta
        return out

class Fuselage:
    def __init__(self, length, width, height, pfactor, roughness, laminarfraction, qfactor=1.0, quantity=1):
        self.length = length
        self.width = width
        self.height = height
        self.pfactor = pfactor
        self.roughness = roughness
        self.laminarfraction = laminarfraction
        self.perimeter = (2*self.width + 2*self.height)*pfactor
        self.diameter = self.perimeter/math.pi
        self.qfactor = qfactor
        self.quantity = quantity
        
    def drag(self, altitude, velocity):
        rlocal = re(altitude, velocity, self.length)
        fratio = self.length/self.diameter
        ff = (1 + 60/fratio**3) + fratio/400
        cfc = skin_friction_cf(rlocal, self.roughness, self.length, self.laminarfraction)
        fuse_term = max(1.0 - 2.0 / max(fratio, 1e-6), 0.0)
        swet = (math.pi * self.diameter * self.length * math.pow(fuse_term, 2.0/3.0) * (1 + 1/(fratio**2)))
        cdo = ff * self.qfactor * cfc
        drag = cdo * 0.5 * Atmosphere(altitude).density[0] * velocity**2 *swet
        return drag

class Powerplant:
    def __init__(self, bcap, neff, pmax):
        self.full = bcap
        self.bcap = bcap
        self.neff = neff
        self.pmax = pmax
    def drawBattery(self, duration, power):
        self.bcap -= duration*power
        return self.bcap
    def validThrust(self,thrust, velocity):
        if (thrust/self.neff)/velocity >= self.pmax: return False
        return True

class Aircraft:
    def __init__(self, altitude, velocity, pplant:Powerplant, mwing:Wing, hwing:HorizontalTail, vtail:VerticalTail, fuselages:list[Fuselage], aoa, trim, xcg, weight, cdomisc):
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
        hForce = self.hwing.forces(self.xcg, self.altitude, self.velocity, self.aoa, downwash=wForce[3], elevator=self.trim, n=res)

        vDrag = 0.0
        if self.vtail is not None:
            vForce = self.vtail.forces(self.xcg, self.altitude, self.velocity, 0.0, n=res)
            vDrag = float(vForce[0])

        fDrag = 0.0
        for f in self.fuselages:
            fDrag += float(f.drag(self.altitude, self.velocity)) * float(getattr(f, "quantity", 1))

        Drag = float(wForce[0]) + float(hForce[0]) + vDrag + fDrag + 0.5 * Atmosphere(self.altitude).density[0] * self.velocity**2 * self.cdomisc * self.mwing.area
        Lift = float(wForce[1]) + float(hForce[1])
        Moment = float(wForce[2]) + float(hForce[2])
        return [Drag, Lift, Moment]

    def solveTrim(self, alpha0=None, de0=None, res=100):
        if alpha0 is None: alpha0 = float(self.aoa)
        if de0 is None: de0 = float(self.trim)
        def residual(x):
            aoa = float(x[0])
            trim = float(x[1])

            self.aoa = aoa
            self.trim = trim

            mf = self.sumFanddM(res=res)
            lres = float(mf[1] - self.weight)
            mres = float(mf[2])
            return np.array([lres, mres], dtype=float)

        sol = scipy.optimize.root(residual, x0=np.array([alpha0, de0], dtype=float), method="hybr")
        if not sol.success:
            return [None, None]
        self.trim = sol.x[1]
        self.aoa = sol.x[0]
        
        return [self.aoa, self.trim]
 
    def solveBestVelocity(self, levelFlightMargin, vguess=20,res=100):
        stall = self.mwing.stallSpeed(self.altitude, self.weight, v0=vguess)[0]
        vmin = float(stall * levelFlightMargin)
        vmax = max(vmin * 2.0, vmin + 1.0)

        def power(v):
            self.velocity = float(v)
            sol = self.solveTrim(res=res)
            if sol==[None, None]:
                return 1e30
            Drag = self.sumFanddM()[0]
            return float(Drag) * self.velocity

        res = scipy.optimize.minimize_scalar(power, bounds=(vmin, vmax), method="bounded", options={"xatol": 0.1})
        self.thrust = self.sumFanddM()[0]
        self.power = res.x * self.thrust
        if(self.power>self.pplant.pmax): print("WARNING: aircraft power exceeds peak power plant power")
        return [float(res.x), self.power, self.thrust]

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
        Cm_alpha = (m_p - m_m) / (2*dalpha) / (qS * cbar)
        return Cm_alpha

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
        CL_alpha = (L_p - L_m) / (2*dalpha) / qS
        return CL_alpha

    def staticMargin(self):
        Cm_a = self.cm_alpha()
        CL_a = self.cl_alpha()
        return -Cm_a / CL_a

    def horizontalTailVolume(self, xhtqc):
        return (self.hwing.area * (xhtqc - self.xcg)) / (self.mwing.area * self.mwing.mac)

    def verticalTailVolume(self, xvtqc):
        return (self.vtail.area * (xvtqc - self.xcg)) / (self.mwing.area * self.span)


'''
wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
mainWing = Wing(wingFoil, 200, 30, 4.5, 0.3, 0.3, 0.2, 0.5, 0, 0, 0, 0.5, 0, 2, True, 0.4, 200, 3.0)
hwing = HorizontalTail(wingFoil, 200, 30, 1.3, 0.2, 0.2, 0.1, 0.5, 0, 0, 0, 0, 0, 0, True, 1.4, 200, 2.0, elevatorDeflection=0, elevatorTau=0.35,cd_deltae_k=0)
vwing = VerticalTail(wingFoil, 200, 30, 0.5, 0.2, 0.23, 0.2, 0.5, 10, 10, 10, 0.5, 0, 0, 1.4, 200, 2.0)
body = Fuselage(1, .3, 0.3, 0.9, 0.00635e-3, 0.3)
booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric = Powerplant(7992000, 0.59, 4000)
commsNode = Aircraft(200, 20, batteryElectric, mainWing, hwing, vwing, [body, booms], 0, 0, 0.5,200, 0.01)

print(commsNode.solveBestVelocity(1.25))
'''



import math
import numpy as np
import scipy.optimize

G = 9.80665

def optimize_endurance(
    wingFoil,
    altitude,
    batteryElectric,
    fuselages,
    xcg,
    cdomisc,
    baseMass,
    totalMassMax,
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    levelFlightMargin=1.25,
    res=60,
    seed=1,
    maxiter=30,
    popsize=6,
    polish=True,
):
    arealDensityMain = 3.0
    arealDensityH = 2.0
    arealDensityV = 2.0

    def build_aircraft(x):
        wingSpan = float(x[0])
        wingChord = float(x[1])

        hSpan = float(x[2])
        hChord = float(x[3])
        xhtqc = float(x[4])

        vHeight = float(x[5])
        vChord = float(x[6])
        xvtqc = float(x[7])

        mainWing = Wing(
            wingFoil, altitude, 0.0,
            wingSpan,
            wingChord, wingChord, wingChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0, True,
            0.0,
            0.0,
            arealDensityMain,
        )

        hwing = HorizontalTail(
            wingFoil, altitude, 0.0,
            hSpan,
            hChord, hChord, hChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0, True,
            xhtqc,
            0.0,
            arealDensityH,
            elevatorDeflection=0.0,
            elevatorTau=0.35,
            cd_deltae_k=0.0,
        )

        vwing = VerticalTail(
            wingFoil, altitude, 0.0,
            vHeight,
            vChord, vChord, vChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0,
            xvtqc,
            0.0,
            arealDensityV,
            eta=1.0,
        )

        totalMass = float(baseMass) + float(mainWing.mass) + float(hwing.mass) + float(vwing.mass)
        weight = totalMass * G

        commsNode = Aircraft(
            altitude,
            20.0,
            batteryElectric,
            mainWing,
            hwing,
            vwing,
            fuselages,
            0.0,
            0.0,
            xcg,
            weight,
            cdomisc,
        )

        return commsNode, totalMass

    def objective(x):
        wingSpan = float(x[0])
        wingChord = float(x[1])
        hSpan = float(x[2])
        hChord = float(x[3])
        xhtqc = float(x[4])
        vHeight = float(x[5])
        vChord = float(x[6])
        xvtqc = float(x[7])
        print("Running objective")
        if wingSpan <= 0.0 or wingChord <= 0.0 or hSpan <= 0.0 or hChord <= 0.0 or vHeight <= 0.0 or vChord <= 0.0:
            return 1e30

        if xhtqc <= xcg or xvtqc <= xcg:
            return 1e30

        try:
            commsNode, totalMass = build_aircraft(x)
        except Exception:
            print("build aircraft failed")
            return 1e30

        if totalMass > float(totalMassMax):
            return 1e30

        try:
            vbest, pwr, thrust = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
        except Exception:
            print("solve best velocity failed")
            return 1e30

        if not np.isfinite(pwr) or pwr <= 0.0:
            return 1e30

        if float(pwr) > float(commsNode.pplant.pmax):
            return 1e30

        try:
            sm = float(commsNode.staticMargin())
        except Exception:
            print("static margin failed")
            return 1e30

        if (sm < float(staticMarginMin)) or (sm > float(staticMarginMax)):
            return 1e30

        return float(pwr)

    bounds = [
        (3.0, 4.5),      # wingSpan
        (0.24, 0.60),    # wingChord
        (0.60, 2.0),     # hSpan
        (0.08, 0.40),    # hChord
        (xcg + 0.2, xcg + 2.0),  # xhtqc
        (0.20, 2.0),     # vHeight
        (0.08, 0.40),    # vChord
        (xcg + 0.2, xcg + 2.0),  # xvtqc
    ]

    rng = np.random.default_rng(int(seed))
    x0 = np.array([(lo + hi) * 0.5 for (lo, hi) in bounds], dtype=float)
    x0 += rng.normal(scale=0.05, size=x0.shape) * np.array([(hi - lo) for (lo, hi) in bounds], dtype=float)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    result = scipy.optimize.differential_evolution(
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
        init="latinhypercube",
        updating="deferred",
        workers=1,
        disp=True
    )

    xbest = result.x
    commsNode, totalMass = build_aircraft(xbest)
    out = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
    sm = float(commsNode.staticMargin())

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "power_W": float(result.fun),
        "vbest_mps": float(out[0]),
        "thrust_N": float(out[2]),
        "totalMass_kg": float(totalMass),
        "staticMargin": float(sm),
        "xbest": {
            "wingSpan": float(xbest[0]),
            "wingChord": float(xbest[1]),
            "hSpan": float(xbest[2]),
            "hChord": float(xbest[3]),
            "xhtqc": float(xbest[4]),
            "vHeight": float(xbest[5]),
            "vChord": float(xbest[6]),
            "xvtqc": float(xbest[7]),
        },
        "result": result,
    }


# Example usage (adjust constants to your case)
wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
body = Fuselage(1, .3, 0.3, 0.9, 0.00635e-3, 0.3)
booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric = Powerplant(7992000, 0.59, 4000)
print("optimizer starting")
best = optimize_endurance(
     wingFoil=wingFoil,
     altitude=200,
    batteryElectric=batteryElectric,
    fuselages=[body, booms],
    xcg=0.45,
    cdomisc=0.01,
    baseMass=18.0,
    totalMassMax=24.9,
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    levelFlightMargin=1.25,
    res=20,
    seed=1,
    maxiter=4,
    popsize=3,
    polish=False
)
print(best)
