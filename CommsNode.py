from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet
import math
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import hashlib

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
            incidence, beta, True, xqc, weight, 2.0*arealDensity
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
        return (self.vtail.area * (xvtqc - self.xcg)) / (self.mwing.area * self.mwing.span)



import math
import numpy as np
import scipy.optimize

G = 9.80665
''' Optimizer
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
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    levelFlightMargin=1.25,
    res=60,
    seed=1,
    maxiter=150,
    popsize=6,
    polish=True,
):
    arealDensityMain = 3.0
    arealDensityH = 1.5
    arealDensityV = 1.5

    boomMassPerM = 0.4
    boomMassFixed = 0.0
    boomLengthMin = 0.05

    evalCount = 0
    bestSeen = {"pwr": 1e30}

    def build_aircraft(x):
        print("Iteration +1")
        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        vHeight = float(x[6])
        vChord = float(x[7])
        xvtqc = xhtqc

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
            0.0, 0.0, True,
            xhtqc,
            0.0,
            arealDensityH,
            elevatorDeflection=0.0,
            elevatorTau=0.35,
            cd_deltae_k=0.0,
        )

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

    def objective(x):
        nonlocal evalCount, bestSeen
        evalCount += 1
        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        vHeight = float(x[6])
        vChord = float(x[7])
        xvtqc = xhtqc

        if wingSpan <= 0.0 or wingChord <= 0.0 or hSpan <= 0.0 or hChord <= 0.0 or vHeight <= 0.0 or vChord <= 0.0:
            return 1e30

        if xwqc <= 0.0:
            return 1e30

        if xhtqc <= xcg or xvtqc <= xcg:
            return 1e30

        if xhtqc <= xwqc:
            return 1e30

        try:
            commsNode, totalMass = build_aircraft(x)
        except Exception:
            return 1e30
        if (commsNode.horizontalTailVolume(xhtqc) < 0.3) or (commsNode.horizontalTailVolume(xhtqc) > 0.9):
            print("Htail reject")
            return 1e30
        
        if (commsNode.verticalTailVolume(xhtqc)<0.02 or commsNode.verticalTailVolume(xhtqc)>0.08):
            print("Vtail reject")
            return 1e30
        
        if totalMass > float(totalMassMax):
            print("mass reject")
            return 1e30

        try:
            vbest, pwr, thrust = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
        except Exception:
            return 1e30

        if not np.isfinite(pwr) or pwr <= 0.0:
            return 1e30

        if float(pwr) > float(commsNode.pplant.pmax):
            print("pwr reject")
            return 1e30

        try:
            sm = float(commsNode.staticMargin())
            cma = float(commsNode.cm_alpha())
        except Exception:
            return 1e30

        if (sm < float(staticMarginMin)) or (sm > float(staticMarginMax)):
            print("sm reject")
            return 1e30

        if cma > 0:
            print("cma reject")
            return 1e30
        
        

        pwr = float(pwr)

        if pwr < bestSeen["pwr"]:
            bestSeen["pwr"] = pwr
            print(f"[best] eval={evalCount} pwr={pwr:.2f} W x={np.array(x, dtype=float)}")

        print("Valid dp")

        return pwr

    bounds = [
        (3.0, 4.5),
        (0.3, 0.40),
        (0.05, xcg + 0.2),
        (0.60, 1.8),
        (0.18, 0.30),
        (xcg + 0.2, xcg + 1.8),
        (0.20, 1.0),
        (0.08, 0.30),
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
            "xwqc": float(xbest[2]),
            "hSpan": float(xbest[3]),
            "hChord": float(xbest[4]),
            "xhtqc": float(xbest[5]),
            "vHeight": float(xbest[6]),
            "vChord": float(xbest[7]),
            "boomLength": float(max(float(xbest[5] - xbest[2]), float(boomLengthMin))),
        },
        "result": result,
    }


wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
body = Fuselage(1, .3, 0.15, 0.9, 0.00635e-3, 0.3)
booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric = Powerplant(7992000, 0.59, 4000)

best = optimize_endurance(
    wingFoil=wingFoil,
    tailFoil=tailFoil,
    altitude=200,
    batteryElectric=batteryElectric,
    fuselages=[body, booms, booms],
    xcg=0.45,
    cdomisc=0.01,
    baseMass=17.5,
    totalMassMax=22.6796,
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    levelFlightMargin=1.25,
    res=2,
    seed=6,
    maxiter=30,
    popsize=30,
    polish=True
)
print(best)
'''

# Actual design point

altitude = 200
bestDesignPoint = []

wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="dae11")
tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
arealDensityMain = 3.0 # kg/m^2
arealDensityH = 2.0 # kg/m^2
arealDensityV = 2.0 # kg/m^2
baseMass = 17.5 # n
boomMassFixed = 0 #n
boomLengthMin = 0.005 #m
cdomisc = 0.01
xcg=0.45
boomMassPerM = 0.4

body = Fuselage(1.0541, 0.3048, 0.21336, 0.9, 0.00635e-3, 0.3)
booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric = Powerplant(7992000, 0.59, 4000)
fuselages = [body, booms, booms]


def build_aircraft(x):
    wingSpan = float(x[0])
    wingChord = float(x[1])
    xwqc = float(x[2])

    hSpan = float(x[3])
    hChord = float(x[4])
    xhtqc = float(x[5])

    vHeight = float(x[6])
    vChord = float(x[7])
    xvtqc = xhtqc

    mainWing = Wing(
        wingFoil, altitude, 0.0,
        wingSpan,
        wingChord, wingChord, 0.8*wingChord, 0.5,
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
        0.0, 0.0, True,
        xhtqc,
        0.0,
        arealDensityH,
        elevatorDeflection=0.0,
        elevatorTau=0.35,
        cd_deltae_k=0.0,
    )

    vwing = VerticalTail(
        tailFoil, altitude, 0.0,
        14.7*0.0254*0.5,
        5.4*0.0254, 9.6*0.0254, 5.4*0.0254, 0.8,
        30, 0.0, -30, 0.8,
        0.0, 0.0,
        xvtqc,
        0.0,
        arealDensityV,
        eta=2.0,
    )

    hwing.incidence = 0
    mainWing.incidence = 1.0

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

design_point = [4.272, 0.306, 0.374, 0.976, 0.194, 1.574, 0.34798, 0.194]
cache_version = 1
cache_path = "aircraft_cache.pkl"

def cache_key():
    fuselage_sig = []
    for f in fuselages:
        fuselage_sig.append((
            float(getattr(f, "length", 0.0)),
            float(getattr(f, "width", 0.0)),
            float(getattr(f, "height", 0.0)),
            float(getattr(f, "pfactor", 0.0)),
            float(getattr(f, "roughness", 0.0)),
            float(getattr(f, "laminarfraction", 0.0)),
            float(getattr(f, "qfactor", 1.0)),
            int(getattr(f, "quantity", 1)),
        ))
    key_data = {
        "version": cache_version,
        "design_point": [float(x) for x in design_point],
        "altitude": float(altitude),
        "xcg": float(xcg),
        "cdomisc": float(cdomisc),
        "baseMass": float(baseMass),
        "arealDensityMain": float(arealDensityMain),
        "arealDensityH": float(arealDensityH),
        "arealDensityV": float(arealDensityV),
        "boomMassFixed": float(boomMassFixed),
        "boomLengthMin": float(boomLengthMin),
        "boomMassPerM": float(boomMassPerM),
        "fuselages": fuselage_sig,
        "battery": (float(batteryElectric.full), float(batteryElectric.neff), float(batteryElectric.pmax)),
        "wingFoil": "psu94097",
        "tailFoil": "S9033",
    }
    return hashlib.sha256(repr(key_data).encode("utf-8")).hexdigest()

def load_cached_aircraft():
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("key") != cache_key():
            return None
        return payload
    except Exception:
        return None

def save_cached_aircraft(payload):
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

def evaluate_ld_at_speed(aircraft, target_velocity):
    saved_state = (aircraft.velocity, aircraft.aoa, aircraft.trim)
    aircraft.velocity = float(target_velocity)
    ld = float("nan")
    try:
        sol = aircraft.solveTrim()
        if sol != [None, None]:
            forces_off = aircraft.sumFanddM()
            drag_off = forces_off[0]
            lift_off = forces_off[1]
            if drag_off:
                ld = lift_off / drag_off
    except Exception:
        pass
    finally:
        aircraft.velocity, aircraft.aoa, aircraft.trim = saved_state
    return ld

def drag_polar_at_speed(aircraft, speed, aoa_samples, trim_guess=0.0):
    saved_state = (aircraft.velocity, aircraft.aoa, aircraft.trim)
    aircraft.velocity = float(speed)
    results = []
    for aoa in aoa_samples:
        aircraft.aoa = float(aoa)
        def moment_sq(t):
            aircraft.trim = float(t)
            m = aircraft.sumFanddM()[2]
            return m * m
        try:
            res = scipy.optimize.minimize_scalar(
                moment_sq,
                bounds=(-10.0, 10.0),
                method="bounded",
                options={"xatol": 0.05},
            )
            aircraft.trim = float(res.x)
            forces = aircraft.sumFanddM()
            results.append((float(aoa), float(aircraft.trim), float(forces[0]), float(forces[1])))
        except Exception:
            results.append((float(aoa), float("nan"), float("nan"), float("nan")))
    aircraft.velocity, aircraft.aoa, aircraft.trim = saved_state
    return results

cached = None
if cached is None:
    epicairplane = build_aircraft(design_point)[0]
    vbest, pwr, thrust = epicairplane.solveBestVelocity(1.25)
    cached = {
        "key": cache_key(),
        "aircraft": epicairplane,
        "vbest": float(vbest),
        "pwr": float(pwr),
        "thrust": float(thrust),
        "aoa": float(epicairplane.aoa),
        "trim": float(epicairplane.trim),
    }
    save_cached_aircraft(cached)
else:
    epicairplane = cached["aircraft"]
    vbest = float(cached.get("vbest", epicairplane.velocity))
    pwr = float(cached.get("pwr", 0.0))
    thrust = float(cached.get("thrust", 0.0))
    epicairplane.velocity = vbest
    if "aoa" in cached and "trim" in cached:
        epicairplane.aoa = float(cached["aoa"])
        epicairplane.trim = float(cached["trim"])
    else:
        epicairplane.solveTrim()

forces = epicairplane.sumFanddM()
drag, lift, moment = forces[0], forces[1], forces[2]
rho = Atmosphere(altitude).density[0]
q = 0.5 * rho * vbest**2
cruise_cl = lift / (q * epicairplane.mwing.area) if q > 0 else float("nan")
moment_coefficient = (
    moment / (q * epicairplane.mwing.area * epicairplane.mwing.mac)
    if q > 0 and epicairplane.mwing.mac > 0
    else float("nan")
)
stall_speed, clmax = epicairplane.mwing.stallSpeed(epicairplane.altitude, epicairplane.weight, v0=vbest)
power_available = epicairplane.pplant.pmax
propulsive_power_elec = (
    pwr / epicairplane.pplant.neff if epicairplane.pplant.neff > 0 else float("inf")
)
excess_power = power_available - pwr
power_fraction = pwr / power_available if power_available else float("nan")
static_margin = epicairplane.staticMargin()
cm_alpha = epicairplane.cm_alpha()
cl_alpha = epicairplane.cl_alpha()
h_tail_volume = epicairplane.horizontalTailVolume(design_point[5])
v_tail_volume = epicairplane.verticalTailVolume(design_point[5])

KNOTS_PER_MPS = 1.9438444924406
LBF_PER_NEWTON = 0.2248089431
IN_PER_M = 39.37007874015748
LB_PER_KG = 2.2046226218487757
IN2_PER_M2 = IN_PER_M * IN_PER_M
LB_PER_IN2_PER_KG_PER_M2 = LB_PER_KG / IN2_PER_M2
MI_PER_NM = 1.1507794480235425
mission_systems_power_w = 50.0
landing_margin = 0.20
battery_capacity_wh = 2220.0

v_knots = vbest * KNOTS_PER_MPS
stall_knots = stall_speed * KNOTS_PER_MPS
drag_lbf = drag * LBF_PER_NEWTON
lift_lbf = lift * LBF_PER_NEWTON
thrust_lbf = thrust * LBF_PER_NEWTON
propulsive_power_fraction = propulsive_power_elec / power_available if power_available else float("nan")
total_power_w = propulsive_power_elec + mission_systems_power_w
available_energy_wh = battery_capacity_wh * (1.0 - landing_margin)
flight_time_h = available_energy_wh / total_power_w if total_power_w > 0.0 else 0.0
flight_time_min = flight_time_h * 60.0
flight_distance_nm = flight_time_h * v_knots
flight_distance_mi = flight_distance_nm * MI_PER_NM
climb_rate_mps = excess_power / epicairplane.weight
FT_PER_MIN_PER_MPS = 196.850394
climb_rate_fpm = climb_rate_mps * FT_PER_MIN_PER_MPS
ld_at_90 = evaluate_ld_at_speed(epicairplane, vbest * 0.9)
ld_at_110 = evaluate_ld_at_speed(epicairplane, vbest * 1.1)

print("Cruise summary:")
print(f"  Velocity: {v_knots:.1f} kt")
print(f"  Stall speed: {stall_knots:.1f} kt, Cl_max: {clmax:.3f}")
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
print(f"Endurance:")
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



# calculate system CER
AFcer = 150 * 6
MOTORcer = 250 * 1.5
FUELcer = 100 * 16
SensorCER = 250 * 0.02
Gccer = 200 * flight_time_h * 3
SysCER = AFcer + MOTORcer + FUELcer + SensorCER + Gccer
# TPMS
TPMcret = min(1, (flight_time_h - 3)/(3))


# SE
SE = (0.34 * TPMcret)/(SysCER/1000)
print("SE score:", SE)
print("TMPEcret:", TPMcret)
print("SysCER:", SysCER)

# simple Mission Envelope
bankAngle = 20 # degrees
velopt = v_knots * 0.514444 # m/s velocity at optimal cruise
velstallopt = stall_knots*0.514444 # m/s stall at optimal straight cruise
load_factor = 1/math.cos(math.radians(bankAngle))
bank_stall = velstallopt * math.sqrt(load_factor)
bank_vopt = velopt * math.sqrt(load_factor)
newpmin = pwr * math.pow(load_factor, 1.5)
truereq = (newpmin/epicairplane.pplant.neff) + mission_systems_power_w
newEndurance = (total_power_w/truereq) * flight_time_h
print(newEndurance)
print(bank_vopt/0.514444)

import math
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
KNOT_TO_MPS = 0.514444
MPS_TO_KNOT = 1.0 / KNOT_TO_MPS
G = 9.80665  # m/s^2
M_TO_FT = 3.28084

# --- Baseline (straight & level) in m/s ---
velopt0_mps    = v_knots * KNOT_TO_MPS
velstall0_mps  = stall_knots * KNOT_TO_MPS

# --- Sweep bank angle ---
bank_angles_deg = np.linspace(0.0, 60.0, 241)  # 0 to 60 deg, 0.25-deg step
phi_rad = np.deg2rad(bank_angles_deg)

load_factor = 1.0 / np.cos(phi_rad)                 # n = 1/cos(phi)
sqrt_n = np.sqrt(load_factor)

# --- Speeds in bank ---
bank_vopt_mps  = velopt0_mps * sqrt_n
bank_stall_mps = velstall0_mps * sqrt_n

# --- Power scaling (assuming pwr is straight-level P_min at Vopt0) ---
newpmin_w = pwr * (load_factor ** 1.5)

# Total power draw (input power) including mission systems
truereq_w = (newpmin_w / epicairplane.pplant.neff) + mission_systems_power_w

# --- Endurance scaling ---
# Option A (your original form): assumes total_power_w is the *baseline draw power* that produced flight_time_h
endurance_h = flight_time_h * (total_power_w / truereq_w)

# --- Turn radii ---
tan_phi = np.tan(phi_rad)
eps = 1e-12

radius_vopt_m = np.where(np.abs(tan_phi) > eps, (bank_vopt_mps ** 2) / (G * tan_phi), np.inf)

stall_margin = 1.25
bank_vstall_margin_mps = bank_stall_mps * stall_margin
radius_vstall_margin_m = np.where(np.abs(tan_phi) > eps, (bank_vstall_margin_mps ** 2) / (G * tan_phi), np.inf)

# Convert to ft
radius_vopt_ft = radius_vopt_m * M_TO_FT
radius_vstall_margin_ft = radius_vstall_margin_m * M_TO_FT

# Crop "crazy" values (near-zero bank -> inf radius) and cap to plot max
phi_min_deg = 2.0
cap_ft = 300.0

mask = bank_angles_deg >= phi_min_deg

radius_vopt_ft_plot = radius_vopt_ft.astype(float).copy()
radius_vstall_margin_ft_plot = radius_vstall_margin_ft.astype(float).copy()

for r in (radius_vopt_ft_plot, radius_vstall_margin_ft_plot):
    r[~np.isfinite(r)] = np.nan
    r[~mask] = np.nan
    r[r > cap_ft] = np.nan

# =========================
# Plot 1: Endurance vs bank
# =========================
plt.figure()
plt.plot(bank_angles_deg, endurance_h, label="Endurance (using total_power_w)")
plt.xlabel("Bank angle (deg)")
plt.ylabel("Endurance (hours)")
plt.title("Endurance vs Bank Angle")
plt.grid(True)
plt.legend()
plt.show()

# ======================
# Plot 2: Vopt vs bank
# ======================
plt.figure()
plt.plot(bank_angles_deg, bank_vopt_mps * MPS_TO_KNOT, label="Vopt (bank)")
plt.plot(bank_angles_deg, bank_stall_mps * MPS_TO_KNOT, label="Vstall (bank)")
plt.xlabel("Bank angle (deg)")
plt.ylabel("Speed (knots)")
plt.title("Optimal & Stall Speed vs Bank Angle")
plt.grid(True)
plt.legend()
plt.show()

# ============================
# Plot 3: Orbit radius vs bank (ft) + overlay + cropped
# ============================
plt.figure()
plt.plot(bank_angles_deg, radius_vopt_ft_plot, label="Radius @ Vopt(bank)")
plt.plot(bank_angles_deg, radius_vstall_margin_ft_plot, linestyle="--", label=f"Radius @ {stall_margin:.2f}*Vstall(bank)")
plt.xlabel("Bank angle (deg)")
plt.ylabel("Turn radius (ft)")
plt.title("Orbit Radius vs Bank Angle (ft)")
plt.grid(True)
plt.ylim(0, cap_ft)
plt.legend()
plt.show()
