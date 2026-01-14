from ambiance import Atmosphere
from PyFoil.airfoil_polars import PolarSet, PolarSet

### Environment Configuration
GroundLevel = 0 # M, ASL
OrbitLevel = 1200 #M, AGL

### Wing Definition

RootChord = 0.4
TipChord = 0.3048
RootThickness = 0.1
TipThickness = 0.1
Span = 3.9624 #M, wing span (both sides)

Taper = TipChord/RootChord
ThicknessRatio = RootThickness/TipThickness
MAC = 2.0/3.0 * RootChord * (1 + Taper + Taper**2)/(1 + Taper)
WingArea = 0.5 * (RootChord + TipChord) * Span
AR = (Span**2)/WingArea

print(WingArea)
print(AR)
print(MAC)



HChord = 0.1 #M
VChord = 0.1 #M

WthickRoot = 0.1 # %
Hthick = 0.1 # %
Vthick = 0.1 # %

FuselageWidth = 0.3 # M
FuselageHeight = 0.15 # M
FuselageLength = 0.8 # M
FuselageDiameter = 0.5 * (FuselageWidth+FuselageHeight)


#### Cdo Simplified Approach: Cdo = Cfe * Swet/Sref
Sref = MAC * Span
Swet = 3.473 #M^2, wetted area
Cfe = 0.007 #Friction coef. for Cdo = Cfe * Swet/Sref
Cdo = Cfe * Swet/Sref
print(Cdo)

#### Cdo Combined Approach 
# Cdo = Sum(Cfc * FFc * Qc * Swetc/Sref)

Components = []
FF = []
Q = []
Cfc = []
Swetc = []

## Cdo Fuselage
Components.append("Fuselage")
Q.append()
Cfc.append()
Swetc.append()




Q =  [1.4, 1.04, 1.04, 1] # Table 13.4
FF = []
Sexp = Sref - FuselageWidth*MAC

ps = PolarSet.from_folder(r".\PyFoil\polars", airfoil="zv15_35")
print(ps.cl(alpha_deg=4.0, reynolds=350_000))
print(ps.best_ld(reynolds=350_000))





Atmosphere(GroundLevel).density
Atmosphere(GroundLevel+OrbitLevel).density
Atmosphere(GroundLevel+OrbitLevel).dynamic_viscosity