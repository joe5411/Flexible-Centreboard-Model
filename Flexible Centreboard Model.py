import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import time


#%% BEND-TWIST PARAMETERS

l = 0.875           # length [m]
a = 0.12             # "height" [m]
b = 0.008            # "width" [m]
E = 10e9            # Young's modulus [Pa / N/m^2]
G = 300e6            # shear modulus [Pa / N/m^2]
Iz = (np.pi/4)*(b**3)*a   # second moment of area about z-axis [m^4]
J = 0.25*(a*b*np.pi)*((a**2)+(b**2))    # polar moment of inertia [m^4]
lever = 0.037        # force lever [m]

mu = 1.19e-6        # kinematic viscosity [m^2/s]
u = 2               # fluid velocity [m/s]
Re = (u*2*a)/mu     # Reynold's number
print('Reynolds number = %.2f \n'%(Re))
AR = (2*l)**2/(l*2*a)

n_pl = 50           # number of point loads
n_e = n_pl + 1      # number of nodes

#%% AREA INCREMENTS

area_circle = np.pi*(0.17**2)*0.5

def areas(y):
    area = 0
    arealist = []
    for i in range(len(y)):
        if y[i] <= 0.705:
            area = (0.34*y[i])
            arealist.append(area)
        elif y[i] > 0.705:
            area1 = (0.705*0.34)
            y2 = np.round(y[i] - 0.705, 6)
            if y2 == 0.17:
                area2 = (np.pi*(0.17**2)*0.5)
            elif y2 <= 0.17:
                theta_rad = 2*np.arccos(y2/0.17)
                area2 = area_circle - ((0.17**2)*0.5*(theta_rad - np.sin(theta_rad)))
            area = area1 + area2 #+ area3
            arealist.append(area)

    newarealist = []
    for i in range(len(arealist) - 1):
        areaincr = np.round(arealist[i + 1] - arealist[i], 7)
        newarealist.append(areaincr)
    
    return np.array(newarealist)

#%% XFOIL AND XFLR5 DATA

dfII = pd.read_excel('Spanwise local Cl Curved Ends 2.xlsx', sheet_name=0, header=0)

leeway_angle = 5    # Degrees
CLfunc = interp1d(dfII.columns[1:], dfII.iloc[0, 1:])
CL = CLfunc(leeway_angle)
yvals = np.linspace(0, l, n_e)
area_vals = areas(yvals)
V = 0.5*1025*(u**2)*CL*area_vals

#%% 3D EFFECTS

def approximation1(s, df=dfII, frac=0.56, delete=0):
    fraclist = []
    for i in np.arange(0.25, 10, 0.25):
        newfrac = dfII[i].iloc[-1]/dfII[i].iloc[0]
        fraclist.append(newfrac)
    
    df1 = pd.DataFrame(fraclist)
    fracdrop = df1.mean()
    
    ylist = [1, fracdrop]
    xlist = [frac, 1]
    
    approxfunc = interp1d(xlist, ylist)
    
    factors = []
    
    for i in s:
        if i <= frac:
            factor = 1
            factors.append(factor)
        elif i > frac:
            factor = float(approxfunc(i))
            factors.append(factor)
    
    return s, factors

def approximation2(s):
    return (-0.8891*(s**3)) + (0.409*(s**2)) - (0.1411*s) + 1

V = list(V * approximation2(np.delete(yvals, 0)/l))

#%% BENDING MODEL

def y_before(V1, a1, x1):
    y = ((V1*(x1**2)*((3*a1)-x1))/(6*E*Iz))
    return y

def y_after(V1, a1, x1):
    y = ((V1*(a1**2)*((3*x1)-a1))/(6*E*Iz))
    return y

def bending(Vdata, n_e):
    column = ['Shear Force [N]']            # Beam elements
    elem = len(Vdata) + 1
    x = np.linspace(0, l, elem)
    r0 = -sum(Vdata)                        # Reaction at support
    Vdata.insert(0, r0)
    df = pd.DataFrame(Vdata, x, column)     # DataFrame
    dpoints = list(range(0, len(df)))
    x_vals = np.linspace(0, l, n_e)        # Deflection
    
    deflist = []
    
    for i in x_vals:
        mylist = []
        for j in dpoints:
            if df.index[j] >= i:    
                y = y_before(float(df.iloc[j]), df.index[j], i)
                mylist.append(y)
            elif df.index[j] < i:
                y = y_after(float(df.iloc[j]), df.index[j], i)
                mylist.append(y)
        yf = sum(mylist)
        deflist.append(yf)

    maxy = max(deflist)
    
    del Vdata[0]
    
    return deflist, maxy

#%% TWIST MODEL

#Torsion
def torque(L_val, lever_val):
    T = L_val*lever_val
    return T

#Angular Deflection of Ellipse Section
def angular_deformation_ellipse(x_val, T):
    phi = (((a**2) + (b**2))*T*x_val)/(np.pi*(a**3)*(b**3)*G)
    phi_deg = phi*(180/np.pi)
    return phi_deg

def twist(Vdata, n_e, lever): # ADD LEVER AS INPUT
    elem = len(Vdata) + 1
    x = list(np.linspace(0, l, elem))
    del(x[0])
    x = np.array(x)
    x_vals = np.linspace(0, l, n_e) # IS THIS NEEDED??
    df = pd.DataFrame(Vdata, x, columns=['Lift [N]'])
    dpoints = list(range(0, len(df)))
    
    # Angular Deflection at given point
    df['Angular Deformation [rad]'] = angular_deformation_ellipse(df.index, torque(df['Lift [N]'], lever))
    
    # Angular Deflections Along Beam
    angdef = []
    
    for i in x_vals:
        sumlist = []
        for j in dpoints:
            if df.index[j] >= i:
                phi = df.iloc[j, 1]*(i/df.index[j])
                sumlist.append(phi)
            elif df.index[j] < i:
                phi = df.iloc[j, 1]
                sumlist.append(phi)
        phif = sum(sumlist)
        angdef.append(phif)
    
    angdef = np.array(angdef) * 0.726
    
    if max(angdef) == 0:
        maxphi = min(angdef)
    elif max(angdef) > 0:
        maxphi = max(angdef)
    
    return list(angdef), maxphi

#%% FSI

def iteration(deflections, df=dfII):
    Vnew = []
    spanvals = np.linspace(0, l, len(deflections))
    load_points = np.delete(spanvals, 0)
    factorvals = approximation2(spanvals/l)
    AoA = np.array(np.delete(df.columns, 0))
    interval = AoA[1]
    for i in range(len(deflections)):
        ang = deflections[i] + leeway_angle
        factorval = factorvals[i]
        for j in AoA:
            if j < ang and j + interval > ang:
                x1 = j
                CL1 = dfII[j].iloc[0] * factorval
                x2 = j + interval
                CL2 = dfII[j + interval].iloc[0] * factorval
                xs = [x1, x2]
                ys = [CL1, CL2]
                func = interp1d(xs, ys)
                CLnew = func(ang)
                L = CLnew*0.5*1025*(u**2)
                Vnew.append(L)
    return list(np.array(Vnew)*area_vals)

start = time.time()

n_iter = 10
i = 1
maxphi_vals = []
maxy_vals = []
while i <= n_iter:
    phi_vals, maxphi = twist(V, n_e, lever)
    y_vals, maxy = bending(V, n_e)
    maxy_vals.append(maxy)
    maxphi_vals.append(maxphi)
    V = iteration(list(np.delete(phi_vals, 0)))
    i += 1

end = time.time()

total_time = end - start
print('Total Execution Time = ', total_time, 's')

iteration_number = list(range(n_iter + 1))
del(iteration_number[0])
plt.figure(figsize=[10, 5])
plt.plot(iteration_number, maxphi_vals, 'kx-')
plt.xlabel('Iteration Number')
plt.ylabel('Maximum Angular Deflection [degrees]')
plt.grid(which='both')
plt.show()

plt.figure(figsize=[10, 5])
plt.plot(iteration_number, maxy_vals, 'kx-')
plt.xlabel('Iteration Number')
plt.ylabel('Tip Deflection [m]')
plt.grid(which='both')
plt.show()

#%% RESULTS FOR REPORT

print('Maximum deflection = ', maxy_vals[n_iter - 1], 'm')
print('Maximum angular deformation = ', maxphi_vals[n_iter - 1], 'degrees')
print('Total lift = ', sum(V), 'N')
