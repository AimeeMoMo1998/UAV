#!/usr/bin/python
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import salem
from scipy.ndimage import gaussian_filter
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
from scipy.spatial import cKDTree as KDTree
from scipy.signal import find_peaks
from scipy.stats import mode
from scipy.interpolate import griddata
#import datetime


###### Constants for UTM conversion #######
import numpy as mathlib
use_numpy = True
#__all__ = ['to_latlon', 'from_latlon']
K0 = 0.9996
E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)
SQRT_E = mathlib.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E
M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)
R = 6378137
ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"



###Function to convert Lat/Lon to CH coordinates
def LatLon2CH(lat,lon):
    #convert from WGS84 to CH LV03
    #equations from "Formulas and constants
    #for the calculation of the swiss conformal cylindrical projection
    #and for the transformation between coordinate systems
    #DAH 10-Nov-2019 original Matlab, 15-June-2020 to Python
    phi = lat*3600    #convert to arcseconds
    lam = lon*3600    #convert to arcseconds
    phip = (phi-169028.66)/10000
    lamp = (lam-26782.5)/10000
    E = 2600072.37 + 211455.93*lamp-10938.51*lamp*phip-0.36*lamp*phip**2-44.54*phip**3
    y = E-2000000.00
    N = 1200147.07+308807.95*phip+3745.25*lamp**2+76.63*phip**2-194.56*lamp**2*phip+119.79*phip**3
    x = N- 1000000.00;
    return x,y

## Define trig functions in degrees
def cosd(deg):
    rad=np.cos(deg*np.pi/180)
    return rad
def sind(deg):
    rad=np.sin(deg*np.pi/180)
    return rad
def tand(deg):
    rad=np.tan(deg*np.pi/180)
    return rad

## Fresnel specular surface reflectivity
def FresnelRefl(tetain, epsin, epsout):
    #Calculate Fresnel Surface Reflectivities
    #tetain = incoming angle (deg)
    #epsin  = permittivity at incoming layer (Epsilon_above in (8))
    #epsout = permittivity at outgoing layer (Epsilon_below in (8))
    A = cosd(tetain)
    B = np.sqrt(1-((1-A**2)*(epsin/epsout)))
    sh = (np.abs((((np.sqrt(epsin))*A)-((np.sqrt(epsout))*B))/(((np.sqrt(epsin))*A)+((np.sqrt(epsout))*B))))**2
    sv = (np.abs((((np.sqrt(epsout))*A)-((np.sqrt(epsin))*B))/(((np.sqrt(epsout))*A)+((np.sqrt(epsin))*B))))**2

# rough interface reflectivities
    hG = 0.2
    #qG = hG*0.05
    qG = 0.1
    #qG = 0
    nGv = 0
    nGh = 0
    sh     = ((sh*(1 - qG)) + (qG*sv))*(np.exp(-hG*((cosd(tetain))**nGh))) # H pol.
    sv     = ((sv*(1 - qG)) + (qG*sh))*(np.exp(-hG*((cosd(tetain))**nGv))) # V pol.
    return sv,sh

def InterfaceRefl(tetain, epsin, epsout, hG, qG, nGv, nGh):
# hG: effective surface roughness dimensionless parameter
# qG: polarization coupling factor.
# nV and nH: polarization-dependent parameters introduced to better account for multi-angular and dual-polarization measurements
    A      = cosd(tetain)
    B      = np.sqrt(1-((1-A**2)*(epsin/epsout)))
    sFresh = (np.abs((((np.sqrt(epsin))*A)-((np.sqrt(epsout))*B))/(((np.sqrt(epsin))*A)+((np.sqrt(epsout))*B))))**2
    sFresv = (np.abs((((np.sqrt(epsout))*A)-((np.sqrt(epsin))*B))/(((np.sqrt(epsout))*A)+((np.sqrt(epsin))*B))))**2
# rough interface reflectivities
    sh     = ((sFresh*(1 - qG)) + (qG*sFresv))*(np.exp(-hG*((cosd(tetain))**nGh))) # H pol.
    sv     = ((sFresv*(1 - qG)) + (qG*sFresh))*(np.exp(-hG*((cosd(tetain))**nGv))) # V pol.
    return sv,sh

def ss_tauomega(W_s,T_s,T_v,alpha,tau,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = (Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2
    return cf

def tt_tauomega(tau,T_s,T_v,alpha,W_s,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = (Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2
    return cf

def st_tauomega(opt,T_s,T_v,alpha,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    W_s = opt[0]
    tau = opt[1]
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = (Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2
    return cf

def st_tauomega_all(opt,T_s,T_v,alpha,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    W_s = opt[0]
    tau = opt[1]
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = np.nansum((Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2)
    return cf

def sto_tauomega_all(opt,T_s,T_v,alpha,Tbhi,Tbvi):
    #Topp et al permittivity
    W_s = opt[0]
    tau = opt[1]
    omega = opt[2]
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = np.nansum((Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2)
    return cf
def stt_tauomega_all(opt,alpha,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    W_s = opt[0]
    tau = opt[1]
    T_s = opt[2] # rerieve single T_g value for veg and soil
    T_v = opt[2]
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = np.nansum((Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2)
    return cf

def stot_tauomega_all(opt,alpha,Tbhi,Tbvi):
    #Topp et al permittivity
    W_s = opt[0]
    tau = opt[1]
    omega = opt[2]
    T_s = opt[3] # rerieve single T_g value for veg and soil
    T_v = opt[3]
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = np.nansum((Tbhf-Tbhi)**2+(Tbvf-Tbvi)**2)
    return cf

def tauomega_all_tb(W_s,tau,T_s,T_v,alpha,omega):
    #Topp et al permittivity
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    return Tbhf,Tbvf

def ss_tauomegav(W_s,T_s,T_v,alpha,tau,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = (Tbvf-Tbvi)**2
    return cf

def tt_tauomegav(tau,T_s,T_v,alpha,W_s,omega,Tbhi,Tbvi):
    #Topp et al permittivity
    eps_s=3.03+9.3*W_s+146.0*W_s**2-76.7*W_s**3
    sv,sh = FresnelRefl(alpha,1,eps_s)
    e_sh = 1.-sh
    e_sv = 1.-sv
    Tbhf = e_sh*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sh)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    Tbvf = e_sv*T_s*np.exp(-tau/(cosd(alpha)))+(1-omega)*T_v*(1-np.exp(-tau/cosd(alpha)))+(1-e_sv)*(1-omega)*T_v*(1-np.exp(-tau/(cosd(alpha))))*np.exp(-tau/cosd(alpha))
    cf = (Tbvf-Tbvi)**2
    return cf

def RetWs(Tbh,Tbv,T_g,alpha,tau,omega,varminw,varmaxw,mode='dual'):
    Ws = np.empty_like(Tbv)
    cfv = np.empty_like(Tbv)
    for i in range(len(Tbh)):
        if np.isnan(Tbh[i]) | np.isnan(Tbv[i]) | np.isnan(alpha[i]):
            Ws[i] = np.nan
            cfv[i] = np.nan
        else:
            #print(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i])
            if mode=='v':
                Wsopt,cfval,ierr,numfunc = scipy.optimize.fminbound(ss_tauomegav,varminw,varmaxw,args=(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i]),full_output=True)
            else:
                Wsopt,cfval,ierr,numfunc = scipy.optimize.fminbound(ss_tauomega,varminw,varmaxw,args=(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i]),full_output=True)
            #Wsopt = scipy.optimize.fminbound(ss_tauomega,0,1,args=(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i]))
            Ws[i] = Wsopt
            cfv[i] = cfval
    return Ws,cfv

def RetTau(Tbh,Tbv,T_g,alpha,W_s,omega,varmintau,varmaxtau,mode='dual'):
    #bounds0 = scipy.optimize.Bounds(varmintau,varmaxtau)
    tau = np.empty_like(Tbv)
    cfv = np.empty_like(Tbv)
    for i in range(len(Tbh)):
        if np.isnan(Tbh[i]) | np.isnan(Tbv[i]) | np.isnan(alpha[i]):
            tau[i] = np.nan
            cfv[i] = np.nan
        else:
            #print(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i])
            if mode=='v':
                tauopt,cfval,ierr,numfunc = scipy.optimize.fminbound(tt_tauomegav,varmintau,vartmaxtau,args=(T_g,T_g,alpha[i],W_s,omega,Tbh[i],Tbv[i]),full_output=True)
            else:
                tauopt,cfval,ierr,numfunc = scipy.optimize.fminbound(tt_tauomega,varmintau,varmaxtau,args=(T_g,T_g,alpha[i],W_s,omega,Tbh[i],Tbv[i]),full_output=True)
            #Wsopt = scipy.optimize.fminbound(ss_tauomega,0,1,args=(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i]))
            tau[i] = tauopt
            cfv[i] = cfval
    return tau,cfv


def RetTauWs(Tbh,Tbv,T_g,alpha,omega,varminw,varmintau,varmaxw,varmaxtau):
    bounds0 = scipy.optimize.Bounds([varminw,varmintau],[varmaxw,varmaxtau])
    Ws = np.empty_like(Tbv)
    Tau = np.empty_like(Tbv)
    cfv = np.empty_like(Tbv)
    x0 = np.array([0.25,0.05])
    for i in range(len(Tbh)):
        if np.isnan(Tbh[i]) | np.isnan(Tbv[i]) | np.isnan(alpha[i]):
            Ws[i] = np.nan
            cfv[i] = np.nan
            Tau[i] = np.nan
        else:
            #print(T_g,T_g,alpha[i],tau,omega,Tbh[i],Tbv[i])
            #optimizeresult = scipy.optimize.minimize(st_tauomega,x0,args=(T_g,T_g,alpha[i],omega,Tbh[i],Tbv[i]),method='trust-constr',bounds=bounds0)
            optimizeresult = scipy.optimize.minimize(st_tauomega,x0,args=(T_g,T_g,alpha[i],omega,Tbh[i],Tbv[i]),bounds=bounds0,options={'maxiter':200})
            Ws[i] = optimizeresult.x[0]
            Tau[i] = optimizeresult.x[1]
            cfval = optimizeresult.fun
            cfv[i] = cfval
            #print('Tau= '+str(Tau[i]))
            #print('WS = '+str(Ws[i]))
    return Ws,Tau,cfv

def RetTauWsAll(Tbh,Tbv,T_g,alpha,omega,varminw,varmintau,varmaxw,varmaxtau):
    bounds0 = scipy.optimize.Bounds([varminw,varmintau],[varmaxw,varmaxtau])
    x0 = np.array([0.25,0.05])
    optimizeresult = scipy.optimize.minimize(st_tauomega_all,x0,args=(T_g,T_g,alpha,omega,Tbh,Tbv),bounds=bounds0,options={'maxiter':500})
    Ws = optimizeresult.x[0]
    Tau = optimizeresult.x[1]
    cfval = optimizeresult.fun
    return Ws,Tau,cfval

def RetTauWsOmegaAll(Tbh,Tbv,T_g,alpha,varminw,varmintau,varmaxw,varmaxtau):
    bounds0 = scipy.optimize.Bounds([varminw,varmintau,0],[varmaxw,varmaxtau,0.5])
    x0 = np.array([0.25,0.05,0.05])
    optimizeresult = scipy.optimize.minimize(sto_tauomega_all,x0,args=(T_g,T_g,alpha,Tbh,Tbv),bounds=bounds0,options={'maxiter':500})
    Ws = optimizeresult.x[0]
    Tau = optimizeresult.x[1]
    Omega = optimizeresult.x[2]
    cfval = optimizeresult.fun
    return Ws,Tau,Omega,cfval

def RetTauWsOmegaTgAll(Tbh,Tbv,alpha,varminw,varmintau,varminomega,varminTgfit,varmaxw,varmaxtau,varmaxomega,varmaxTgfit):
    bounds0 = scipy.optimize.Bounds([varminw,varmintau,varminomega,varminTgfit],[varmaxw,varmaxtau,varmaxomega,varmaxTgfit])
 # initial guess  (Ws, Tau, Omega, T_g)
    x0 = np.array([0.25,0.05,0.05,300.])
    optimizeresult = scipy.optimize.minimize(stot_tauomega_all,x0,args=(alpha,Tbh,Tbv),bounds=bounds0,options={'maxiter':1e8})
    Ws = optimizeresult.x[0] 
    Tau = optimizeresult.x[1]
    Omega = optimizeresult.x[2]
    T_g = optimizeresult.x[3]
    #T_v = optimizeresult.x[4]
    cfval = optimizeresult.fun
    return Ws,Tau,Omega,T_g,cfval

def RetTauWsTgAll(Tbh,Tbv,alpha,omega,varminw,varmintau,varminTgfit,varmaxw,varmaxtau,varmaxTgfit):
    bounds0 = scipy.optimize.Bounds([varminw,varmintau,varminTgfit],[varmaxw,varmaxtau,varmaxTgfit])
 # initial guess  (Ws, Tau, Omega, T_g)
    x0 = np.array([0.25,0.05,300.])
    optimizeresult = scipy.optimize.minimize(stt_tauomega_all,x0,args=(alpha,omega,Tbh,Tbv),bounds=bounds0,options={'maxiter':1e8})
    Ws = optimizeresult.x[0]
    Tau = optimizeresult.x[1]
    T_g = optimizeresult.x[2]
    #T_v = optimizeresult.x[4]
    cfval = optimizeresult.fun
    return Ws,Tau,T_g,cfval

def madfilter(a,nmad,madwindowsz):
    #### MAD filter ####
    arollmed = pd.Series(a).rolling(madwindowsz, min_periods=3, center=True).median()
    arollstd = pd.Series(a).rolling(madwindowsz, min_periods=3, center=True).std()
    prefilt=np.sum(np.isnan(a))
    a[(a <= arollmed-nmad*arollstd) | (a >= arollmed+nmad*arollstd)]=np.nan
    postfilt = np.sum(np.isnan(a))
    nfilt = postfilt-prefilt
    return a,nfilt

def getTCL(hclfile,vclfile):
    hpars = np.genfromtxt(hclfile)
    vpars = np.genfromtxt(vclfile)
    return hpars,vpars

#calculate most common 1 degree yaw elements for two main flight paths
def yawpks(yaw):
    yawint = np.rint(yaw)
    mres = mode(yawint)
    mode1 = mres.mode
    yawint=yawint[yawint!=mode1]
    mres2 = mode(yawint)
    mode2 = mres2.mode
    return mode1,mode2

def getPlotMinMax(x,y,border):
    xrange = np.nanmax(x)-np.nanmin(x)
    yrange = np.nanmax(y)-np.nanmin(y)
    txmin = np.nanmin(x)-border*xrange
    txmax = np.nanmax(x)+border*xrange
    tymax = np.nanmax(y)+border*yrange
    tymin = np.nanmin(y)-border*yrange
    xsize = txmax-txmin
    print('X size of plot area: ',xsize)
    ysize = tymax-tymin
    print('Y size of plot area: ',ysize)
    #make plot boundaries square and properly adjusted to 0m-1200m google grid
    if xsize>ysize:
        tymaxp = (tymax+tymin)/2+xsize/2
        tyminp = (tymax+tymin)/2-xsize/2
        txmaxp = txmax
        txminp = txmin
        if tymaxp > 1200:
            tyminp = tyminp+(tymaxp-1200)
            tymaxp = 1200
        if tyminp < 0:
            tymaxp = tymaxp-(tyminp)
            tyminp = 0
    elif ysize>=xsize:
        txmaxp = (txmax+txmin)/2+ysize/2
        txminp = (txmax+txmin)/2-ysize/2
        tyminp = tymin
        tymaxp = tymax
        if txmaxp > 1200:
            txminp = txminp+(txmaxp-1200)
            txmaxp = 1200
        if txminp < 0:
            txmaxp = txmaxp-(txminp)
            txminp = 0
    return txminp,txmaxp,tyminp,tymaxp

def getMapParams(latr,lonr):
    border = 0.1 #percent beyond points
    latrange = np.nanmax(latr)-np.nanmin(latr)      # define range of latitude for plot
    lonrange = np.nanmax(lonr)-np.nanmin(lonr)      # define range of longitude for plot
    maplonmin = np.nanmin(lonr)-border*lonrange     # define bounds of lat/lon
    maplonmax = np.nanmax(lonr)+border*lonrange
    maplatmin = np.nanmin(latr)-border*latrange
    maplatmax = np.nanmax(latr)+border*latrange
    # get google map from salem function
    g = GoogleVisibleMap(x=[maplonmin, maplonmax], y=[maplatmin, maplatmax],scale=2,maptype='satellite',key='AIzaSyDRWkMnt5bRPlaOOUE-G8er4RpdY40N7vM')  # try out also: 'terrain'
    ggl_img = g.get_vardata()
    # sm is the parameter that is returned ready to map
    sm = Map(g.grid, factor=1, countries=False)
    sm.set_rgb(ggl_img)  # add the background rgb image
    x, y = sm.grid.transform(lonr, latr) #get x and y of google map grid
    #sm.set_scale_bar(location=(0.88, 0.94))  # add scale
    txminp,txmaxp,tyminp,tymaxp = getPlotMinMax(x,y,border) # Get min and max for square plot
    return sm,x,y,txminp,txmaxp,tyminp,tymaxp,maplonmin,maplonmax,maplatmin,maplatmax

###### This function creates a scatter plot of raw data over the google map image
def plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,plotvar,varmin,varmax,fname,varname,unitstr,fpath,savefigbool):
    print("Creating Map Overlay of Raw Data: "+varname+"...",end='',flush=True)
    f, (ax1) = plt.subplots(1,1,figsize=(9.5,9.5))  # create figure for plotting
    sm.visualize(ax=ax1)  # plot it
    ax1.set_title(varname)
    if unitstr=='$T_B (K)$':
        cm = plt.cm.get_cmap('Spectral_r')
    else:
        cm = plt.cm.get_cmap('Spectral')
    scat = ax1.scatter(x, y, c=plotvar, s=100, edgecolors='none', linewidths=1,cmap=cm,alpha=1,vmin=varmin,vmax=varmax)
    clb = plt.colorbar(scat,fraction=0.046,pad=0.04)
    clb.set_label(unitstr, labelpad=-40, y=1.05, rotation=0)
    plt.tight_layout()
    plt.xlim((txminp,txmaxp))
    plt.ylim((tymaxp,tyminp))
    if savefigbool:
        plt.savefig(fpath+fname)
    else:
        plt.show(block=False)
    print("Done.")

###### This function applies gridding and Gaussian filtering
def GaussFilterTest(x,y,plotvar,sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0):
    xi = np.linspace(x.min(),x.max(),300)
    yi = np.linspace(y.min(),y.max(),300)
    # grid the data.
    print("Interpolating Data to Grid...",end='',flush=True)
    zi = griddata((np.array(x), np.array(y)), plotvar, (xi[None,:], yi[:,None]), method='linear')
    zinan = griddata((np.array(x), np.array(y)), plotvar, (xi[None,:], yi[:,None]), method='linear')
    print("Done.")
    #zi[zi>plotvar.max()]=np.nan
    #zi[zi<plotvar.min()]=np.nan
    #### Apply 2D Gaussian Filter to raw soil moisture data
    print("Applying 2D Gaussian Filter...",end='',flush=True)
    #zifilt = gaussian_filter(zi,sigma=sigma0,mode='mirror')
    #zifilt = filter_nan_gaussian_conserving(zi,sigma=sigma0)
    zifilt = filter_nan_gaussian_david(zi,sigma=sigma0)
    #put nans back from original linear interp
    zifilt[np.isnan(zinan)]=np.nan
    print("Done.")
    return xi,yi,zifilt

def GaussFilterTestConv(x,y,xiv,yiv,plotvar,sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0):
    # grid the data.
    xi,yi = np.meshgrid(xiv,yiv)
    print("Interpolating Data to Grid...",end='',flush=True)
    zi = griddata((np.array(x), np.array(y)), plotvar, (xi, yi), method='linear')
    #zinan = griddata((np.array(x), np.array(y)), plotvar, (xi[None,:], yi[:,None]), method='linear')
    # Makes sure we dont interpolate over convex shapes
    tree = KDTree(np.c_[x, y])
    dist, _ = tree.query(np.c_[xi.ravel(), yi.ravel()], k=1)
    dist = dist.reshape(xi.shape)
    #print(dist.max())
    #print(dist.min())
    #print(np.median(dist))
    zi[dist > distthresh*np.median(dist)] = np.nan
    print("Done.")
    #zi[zi>plotvar.max()]=np.nan
    #zi[zi<plotvar.min()]=np.nan
    #### Apply 2D Gaussian Filter to raw soil moisture data
    print("Applying 2D Gaussian Filter...",end='',flush=True)
    #zifilt = gaussian_filter(zi,sigma=sigma0,mode='mirror')
    zifilt = filter_nan_gaussian_david(zi,sigma=sigma0)
    #zifilt = filter_nan_gaussian_conserving(zi,sigma=sigma0)
    #put nans back from original linear interp
    #zifilt[np.isnan(zinan)]=np.nan
    print("Done.")
    return xi,yi,zifilt

###### This functions plots the contour map of gridded/cleaned/averaged data
def plotcontmap(xi,yi,zi,bool2b1,sm,varmin,varmax,txminp,txmaxp,tyminp,tymaxp,fname,varname,unitstr,darkmode,fpath,savefigbool):
    if varmax<5.0:
        nconts = (varmax-varmin)/0.01
    else:
        nconts = 4.*(varmax-varmin)
    if bool2b1=="False":  #if we only want one big map image
        f, (ax1) = plt.subplots(1,1,figsize=(9.5,9.5))
        if darkmode=='True':
            ax1.set_facecolor((49/255.,51/255.,53/255.))   #comment out for transparents with white text
            fontcolor='white'
        else:
            ax1.set_facecolor((255/255.,255/255.,255/255.))     #comment out for transparent with black text
            fontcolor = 'black'
        #sm.set_rgb(ggl_img)  # add the background rgb image
        sm.set_scale_bar(location=(0.7, 0.73),color='white')  # add scale
        sm.visualize(ax=ax1)  # plot it
        levels = np.linspace(varmin,varmax,int(nconts+1))
        if unitstr=='$K$':  #if temperatures are being plot reverse the colorbar for blue cold red hot
            cm = plt.cm.get_cmap('Spectral_r')
        else:
            cm = plt.cm.get_cmap('Spectral')
        #plot contour of filtered data
        CS = plt.contourf(xi,yi,zi,levels,cmap=cm,origin='upper',alpha=0.9,antialiased = True)

        clb = plt.colorbar(CS,fraction=0.046,pad=0.04)

        cbytick = plt.getp(clb.ax.axes,'yticklabels')
        plt.setp(cbytick,color=fontcolor)
        ytl = plt.getp(ax1,'yticklabels')
        plt.setp(ytl,color=fontcolor)
        xtl = plt.getp(ax1,'xticklabels')
        plt.setp(xtl,color=fontcolor)
        clb.set_label(unitstr, labelpad=-40, y=1.05, rotation=0,color=fontcolor)
        ax1.set_title(varname,color=fontcolor)
        plt.tight_layout()
        plt.xlim((txminp,txmaxp))
        plt.ylim((tymaxp,tyminp))
        #for c in CS.collections:
    #        c.set_edgecolor("face",alpha=0.7)
        if savefigbool:
            plt.savefig(fpath+fname,transparent=True,facecolor=ax1.get_facecolor(),edgecolor='none')
            #plt.savefig(fpath+fname,transparent=True)      #for transparent output background
        else:
            plt.show(block=False)
        return f
    elif bool2b1=="True":  #True for 2 by 1 image with blank map and overlay
        f, (ax1,ax2) = plt.subplots(1,2,figsize=(17,7))
        # make a map of the same size as the image (no country borders)
        sm.set_scale_bar(location=(0.5, 0.5))  # add scale
        sm.visualize(ax=ax1)  # plot it
        #plt.sca(ax1)
        plt.xlim((txminp,txmaxp))
        plt.ylim((tymaxp,tyminp))
        plt.tight_layout()
        #plt.sca(ax2)
        sm.visualize(ax=ax2)  # plot it
        levels = np.linspace(varmin,varmax,31)
        cm = plt.cm.get_cmap('Spectral')
        #plot contour of filtered data
        CS = plt.contourf(xi,yi,zi,levels,cmap=cm,origin='upper')
        clb = plt.colorbar(CS,fraction=0.046,pad=0.04)
        clb.set_label(unitstr, labelpad=-40, y=1.05, rotation=0)
        plt.tight_layout()
        plt.xlim((txminp,txmaxp))
        plt.ylim((tymaxp,tyminp))
        if savefigbool:
            plt.savefig(fpath+fname)
        else:
            plt.show(block=False)
def filter_nan_gaussian_david(arr, sigma):
    """Allows intensity to leak into the nan area.
    According to Davids answer:
        https://stackoverflow.com/a/36307291/7128154
    """
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)

    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = np.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[np.isnan(arr)] = np.nan
    return gauss
def filter_nan_gaussian_conserving(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss

def plottbtime(TBH,TBV,datet,fpath,savefigbool):
    print("Creating Brightness Temp Series of Raw Data...",end='',flush=True)
    f, (ax1,ax2) = plt.subplots(2,1,figsize=(9.5,9.5))  # create figure for plotting
    lins = ax1.plot(datet, TBH)
    ax1.set_ylabel('H pol TB (K)')
    ax1.grid()
    plt.tight_layout()
    #plt.xlim((txminp,txmaxp))
    #plt.ylim((tymaxp,tyminp))
    lins = ax2.plot(datet, TBV)
    ax2.set_ylabel('V pol TB (K)')
    ax2.grid()
    if savefigbool:
        plt.savefig(fpath+'TBs.png')
    else:
        plt.show(block=False)
    print("Done.")

############################ Conversion to UTM package ########################


def in_bounds(x, lower, upper, upper_strict=False):
    if upper_strict and use_numpy:
        return lower <= mathlib.min(x) and mathlib.max(x) < upper
    elif upper_strict and not use_numpy:
        return lower <= x < upper
    elif use_numpy:
        return lower <= mathlib.min(x) and mathlib.max(x) <= upper
    return lower <= x <= upper


def check_valid_zone(zone_number, zone_letter):
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError('zone letter out of range (must be between C and X)')


def mixed_signs(x):
    return use_numpy and mathlib.min(x) < 0 and mathlib.max(x) >= 0


def negative(x):
    if use_numpy:
        return mathlib.max(x) < 0
    return x < 0


def mod_angle(value):
    """Returns angle in radians to be between -pi and pi"""
    return (value + mathlib.pi) % (2 * mathlib.pi) - mathlib.pi


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, strict=True):
    """This function converts UTM coordinates to Latitude and Longitude

        Parameters
        ----------
        easting: int or NumPy array
            Easting value of UTM coordinates

        northing: int or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented with global map numbers of a UTM zone
            numbers map. For more information see utmzones [1]_

        zone_letter: str
            Zone letter can be represented as string values.  UTM zone
            designators can be seen in [1]_

        northern: bool
            You can set True or False to set this parameter. Default is None

        strict: bool
            Raise an OutOfRangeError if outside of bounds

        Returns
        -------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).


       .. _[1]: http://www.jaworski.ca/utmzones.htm

    """
    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if strict:
        if not in_bounds(easting, 100000, 1000000, upper_strict=True):
            raise OutOfRangeError('easting out of range (must be between 100,000 m and 999,999 m)')
        if not in_bounds(northing, 0, 10000000):
            raise OutOfRangeError('northing out of range (must be between 0 m and 10,000,000 m)')

    check_valid_zone(zone_number, zone_letter)

    if zone_letter:
        zone_letter = zone_letter.upper()
        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * mathlib.sin(2 * mu) +
             P3 * mathlib.sin(4 * mu) +
             P4 * mathlib.sin(6 * mu) +
             P5 * mathlib.sin(8 * mu))

    p_sin = mathlib.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = mathlib.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = mathlib.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = E_P2 * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                 d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    longitude = mod_angle(longitude + mathlib.radians(zone_number_to_central_longitude(zone_number)))

    return (mathlib.degrees(latitude),
            mathlib.degrees(longitude))


def from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None):
    """This function converts Latitude and Longitude to UTM coordinate

        Parameters
        ----------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).

        force_zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. You may force conversion to be included within one
            UTM zone number.  For more information see utmzones [1]_

        force_zone_letter: str
            You may force conversion to be included within one UTM zone
            letter.  For more information see utmzones [1]_

        Returns
        -------
        easting: float or NumPy array
            Easting value of UTM coordinates

        northing: float or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented by global map numbers of a UTM zone
            numbers map. More information see utmzones [1]_

        zone_letter: str
            Zone letter is represented by a string value. UTM zone designators
            can be accessed in [1]_


       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    if not in_bounds(latitude, -80.0, 84.0):
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not in_bounds(longitude, -180.0, 180.0):
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = mathlib.radians(latitude)
    lat_sin = mathlib.sin(lat_rad)
    lat_cos = mathlib.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    if force_zone_letter is None:
        zone_letter = latitude_to_zone_letter(latitude)
    else:
        zone_letter = force_zone_letter

    lon_rad = mathlib.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = mathlib.radians(central_lon)

    n = R / mathlib.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * mathlib.sin(2 * lat_rad) +
             M3 * mathlib.sin(4 * lat_rad) -
             M4 * mathlib.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if mixed_signs(latitude):
        raise ValueError("latitudes must all have the same sign")
    elif negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(latitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if use_numpy and isinstance(latitude, mathlib.ndarray):
        latitude = latitude.flat[0]

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if use_numpy:
        if isinstance(latitude, mathlib.ndarray):
            latitude = latitude.flat[0]
        if isinstance(longitude, mathlib.ndarray):
            longitude = longitude.flat[0]

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3


import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

def exportgeotiff(lat, lon, array,filename):

    # For each pixel I know it's latitude and longitude.
    # As you'll see below you only really need the coordinates of
    # one corner, and the resolution of the file.

    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    nrows,ncols = np.shape(array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???

    output_raster = gdal.GetDriverByName('GTiff').Create(filename,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                 # Anyone know how to specify the
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster
    output_raster.FlushCache()
    return

def find_nearest(array, value):
    #find value nearest to value in array (for finding PoLRa coordinate TIFF value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]

def do_kdtree(y_array,x_array,points):
    combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
    points_list = list(points)
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

def load_geotiff(filename,xgrid,ygrid,sm,xiv,yiv,x,y,lat,lon,sigma0):
    #filename = path and filename to geoTiff file
    # xdarr = array of UTM coordinates (from PoLRa) for regrid
    # ydarr = array of UTM coordinates (from PoLRa)
    # sigma0 = Gaussian blur parameter
    dataset = gdal.Open(filename)
    print("Loaded GeoTiff number of channels: ",dataset.RasterCount,"  X size: ",dataset.RasterXSize,"   Y size: ",dataset.RasterYSize)
    # Note GetRasterBand() takes band no. starting from 1 not 0
    #load and get band from geotiff
    band = dataset.GetRasterBand(1)
    gt = dataset.GetGeoTransform()
    arr = band.ReadAsArray()
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    #get parameters from geoTiff
    gt = dataset.GetGeoTransform()
    # define limits of geotiff
    minlon = gt[0]
    minlat = gt[3] + width*gt[4] + height*gt[5]
    maxlon = gt[0] + width*gt[1] + height*gt[2]
    maxlat = gt[3]
    (xu,yu,zonenum,zonelet)=from_latlon(np.array([minlat,maxlat]),np.array([minlon,maxlon]))
    # og file coord. arrays
    latarr = np.linspace(minlat,maxlat,arr.shape[0])
    lonarr = np.linspace(minlon,maxlon,arr.shape[1])
    longrid,latgrid = np.meshgrid(latarr,lonarr)

    # downsample image (for time savings)
    print("Size of GeoTIFF pre-downsample: ",arr.shape)
    ds = int(5) # downsample ratio
    downsamp = arr[::ds,::ds]
    #xgriddown = xgrid[::ds,::ds]
    #ygriddown = ygrid[::ds,::ds]
    longriddown = latgrid[::ds,::ds]
    latgriddown = longrid[::ds,::ds]

    print("Size of GeoTIFF post-downsample: ",downsamp.shape)
    print("Getting overlay coordinates for GeoTIFF File.")

    xmap, ymap = sm.grid.transform(longriddown.flatten(), latgriddown.flatten()) #get x and y of google map grid

    polgrid = griddata((np.array(xmap),np.array(ymap)),np.fliplr(np.transpose(downsamp)).flatten(),(xgrid,ygrid), method='linear')
    polgridblur = filter_nan_gaussian_david(polgrid,sigma=sigma0)

    #rasmple at radiometer points
    # find closest coordinate to each lat lon point
    indexes = do_kdtree(xgrid,ygrid,np.column_stack((x,y)))
    polpts = polgridblur.flatten()[indexes-1]

# UTM vectors,Lat/Lon vectors, UTM coord array, lat/lon coord array
    return polgrid,polgridblur,polpts

###### This function creates a scatter plot of raw data over the google map image
def plotflightmap(sm,x,y,xr,yr,txminp,txmaxp,tyminp,tymaxp,fname,fpath,savefigbool):
    print("Creating Map Overlay of Raw Flightpath.",end='',flush=True)
    f, (ax1) = plt.subplots(1,1,figsize=(9.5,9.5))  # create figure for plotting
    sm.visualize(ax=ax1)  # plot it
    ax1.set_title("Flightpath (projected)")
    scat = ax1.scatter(x, y, c='black', s=10, edgecolors='none', linewidths=1,alpha=1)
    scat2 = ax1.scatter(xr, yr, c='blue', s=10, edgecolors='none', linewidths=1,alpha=1)
    plt.tight_layout()
    plt.xlim((txminp,txmaxp))
    plt.ylim((tymaxp,tyminp))
    if savefigbool:
        plt.savefig(fpath+fname)
    else:
        plt.show(block=False)
    print("Done.")
