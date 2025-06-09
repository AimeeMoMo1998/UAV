#!/usr/bin/python3
from libdrone import *
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.signal import argrelmax
# import warnings
# np.warnings = warnings
fpath = sys.argv[1]
flogfile = sys.argv[2]
radfile = sys.argv[3]
sys.path.append(fpath)

#define some defaults before importing config
cablecorrbool = 'True'

from config import *
## import graphic handler if saving figures as pngs
if savefigbool=='True':
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

fieldpath = fpath #output files to same directory as data?
print("Loading Flight Log...")

# apply a time offset if defined in config file
if 'toffset' not in locals():
    print("no time offset")
    toffset = 0

#### check extension of flightlog: #####

# .bin is MAVLink format (needs conversion)
# .csv is Litchi format
if flogfile.endswith('.bin') or flogfile.endswith('.BIN') or flogfile.endswith('.Bin'):
    logtype="MAV"
    UTCerror = 1 # PoLRa UTC timestamp seems 1 hour ahead?
    #convert with mavlogdump.py and pymavlink package
    print("Detected MAVLink format flightlog (Aurelia, etc..).")
    cmd = "python mavlogdump.py --format=csv --types=AHR2 "+fpath+flogfile+" > "+fpath+"templog.csv"
    print("Running: ",cmd," to decode MAVLink logfile")
    os.system(cmd)
    flog = np.genfromtxt(fpath+"templog.csv",delimiter=",",skip_header=1,dtype=str,usecols=range(11),invalid_raise=False,filling_values=np.nan)
    dronetnum = flog[:,0].astype(float) # OR COLUMN 12 is Local
    lat = flog[:,6].astype(float)   # get drone latitude
    lon = flog[:,7].astype(float)   # get drone longitude
    altabs = flog[:,5].astype(float)   #absolute altitude
    alt = altabs-np.mean(altabs[0:10])
    #speed = flog[:,4].astype(float)
    #flystate = flog[:,66].astype(float)
    pitch = flog[:,3].astype(float)
    roll = flog[:,2].astype(float)
    yaw = flog[:,4].astype(float)

    dronet = [datetime.datetime.utcfromtimestamp(ts) for ts in dronetnum]
    lonraw = lon # save raw lat lon for flightpath plot
    latraw = lat
    print("")
    print("Drone Start Time [UTC]:")
    print(dronet[0])
    print("")
    print("Drone End Time:")
    print(dronet[-1])
    print("")
    print("Done.")

#### v2.csv is Litchi's flightlog ####
if flogfile.endswith('v2.csv') or flogfile.endswith('V2.CSV'):
    logtype="Litchi"
    UTCerror = 0
    #call function to load and convert Litchi format flightlog
    print("Detected .csv format flightlog, proceeding with Litchi format.")

    flog = np.genfromtxt(fpath+flogfile,delimiter=",",skip_header=1,dtype=str,usecols=range(67),invalid_raise=False,filling_values=np.nan)
    datestrings = flog[:,11] # OR COLUMN 12 is Local
    tms = flog[:,10].astype(float)  # drone flight ticker time in milliseconds
    lat = flog[:,0].astype(float)   # get drone latitude
    lon = flog[:,1].astype(float)   # get drone longitude
    alt = flog[:,2].astype(float)   #altitude above ground (m)
    speed = flog[:,4].astype(float)
    flystate = flog[:,66].astype(float)
    pitch = flog[:,22].astype(float)
    roll = flog[:,23].astype(float)
    yaw = flog[:,24].astype(float)
    dronet = pd.to_datetime(datestrings)+datetime.timedelta(hours=0)

    print(dronet[-1])
    dronet=dronet+datetime.timedelta(seconds=0)+datetime.timedelta(hours=0)
    dronetnum = (dronet - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1.)
    lonraw = lon
    latraw = lat
    print("")
    print("Drone Start Time:")
    print(dronet[0])
    print("")
    print("Drone End Time:")
    print(dronet[-1])
    print("")
    print("Done.")

#### p31gps.csv is the ending for vehcile-based systems. ####
if flogfile.endswith('p31gps.csv') or flogfile.endswith('P31GPS.CSV') or flogfile.endswith('P31GPS.csv'):
    logtype="P31GPS"
    UTCerror = 0
    #call function to load and convert Litchi format flightlog
    print("Detected Ground-based GPS Telemetry file.")

    flog = np.genfromtxt(fpath+flogfile,delimiter=",",skip_header=1,dtype=str,invalid_raise=False,filling_values=np.nan)
    day = flog[:,0].astype(int)    # drone flight ticker time in milliseconds
    month = flog[:,1].astype(int)  # get drone latitude
    year = flog[:,2].astype(int)   # get drone longitude
    hour = flog[:,3].astype(int)   #altitude above ground (m)
    minute = flog[:,4].astype(int)
    second = flog[:,5].astype(int)
    lat = flog[:,6].astype(float)
    lon = flog[:,7].astype(float)
    altabs = flog[:,8].astype(float)
    speed = flog[:,9].astype(float)

    course = flog[:,10].astype(float)
    HDOP = flog[:,11].astype(float)
    numsat = flog[:,12].astype(int)

    alt = altabs-np.mean(altabs[0:10])

    pitch = np.zeros_like(lat)
    roll = np.zeros_like(lat)
    yaw = np.zeros_like(lat)

    #dronet=dronet+datetime.timedelta(seconds=0.0)+datetime.timedelta(hours=0)
    #dronetnumdelta = datetime.datetime(year,month,day,hour,minute,second)-datetime.datetime(1970, 1, 1,0,0,0)
    datearr      = np.column_stack((year,month,day,hour,minute,second))
    dt_ref    = datetime.datetime(1970, 1, 1,0,0,0)
    dronetnum  = np.array([(datetime.datetime(y,m,d,h,mi,s)-dt_ref).total_seconds() for y,m,d,h,mi,s in datearr])
    dronet    = [datetime.datetime.utcfromtimestamp(ts) for ts in dronetnum]
    #dronetnum = dronetnumdelta.total_seconds()
    lonraw = lon
    latraw = lat

    print("")
    print("GPS file Start Time:")
    print(dronet[0])
    print("")
    print("GPS file End Time:")
    print("")
    print(dronet[-1])

    print("Done.")


print("Processing Flightlog Data...",end='',flush=True)

# may not want to project to ground if measuring tree canopy Tau.
if 'meancanopyheight'in locals():
    alt = alt-meancanopyheight

if logtype=="P31GPS":
    alt = 1.5*np.ones_like(alt) # m above ground for fixed (vehicle-based) sensor

# Convert from Lat/Lon to UTM coordinates in meters
(x,y,zonenum,zonelet)=from_latlon(lat,lon)
yaw_offset = 0  # compass error
# Calculate vector corresponding to front pointing direction
xfront = cosd(yaw+yaw_offset)
xfront = np.reshape(xfront,(len(xfront),1))
yfront = sind(yaw+yaw_offset)
yfront = np.reshape(yfront,(len(yfront),1))
vecfront = np.concatenate((xfront,yfront),axis=1)   #vector pointing of drone forward
vecant = np.concatenate((xfront,yfront),axis=1)                           #rotate 90 deg counter-clockwise (based on installation of PoLRa on drone)

xv = np.concatenate(([0],np.diff(x))) # amke difference vector with same length as x
xv = np.reshape(xv,(len(xv),1))
yv = np.concatenate(([0],np.diff(y))) # make difference vector with same length as y
yv = np.reshape(yv,(len(yv),1))
xyv = np.concatenate((xv,yv),axis=1)
velnorm = np.concatenate(([0],np.sqrt(np.diff(x)**2+np.diff(y)**2))) # normalization factor
velnorm = np.reshape(velnorm,(len(velnorm),1))

velvec = np.divide(xyv,velnorm,out=np.zeros_like(xyv), where=velnorm!=0) #velocity vector of drone shape=(nmeas,2)

#vecrhs = np.array([-velvec[:,1],velvec[:,0]]) #rorate vector 90 deg counter-clockwise (antenna looks left of forward)
vecrhs = np.array([velvec[:,0],velvec[:,1]]) #(antenna looks towards velocity vector)


d_gndpixel = alt*tand(viewangle-roll)
d_gndpixel = np.reshape(d_gndpixel,(len(d_gndpixel),1))
pitchcorr = alt*tand(pitch)
pitchcorr = np.reshape(pitchcorr,(len(pitchcorr),1))
#######method 2 uses yaw (arbitrary flight orientation)
v_gndpixely = vecant*d_gndpixel+velvec*pitchcorr
xpixely = x+v_gndpixely[:,0]
ypixely = y+v_gndpixely[:,1]

# Apply correction to goe-position from config file
if 'xoffset' in locals():
    xpixely = xpixely+xoffset
    print("Applying X offset of ",xoffset," m")
if 'yoffset' in locals():
    print("Applying Y offset of ",yoffset," m")
    ypixely = ypixely+yoffset

#### convert UTM back to lat lon with correction
lat,lon = to_latlon(xpixely, ypixely, zonenum, zonelet)
print("Done.")
###
print("Loading PoLRa Data...",end='',flush=True)
DATA = np.genfromtxt(fpath+radfile,delimiter=None,autostrip=True)
print("Done.")
print("Processing & Calibrating Radiometer Data...")

#extract POSIX timestamp as it has sub-second accuracy
unixtime = DATA[:,4].astype(float)-UTCerror*3600+toffset
if 'polra2bool' in locals():  # POLRA2 had different channels
    if polra2bool == 'True':
        ml = DATA[:,5]     #matched load
        cl = DATA[:,6]     #cold load
        rv = DATA[:,7]     #v pol
        rh = DATA[:,8]     #h pol
        Tdet = DATA[:,9]   #detector physical temp
        Tml = DATA[:,10]    #matched load physical temp
        Tcl = DATA[:,11]    #cold load physical temp
        Tpam1 = DATA[:,12]  #external/antenna temp 1
        Tpam2 = DATA[:,13]  #external / antenna temp 2
        Tpam = (Tpam1+Tpam2)/2
        uml = DATA[:,14]   #matched load
        ucl = DATA[:,15]   #cold load
        urv = DATA[:,16]   #v pol
        urh = DATA[:,17]   #h pol

else:  # standard column format since PoLRa3
    ###radiometer voltages
    cl = DATA[:,5]     #cold load
    ml = DATA[:,6]     #matched load
    rv = DATA[:,7]     #v pol
    rh = DATA[:,8]     #h pol

    ###extract physical temperatures and save mean to vector
    Tdet = DATA[:,9]   #detector physical temp (null for PoLRa3)
    Tml = DATA[:,10]    #matched load physical temp
    Tcl = DATA[:,11]    #cold load physical temp
    Tpam1 = DATA[:,12]  #aux board temp / ext. 1
    Tpam2 = DATA[:,13]  #external temp
    Tpam = (Tpam1+Tpam2)/2

    ucl = DATA[:,14]   #cold load
    uml = DATA[:,15]   #matched load
    urv = DATA[:,16]   #v pol
    urh = DATA[:,17]   #h pol


date_time = np.array([datetime.datetime.utcfromtimestamp(i) for i in unixtime])  #convert unix time to datetimes
print("")
print("PoLRa Start Time [UTC]:")
print(date_time[0])
print("")
print("PoLRa End Time:")
print(date_time[-1])
#### get time matchups and start interpolating stuff
radtnum = np.array(unixtime)
drntnum = np.array(dronetnum)
xradt = np.interp(radtnum,drntnum,xpixely,left=np.nan,right=np.nan)
yradt = np.interp(radtnum,drntnum,ypixely,left=np.nan,right=np.nan)

# Decide which velocity to use (self calculated or litchi log)
if logtype=="Litchi":
    vradt = np.interp(radtnum,drntnum,speed,left=np.nan,right=np.nan)
    flystateradt = np.interp(radtnum,drntnum,flystate,left=np.nan,right=np.nan)
if logtype=="MAV" or logtype=="P31GPS":
    vradt = np.interp(radtnum,drntnum,np.array(velnorm).flatten(),left=np.nan,right=np.nan)
    flystateradt = 14*np.ones_like(radtnum)

yawradt = np.interp(radtnum,drntnum,yaw,left=np.nan,right=np.nan)
rollradt = np.interp(radtnum,drntnum,roll,left=np.nan,right=np.nan)
pitchradt = np.interp(radtnum,drntnum,pitch,left=np.nan,right=np.nan)
altradt = np.interp(radtnum,drntnum,alt,left=np.nan,right=np.nan)

latrad = np.interp(radtnum,drntnum,lat,left=np.nan,right=np.nan)
lonrad = np.interp(radtnum,drntnum,lon,left=np.nan,right=np.nan)

radtnum_flight = np.array(radtnum)
radtnum_flight[latrad==np.nan]=np.nan

tsec = unixtime-unixtime[0]
tmin = tsec/60
#print("Total Flight Time: "+str(tmin)+" Minutes.")
totalt = unixtime[-1:]-unixtime[0]

int1 = DATA[:,19]
inttot = DATA[:,20]
mnwindowsz = 1000*int_time/np.nanmean(inttot)
print("Done.")

####standard deviations of each individual measurement


#### smooth calibration voltages
cl[ucl>stdthresh]=np.nan
ml[uml>stdthresh]=np.nan

rh[urh>stdthresh]=np.nan
rv[urv>stdthresh]=np.nan

if madfiltbool == 'True':
    # run median absolute deviation on raw voltages to remove any spikes
    nmad = 2.0
    madwindowsz = 30
    cl,nfiltcl = madfilter(cl,nmad,madwindowsz)
    ml,nfiltml = madfilter(ml,nmad,madwindowsz)
    rh,nfiltrh = madfilter(rh,nmad,madwindowsz)
    rv,nfiltrh = madfilter(rv,nmad,madwindowsz)
    #print(nfilth,' H, ',nfiltv,' V values filtered by MAD filter.')

cl = pd.Series(cl).fillna(pd.Series(cl).rolling(mnwindowsz_cal, min_periods=1,win_type='gaussian').mean(std=mnwindowsz_cal))
ml = pd.Series(ml).fillna(pd.Series(ml).rolling(mnwindowsz_cal, min_periods=1,win_type='gaussian').mean(std=mnwindowsz_cal))

cl_smth = pd.Series(cl).rolling(mnwindowsz_cal, min_periods=mnwindowsz_cal-mnwindowsz_miss,center=True,win_type='gaussian').mean(std=mnwindowsz_cal)
ml_smth =pd.Series(ml).rolling(mnwindowsz_cal, min_periods=mnwindowsz_cal-mnwindowsz_miss,center=True,win_type='gaussian').mean(std=mnwindowsz_cal)
Tml_smth = pd.Series(Tml).rolling(mnwindowsz_cal, min_periods=mnwindowsz_cal-mnwindowsz_miss,center=True,win_type='gaussian').mean(std=mnwindowsz_cal)
Tcl_smth = pd.Series(Tcl).rolling(mnwindowsz_cal, min_periods=mnwindowsz_cal-mnwindowsz_miss,center=True,win_type='gaussian').mean(std=mnwindowsz_cal)

## Apply raw voltage correction (matched load test) if in config
if 'V_rawoffset' in locals():
    rv = rv+V_rawoffset
if 'H_rawoffset' in locals():
    rh = rh+H_rawoffset
#load temp dependent CL files
#contains linear fit params mx+b and cable loss in db
hpars,vpars = getTCL(hfile,vfile)
Tcl_H = hpars[0]*Tcl_smth+hpars[1]
Tcl_V = vpars[0]*Tcl_smth+vpars[1]

slopeh=(Tml_smth-Tcl_H)/(ml_smth-cl_smth)
offh = slopeh*(-ml_smth)+Tml_smth
Tb_hraw = offh+rh*slopeh                    #calibrated h pol

slopev=(Tml_smth-Tcl_V)/(ml_smth-cl_smth)
offv = slopev*(-ml_smth)+Tml_smth
Tb_vraw = offv+rv*slopev                    #calibrated v pol

##### FILTERING AVERAGING AND QUALITY CONTROL %%%%%%
cleanTbh = np.array(Tb_hraw) #copy raw data array
cleanTbv = np.array(Tb_vraw)

cleanTbh[urv>stdthresh]=np.nan  #apply standard deviation filter
cleanTbh[urh>stdthresh]=np.nan  #both polarizations filtered together
cleanTbv[urv>stdthresh]=np.nan  #apply standard deviation filter
cleanTbv[urh>stdthresh]=np.nan  #both polarizations filtered together

#cleanTbh[uml>stdthresh]=np.nan  #apply standard deviation filter
#cleanTbh[ucl>stdthresh]=np.nan  # also for calibration voltages
#cleanTbv[uml>stdthresh]=np.nan
#cleanTbv[ucl>stdthresh]=np.nan

numstd =  (urv>stdthresh).sum()+(urh>stdthresh).sum()
print(numstd,' values filtered by std filter.')

##### Outlier filtering!
#Houties = isoutlier(cleanTbh,'movmedian',MADfiltsz);
#Vouties = isoutlier(cleanTbv,'movmedian',MADfiltsz);
#cleanTbh(Houties)=NaN;
#cleanTbv(Houties)=NaN;
#cleanTbh(Vouties)=NaN;
#cleanTbv(Vouties)=NaN;

# np.warnings.filterwarnings('ignore') #turn off warnings for NaN values
##### Filter by Roll
if rollfiltbool == 'True':
    rollnonan = rollradt[~np.isnan(yawradt)]
    #rdir1,rdir2=yawpks(rollnonan)
    rollmn = np.nanmean(rollradt)
    rmin1 = rollmn-roll_tolerance   #define limits for yaw angle based on tolerance and most seen angles
    rmax1 = rollmn+roll_tolerance
    print("Mean Roll: ",rollmn," degrees.")
    cleanTbh[((rollradt<rmin1) | (rollradt>rmax1))]=np.nan
    cleanTbv[((rollradt<rmin1) | (rollradt>rmax1))]=np.nan
    print(((rollradt<rmin1) | (rollradt>rmax1)).sum(),' values filtered by roll filter.')
##### Filter by Pitch
if pitchfiltbool == 'True':
    pitchnonan = pitchradt[~np.isnan(yawradt)]
    pdir1,pdir2=yawpks(pitchnonan)
    pmin1 = pdir1-pitch_tolerance   #define limits for yaw angle based on tolerance and most seen angles
    pmax1 = pdir1+pitch_tolerance
    print("Most occuring pitch direction: ",pdir1)
    cleanTbh[((pitchradt<pmin1) | (pitchradt>pmax1))]=np.nan
    cleanTbv[((pitchradt<pmin1) | (pitchradt>pmax1))]=np.nan
    print(((pitchradt<pmin1) | (pitchradt>pmax1)).sum(),' values filtered by pitch filter.')
#####filter by yaw
if yawfiltbool == 'True':
    yawint = np.rint(yawradt)
    yawnonan = yawradt[~np.isnan(yawradt)]
    dir1,dir2=yawpks(yawnonan)
    print("Most occuring yaw directions: ",dir1,dir2)
    ymin1 = dir1-yaw_tolerance   #define limits for yaw angle based on tolerance and most seen angles
    ymax1 = dir1+yaw_tolerance
    ymin2 = dir2-yaw_tolerance
    ymax2 = dir2+yaw_tolerance
    if pushbroombool == 'True': # if pushbroom or constant sidelook (2 yaw angles for all sweeps)
        cleanTbh[((yawradt<ymin1) | (yawradt>ymax1)) & ((yawradt<ymin2) | (yawradt>ymax2))]=np.nan
        cleanTbv[((yawradt<ymin1) | (yawradt>ymax1)) & ((yawradt<ymin2) | (yawradt>ymax2))]=np.nan
        print((((yawradt<ymin1) | (yawradt>ymax1)) & ((yawradt<ymin2) | (yawradt>ymax2))).sum(),' values filtered by yaw filter.')
    if constyawbool == 'True': #if constant yaw angle for all sweeps
        # this is the case for non-continuous yaw (East-West flightlines)
        if (dir1 >= 180-yaw_tolerance) or (dir1 <= -180+yaw_tolerance):
            yawradt[yawradt<0]=yawradt[yawradt<0]+360
            if (dir1 <= -180+yaw_tolerance):
                dir1 = dir1+360
                ymin1 = dir1-yaw_tolerance   #define limits for yaw angle based on tolerance and most seen angles
                ymax1 = dir1+yaw_tolerance
            cleanTbh[((np.abs(yawradt)<np.abs(ymin1)) | (yawradt>ymax1))]=np.nan
            cleanTbv[((yawradt<ymin1) | (yawradt>ymax1))]=np.nan
        # normal arbitrary case
        else:
            cleanTbh[((yawradt<ymin1) | (yawradt>ymax1))]=np.nan
            cleanTbv[((yawradt<ymin1) | (yawradt>ymax1))]=np.nan
            print(((yawradt<ymin1) | (yawradt>ymax1)).sum(),' values filtered by yaw filter.')

#### Filter by fly state
if flystatefiltbool == 'True':
    wpstate = 14 # means flying waypoint mission for Litchi
    cleanTbh[flystateradt!=wpstate]=np.nan
    cleanTbv[flystateradt!=wpstate]=np.nan
    print((flystateradt!=wpstate).sum(),' values filtered by fly state filter.')

#####filter by velocity
if velfiltbool == 'True':
    cleanTbh[(vradt<min_vel) | (vradt>max_vel)]=np.nan
    cleanTbv[(vradt<min_vel) | (vradt>max_vel)]=np.nan
    print(((vradt<min_vel) | (vradt>max_vel)).sum(),' values filtered by velocity filter.')

if 'altfiltbool' in locals():
    if altfiltbool == 'True':                #set mode if it is defined in config
        cleanTbh[(altradt<min_alt) | (altradt>max_alt)]=np.nan
        cleanTbv[(altradt<min_alt) | (altradt>max_alt)]=np.nan
        print(((altradt<min_alt) | (altradt>max_alt)).sum(),' values filtered by Altitude filter.')
##apply and insert moving median over missing values
min_nonnan = 3
# we are not averaging at least 3 samples we take std over 3
if int(mnwindowsz)<3:
    mnwindowsz = 3
    min_nonnan = int(mnwindowsz)

if madfiltbool == 'True':
    # filter brightness temperatures for MAD spikes also
    nmad = 3
    madwindowsz = 30
    cleanTbh,nfilth = madfilter(cleanTbh,nmad,madwindowsz)
    cleanTbv,nfiltv = madfilter(cleanTbv,nmad,madwindowsz)
    print(nfilth,' H, ',nfiltv,' V values filtered by MAD filter.')

if 'spikefilterbool' in locals():
    if spikefilterbool == 'True':
        nmad = 3
        # apply mad filter to entire time series
        cleanTbh,nfilth = madfilter(cleanTbh,nmad,len(cleanTbh))
        cleanTbv,nfiltv = madfilter(cleanTbv,nmad,len(cleanTbv))
        print(nfilth,' H, ',nfiltv,' V values filtered by Spike filter.')


#####moving average
windowstd = int(np.around(mnwindowsz/3))
cleanTbv = pd.Series(cleanTbv).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
cleanTbh = pd.Series(cleanTbh).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
# must also average angles to be looking at same time periods
rollradt = pd.Series(rollradt).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
pitchradt = pd.Series(pitchradt).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)

#####remove constant bias
cleanTbv = cleanTbv-V_bias;
cleanTbh = cleanTbh-H_bias;

#####calculate rolling standard deviation

#sigTbv = pd.Series(cleanTbv).rolling(9*mnwindowsz_cal, min_periods=3*mnwindowsz_cal-3*mnwindowsz_miss).std()
#sigTbh = pd.Series(cleanTbh).rolling(9*mnwindowsz_cal, min_periods=3*mnwindowsz_cal-3*mnwindowsz_miss).std()
sigTbv = pd.Series(cleanTbv).rolling(int(mnwindowsz), min_periods=3).std()
sigTbh = pd.Series(cleanTbh).rolling(int(mnwindowsz), min_periods=3).std()
print("Done.")

##### Apply cable loss correction
if cablecorrbool == 'True':
    loss_ratio = 0.5 #may depend on system! (whitey had therm outside Alu)
    #loss_ratio = 0.95 #ratio of total transmission loss on antenna temp side
    #Teff = loss_ratio*Tpam+(1-loss_ratio)*Tml #effective temperature of transmission line loss
    Teff = loss_ratio*Tpam+(1-loss_ratio)*Tml #effective temperature of transmission line loss
    print("Applying Cable Loss Correction...",end='',flush=True)
    Loss_V = vpars[2]
    Loss_H = hpars[2]
    t_cablev = 10**(Loss_V/10)
    t_cableh = 10**(Loss_H/10)
    cleanTbh = (cleanTbh-(1-t_cableh)*Teff)/(t_cableh)
    cleanTbv = (cleanTbv-(1-t_cablev)*Teff)/(t_cablev)
    print("Done.")

# calculate Microwave Polarization difference index
MPDI = (cleanTbv-cleanTbh)/(cleanTbv+cleanTbh)

####### Retrievals ######
cleanTbv[np.isnan(cleanTbh)]=np.nan   #make sure we have both Tbs or both NaN
cleanTbh[np.isnan(cleanTbv)]=np.nan
#alpha = np.empty_like(rollradt)

##### caerful with mount orientation
alpha = viewangle-rollradt            #incidence angle after roll correction
# boolean for running with reversed mount direction
if 'reversemountbool' in locals():
    if reversemountbool == 'True':
        print("Reversing Pitch Angle.")
        alpha = viewangle+rolldradt
# nominal mount direction

print("mean look angle:",np.nanmean(alpha))
##### Get plot parameters for map overlays (plot coordinates) from Salem package
sm,x,y,txminp,txmaxp,tyminp,tymaxp,maplonmin,maplonmax,maplatmin,maplatmax = getMapParams(latrad,lonrad)  #sm is the variable containing map data txxx are the limits of the plot
xr, yr = sm.grid.transform(lonraw, latraw) #get x and y of google map grid

# vector and grid for google map overlay coordinates
xiv = np.linspace(np.nanmin(x),np.nanmax(x),300) # define vector for smoothed / interpd data
yiv = np.linspace(np.nanmin(y),np.nanmax(y),300) # define vector for smoothed / interpd data
xgrid,ygrid = np.meshgrid(xiv,yiv)
plotflightmap(sm,x,y,xr,yr,txminp,txmaxp,tyminp,tymaxp,"flight_path.png",fpath,savefigbool)
# If we want to do processing on any comparison GeoTIFF files this is where we input them.
if 'tiffloadbool' in locals():
    if tiffloadbool == 'True':
        print('Starting Coordinates of flight: (',np.nanmean(latrad[~np.isnan(latrad)][0:50]),' deg Lat., ',np.nanmean(lonrad[~np.isnan(lonrad)][0:50]),' deg Lon.)')
        plotflightmap(sm,x,y,xr,yr,txminp,txmaxp,tyminp,tymaxp,"flight_path.png",fpath,savefigbool)
        if useirtemp:
            print("Loading Thermal IR GeoTIFF for use in retrieval.")

            irgrid,irgridblur,irpts = load_geotiff(irfilename,xgrid,ygrid,sm,xiv,yiv,x,y,latrad,lonrad,sigma0)
            # convert from degrees celcius to Kelvin
            irgrid = irgrid+273.15
            irgridblur = irgridblur+273.15
            irpts = irpts+273.15
            # print min/max for sanity check
            minir = np.nanmin(irgrid)
            maxir = np.nanmax(irgrid)
            minirblur = np.nanmin(irgridblur)
            maxirblur = np.nanmax(irgridblur)
            print("Max IR temp: ",maxir," Min IR temp: ",minir)
            print("Max Filtered IR temp: ",maxirblur," Min IR temp: ",minirblur)
            # makes plots of raw regridded / filtered / and pointwise interpolations
            figir= plotcontmap(xgrid,ygrid,irgrid,     bool2b1,sm,minir,maxir,txminp,txmaxp,tyminp,tymaxp,"IR_rawinterp.png",'IR Temp','$K$',darkmode,fpath,savefigbool)
            figirs=plotcontmap(xgrid,ygrid,irgridblur, bool2b1,sm,minir,maxir,txminp,txmaxp,tyminp,tymaxp,"IR_filtered.png",'IR Temp','$K$',darkmode,fpath,savefigbool)
            figirpt=plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,irpts,minir,maxir,"IR_pointwise.png",'IR Temp','$K$',fpath,savefigbool)

            ###### Plot histogram of IR temps
            plt.figure()
            plt.ylabel('Number of Occurances')
            plt.xlabel('IR Temp. in Raw Regridded (K)')
            counts, bins,_ = plt.hist(irgrid.flatten(), bins='auto')
            #pkind = argrelmax(counts)
            #print("peaks in IR hist: ",bins[pkind[0]],bins[pkind[1]])
            plt.grid()
            if savefigbool=='True':
                plt.savefig(fieldpath+'IRtemp_Histogram.png')
            if showfigbool=='True':
                plt.show(block=False)

        if ndvibool:
            print("Loading NDVI GeoTIFF for comparison.")
            sigma_ndvi = 12
            ndvigrid,ndvigridblur,ndvipts = load_geotiff(ndvifilename,xgrid,ygrid,sm,xiv,yiv,x,y,latrad,lonrad,sigma_ndvi)
            minndvi = np.nanmin(ndvigrid)
            maxndvi = np.nanmax(ndvigrid)
            minndviblur = np.nanmin(ndvigridblur)
            maxndviblur = np.nanmax(ndvigridblur)
            print("Max NDVI: ",maxndvi," Min NDVI: ",minndvi)
            print("Max Filtered NDVI: ",maxndviblur," Min NDVI: ",minndviblur)
            fign1=plotcontmap(xgrid,ygrid,ndvigrid, bool2b1,sm,minndvi,maxndvi,txminp,txmaxp,tyminp,tymaxp,"NDVI_rawinterp.png",'NDVI','(-)',darkmode,fpath,savefigbool)
            fign2=plotcontmap(xgrid,ygrid,ndvigridblur, bool2b1,sm,minndviblur,maxndviblur,txminp,txmaxp,tyminp,tymaxp,"NDVI_filtered.png",'Filtered NDVI','(-)',darkmode,fpath,savefigbool)
            fign3=plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,ndvipts,minndviblur,maxndviblur,"NDVI_pointwise.png",'NDVI Interpolated','(-)',fpath,savefigbool)


if 'mode' in locals():                #set mode if it is defined in config
    mode=mode
else:
    mode='dual'                       # else default to dual pol mode
if 'retmeanbool' in locals():
    if retmeanbool == 'True':
        #Wsmn,Taumn,T_g,cfv = RetTauWsTgAll(cleanTbh,cleanTbv,alpha,omega,varminwfit,varmintaufit,varminTgfit,varmaxwfit,varmaxtaufit,varmaxTgfit)
        Wsmn,Taumn,omega,T_g,cfv = RetTauWsOmegaTgAll(cleanTbh,cleanTbv,alpha,varminwfit,varmintaufit,varminomegafit,varminTgfit,varmaxwfit,varmaxtaufit,varmaxomegafit,varmaxTgfit)
        Tbhf,Tbvf = tauomega_all_tb(Wsmn,Taumn,T_g,T_g,alpha,omega)
        print("Mean Soil Moisture Accross Flight: ",Wsmn)
        print("Mean L-VOD (Tau) Accross Flight: ",Taumn)
        print("Mean Single Scattering Albedo Accross Flight: ",omega)
        print("Mean Emission Temperature Accross Flight: ",T_g)
        print("Cost Function Value: ",np.sqrt(cfv/len(cleanTbh[~np.isnan(cleanTbh)])))
        # set values if doing a 1 parameter retrieval
        W_s = Wsmn
        tau = Taumn
if retbool=='True':
    if retstring == 'WS':
        print("Retreiving Soil Moisture for ",len(alpha[~np.isnan(cleanTbv)])," points...",end='',flush=True)
        Ws,cfv = RetWs(cleanTbh,cleanTbv,T_g,alpha,tau,omega,varminwfit,varmaxwfit,mode)
        ## filter cost function values above threshold
        Ws[cfv>cf_thresh]=np.nan
        #Smooth retrieval similar to brightness temperatures
        Ws = pd.Series(Ws).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
        nonnans = ~np.isnan(Ws)
        Wsp = Ws[nonnans]             #pass only non-nan to plotting function
        print("Done.")
        print((cfv>cf_thresh).sum(),' values filtered by cost function value.')
    if retstring == 'WSTAU':
        print("Retreiving Soil Moisture and Vegetation Depth for ",len(alpha[~np.isnan(cleanTbv)])," points...",end='',flush=True)
        Ws,Tau,cfv = RetTauWs(cleanTbh,cleanTbv,T_g,alpha,omega,varminwfit,varmintaufit,varmaxwfit,varmaxtaufit)
        #Smooth retrieval similar to brightness temperatures
        Ws = pd.Series(Ws).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
        Tau = pd.Series(Tau).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
        ## filter cost function values above threshold
        Ws[cfv>cf_thresh]=np.nan
        Tau[cfv>cf_thresh]=np.nan
        nonnans = ~np.isnan(Ws)
        Wsp = Ws[nonnans]             #pass only non-nan to plotting function
        Taup = Tau[nonnans]
        print("Done.")
        print((cfv>cf_thresh).sum(),' values filtered by cost function value.')

    if retstring == 'TAU':
        print("Retreiving Vegetation Optical Depth for ",len(alpha[~np.isnan(cleanTbv)])," points...",end='',flush=True)
        Tau,cfv = RetTau(cleanTbh,cleanTbv,T_g,alpha,W_s,omega,varmintaufit,varmaxtaufit,mode)
        #Smooth retrieval similar to brightness temperatures
        Tau = pd.Series(Tau).rolling(int(mnwindowsz), min_periods=min_nonnan,center=True,win_type='gaussian').mean(std=windowstd)
        ## filter cost function values above threshold
        Tau[cfv>cf_thresh]=np.nan
        Tau,nfiltv = madfilter(Tau,nmad,madwindowsz)
        nonnans = ~np.isnan(Tau)
        Taup = Tau[nonnans]
        print("Done.")
        print((cfv>cf_thresh).sum(),' values filtered by cost function value.')
else:
    #nonnans = ~np.isnan(latrad)
    nonnans = ~np.isnan(cleanTbv)


latradp = latrad[nonnans]     #remove nans from lat/lon according to Ws
lonradp = lonrad[nonnans]     #
x = np.array(x[nonnans])
y = np.array(y[nonnans])
cleanTbhp = np.array(cleanTbh[nonnans])
cleanTbvp = np.array(cleanTbv[nonnans])

sigTbvp = np.array(sigTbv[nonnans])
sigTbhp = np.array(sigTbh[nonnans])
if retbool=='True':
    cfvp = np.array(cfv[nonnans])
date_time_f = np.array(date_time[nonnans])
rollplot = np.array(rollradt[nonnans])
MPDIp =  MPDI[nonnans]#

#### Plots for no internet connection
# f, (ax1) = plt.subplots(1,1,figsize=(9.5,9.5))  # create figure for plotting plot it
# cm = plt.cm.get_cmap('Spectral')
# scat = ax1.scatter(lonradp, latradp, c=cleanTbvp, s=100, edgecolors='none', linewidths=1,cmap=cm,alpha=1,vmin=272,vmax=278)
# #scat = ax1.scatter(lonradp, latradp, c=cleanTbvp, s=100, edgecolors='none', linewidths=1,cmap=cm,alpha=1,vmin=278,vmax=288)
# clb = plt.colorbar(scat,fraction=0.046,pad=0.04)
# plt.tight_layout()
# plt.savefig('TBvtemp.png')
#
# f, (ax1) = plt.subplots(1,1,figsize=(9.5,9.5))  # create figure for plotting plot it
# cm = plt.cm.get_cmap('Spectral')
# scat = ax1.scatter(lonradp, latradp, c=cleanTbhp, s=100, edgecolors='none', linewidths=1,cmap=cm,alpha=1,vmin=272,vmax=278)
# #scat = ax1.scatter(lonradp, latradp, c=cleanTbhp, s=100, edgecolors='none', linewidths=1,cmap=cm,alpha=1,vmin=278,vmax=288)
# clb = plt.colorbar(scat,fraction=0.046,pad=0.04)
# plt.tight_layout()
# plt.savefig('TBhtemp.png')

### debug plots
if plotdebug == 'True':
    plt.figure()
    plt.plot(tmin,xradt-xradt[0],'-.')
    plt.plot(tmin,yradt-yradt[0],'-.')
    plt.plot(tmin,altradt,'-.')
    plt.legend(['deltaX','deltaY','deltaAltitude'])
    if savefigbool=='True':
        plt.savefig(fieldpath+'XYAlt.png')
    if showfigbool=='True':
        plt.show(block=False)
    ##########################################
    plt.figure()
    plt.plot(tmin,rollradt,'-.')
    plt.plot(tmin,pitchradt,'-.')
    plt.plot(tmin,yawradt,'-.')
    plt.legend(['Roll','Pitch','Yaw'])
    if savefigbool=='True':
        plt.savefig(fieldpath+'RollPitchYaw.png')
    if showfigbool=='True':
        plt.show(block=False)
        ##########################################
    plt.figure()
    plt.plot(alpha,cleanTbv,'.')
    plt.plot(alpha,cleanTbh,'.')
    if 'retmeanbool' in locals():
        if retmeanbool == 'True':
            plt.plot(alpha,Tbvf,'-')
            plt.plot(alpha,Tbhf,'-')
            legstr = ['V pol.','H pol.','V pol. Theory (mean)','H pol. Theory (mean)']
        else:
            legstr = ['V pol.','H pol.']
    else:
        legstr = ['V pol.','H pol.']
    plt.ylabel('T_B (K)')
    plt.title('Alpha vs. T_B')
    plt.grid()
    #plt.legend(['V pol.','H pol.'])
    plt.legend(legstr)
    if savefigbool=='True':
        plt.savefig(fieldpath+'Alpha_vs_TB.png')
    if showfigbool=='True':
        plt.show(block=False)
    ##########################################
    plt.figure()
    plt.plot(alpha,MPDI,'.')
    if 'retmeanbool' in locals():
        if retmeanbool == 'True':
            plt.plot(alpha,(Tbvf-Tbhf)/(Tbvf+Tbhf),'-')
            plt.legend(['MPDI Meas.','MPDI Mean Theory'])
    plt.ylabel('MPDI (-)')
    plt.title('Alpha vs. MPDI')
    plt.grid()

    #plt.legend(['V pol.','H pol.','V pol. Theory (mean)','H pol. Theory (mean)'])
    if savefigbool=='True':
        plt.savefig(fieldpath+'Alpha_vs_MPDI.png')
    if showfigbool=='True':
        plt.show(block=False)
    ##########################################
    plt.figure()
    plt.plot(tmin,ml)
    plt.plot(tmin,cl)
    plt.plot(tmin,rv)
    plt.plot(tmin,rh)
    plt.ylabel('Detector Voltage (mV)')
    plt.legend(['Matched Load','Cold Load','Antenna V','Antenna H'])
    plt.title('Raw Voltages')
    plt.grid()
    #plt.ylim((1410,1465))
    if savefigbool=='True':
        plt.savefig(fieldpath+'Raw_voltages.png')
    if showfigbool=='True':
        plt.show(block=False)

    ##%%%%%%%%%%%%%% plot physical temperatures %%%%%%%%%
    plt.figure()
    plt.plot(tmin,Tml,'-o')
    plt.plot(tmin,Tcl,'-o')
    plt.plot(tmin,Tpam,'-o')
    plt.ylabel('Physical Temperatures (K)')
    plt.legend(['Matched Load','Cold Load','Antenna','Detector'])
    plt.grid()
    if savefigbool=='True':
        plt.savefig(fieldpath+'Temps.png')
    if showfigbool=='True':
        plt.show(block=False)

    ##%%%%%%%%%%%%%% plot difference versus physical temperature %%%%%%%%%
    plt.figure()
    plt.plot(Tpam,ml-cl,'.',linewidth=0.5)
    plt.plot(Tcl,ml-cl,'.',linewidth=0.5)
    plt.ylabel('ml-cl (mV)')
    plt.xlabel('Physical Temperature (K)')
    plt.legend(['Antenna Temp','Cold Load Temp'])
    plt.grid()
    if savefigbool=='True':
        plt.savefig(fieldpath+'DiffvsTemp.png')
    if showfigbool=='True':
        plt.show(block=False)

    ##%%%%%%%%%%%%%% scatter CL vs T %%%%%%%%%
    plt.figure()
    plt.plot(Tml,ml,'.',linewidth=0.5)
    plt.plot(Tcl,cl,'.',linewidth=0.5)
    plt.plot(Tcl,Tcl_H,'.',linewidth=0.5)
    plt.ylabel('ml-cl (mV)')
    plt.ylabel('Physical Temperature (K)')
    plt.legend(['RS','ACS','ACS TB'])
    plt.grid()
    if savefigbool=='True':
        plt.savefig(fieldpath+'RawVvsTemp.png')
    if showfigbool=='True':
        plt.show(block=False)

    ############Plot scatter of angle versus TB %%%%%%%%%%%%%%%%%%%%%%%%%
    # plt.figure()
    # plt.plot(alpha[cleanTbv<250],cleanTbv[cleanTbv<250],'.')
    # plt.plot(alpha[cleanTbh<150],cleanTbh[cleanTbh<150],'.')
    # plt.xlabel('Look Angle (deg)')
    # plt.ylabel('TB (K)')
    # plt.title('Look Angle versus $T_{Ant}$')
    # plt.legend(['V pol.','H pol.'])
    # plt.grid()
    # plt.savefig(fieldpath+'Alpha_vs_TB.png')

    plt.figure()
    #plt.plot(pitchradt[cleanTbv<250],cleanTbv[cleanTbv<250],'.')
    #plt.plot(pitchradt[cleanTbh<150],cleanTbh[cleanTbh<150],'.')
    plt.plot(pitchradt,cleanTbv,'.')
    plt.plot(pitchradt,cleanTbh,'.')
    plt.xlabel('Pol. Misalignment Angle (deg)')
    plt.ylabel('TB (K)')
    plt.title('Pitch versus TB')
    plt.legend(['V pol.','H pol.'])
    plt.grid()
    if savefigbool=='True':
        plt.savefig(fieldpath+'Pitch_vs_TB.png')
    if showfigbool=='True':
        plt.show(block=False)


##### Scatter and Gaussian smoothed contour plots of retrievals
if 'csvbool' in locals():                #set if csv bool is defined in config
    csv=csvbool
else:
    csv='False'                      # else default to no csv
if csv=='True':
    if retbool=='True':
        if retstring == 'WS' or retstring =='WSTAU':
            Tau = np.ones_like(cleanTbv)*np.nan
        if retstring =='TAU':
            Ws = np.ones_like(cleanTbv)*np.nan
    else:
        Tau = np.ones_like(cleanTbv)*np.nan
        Ws = np.ones_like(cleanTbv)*np.nan

    outarr = np.column_stack((radtnum,xradt,yradt,latrad,lonrad,cleanTbv,cleanTbh,Ws,Tau,alpha,pitchradt,yawradt,Tdet,Tml,Tcl,Tpam1,Tpam2,ml,cl,rv,rh,uml,ucl,urv,urh,slopeh,offh,vradt,flystateradt))
    radfileroot=os.path.splitext(radfile)[0]
    outarr = outarr[~np.isnan(xradt),:]
    outarr = outarr[~np.isnan(outarr[:,6]),:]
    np.savetxt(fieldpath+radfileroot+'_processed.csv', outarr, delimiter=',', header="posix time,X (m) UTM,Y (m) UTM,Latitude,Longitude,TBV (K),TBH (K),Soil Moisture (m^3/m^3),Tau (-),Nadir Angle (deg),Roll angle (deg),Yaw angle (deg),Detector Temp (K),T_RS,T_ACS,T_Ant1,T_Ant2,u_RS (mV),u_ACS (mV),u_V (mV),u_H (mV),STDOM(u_RS) (mV),STDOM(u_ACS) (mV),STDOM(u_V) (mV),STDOM(u_H) (mV),Gain (K/mV),Offset (K),Velocity (m/s),Flystate (-)")


if retbool=='True':
    if retstring == 'WS':
        mask = np.logical_and.reduce((~np.isnan(x),~np.isnan(y),~np.isnan(Wsp)))
        # make scatter map of Ws
        plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,Wsp,varminw,varmaxw,fname1,'Raw Soil Moisture',unitstrsm,fieldpath,savefigbool)
        # Apply Gaussian smoothing filter to Retrevials
        xi,yi,zifilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,Wsp[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
        print("Output Ws contour map overlay...",end='',flush=True)
        wsfig=plotcontmap(xi,yi,zifilt,bool2b1,sm,varminw,varmaxw,txminp,txmaxp,tyminp,tymaxp,fname2,'Soil Moisture',unitstrsm,darkmode,fieldpath,savefigbool)
        print("Done.")

    if retstring == 'WSTAU':
        mask = np.logical_and.reduce((~np.isnan(x),~np.isnan(y),~np.isnan(Wsp),~np.isnan(Taup)))
        plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,Taup,varmintau,varmaxtau,fname3,'Raw Vegetation Optical Depth',unitstrtau,fieldpath,savefigbool)
        plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,Wsp,varminw,varmaxw,fname1,'Raw Soil Moisture',unitstrsm,fieldpath,savefigbool)
        #if plotdebug == 'True':
            #plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,corr1dp,-1.0,1.0,'Correlation','Ws/Tau Correlation','',fieldpath,savefigbool)
            #plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,corr1dtbp,-1.0,1.0,'TBCorrelation','H/V Correlation','',fieldpath,savefigbool)

        xi,yi,taufilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,Taup[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
        print("Output Tau contour map overlay...",end='',flush=True)
        plotcontmap(xi,yi,taufilt,bool2b1,sm,varmintau,varmaxtau,txminp,txmaxp,tyminp,tymaxp,fname4,'Vegetation Optical Depth',unitstrtau,darkmode,fieldpath,savefigbool)
        print("Done.")
        xi,yi,zifilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,Wsp[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
        print("Output Ws contour map overlay...",end='',flush=True)
        plotcontmap(xi,yi,zifilt,bool2b1,sm,varminw,varmaxw,txminp,txmaxp,tyminp,tymaxp,fname2,'Soil Moisture',unitstrsm,darkmode,fieldpath,savefigbool)
        print("Done.")

    if retstring =='TAU':
        mask = np.logical_and.reduce((~np.isnan(x),~np.isnan(y),~np.isnan(Taup)))
        plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,Taup,varmintau,varmaxtau,fname3,'Raw Vegetation Optical Depth',unitstrtau,fieldpath,savefigbool)
        xi,yi,taufilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,Taup[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
        print("Output Tau contour map overlay...",end='',flush=True)
        plotcontmap(xi,yi,taufilt,bool2b1,sm,varmintau,varmaxtau,txminp,txmaxp,tyminp,tymaxp,fname4,'Vegetation Optical Depth',unitstrtau,darkmode,fieldpath,savefigbool)
        print("Done.")
    if plotrawcfbool == 'True':
        plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,cfvp,varmincf,varmaxcf,"cf_raw.png",'Cost Function Value','$(K^2)$',fieldpath,savefigbool)

if booltbplot == 'True':
    #Plot TB versus time
    plottbtime(cleanTbhp,cleanTbvp,date_time_f,fieldpath,savefigbool)

if plotrawtbbool == 'True':
    #Raw TB scatter maps
    plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,cleanTbhp,varminh,varmaxh,"TB_raw_H_pol.png",'Brightness Temperature (H pol)','$T_B (K)$',fieldpath,savefigbool)
    plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,cleanTbvp,varminv,varmaxv,"TB_raw_V_pol.png",'Brightness Temperature (V pol)','$T_B (K)$',fieldpath,savefigbool)
    plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,MPDIp,-0.1,0.1,"Pol_ratio.png",'Polarizarion Difference Index MPDI','$(-))$',fieldpath,savefigbool)
if plotrawsigbool == 'True':
    #Raw sigma_TB scatter maps
    plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,sigTbhp,varminsg,varmaxsg,"sigmaTB_raw_H_pol.png",'Standard Deviation of TB (H pol)','$(K)$',fieldpath,savefigbool)
    plotscattermap(sm,x,y,txminp,txmaxp,tyminp,tymaxp,sigTbvp,varminsg,varmaxsg,"sigmaTB_raw_V_pol.png",'Standard Deviation of TB (V pol)','$(K)$',fieldpath,savefigbool)

if plotsmthtbbool == 'True':
    #Gaussian filtered TB contours
    mask = np.logical_and.reduce((~np.isnan(cleanTbhp),~np.isnan(cleanTbvp),~np.isnan(x),~np.isnan(y)))
    xi,yi,thfilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,cleanTbhp[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
    xi,yi,tvfilt = GaussFilterTestConv(x[mask],y[mask],xiv,yiv,cleanTbvp[mask],sm,txminp,txmaxp,tyminp,tymaxp,distthresh,sigma0)
    print("Output TB contour map overlays...",end='',flush=True)
    fnameh = 'Filtered_TB_H.png'
    fnamev = 'Filtered_TB_V.png'
    plotcontmap(xi,yi,thfilt,bool2b1,sm,varminh,varmaxh,txminp,txmaxp,tyminp,tymaxp,fnameh,'H pol. Smoothed',unitstrtb,darkmode,fieldpath,savefigbool)
    plotcontmap(xi,yi,tvfilt,bool2b1,sm,varminv,varmaxv,txminp,txmaxp,tyminp,tymaxp,fnamev,'V pol. Smoothed',unitstrtb,darkmode,fieldpath,savefigbool)
    print("Done.")

if 'ndvibool' in locals():
    if (ndvibool == 'True') and (tiffloadbool == 'True'):
        ##%%%%%%%%%%%%%% scatter NDVI vs Tau %%%%%%%%%
        # custom integration for UCD comparison with other drone data
        plt.figure()
        plt.scatter(taufilt.flatten(),ndvigridblur.flatten(),s=5,c='red')
        plt.scatter(Tau,ndvipts,s=5,c=cfv)
        plt.ylabel('NDVI (-)')
        plt.xlabel('L-VOD (Tau) (-)')
        plt.legend(['Gaussian Filtered Data','Raw L-VOD/Pointwise NDVI'])
        plt.grid()
        if savefigbool=='True':
            plt.savefig(fieldpath+'NDVI_Tau_scatter.png')
        if showfigbool=='True':
            plt.show(block=False)
        mask = ~np.isnan(taufilt.flatten()) & ~np.isnan(ndvigridblur.flatten())
        rr,p = pearsonr(taufilt.flatten()[mask],ndvigridblur.flatten()[mask])
        maskr = ~np.isnan(Tau) & ~np.isnan(ndvipts)
        rrp,pp = pearsonr(Tau[maskr],ndvipts[maskr])
        print("Correlation Filtered NDVI / Tau: ",rr)
        print("P-Value: ",p)
        print("Correlation Pointwise NDVI / Tau: ",rrp)
        print("P-Value: ",pp)
########################################
if 'irbool' in locals():
    if (irbool == 'True') and (tiffloadbool == 'True'):
        ##%%%%%%%%%%%%%% scatter NDVI vs Tau %%%%%%%%%
        # custom integration for UCD comparison with other drone data
        plt.figure()
        #plt.plot(taufilt.flatten(),ndvigridblur.flatten(),'.',linewidth=0.5)
        plt.scatter(Tau,irpts,s=5,c=cfv)
        plt.ylabel('IR Temp. (K)')
        plt.xlabel('L-VOD (Tau) (-)')
        #plt.legend(['RS','ACS','ACS TB'])
        plt.grid()
        if savefigbool=='True':
            plt.savefig(fieldpath+'IRtemp_Tau_scatter.png')
        if showfigbool=='True':
            plt.show(block=False)
        mask = ~np.isnan(taufilt.flatten()) & ~np.isnan(irgridblur.flatten())
        rr,p = pearsonr(taufilt.flatten()[mask],irgridblur.flatten()[mask])
        maskr = ~np.isnan(Tau) & ~np.isnan(irpts)
        rrp,pp = pearsonr(Tau[maskr],irpts[maskr])
        print("Correlation Filtered IR / Tau: ",rr)
        print("P-Value: ",p)
        print("Correlation Pointwise IR / Tau: ",rrp)
        print("P-Value: ",pp)



if savebinout == 'True':
    #pickle.dump(wsfig,open('FigureObject.wsfig.pickle','wb'))
    lxi = np.linspace(maplonmin,maplonmax,300)
    lyi = np.linspace(maplatmin,maplatmax,300)
    lxiv,lyiv = np.meshgrid(lxi,lyi)

    np.save(fieldpath+'lat.npy',lyiv.flatten())
    np.save(fieldpath+'lon.npy',lxiv.flatten())
    if retbool=='True':
        if retstring == 'WS' or retstring =='WSTAU':
            np.save(fieldpath+'wsraw.npy',Wsp)
            np.save(fieldpath+'ws.npy',np.flipud(zifilt).flatten())
        if retstring =='TAU':
            np.save(fieldpath+'tau.npy',np.flipud(taufilt).flatten())
        if retstring =='WSTAU':
            np.save(fieldpath+'wsraw.npy',Wsp)
            np.save(fieldpath+'ws.npy',np.flipud(zifilt).flatten())
            np.save(fieldpath+'tau.npy',np.flipud(taufilt).flatten())
    np.save(fieldpath+'latraw.npy',latradp)
    np.save(fieldpath+'lonraw.npy',lonradp)
    np.save(fieldpath+'tbh.npy',np.flipud(thfilt).flatten())
    np.save(fieldpath+'tbv.npy',np.flipud(tvfilt).flatten())

# if 'csvbool' in locals():                #set if csv bool is defined in config
#     csv=csvbool
# else:
#     csv='False'                      # else default to no csv
# if csv=='True':
#     if retbool=='True':
#         if retstring == 'WS' or retstring =='WSTAU':
#             Tau = np.ones_like(cleanTbv)*np.nan
#         if retstring =='TAU':
#             Ws = np.ones_like(cleanTbv)*np.nan
#     else:
#         Tau = np.ones_like(cleanTbv)*np.nan
#         Ws = np.ones_like(cleanTbv)*np.nan

#     outarr = np.column_stack((radtnum,xradt,yradt,latrad,lonrad,cleanTbv,cleanTbh,Ws,Tau,alpha,pitchradt,yawradt,Tdet,Tml,Tcl,Tpam1,Tpam2,ml,cl,rv,rh,uml,ucl,urv,urh,slopeh,offh,vradt,flystateradt))
#     radfileroot=os.path.splitext(radfile)[0]
#     outarr = outarr[~np.isnan(xradt),:]
#     outarr = outarr[~np.isnan(outarr[:,6]),:]
#     np.savetxt(fieldpath+radfileroot+'_processed.csv', outarr, delimiter=',', header="posix time,X (m) UTM,Y (m) UTM,Latitude,Longitude,TBV (K),TBH (K),Soil Moisture (m^3/m^3),Tau (-),Nadir Angle (deg),Roll angle (deg),Yaw angle (deg),Detector Temp (K),T_RS,T_ACS,T_Ant1,T_Ant2,u_RS (mV),u_ACS (mV),u_V (mV),u_H (mV),STDOM(u_RS) (mV),STDOM(u_ACS) (mV),STDOM(u_V) (mV),STDOM(u_H) (mV),Gain (K/mV),Offset (K),Velocity (m/s),Flystate (-)")

if 'exporttifbool' in locals():                #set if csv bool is defined in config
    tifexp=exporttifbool
else:
    tifexp='False'                      # else default to no csv
if tifexp=='True':
    print("Output GeoTIFF Files...",end='',flush=True)
    lxi = np.linspace(maplonmin,maplonmax,300)
    lyi = np.linspace(maplatmin,maplatmax,300)
    lxiv,lyiv = np.meshgrid(lxi,lyi)
    if retbool=='True':
        if retstring == 'WS':
            exportgeotiff(lyiv, lxiv, zifilt,fieldpath+'ws.tif')
            exportgeotiff(lyiv, lxiv, thfilt,fieldpath+'tbh.tif')
            exportgeotiff(lyiv, lxiv, tvfilt,fieldpath+'tbv.tif')
        elif retstring =='TAU':
            exportgeotiff(lyiv, lxiv, taufilt,fieldpath+'tau.tif')
            exportgeotiff(lyiv, lxiv, thfilt,fieldpath+'tbh.tif')
            exportgeotiff(lyiv, lxiv, tvfilt,fieldpath+'tbv.tif')
        elif retstring =='WSTAU':
            exportgeotiff(lyiv, lxiv, zifilt,fieldpath+'ws.tif')
            exportgeotiff(lyiv, lxiv, taufilt,fieldpath+'tau.tif')
            exportgeotiff(lyiv, lxiv, thfilt,fieldpath+'tbh.tif')
            exportgeotiff(lyiv, lxiv, tvfilt,fieldpath+'tbv.tif')
    else:
            exportgeotiff(lyiv, lxiv, thfilt,fieldpath+'tbh.tif')
            exportgeotiff(lyiv, lxiv, tvfilt,fieldpath+'tbv.tif')
    print("Done.")

#if not savefigbool:
plt.show()
