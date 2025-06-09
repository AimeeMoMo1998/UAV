##### Retrieval Variables

# TYPE OF RETRIEVAL
printbool = False
retbool = "True"
retstring = 'WS'        # set the retrieval string ('WSTAU' or 'WS' or 'TAU') for 1 parameter or two
savefigbool = 'True'
showfigbool = 'False'
#W_s = 0.33                 # Soil Moisture    only used if retstring = 'TAU'
tau = 0.10                 # L-VOD            only used if retstring = 'WS'

# mode = 'v'               # option to use only 1 of two antennas
T_g = 273.15+23            # Ground Temperature in (K) ( can approximate with air temperature )

omega = 0.00               # single scattering albedo (Tau-Omega emission model)

# INCIDENCE ANGLE
viewangle = 40.0           # degrees INCIDENCE ANGLE OF ANTENNA

#files for cold load
hfile = 'TclH.txt'         # files containing fit parameters for cold load TB
vfile = 'TclV.txt'

# Calibration offset from water flight
H_bias = 40 #0. # (K)
V_bias = 10 #0. # (K)
H_rawoffset = -0.72
V_rawoffset = 1.71

#### Min and Max contraints on fitting
varminwfit = 0.0     # Soil moisture min
varmaxwfit = 1.0     # Soil moisture max
varmintaufit = 0.0   # Tau min
varmaxtaufit = 1.0   # Tau max

#### Smoothing and Filtering
int_time  = 0.3       # Seconds INTEGRATION TIME
sigma0    = 8.        # Gaussian filter sigma (size of 2D smoothing)
cf_thresh = 1e8       # threshold for Cost Function value (filter)

#### Standard Deviation Filter
stdthresh =  1.    # mV of raw voltages
distthresh = 3       # multiplier of median distance between points for filter

#### Calibration Reference Smoothing
mnwindowsz_cal = 1  # number of samples to average for calibration loads
mnwindowsz_miss = 1 # the minimum allowed non-nan size of window

#### Velocity Filter (removes turns where angle is highly variable)
velfiltbool = 'False'
min_vel = 3.5     # m/s
max_vel = 8       # m/s

#### Flystate Filter
flystatefiltbool = 'True' # removes flying back home and takeoff from results

#roll angle filter
rollfiltbool = 'False'    # removes points where drone is excessively tilted
roll_tolerance = 8.       # deg +/-
pitchfiltbool = 'False'    # removes points where drone is excessively off normal for polarizations
pitch_tolerance = 4.0      # deg +/-

# 2 yaw angles filter
constyawbool = 'True'     # for flights with only 1 yaw angle
pushbroombool = 'False'   # for flights with 2 main yaw angles (pushbroom)
yawfiltbool = 'False'     # filter for 2 yaw positions?
yaw_tolerance = 3.        # tolerance for yaw angle +/-

altfiltbool = 'False'
min_alt = 5
max_alt = 11

lonfiltbool = 'False'
min_lon = 8.660
max_lon = 8.664

madfiltbool = 'True'
#madfiltbool = 'False'
#cable correction?
cablecorrbool = 'True'    # defaults to True

####display paramters if plotting TBs
varminw = 0.0             # Soil moisture min
varmaxw = 1.0            # Soil moisture max
varmintau = 0.0           # Tau min
varmaxtau = 0.6           # Tau max

# Min and Max Values for Plots
varminh = 120.   # K H pol min TB
varmaxh = 290.   # K H pol max TB
varminv = 120.   # K V pol min TB
varmaxv = 310.   # K V pol max TB

varminsg = 0.    # K standard deviation min
varmaxsg = 2.    # K standard deviation max
varmincf = 0     # Min cost function value
varmaxcf = 100.  # Max cost function value

# unit string for plot colorbar
unitstrsm = '$cm^3/cm^3$'   # Soil Moisture Plot
unitstrtau = ''             # VOD plot
unitstrtb = '$K$'           # TB plot

##### Filenames for Plots
fname1 = "Raw_Soil_Moisture.png"
fname2 = "Filtered_Soil_Moisture.png"
fname3 = "Raw_Tau.png"
fname4 = "Filtered_Tau.png"

#### Booleans for plots
bool2b1 = "False"       # Plot side by side contour
plotcfbool = 'False'     # Plot cost function
darkmode = 'False'      # plot filtered WS with dark gray background
plotsmthtbbool = 'True' # Plot filtered TBs
plotrawtbbool = 'True'  # Plot raw TB
plotrawsigbool = 'False' # Plot standard deviations of raw TB
plotrawcfbool = 'False'  # Plot raw cost function
booltbplot = 'True'       # Plot TB time series
savebinout = 'False'    # Save binary output files for Dash web app display
plotdebug = 'True'      # create "advanced user" plots for debugging or more detailed analysis
csvbool = 'True'        # output CSV containing raw data points
exporttifbool = 'True'  # output geotiff files of smoothed brightness temperatures and WS / Tau
