#!/bin/python
import os
import sys
import platform

fpath="./data/POLRA3_20240704_15_59_28/"
radfile = "POLRA3_20240704_15_59_28.dat"
flogfile = "POLRA3_20240704_15_59_28_p31gps.csv" #flogfile1
# flogfile = "2024-07-04_16-00-33_v2.csv" #flogfile2


if platform.system()!="Windows":
    cmd = "python3 ./pypolra31.py "+fpath+" "+flogfile+" "+radfile
else:
    cmd = "python ./pypolra31.py "+fpath+" "+flogfile+" "+radfile
os.system(cmd)

#######################################
