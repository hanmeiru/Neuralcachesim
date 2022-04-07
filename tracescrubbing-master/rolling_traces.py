from pickletools import uint8
import os
import numpy as np 
import sys

#Through command input a (long/complete/undivided) trace textfile, ex. ls_L1D 
#take the first 1000 traces (rows 0~999) to make a new textfile, put in RollingTraces
#take rows 300~1299 to make a new textfile 
#take rows 600~1599 to make a new textfile 
#...
#results in a directory RollingTraces with textfiles, each 1000 lines 

#(can run rolling_traces multiple times on different traces, ex. mkdir_L1D,
# and use multi_heatmaps on RollingTraces to create heatmaps) 


o_trace_folder = "/Users/hanmeiru/Desktop/research/RollingTraces"

#from command input some directory with trace text files
assert sys.argv[1]
i_trace_folder = sys.argv[1]
for trace in os.listdir(i_trace_folder):
    if (trace == ".DS_Store"):
        continue
    trace_path = i_trace_folder + "/" + trace
    f = open(trace_path, "r")
    lines = f.readlines()
    flength = len(lines) #number of lines
    #print(lines)
    for i in range(0,flength-100000,20000):
        output_name = o_trace_folder + "/" + os.path.basename(trace_path) + "_" + str(i)
        ofile = open(output_name, "wt")
        for line in lines[i:i+100000]:
            ofile.write(str(line))
        ofile.close()
    f.close()



















