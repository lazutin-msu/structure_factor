import numpy as np
# import cupy as cp
import time
# import sys
# import matplotlib.pyplot as plt
import re
# import glob
import lammps_trajectory as tr
# import tracemalloc
import argparse

parser = argparse.ArgumentParser(description = 'Calculate structure factor')
# parser.add_argument('file', type = str, nargs = 2, help = 'input and output files')
parser.add_argument('infile', type=str, help = 'input LAMMPS data file')
parser.add_argument('outfile', nargs='?', type=str, help = 'output .npz file',
                    default='')
parser.add_argument('-d', '--duplicate', action='store_true', help = 'whether duplicate cell')
parser.add_argument('-q', nargs=2, type=float, help='q range', default=[0.0,1.0])
parser.add_argument('--dir', type=str, help = 'output directory')
parser.add_argument('-n','--numq',type=int,help='number of points in q range', default=1000)
args = parser.parse_args()

qs = np.linspace(args.q[0], args.q[1], num = args.numq, endpoint=False)

file = args.infile
if args.outfile:
    savename = args.outfile
else:
    m = re.search('B_(\w*)_t', file)
    if m:
        label = m.group(1)
    else:
        label = 'unknown'
    dup = '_dup' if args.duplicate else ''
    savename = label + '_q{}_{}_n{}'.format(args.q[0], args.q[1], args.numq) + dup 

frames = tr.readdata(file)
# print(len(frames))
# for iframe in range(len(frames)):
 
# frame = frames[iframe]
frame = frames[0]
x,y,z = tr.get_xyz_from_frame(frame)

x = x.reshape(-1,1)
y = y.reshape(-1,1)
z = z.reshape(-1,1)
f = np.ones((x.shape[0],1))    
rf = np.concatenate((x,y,z,f),axis = 1)   

if args.duplicate:
    rf = tr.cell_duplicate(rf, frame['xhi']-frame['xlo'], frame['yhi']-frame['ylo'], frame['zhi']-frame['zlo'])

s = time.time()
#res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress(rf, qs, 36, 18, 0.0)
res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci(rf, qs, 36 * 18, 0.0)
e = time.time()
print('{} s used'.format(e-s))

if np.array_equal(qs,res_qs):
   print('qs ok')
else:
   print('qs is not ok')
   print(qs)
   print(res_qs) 
 


 

# savename = label+'_i{}'.format(iframe)+'_nq{}'.format(numq)+'_sf_loz'
# print(savename)

if(args.dir):
   np.savez(args.dir + savename,qs=qs,res_abs=res_abs,res_sq=res_sq)
   print('save as {}'.format(args.dir + savename))
else:
   np.savez( savename,qs=qs,res_abs=res_abs,res_sq=res_sq)
   print('save as {}'.format(savename))



