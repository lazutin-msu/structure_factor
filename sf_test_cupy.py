import numpy as np
try:
    import cupy as cp
#    print("module 'cupy' is installed")
except ModuleNotFoundError:
    print("module 'cupy' is not installed")
    quit(1)

import os.path
import time
# import sys
# import matplotlib.pyplot as plt
import re
# import glob
import lammps_trajectory_cupy as tr
# import tracemalloc
import argparse

parser = argparse.ArgumentParser(description = 'Calculate structure factor')
# parser.add_argument('file', type = str, nargs = 2, help = 'input and output files')
parser.add_argument('infile', type=str, help = 'input npz data file')
parser.add_argument('outfile', nargs='?', type=str, help = 'output .npz file',
                    default='')
parser.add_argument('--xyz', type=str, help = 'output .xyz file')
#parser.add_argument('-d', '--duplicate', action='store_true', help = 'whether duplicate cell')
parser.add_argument('--diff', action='store_true', help = 'whether to calculate differential structure factor')
#parser.add_argument('--dupdup', action='store_true', help = 'whether duplicate cell')
#parser.add_argument('-s', '--shift', type=float, help = 'whether duplicate cell')
parser.add_argument('-q', nargs=2, type=float, help='q range', default=[0.0,1.0])
parser.add_argument('--dir', type=str, help = 'output directory')
parser.add_argument('-n','--numq',type=int,help='number of points in q range', default=1000)
parser.add_argument('--vec', type=int, help='number of q direction to average', default=36*18)
parser.add_argument('--gpu',type=int,help='which gpu to use')
parser.add_argument('--directions', action='store_true', help = 'if no averaging over q directions is needed')
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
    diff = '_diff' if args.diff else ''
    savename = label + '_q{}_{}_n{}'.format(args.q[0], args.q[1], args.numq) + dup + diff

#if file.lower().endswith('.xyz'):
# frames = tr.readxyz(file)
#else:

#frames = tr.readdata(file)


# print(len(frames))
# for iframe in range(len(frames)):
 
# frame = frames[iframe]
#frame = frames[0]
#x,y,z,f = tr.get_xyzf_from_frame(frame)

#x = x.reshape(-1,1)
#y = y.reshape(-1,1)
#z = z.reshape(-1,1)
#f = np.ones((x.shape[0],1))    
#f = f.reshape(-1,1)
#rf = np.concatenate((x,y,z,f),axis = 1)   
if not os.path.isfile(file):
    print('No file "{}" exist'.format(file))
    exit(2)
npzfile = np.load(file)
rf = npzfile['rf']

if not args.diff:
    rf = rf[rf[:, 3] > 0.0, :]

#lx = frame['xhi']-frame['xlo']
#ly = frame['yhi']-frame['ylo']
#lz = frame['zhi']-frame['zlo']

#if args.duplicate:
#    rf = tr.cell_duplicate(rf, frame['xhi']-frame['xlo'], frame['yhi']-frame['ylo'], frame['zhi']-frame['zlo'])

#if args.dupdup:
#    rf = tr.cell_duplicate(rf, lx, ly, lz)
#    rf = tr.cell_duplicate(rf, 2*lx, 2*ly, 2*lz)

#if args.shift:
#    rf = tr.cell_duplicate(rf, args.shift, args.shift, args.shift )

s = time.time()
#res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_cupy(rf, qs, 36, 18, 0.0)

if args.gpu is not None:
  with cp.cuda.Device(args.gpu):
   if not args.diff:
       if not args.directions:
           res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf, qs, args.vec, 0.0)
       else:
           res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf, qs, args.vec, 0.0)
   else:
       f = rf[:,3]
       if not args.directions:
           res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf, qs, args.vec, f.mean())
       else:
           res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf, qs, args.vec, f.mean())

else:
   if not args.diff:
       if not args.directions:
           res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf, qs, args.vec, 0.0)
       else:
           res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf, qs, args.vec, 0.0)
   else:
       f = rf[:,3]
       if not args.directions:
           res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf, qs, args.vec, f.mean())
       else:
           res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf, qs, args.vec, f.mean())

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
   output_savename = args.dir + savename
else:
   output_savename = savename

if args.xyz:
   if(args.dir):
       xyz_savename = args.dir + args.xyz
   else:
       xyz_savename = args.xyz

#print('res_qs',res_qs.shape)
#print('res_abs',res_abs.shape)
#print('res_sq',res_sq.shape)
#print('xyz_all',xyz_all.shape)
#print('xyz_sphere',xyz_sphere.shape)
#print('phi_theta',phi_theta.shape)

if(not args.directions):
   np.savez(output_savename,qs=qs,res_abs=res_abs,res_sq=res_sq)
else:
   np.savez(output_savename,qs=qs,res_abs=res_abs,res_sq=res_sq,xyz_all=xyz_all,xyz_sphere=xyz_sphere,phi_theta=phi_theta)

print('saved as {}'.format(output_savename))


if args.xyz:
   tr.write_xyz(xyz_savename,rf)
   





