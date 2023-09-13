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
parser.add_argument('infile', type=str, help = 'input LAMMPS data file')
parser.add_argument('outfile', nargs='?', type=str, help = 'output .npz file',
                    default='')
parser.add_argument('--xyz', type=str, help = 'output .xyz file')
parser.add_argument('-d', '--duplicate', action='store_true', help = 'whether duplicate cell')
parser.add_argument('--diff', action='store_true', help = 'whether to calculate differential structure factor')
parser.add_argument('--dupdup', action='store_true', help = 'whether duplicate cell')
parser.add_argument('-s', '--shift', type=float, help = 'whether duplicate cell')
parser.add_argument('-q', nargs=2, type=float, help='q range', default=[0.0,1.0])
parser.add_argument('--dir', type=str, help = 'output directory')
parser.add_argument('-n','--numq',type=int,help='number of points in q range', default=1000)
parser.add_argument('--type',type=str,help='type of input file: data or trj')
parser.add_argument('--vec', type=int, help='number of q direction to average', default=36*18)
parser.add_argument('--gpu',type=int,help='which gpu to use')
parser.add_argument('--sphere', action='store_true', help = 'cut cell to sphere')
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
    directions = '_directions' if args.directions else ''
    savename = label + '_q{}_{}_n{}'.format(args.q[0], args.q[1], args.numq) + dup + diff + directions

#if file.lower().endswith('.xyz'):
# frames = tr.readxyz(file)
#else:

if not os.path.isfile(file):
    print('No file "{}" exist'.format(file))
    exit(2)

if args.type:
   if args.type=='data':
     frames = tr.readdata(file)
   elif args.type=='trj':
     frames = tr.readfile(file)
   else:
     print("type should be data or trj")
     quit
else:
   if file.endswith('.data'):
     frames = tr.readdata(file)
   elif file.endswith('.lammpstrj'):
     frames = tr.readfile(file)
   else:
     print("unknown file extension: use --type argument")
     quit

#frames = tr.readdata(file)

n_frames = len(frames)
# print(len(frames))
for iframe in range(len(frames)):
 if n_frames>1:
   savename_out = savename + '_t{}'.format(iframe)
 else:
   savename_out = savename
 frame = frames[iframe]
#frame = frames[0]
 x,y,z,f = tr.get_xyzf_from_frame_data_or_trj(frame)

 x = x.reshape(-1,1)
 y = y.reshape(-1,1)
 z = z.reshape(-1,1)
#f = np.ones((x.shape[0],1))    
 f = f.reshape(-1,1)
 rf = np.concatenate((x,y,z,f),axis = 1)   

 if not args.diff:
    rf = rf[rf[:, 3] > 0.0, :]

 lx = frame['xhi']-frame['xlo']
 ly = frame['yhi']-frame['ylo']
 lz = frame['zhi']-frame['zlo']

 if args.duplicate:
    rf = tr.cell_duplicate(rf, lx, ly, lz)
    if args.sphere:
        rf = tr.cell_sphere(rf, frame['xlo']+lx/2.,frame['ylo']+ly/2.,frame['zlo']+lz/2.,min(lx/2.,ly/2.,lz/2.))

 elif args.dupdup:
    rf = tr.cell_duplicate(rf, lx, ly, lz)
    rf = tr.cell_duplicate(rf, 2*lx, 2*ly, 2*lz)
    if args.sphere:
        rf = tr.cell_sphere(rf, frame['xlo']+lx,frame['ylo']+ly,frame['zlo']+lz,min(lx,ly,lz))

 elif args.shift:
    rf = tr.cell_duplicate(rf, args.shift, args.shift, args.shift )
    if args.sphere:
        rf = tr.cell_sphere(rf, frame['xlo']+(lx+args.shift)/2.,frame['ylo']+(ly+args.shift)/2.,frame['zlo']+(lz+args.shift)/2.,args.shift/2.+min(lx/2.,ly/2.,lz/2.))

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
       if not args.directions:
           res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf, qs, args.vec, f.mean())
       else:
           res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = tr.structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf, qs, args.vec, f.mean())

 e = time.time()
 print('{:.1f} s used'.format(e-s), end='')

 if not np.array_equal(qs,res_qs):
#   print('qs ok')
# else:
   print('qs is not ok')
   print(qs)
   print(res_qs) 
 


# savename = label+'_i{}'.format(iframe)+'_nq{}'.format(numq)+'_sf_loz'
# print(savename)
 if(args.dir):
   output_savename = args.dir + savename_out
 else:
   output_savename = savename_out

 if args.xyz:
   if(args.dir):
       xyz_savename = args.dir + args.xyz
   else:
       xyz_savename = args.xyz

 if(not args.directions):
   np.savez(output_savename,qs=qs,res_abs=res_abs,res_sq=res_sq)
 else:
   np.savez(output_savename,qs=qs,res_abs=res_abs,res_sq=res_sq,xyz_all=xyz_all,xyz_sphere=xyz_sphere,phi_theta=phi_theta)

 print('saved as {}'.format(output_savename))


 if args.xyz:
   tr.write_xyz(xyz_savename,rf)






