import numpy as np
try:
    import cupy as cp
#    print("module 'cupy' is installed")
except ModuleNotFoundError:
    print("module 'cupy' is not installed")
    quit(1)

import os.path
import time
import re
import lammps_trajectory_cupy as tr
import structure_factor as sf
import sys
import scipy.signal as si

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == "__main__":
    import argparse
    import tracemalloc
    import linecache
    
    tracemalloc.start()
    
    parser = argparse.ArgumentParser(description = 'Calculate structure factor')
    # parser.add_argument('file', type = str, nargs = 2, help = 'input and output files')
    parser.add_argument('infile', type=str, help = 'input LAMMPS data file')
    parser.add_argument('outfile', nargs='?', type=str, help = 'output .npz file',
                        default='')
    parser.add_argument('--xyz', type=str, help = 'output .xyz file')
#    parser.add_argument('-d', '--duplicate', action='store_true', help = 'whether duplicate cell')
    parser.add_argument('--diff', action='store_true', help = 'whether to calculate differential structure factor')
    parser.add_argument('--dupdup', action='store_true', help = 'whether duplicate cell')
    parser.add_argument('-s', '--shift', type=float, help = 'whether duplicate cell')
    parser.add_argument('-q', nargs=2, type=float, help='q range', default=[-10.0,10.0])
    parser.add_argument('--dir', type=str, help = 'output directory')
    parser.add_argument('-n','--numq',type=int,help='number of points in q range', default=101)
    parser.add_argument('--type',type=str,help='type of input file: data or trj')
    parser.add_argument('--vec', type=int, help='number of q direction to average', default=36*18)
    parser.add_argument('--gpu',type=int,help='which gpu to use')
    parser.add_argument('--dframe',type=int,help='step on time frames to calculate', default=1)
    parser.add_argument('--prominence', type=float, help = 'prominence level',default=0.5)
#    parser.add_argument('--sphere', action='store_true', help = 'cut cell to sphere')
#    parser.add_argument('--single', action='store_true', help = 'output single file for all frames')
#    parser.add_argument('--directions', action='store_true', help = 'if no averaging over q directions is needed')
    parser.add_argument('--select', type=int, help='beads form factor selection: 0 - 4=1 2,3=-1 ; 1 - 4,5=1 1,2,3=-1 ; 2 - 1=1 2=-1 ; 3 - 1=-1 2=1 ', default=0)
    args = parser.parse_args()
    
#    qs = np.linspace(args.q[0], args.q[1], num = args.numq, endpoint=False)
    qsx = np.linspace(args.q[0], args.q[1], num = args.numq, endpoint=True)
    qsy = np.linspace(args.q[0], args.q[1], num = args.numq, endpoint=True)
    qsz = np.linspace(args.q[0], args.q[1], num = args.numq, endpoint=True)
    
    QX,QY,QZ = np.meshgrid(qsx,qsy,qsz)
    
    QX,QY,QZ = np.meshgrid(qsx,qsy,qsz)

    QQX = QX.flatten()
    QQY = QY.flatten()
    QQZ = QZ.flatten()

    xyz = np.stack([QQX,QQY,QQZ],axis=1)
    
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
         frames = sf.readdata(file)
       elif args.type=='trj':
         rf_all,df = sf.readtrj2np2(file,args.select)
       elif args.type=='npz':
         rf_all,df = sf.readnpz2np2(file,args.select)
       else:
         print("type should be data or trj")
         quit
    else:
       if file.endswith('.data'):
         frames = sf.readdata(file)
       elif file.endswith('.lammpstrj'):
         rf_all,df = sf.readtrj2np2(file,args.select)
       elif file.endswith('.npz'):
         rf_all,df = sf.readnpz2np2(file,args.select)
       else:
         print("unknown file extension: use --type argument")
         quit
    
    #frames = tr.readdata(file)
#    snapshot = tracemalloc.take_snapshot()
#    display_top(snapshot)
    
#    n_frames = len(frames)
#    n_frames = rf_all.shape[2]
    n_frames = len(rf_all)
    
    res_abs_all_scalar = []
    res_sq_all_scalar = []
    res_qs_all_scalar = []


    res_abs_all_vec = []
    res_sq_all_vec = []
#    res_qs_all_vec = []
    res_xyz_all_vec = []
    
    peaks_all = []
    proms_all = []
    timesteps_all = []
    rf_all_vec = []

    s = time.time()

    length = len(list(range(0,n_frames,args.dframe)))
    sf.printProgressBar(0, 2*length, prefix = 'Progress:', suffix = 'Complete', length = 50)

#    for iframe in range(0,n_frames,args.dframe):
    for iframe in [-1]:
    
     frame = df.iloc[iframe]
     rf = rf_all[iframe]
    
     rf = rf[rf[:, 3] > 0.0, :]
    
     lx = frame['xhi']-frame['xlo']
     ly = frame['yhi']-frame['ylo']
     lz = frame['zhi']-frame['zlo']
    
#     rf = tr.cell_duplicate(rf, lx, ly, lz)
     rf = tr.cell_sphere(rf, frame['xlo']+lx/2.,frame['ylo']+ly/2.,frame['zlo']+lz/2.,min(lx/2.,ly/2.,lz/2.))
#     rf = tr.cell_sphere(rf, frame['xlo']+lx,frame['ylo']+ly,frame['zlo']+lz,min(lx,ly,lz))
    
     s = time.time()
    #res_abs,res_sq,res_qs = tr.structure_factor_cuda_better_wrap2_progress_cupy(rf, qs, 36, 18, 0.0)
    
     if args.gpu is not None:
      with cp.cuda.Device(args.gpu):
#           if not args.directions:
#               res_abs,res_sq,res_qs = sf.structure_factor_cupy_everything(rf, qs, args.vec, 0.0, False)
#           else:
               res_abs,res_sq,xyz_all = sf.structure_factor_cupy_xyz(rf, xyz, 0.0, print=True)
     else:
#           if not args.directions:
#               res_abs,res_sq,res_qs = sf.structure_factor_cupy_everything(rf, qs, args.vec, 0.0, False)
#           else:
               res_abs,res_sq,xyz_all = sf.structure_factor_cupy_xyz(rf, xyz, 0.0, print=True)

     res_abs_all_vec.append(res_abs)
     res_sq_all_vec.append(res_sq)
#     res_qs_all_vec.append(qs)
     res_xyz_all_vec.append(xyz_all)
     timesteps_all.append(frame['timestep'])
     rf_all_vec.append(rf)
    
     e = time.time()
     print('{:.1f} s used '.format(e-s), end='')
    
     if not np.array_equal(xyz_all,xyz):
    #   print('qs ok')
    # else:
       print('xyz is not ok')
       print(xyz_all)
       print(xyz) 

     sf.printProgressBar(iframe+1, 2*length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    e = time.time()
    print('{:.1f} s used '.format(e-s), end='')

    res_abs_vector = np.stack(res_abs_all_vec,axis=0)
    res_sq_vector  = np.stack(res_sq_all_vec,axis=0)
#    res_qs_vector  = np.stack(res_qs_all_vec,axis=0)
    res_xyz_vector = np.stack(res_xyz_all_vec,axis=0)
    rf_all_vector  = np.stack(rf_all_vec,axis=0)

    timesteps_out = np.array(timesteps_all)

    if(args.dir):
           output_savename = args.dir + savename
    else:
           output_savename = savename
        
#    np.savez(output_savename,qs_scalar=res_qs_scalar,res_abs_scalar=res_abs_scalar,res_sq_scalar=res_sq_scalar,
#    qs_vector=res_qs_vector,res_abs_vector=res_abs_vector,res_sq_vector=res_sq_vector,xyz_all=xyz_all,xyz_sphere=xyz_sphere,phi_theta=phi_theta,peaks_index=peaks_unique,
#    timesteps_out=timesteps_out)

    np.savez(output_savename,
    res_abs_vector=res_abs_vector,res_sq_vector=res_sq_vector,xyz=xyz,res_xyz_vector=res_xyz_vector,
    timesteps_out=timesteps_out,
    rf=rf_all_vector)
    
    print('saved as {}'.format(output_savename))
        
        
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)


