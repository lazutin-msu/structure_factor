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
import pandas as pd

def readtrj(filename):

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
    
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x),'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x)}
    
    iline = 0
    
    while iline<len(lines):
        line = lines[iline]
        line = line.lstrip() 
        if not line.startswith('ITEM: TIMESTEP'):
            print("should be TIMESTEP")
            return 0
        else:
            framenum = int(lines[iline+1].lstrip())
            iline += 2
            line = lines[iline]
            line = line.lstrip() 
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                print("should be NUMBER OF ATOMS")
                quit
            else:
                atomnum = int(lines[iline+1].lstrip())
                iline += 2
                line = lines[iline]
                line = line.lstrip() 
                if not line.startswith('ITEM: BOX BOUNDS pp pp pp'):
                    print("should be BOX BOUNDS pp pp pp")
                    quit
                else:
                    (xlo,xhi) = (lines[iline+1].lstrip()).split()
                    (ylo,yhi) = (lines[iline+2].lstrip()).split()
                    (zlo,zhi) = (lines[iline+3].lstrip()).split()
                    iline += 4
                    line = lines[iline]
                    line = line.lstrip() 
                    if not line.startswith('ITEM: ATOMS'):
                        print("should be ATOMS")
                        quit
                    else:
                        heads=line[11:].lstrip().split()
                        iline += 1
                        atoms = []
                        for iatom in range(atomnum):
                            arr = lines[iline].lstrip().split()
                            
                            d = dict(zip(heads,arr))
                            for key in conv:
                                if key in d:
                                    d[key] = conv[key](d[key])
                            atoms.append(d)
                            iline += 1
                        atoms2 = sorted(atoms,key= lambda d: int(d['id']))

            d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }

            frames.append(d1)
            printCounter(framenum,'Read: timestep','')
    print()
    return frames

def readtrj2np(filename,select):
    def func(t,select):
          if select == 0:
            if t==4:
              f = 1.0
            elif t==2 or t==3 :
              f = -1.0
          elif select == 1:
            if t==4 or t==5:
              f = 1.0
            elif t==2 or t==3 or t==1 :
              f = -1.0
          elif select == 2:
            if t==1:
              f = 1.0
            else:
              f = -1.0
          elif select == 3:
            if t==2:
              f = 1.0
            else:
              f = -1.0
            
          return f

    t2f = np.vectorize(func)

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
    
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x),'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x)}
    
    iline = 0
    rf_out = []
    df_list = []
    
    while iline<len(lines):
        line = lines[iline]
        line = line.lstrip() 
        if not line.startswith('ITEM: TIMESTEP'):
            print("should be TIMESTEP")
            return 0
        else:
            framenum = int(lines[iline+1].lstrip())
            iline += 2
            line = lines[iline]
            line = line.lstrip() 
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                print("should be NUMBER OF ATOMS")
                quit
            else:
                atomnum = int(lines[iline+1].lstrip())
                iline += 2
                line = lines[iline]
                line = line.lstrip() 
                if not line.startswith('ITEM: BOX BOUNDS pp pp pp'):
                    print("should be BOX BOUNDS pp pp pp")
                    quit
                else:
                    (xlo,xhi) = (lines[iline+1].lstrip()).split()
                    (ylo,yhi) = (lines[iline+2].lstrip()).split()
                    (zlo,zhi) = (lines[iline+3].lstrip()).split()
                    iline += 4
                    line = lines[iline]
                    line = line.lstrip() 
                    if not line.startswith('ITEM: ATOMS'):
                        print("should be ATOMS")
                        quit
                    else:
                        heads=line[11:].lstrip().split()
                        iline += 1
                        atoms = []
                        for iatom in range(atomnum):
                            arr = lines[iline].lstrip().split()
                            
                            d = dict(zip(heads,arr))
                            for key in conv:
                                if key in d:
                                    d[key] = conv[key](d[key])
                            atoms.append(d)
                            iline += 1
                        atoms2 = sorted(atoms,key= lambda d: int(d['id']))

            d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }

            d2 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum }

            #frames.append(d1)
            df_list.append(d2)
            x,y,z,t = get_xyzt_from_frame_data_or_trj(d1)

            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            z = z.reshape(-1,1)
            #f = np.ones((x.shape[0],1))    
            t = t.reshape(-1,1)
            f = t2f(t,select)
            rf = np.concatenate((x,y,z,f),axis = 1)   
            rf_out.append(rf)
            
            printCounter(framenum,'Read: timestep','')
    print()
#    return r,t
    rfnew = np.stack(rf_out,axis = 2)
    
    df = pd.DataFrame(df_list)

    return rfnew,df

def readnpz2np2(filename,select):

    rf_out = []
    df_list = []
    data = np.load(filename)
    rf = data['xyzf']
    d2 = {'timestep':0, 'xlo':0.0, 'xhi':35.0, 'ylo':0.0, 'yhi':35.0, 'zlo':0.0, 'zhi':35.0, 'natoms': rf.shape[0] }
    rf_out.append(rf)
    df_list.append(d2)
    df = pd.DataFrame(df_list)

    return rf_out,df


def readtrj2np2(filename,select):
    def func(t,select):
          if select == 0:
            if t==4:
              f = 1.0
            elif t==2 or t==3 :
              f = -1.0
          elif select == 1:
            if t==4 or t==5:
              f = 1.0
            elif t==2 or t==3 or t==1 :
              f = -1.0
          return f

    t2f = np.vectorize(func)

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
    
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x),'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x)}
    
    iline = 0
    rf_out = []
    df_list = []
    
    while iline<len(lines):
        line = lines[iline]
        line = line.lstrip() 
        if not line.startswith('ITEM: TIMESTEP'):
            print("should be TIMESTEP")
            return 0
        else:
            framenum = int(lines[iline+1].lstrip())
            iline += 2
            line = lines[iline]
            line = line.lstrip() 
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                print("should be NUMBER OF ATOMS")
                quit
            else:
                atomnum = int(lines[iline+1].lstrip())
                iline += 2
                line = lines[iline]
                line = line.lstrip() 
                if not line.startswith('ITEM: BOX BOUNDS pp pp pp'):
                    print("should be BOX BOUNDS pp pp pp")
                    quit
                else:
                    (xlo,xhi) = (lines[iline+1].lstrip()).split()
                    (ylo,yhi) = (lines[iline+2].lstrip()).split()
                    (zlo,zhi) = (lines[iline+3].lstrip()).split()
                    iline += 4
                    line = lines[iline]
                    line = line.lstrip() 
                    if not line.startswith('ITEM: ATOMS'):
                        print("should be ATOMS")
                        quit
                    else:
                        heads=line[11:].lstrip().split()
                        iline += 1
                        atoms = []
                        for iatom in range(atomnum):
                            arr = lines[iline].lstrip().split()
                            
                            d = dict(zip(heads,arr))
                            for key in conv:
                                if key in d:
                                    d[key] = conv[key](d[key])
                            atoms.append(d)
                            iline += 1
                        atoms2 = sorted(atoms,key= lambda d: int(d['id']))

            d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }

            d2 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum }

            #frames.append(d1)
            df_list.append(d2)
            x,y,z,t = get_xyzt_from_frame_data_or_trj(d1)

            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            z = z.reshape(-1,1)
            #f = np.ones((x.shape[0],1))    
            t = t.reshape(-1,1)
            f = t2f(t,select)
            rf = np.concatenate((x,y,z,f),axis = 1)   
            rf_out.append(rf)
            
            printCounter(framenum,'Read: timestep','')
    print()
#    return r,t
#    rfnew = np.stack(rf_out,axis = 2)
    
    df = pd.DataFrame(df_list)

#    return rfnew,df
    return rf_out,df

def readdata(filename):

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
        
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x), 'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x),'charge': lambda x: int(x)}
    
    iline = 0

    line = lines[iline]
    line = line.lstrip() 
    m = re.search('timestep = (\d*)', line)
    if m:
        framenum = int(m.group(1))
    else:
        framenum = -1
        
    iline += 1


    while iline<len(lines):
        line = lines[iline]
        line = line.lstrip() 
        m = re.search('^\s*(\d*)\s+atoms$', line)
        if not m:
            
            iline += 1
            continue
        else:
            break
        
    if m:
        atomnum = int(m.group(1))
    else:
        print("NUMBER OF ATOMS not found")
        quit
    
    iline += 1

    while iline<len(lines):
        line = lines[iline]
        line = line.lstrip() 
        m = re.search('^\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+xlo\s+xhi$', line)
        if not m:
            iline += 1
            continue
        else:
            break
    if m:
        xlo = float(m.group(1))
        xhi = float(m.group(2))
    else:
        print("xlo xhi not found")
        quit

    iline += 1
    line = lines[iline]
    line = line.lstrip() 
    m = re.search('^\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+ylo\s+yhi$', line)
    if m:
        ylo = float(m.group(1))
        yhi = float(m.group(2))
    else:
        print("ylo yhi not found")
        quit

    iline += 1
    line = lines[iline]
    line = line.lstrip() 
    m = re.search('^\s*(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+(-?\d+\.?\d*[Ee]?[+-]?\d*)\s+zlo\s+zhi$', line)
    if m:
        zlo = float(m.group(1))
        zhi = float(m.group(2))
    else:
        print("zlo zhi not found")
        quit

    iline += 1
    while iline<len(lines):        
        line = lines[iline]
        line = line.lstrip() 
        m = re.search('^Atoms # (\w*)$', line)
        if not m:
            
            iline += 1
            continue
        else:
            break
    if m:
        datatype = m.group(1)
    else:
        print("Atoms list not found")
        quit

    if(datatype=='full'):
        heads = ['id','mol','type','charge','x','y','z','ix','iy','iz']
    else:
        print("Type of Atoms list is not not found")
        quit

    # print('atoms found')
 
    iline += 2
    atoms = []
    for iatom in range(atomnum):
        arr = lines[iline].lstrip().split()
        d = dict(zip(heads,arr))
        for key in conv:
            if key in d:
                d[key] = conv[key](d[key])
        atoms.append(d)
        iline += 1
    atoms2 = sorted(atoms,key= lambda d: int(d['id']))

    d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }
    frames.append(d1)
    return frames    

def get_xyzf_from_frame_data_or_trj(frame):
    def get_xyz_from_atom(atom,lx,ly,lz):
            if 'x' in atom.keys():
              xt = atom['x']
            elif 'xu' in atom.keys():
              xt = atom['xu']
            elif 'xs' in atom.keys():
    #          print(atom['xs'])
    #          print(type(atom['xs']))
              xt = atom['xs']*lx
            else:
              print('no x or xu value')
              print(atom)
              quit
            if 'y' in atom.keys():
              yt = atom['y']
            elif 'yu' in atom.keys():
              yt = atom['yu']
            elif 'ys' in atom.keys():
              yt = atom['ys']*ly
            else:
              print('no y or yu value')
              print(atom)
              quit
            if 'z' in atom.keys():
              zt = atom['z']
            elif 'zu' in atom.keys():
              zt = atom['zu']
            elif 'zs' in atom.keys():
              zt = atom['zs']*lz
            else:
              print('no z or zu value')
              print(atom)
              quit
            return xt,yt,zt
    # frames = readdata(dire+file)
    # frame = frames[0]
    atoms = frame['atoms']
    lx = frame['xhi']-frame['xlo']
    ly = frame['yhi']-frame['ylo']
    lz = frame['zhi']-frame['zlo']
    # print(atoms)
    # print(atoms)
    x = []
    y = []
    z = []
    f = []
    for atom in atoms:
        xt,yt,zt = get_xyz_from_atom(atom,lx,ly,lz)
        if select == 0:
          if atom['type']==4:
            x.append(xt)
            y.append(yt)
            z.append(zt)
            f.append(1.0)
          elif atom['type']==2 or atom['type']==3 :
            x.append(xt)
            y.append(yt)
            z.append(zt)
            f.append(-1.0)
        elif select == 1:
          if atom['type']==4 or atom['type']==5:
            x.append(xt)
            y.append(yt)
            z.append(zt)
            f.append(1.0)
          elif atom['type']==2 or atom['type']==3 or atom['type']==1 :
            x.append(xt)
            y.append(yt)
            z.append(zt)
            f.append(-1.0)
        else:
          print('wrong select type {}'.format(select))
          sys.exit(1)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)
    return x,y,z,f

def get_xyzt_from_frame_data_or_trj(frame):
    def get_xyz_from_atom(atom,lx,ly,lz):
            if 'x' in atom.keys():
              xt = atom['x']
            elif 'xu' in atom.keys():
              xt = atom['xu']
            elif 'xs' in atom.keys():
    #          print(atom['xs'])
    #          print(type(atom['xs']))
              xt = atom['xs']*lx
            else:
              print('no x or xu value')
              print(atom)
              quit
            if 'y' in atom.keys():
              yt = atom['y']
            elif 'yu' in atom.keys():
              yt = atom['yu']
            elif 'ys' in atom.keys():
              yt = atom['ys']*ly
            else:
              print('no y or yu value')
              print(atom)
              quit
            if 'z' in atom.keys():
              zt = atom['z']
            elif 'zu' in atom.keys():
              zt = atom['zu']
            elif 'zs' in atom.keys():
              zt = atom['zs']*lz
            else:
              print('no z or zu value')
              print(atom)
              quit
            return xt,yt,zt
    # frames = readdata(dire+file)
    # frame = frames[0]
    atoms = frame['atoms']
    lx = frame['xhi']-frame['xlo']
    ly = frame['yhi']-frame['ylo']
    lz = frame['zhi']-frame['zlo']
    # print(atoms)
    # print(atoms)
    x = []
    y = []
    z = []
    t = []
    for atom in atoms:
        xt,yt,zt = get_xyz_from_atom(atom,lx,ly,lz)
        tt = atom['type']
        x.append(xt)
        y.append(yt)
        z.append(zt)
        t.append(tt)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    t = np.array(t,dtype=int)
    return x,y,z,t

def Cartesian_np(rtp):
    ptsnew = np.zeros(rtp.shape)
    ptsnew[:,0] = rtp[:,0] * np.sin(rtp[:,1]) * np.cos(rtp[:,2])
    ptsnew[:,1] = rtp[:,0] * np.sin(rtp[:,1]) * np.sin(rtp[:,2])
    ptsnew[:,2] = rtp[:,0] * np.cos(rtp[:,1]) 
    return ptsnew

def structure_factor_cupy_everything(rf,qs,points_num,fmean,directions=False, print=True):

# input
# rf   shape N,4   N - number of beads, 4 - x,y,z,f
# qs shape Nq    - number of q modules
# points_num - number of directions at each q module
# fmean - will be substracted from f_i 

# output
# directions == True
# res_qs_out  - shape len(qs) = qs
# res_abs_out - shape len(qs),points_num   - |qs|, directions          - S_abs(q) 
# res_sq_out  - shape len(qs),points_num   - |qs|, directions          - S_sq(q)
# res_xyz_out - shape len(qs),points_num,3 - |qs|, directions, x;y;z   - vector q
# xyz         - shape points_num,3         - directions, x;y;z         - vector q for |qs| = 1
# phi_theta   - shape points_num,3         - directions, rho;theta;phi - vector q for |qs| = 1

# directions == False
# res_qs_out  - shape len(qs) = qs
# res_abs_out - shape len(qs) - S_abs(q)
# res_sq_out  - shape len(qs) - S_sq(q)
  
  s = time.time()
  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]

  z2_arr, z2_arr_step = np.linspace(1,-1,num=points_num,endpoint=False,retstep=True)
  z2_arr = z2_arr + 0.5*z2_arr_step

  sp_dlong = np.pi*(3.0-np.sqrt(5.0))

  phi2_arr, phi2_arr_step = np.linspace(0,points_num*sp_dlong,num=points_num,endpoint=False,retstep=True)

  r2_arr = np.sqrt(1.0-z2_arr*z2_arr)
  rho2_arr = np.sqrt(r2_arr*r2_arr+z2_arr*z2_arr)
  theta2_arr = np.arctan2(r2_arr,z2_arr)

  pt_pairn = np.empty((phi2_arr.shape[0],3))

  pt_pairn[:,0] = rho2_arr
  pt_pairn[:,1] = theta2_arr
  pt_pairn[:,2] = phi2_arr

  phi_theta = pt_pairn

  xyz = Cartesian_np(phi_theta)

  r_mys = np.array_split(qs,qs.shape[0])

  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  res_xyz_arr = []
  l = len(r_mys)
  if print:
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
  for ir_my,r_my in enumerate(r_mys):
    xyz2 = np.einsum('i,jk',r_my,xyz)
        
    xyz2_gpu = cp.asarray(xyz2)
    r_gpu = cp.asarray(r)

    res_gpu = cp.einsum('ik,zlk',r_gpu,xyz2_gpu)
    res2_gpu = cp.exp(1j*res_gpu)
    
    f_gpu = cp.asarray(f)
    res3_gpu = cp.einsum('ikz,i',res2_gpu,f_gpu)
    
    res3_gpu = res3_gpu * np.conjugate(res3_gpu)
    
    res3 = cp.asnumpy(res3_gpu)

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2

    if(not directions):
        res4_abs = np.mean(res3_abs,axis=0)
        res4_sq = np.mean(res3_sq,axis=0)
    else:
        res4_abs = res3_abs
        res4_sq  = res3_sq

    res4_abs = res4_abs / natom
    res4_sq = res4_sq / natom / natom

    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)

    if directions:
        res_xyz_arr.append(xyz2)
   
    if print:
      printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

  if directions:
      res_abs_out = np.squeeze(np.stack(res_abs_arr,axis=0))
      res_sq_out  = np.squeeze(np.stack(res_sq_arr,axis=0))
      res_qs_out  = np.squeeze(np.stack(res_qs_arr,axis=0))
      res_xyz_out = np.squeeze(np.stack(res_xyz_arr,axis=0))

      return res_abs_out,res_sq_out,res_qs_out,res_xyz_out,xyz,phi_theta
  else:
      res_abs_out = np.concatenate(res_abs_arr,axis=None)
      res_sq_out = np.concatenate(res_sq_arr,axis=None)
      res_qs_out = np.concatenate(res_qs_arr,axis=None)

      return res_abs_out,res_sq_out,res_qs_out



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def printCounter (iteration, prefix = '', suffix = '', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    print(f'\r{prefix} {iteration} {suffix}', end = printEnd)
    # Print New Line on Complete
#    if iteration == total: 
#        print()

if __name__ == "__main__":
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
    parser.add_argument('--single', action='store_true', help = 'output single file for all frames')
    parser.add_argument('--directions', action='store_true', help = 'if no averaging over q directions is needed')
    parser.add_argument('--select', type=int, help='beads form factor selection: 0 - 4=1 2,3=-1 ; 1 - 4,5=1 1,2,3=-1 ', default=0)
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
         frames = readdata(file)
       elif args.type=='trj':
         frames = readtrj(file)
       else:
         print("type should be data or trj")
         quit
    else:
       if file.endswith('.data'):
         frames = readdata(file)
       elif file.endswith('.lammpstrj'):
         frames = readtrj(file)
       else:
         print("unknown file extension: use --type argument")
         quit
    
    #frames = tr.readdata(file)
    
    n_frames = len(frames)
    
    if n_frames>1 and args.single:
        if not args.directions:
            res_abs_all = []
            res_sq_all = []
            res_qs_all = []
        else:
            res_abs_all = []
            res_sq_all = []
            res_qs_all = []
            # xyz_all_all = []
            # xyz_sphere_all = []
            # phi_theta_all = []
    
    # print(len(frames))
    for iframe in range(len(frames)):
     if n_frames>1 and not args.single:
       savename_out = savename + '_t{}'.format(iframe)
     else:
       savename_out = savename
     frame = frames[iframe]
    #frame = frames[0]
     x,y,z,f = get_xyzf_from_frame_data_or_trj(frame, args.select)
    
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
               res_abs,res_sq,res_qs = structure_factor_cupy_everything(rf, qs, args.vec, 0.0, False)
           else:
               res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = structure_factor_cupy_everything(rf, qs, args.vec, 0.0, True)
       else:
           if not args.directions:
               res_abs,res_sq,res_qs = structure_factor_cupy_everything(rf, qs, args.vec, f.mean(), False)
           else:
               res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = structure_factor_cupy_everything(rf, qs, args.vec, f.mean(),True)
     else:
       if not args.diff:
           if not args.directions:
               res_abs,res_sq,res_qs = structure_factor_cupy_everything(rf, qs, args.vec, 0.0, False)
           else:
               res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = structure_factor_cupy_everything(rf, qs, args.vec, 0.0, True)
       else:
           if not args.directions:
               res_abs,res_sq,res_qs = structure_factor_everything(rf, qs, args.vec, f.mean(), False)
           else:
               res_abs,res_sq,res_qs,xyz_all,xyz_sphere,phi_theta = structure_factor_cupy_everything(rf, qs, args.vec, f.mean(), True)
    
     e = time.time()
     print('{:.1f} s used '.format(e-s), end='')
    
     if not np.array_equal(qs,res_qs):
    #   print('qs ok')
    # else:
       print('qs is not ok')
       print(qs)
       print(res_qs) 
     
     if not args.single:
    
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
     else:
        res_abs_all.append(res_abs)
        res_sq_all.append(res_sq)
        res_qs_all.append(qs)
    
    if args.single:
         res_abs_out = np.stack(res_abs_all,axis=0)
         res_sq_out  = np.stack(res_sq_all,axis=0)
         res_qs_out  = np.stack(res_qs_all,axis=0)
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
           np.savez(output_savename,qs=qs,res_abs=res_abs_out,res_sq=res_sq_out)
         else:
           np.savez(output_savename,qs=res_qs_out,res_abs=res_abs_out,res_sq=res_sq_out,xyz_all=xyz_all,xyz_sphere=xyz_sphere,phi_theta=phi_theta)
        
         print('saved as {}'.format(output_savename))
        
        
         if args.xyz:
           tr.write_xyz(xyz_savename,rf)





