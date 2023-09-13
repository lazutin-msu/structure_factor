import numpy as np
import cupy as cp
import re
import time

def write_xyz(filename,rf):
    f = open(filename, 'w')
    
    n = rf.shape[0]
    f.write('{}\n'.format(n))
    f.write('test model\n')
    for i in range(n):
        f.write(' C {} {} {}\n'.format(rf[i,0],rf[i,1],rf[i,2]))

    f.close()

def lam(row):
    y = row[2] // 2.0
    if(int(y)%2==0):
        return 1
    else:
        return -1

def lam_test(xa):
   ret = []
   for x in xa:  
    y = x // 2.0
    if(int(y)%2==0):
        ret.append(1)
    else:
        ret.append( -1)
   return ret

def Spherical_np(xyz):
#    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def Cartesian_np(rtp):
#    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    ptsnew = np.zeros(rtp.shape)
    ptsnew[:,0] = rtp[:,0] * np.sin(rtp[:,1]) * np.cos(rtp[:,2])
    ptsnew[:,1] = rtp[:,0] * np.sin(rtp[:,1]) * np.sin(rtp[:,2])
    ptsnew[:,2] = rtp[:,0] * np.cos(rtp[:,1]) 
    return ptsnew


def structure_factor(rf,qs,phi_num,theta_num):

  r = rf[:,:3]
  f = rf[:,3] 

  r = r[:, np.newaxis] - r
  f = f[:, np.newaxis] * f
  #rf = np.concatenate((r,f),axis = 1)
  #print(r)
  #print(r[0])
  #print(rf)
  
  # phi_num = 4
  # theta_num = 4

  phi, phi_step = np.linspace(-np.pi, np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)

#phi_theta = np.meshgrid(phi, theta)
#phi_theta = np.dot(phi.T,theta)
#print(phi_theta) 

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

# phi_theta =  [[1.0, theta[0], phi[0]],
#               [1.0, theta[0], phi[1]]]
# phi_theta = np.array(phi_theta)

  r_q = np.ones((pt_pairs.shape[0],1))
  # print(r_q)
  # print(pt_pairs)
  #pt_pairs = np.concatenate((np.ones((pt_pairs.shape[0],1)),pt_pairs).axis=1)
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  # print(phi_theta)

  xyz = Cartesian_np(phi_theta)

  # print(xyz)
  
 
  # r_my = np.linspace(1, 5, num = 5, endpoint=True)
  
  r_my = qs

  xyz2 = np.einsum('i,jk',r_my,xyz)
  # print(r.shape, xyz.shape)
  # print(r.shape, xyz2.shape)

  # res = np.einsum('ijk,lk',r,xyz)
  res = np.einsum('ijk,zlk',r,xyz2)
#res = np.tensordot(r,xyz,axes=([2],[1]))

  # print(res)
  # print(res.shape)

  res2 = np.exp(2*np.pi*1j*res)
  # print(res2)

  # print(f)
  # print(f.shape,res2.shape)

  #res3 = np.tensordot(f,res2,axes=([2],[2]))
  res3 = np.einsum('ijkz,ij',res2,f)

  # print(res3)
  # print(res3.sum)
  res3_abs = np.abs(res3)
  res3_sq = res3.real**2 + res3.imag**2

  # print(res3_abs)
  # print(res3_abs.shape)

  # res4_abs = np.sum(res3_abs,axis=(0,1))
  res4_abs = np.sum(res3_abs,axis=0)
  res4_sq = np.sum(res3_sq,axis=0)

  res4_abs = res4_abs / (phi_num * theta_num)
  res4_sq = res4_sq / (phi_num * theta_num)
  # print(res4_abs)
  # print(res4_abs.shape)

  return res4_abs,res4_sq


def structure_factor_cuda(rf,qs,phi_num,theta_num,fmean):

  r = rf[:,:3]
  f = rf[:,3] 

  r = r[:, np.newaxis] - r
  f = f[:, np.newaxis] * f 
  f = f - fmean * fmean
  
   # print(f)
  
  f1 = np.triu(f, k=1)
  
  # print(f1)
  
  f = f1
  
  natom = f.shape[0]
  # print(natom)
  #rf = np.concatenate((r,f),axis = 1)
  #print(r)
  #print(r[0])
  #print(rf)
  
  # phi_num = 4
  # theta_num = 4

  # phi, phi_step = np.linspace(-np.pi, np.pi, num = phi_num, endpoint=False, retstep=True)
  # phi = phi + 0.5 * phi_step

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  # phi = phi + 0.5 * phi_step

  # print(phi)
  
  # sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  # sintheta = sintheta + 0.5 *sintheta_step
  # theta = np.arcsin(sintheta)

  theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)
  
  # print(theta)

#phi_theta = np.meshgrid(phi, theta)
#phi_theta = np.dot(phi.T,theta)
#print(phi_theta) 

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

# phi_theta =  [[1.0, theta[0], phi[0]],
#               [1.0, theta[0], phi[1]]]
# phi_theta = np.array(phi_theta)

  r_q = np.ones((pt_pairs.shape[0],1))
  # print(r_q)
  # print(pt_pairs)
  #pt_pairs = np.concatenate((np.ones((pt_pairs.shape[0],1)),pt_pairs).axis=1)
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  # print(phi_theta)

  xyz = Cartesian_np(phi_theta)

  # print(xyz)
  
 
  # r_my = np.linspace(1, 5, num = 5, endpoint=True)
  
  r_my = qs

  xyz2 = np.einsum('i,jk',r_my,xyz)
  # print(r.shape, xyz.shape)
  # print(r.shape, xyz2.shape)

  # res = np.einsum('ijk,lk',r,xyz)
  res = np.einsum('ijk,zlk',r,xyz2)
#res = np.tensordot(r,xyz,axes=([2],[1]))

  # print(res)
  # print(res.shape)

  res2 = np.exp(2*np.pi*1j*res)
  # print(res2)

  # print(f)
  # print(f.shape,res2.shape)

  #res3 = np.tensordot(f,res2,axes=([2],[2]))
  res3 = np.einsum('ijkz,ij',res2,f)

  # print(res3)
  # print(res3.sum)
  res3_abs = np.abs(res3)
  res3_sq = res3.real**2 + res3.imag**2

  # print(res3_abs)
  # print(res3_abs.shape)

  # res4_abs = np.sum(res3_abs,axis=(0,1))
  res4_abs = np.sum(res3_abs,axis=0)
  res4_sq = np.sum(res3_sq,axis=0)

  res4_abs = res4_abs / (phi_num * theta_num) / natom
  res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
  # print(res4_abs)
  # print(res4_abs.shape)

  return res4_abs,res4_sq

def structure_factor_cuda_full(rf,qs,phi_num,theta_num,fmean):

  r = rf[:,:3]
  f = rf[:,3] 

  r = r[:, np.newaxis] - r
  f = f[:, np.newaxis] * f 
  f = f - fmean * fmean
  
  # print(f)
  
  # f1 = np.triu(f, k=1)
  
  # print(f1)
  
  # f = f1
  
  natom = f.shape[0]
  # print(natom)
  #rf = np.concatenate((r,f),axis = 1)
  #print(r)
  #print(r[0])
  #print(rf)
  
  # phi_num = 4
  # theta_num = 4

  # phi, phi_step = np.linspace(-np.pi, np.pi, num = phi_num, endpoint=False, retstep=True)
  # phi = phi + 0.5 * phi_step

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  # phi = phi + 0.5 * phi_step

  # print(phi)
  
  # sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  # sintheta = sintheta + 0.5 *sintheta_step
  # theta = np.arcsin(sintheta)

  theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)
  
  # print(theta)

#phi_theta = np.meshgrid(phi, theta)
#phi_theta = np.dot(phi.T,theta)
#print(phi_theta) 

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

# phi_theta =  [[1.0, theta[0], phi[0]],
#               [1.0, theta[0], phi[1]]]
# phi_theta = np.array(phi_theta)

  r_q = np.ones((pt_pairs.shape[0],1))
  # print(r_q)
  # print(pt_pairs)
  #pt_pairs = np.concatenate((np.ones((pt_pairs.shape[0],1)),pt_pairs).axis=1)
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  # print(phi_theta)

  xyz = Cartesian_np(phi_theta)

  # print(xyz)
  
 
  # r_my = np.linspace(1, 5, num = 5, endpoint=True)
  
  r_my = qs

  xyz2 = np.einsum('i,jk',r_my,xyz)
  # print(r.shape, xyz.shape)
  # print(r.shape, xyz2.shape)

  # res = np.einsum('ijk,lk',r,xyz)
  res = np.einsum('ijk,zlk',r,xyz2)
#res = np.tensordot(r,xyz,axes=([2],[1]))

  # print(res)
  # print(res.shape)

  res2 = np.exp(2*np.pi*1j*res)
  # print(res2)

  # print(f)
  # print(f.shape,res2.shape)

  #res3 = np.tensordot(f,res2,axes=([2],[2]))
  res3 = np.einsum('ijkz,ij',res2,f)

  # print(res3)
  # print(res3.sum)
  res3_abs = np.abs(res3)
  res3_sq = res3.real**2 + res3.imag**2

  # print(res3_abs)
  # print(res3_abs.shape)

  # res4_abs = np.sum(res3_abs,axis=(0,1))
  res4_abs = np.sum(res3_abs,axis=0)
  res4_sq = np.sum(res3_sq,axis=0)

  res4_abs = res4_abs / (phi_num * theta_num) / natom
  res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
  # print(res4_abs)
  # print(res4_abs.shape)

  return res4_abs,res4_sq

def structure_factor_cuda_better(rf,qs,phi_num,theta_num,fmean):

  r = rf[:,:3]
  f = rf[:,3] 

  # r = r[:, np.newaxis] - r
  # f = f[:, np.newaxis] * f 
  # f = f - fmean * fmean
  
  # f1 = np.triu(f, k=1)
 
  # f = f1
  
  natom = f.shape[0]
  # print(natom)

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  # phi = phi + 0.5 * phi_step
  # sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  # sintheta = sintheta + 0.5 *sintheta_step
  # theta = np.arcsin(sintheta)
  theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  r_my = qs

  xyz2 = np.einsum('i,jk',r_my,xyz)

  print(r.shape,xyz2.shape)

  res = np.einsum('ik,zlk',r,xyz2)

  res2 = np.exp(-2*np.pi*1j*res)

  res3 = np.einsum('ikz,i',res2,f)

  res3 = res3 * np.conjugate(res3)

  res3_abs = np.abs(res3)
  res3_sq = res3.real**2 + res3.imag**2

  res4_abs = np.sum(res3_abs,axis=0)
  res4_sq = np.sum(res3_sq,axis=0)

  res4_abs = res4_abs / (phi_num * theta_num) / natom
  res4_sq = res4_sq / (phi_num * theta_num) / natom / natom

  return res4_abs,res4_sq


def readfile(filename):

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
                        #print(heads)
                        iline += 1
                        atoms = []
                        for iatom in range(atomnum):
                            arr = lines[iline].lstrip().split()
                            
                            d = dict(zip(heads,arr))
                            for key in conv:
                                if key in d:
                                    d[key] = conv[key](d[key])
                            #print(d)
                            #atoms[int(d['id'])] = d
                            atoms.append(d)
                            iline += 1
                        atoms2 = sorted(atoms,key= lambda d: int(d['id']))
                        #print(atoms2)
            d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }
            #print(framenum)
            #if(framenum>=81500000):
            frames.append(d1)
            printCounter(framenum,'Read: timestep','')
    print()
    return frames

#def readxyz(filename):

#    f = open(filename, 'r')

#    lines = f.read().splitlines()

#    f.close()
    
    # print('alive')
    
#    frames = []


def readdata(filename):

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
    
    # print('alive')
    
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
 
    # return 1
    iline += 2
    atoms = []
    # print(atomnum)
    for iatom in range(atomnum):
        arr = lines[iline].lstrip().split()
        #print(len(arr))
        d = dict(zip(heads,arr))
        for key in conv:
            if key in d:
                d[key] = conv[key](d[key])
        # print(d)
        #atoms[int(d['id'])] = d
        atoms.append(d)
        iline += 1
    # print(atoms)
    # for atom in atoms:
    #     if not 'id' in atom:
    #         print(atom)
    atoms2 = sorted(atoms,key= lambda d: int(d['id']))
    #print(atoms2)        

    d1 = {'timestep':framenum, 'xlo':float(xlo), 'xhi':float(xhi), 'ylo':float(ylo), 'yhi':float(yhi), 'zlo':float(zlo), 'zhi':float(zhi), 'natoms': atomnum, 'atoms' : atoms2 }
    #print(framenum)
    #if(framenum>=81500000):
    frames.append(d1)
    return frames    

def get_xyz_arrays(frames):
    # frames = readdata(dire+file)
    frame = frames[0]
    atoms = frame['atoms']
    # print(atoms)
    x = []
    y = []
    z = []
    for atom in atoms:
        if atom['type']==4:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x,y,z

def get_xyz_from_frame(frame):
    # frames = readdata(dire+file)
    # frame = frames[0]
    atoms = frame['atoms']
    # print(atoms)
    # print(atoms)
    x = []
    y = []
    z = []
    for atom in atoms:
        if atom['type']==4:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x,y,z

def get_xyz_from_frame(frame):
    # frames = readdata(dire+file)
    # frame = frames[0]
    atoms = frame['atoms']
    # print(atoms)
    # print(atoms)
    x = []
    y = []
    z = []
    for atom in atoms:
        if atom['type']==4:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x,y,z

def get_xyzf_from_frame(frame):
    # frames = readdata(dire+file)
    # frame = frames[0]
    atoms = frame['atoms']
    # print(atoms)
    # print(atoms)
    x = []
    y = []
    z = []
    f = []
    for atom in atoms:
        if atom['type']==4:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'])
            f.append(1.0)
        else:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'])
            f.append(-1.0)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)
    return x,y,z,f

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
   

def get_xyzf_from_frame_data_or_trj(frame):
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
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    f = np.array(f)
    return x,y,z,f


def structure_factor_cuda_better_wrap(rf,qs,phi_num,theta_num,fmean,nslice=0):

  r = rf[:,:3]
  f = rf[:,3] 

  natom = f.shape[0]

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  if nslice==0:
     r_mys = [qs]
  else:
     r_mys = np.array_split(qs,qs.shape[0] // nslice + 1)

  print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  for r_my in r_mys:

    xyz2 = np.einsum('i,jk',r_my,xyz)

    # print(r.shape,xyz2.shape)

    res = np.einsum('ik,zlk',r,xyz2)

    res2 = np.exp(-2*np.pi*1j*res)

    res3 = np.einsum('ikz,i',res2,f)

    res3 = res3 * np.conjugate(res3)

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)

    res4_abs = res4_abs / (phi_num * theta_num) / natom
    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2(rf,qs,phi_num,theta_num,fmean):

  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  r_mys = np.array_split(qs,qs.shape[0])

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  for r_my in r_mys:

    xyz2 = np.einsum('i,jk',r_my,xyz)

    # print(r.shape,xyz2.shape)

    res = np.einsum('ik,zlk',r,xyz2)

    res2 = np.exp(-2*np.pi*1j*res)

    res3 = np.einsum('ikz,i',res2,f)

    res3 = res3 * np.conjugate(res3)

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)

    res4_abs = res4_abs / (phi_num * theta_num) / natom
    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2_progress(rf,qs,phi_num,theta_num,fmean):

  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  r_mys = np.array_split(qs,qs.shape[0])

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  l = len(r_mys)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
  for ir_my,r_my in enumerate(r_mys):

    xyz2 = np.einsum('i,jk',r_my,xyz)

    # print(r.shape,xyz2.shape)

    res = np.einsum('ik,zlk',r,xyz2)

    res2 = np.exp(-2*np.pi*1j*res)

    res3 = np.einsum('ikz,i',res2,f)

    res3 = res3 * np.conjugate(res3)

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)

    res4_abs = res4_abs / (phi_num * theta_num) / natom
    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)
    printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2_progress_time(rf,qs,phi_num,theta_num,fmean):
  
  s = time.time()
  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  r_mys = np.array_split(qs,qs.shape[0])

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  l = len(r_mys)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
  e = time.time()
  
  print('init: {}'.format(e-s))
  c0 = e-s
  c1 = c2 = c3 = c4 = c5 =c6 =c7 = c8 =  0
  
  for ir_my,r_my in enumerate(r_mys):
    s = time.time()
    xyz2 = np.einsum('i,jk',r_my,xyz)
    e = time.time()
    # print('xyz2: {}'.format(e-s))
    c1 += e-s
    # print(r.shape,xyz2.shape)

    res = np.einsum('ik,zlk',r,xyz2)
    e2 = time.time()
    # print('res: {}'.format(e2-e))
    c2 += e2-e
    res2 = np.exp(-2*np.pi*1j*res)
    
    e3 = time.time()
    # print('res2: {}'.format(e3-e2))
    c3 += e3-e2
    
    res3 = np.einsum('ikz,i',res2,f)
    
    e4 = time.time()
    # print('res3: {}'.format(e4-e3))
    c4 += e4-e3

    res3 = res3 * np.conjugate(res3)
    
    e5 = time.time()
    # print('res3 conj: {}'.format(e5-e4))
    c5 += e5-e4

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2
    
    e6 = time.time()
    # print('res3 abs: {}'.format(e6-e5))
    c6 += e6-e5

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)
    
    e7 = time.time()
    # print('res4: {}'.format(e7-e6))
    c7 += e7-e6
    
    res4_abs = res4_abs / (phi_num * theta_num) / natom
    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)
    
    e8 = time.time()
    # print('append: {}'.format(e8-e7))
    c8 += e8-e7
    
    printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)
  print('xyz2: {}'.format(c1))
  print('res: {}'.format(c2))
  print('res2: {}'.format(c3))
  print('res3: {}'.format(c4))
  print('res3 conj: {}'.format(c5))
  print('res3 abs: {}'.format(c6))
  print('res4: {}'.format(c7))
  print('append: {}'.format(c8))
  e9 = time.time()
  print('finish: {}'.format(e9-e8))
  print('total: {}'.format(c0+c1+c2+c3+c4+c5+c6+c7+c8+e9-e8))

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2_progress_cupy(rf,qs,phi_num,theta_num,fmean):
  
  s = time.time()
  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]

  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
  phi = phi + 0.5 * phi_step
  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
  sintheta = sintheta + 0.5 *sintheta_step
  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

  ptp, ptt = np.meshgrid(phi, theta)
  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

  r_q = np.ones((pt_pairs.shape[0],1))
  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

  phi_theta = pt_pairs

  xyz = Cartesian_np(phi_theta)

  r_mys = np.array_split(qs,qs.shape[0])

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  l = len(r_mys)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
  for ir_my,r_my in enumerate(r_mys):
    xyz2 = np.einsum('i,jk',r_my,xyz)
    
#    print(xyz2.shape)
    
    xyz2_gpu = cp.asarray(xyz2)
    r_gpu = cp.asarray(r)
#    print(xyz2_gpu.shape,r_gpu.shape)

    res_gpu = cp.einsum('ik,zlk',r_gpu,xyz2_gpu)
#    print(res_gpu.shape)
    res2_gpu = cp.exp(-2*np.pi*1j*res_gpu)
#    print(res2_gpu.shape)
    
    
    f_gpu = cp.asarray(f)
    res3_gpu = cp.einsum('ikz,i',res2_gpu,f_gpu)
#    print(res3_gpu.shape)
    
    res3_gpu = res3_gpu * np.conjugate(res3_gpu)
#    print(res3_gpu.shape)
    
    res3 = cp.asnumpy(res3_gpu)
#    print(res3.shape)
 #   print(res3.shape, end = '\r')

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2
#    print('res3_sq.shape',res3_sq.shape)
    

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)
 #   print(res4_sq.shape)
    
    res4_abs = res4_abs / (phi_num * theta_num) / natom
    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom

 #   print(res4_sq.shape)
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)
    
   
    printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)

 # print(res_abs_out.shape, res_sq_out.shape, res_qs_out.shape)

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2_progress_fibonacci_cupy(rf,qs,points_num,fmean):
  
  s = time.time()
  r = rf[:,:3]
  f = rf[:,3] 
  
  f = f - fmean

  natom = f.shape[0]
#  print('natom {}'.format(natom))

#  phi, phi_step = np.linspace(0, 2*np.pi, num = phi_num, endpoint=False, retstep=True)
#  phi = phi + 0.5 * phi_step
#  sintheta, sintheta_step = np.linspace(-1, 1, num = theta_num, endpoint=False, retstep=True)
#  sintheta = sintheta + 0.5 *sintheta_step
#  theta = np.arcsin(sintheta)
  # theta, theta_step = np.linspace(-np.pi/2.0, np.pi/2.0, num = theta_num, endpoint=False, retstep=True)

#  ptp, ptt = np.meshgrid(phi, theta)
#  pt_pairs = np.dstack([ptp, ptt]).reshape(-1, 2)

#  r_q = np.ones((pt_pairs.shape[0],1))
#  pt_pairs = np.concatenate((r_q,pt_pairs),axis=1)

#  phi_theta = pt_pairs

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

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  l = len(r_mys)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
  for ir_my,r_my in enumerate(r_mys):
    xyz2 = np.einsum('i,jk',r_my,xyz)
    
#    print(xyz2.shape)
    
    xyz2_gpu = cp.asarray(xyz2)
    r_gpu = cp.asarray(r)
#    print(xyz2_gpu.shape,r_gpu.shape)

    res_gpu = cp.einsum('ik,zlk',r_gpu,xyz2_gpu)
#    print(res_gpu.shape)
#    res2_gpu = cp.exp(-2*np.pi*1j*res_gpu)
    res2_gpu = cp.exp(1j*res_gpu)
#    print(res2_gpu.shape)
    
    
    f_gpu = cp.asarray(f)
    res3_gpu = cp.einsum('ikz,i',res2_gpu,f_gpu)
#    print(res3_gpu.shape)
    
    res3_gpu = res3_gpu * np.conjugate(res3_gpu)
#    print(res3_gpu.shape)
    
    res3 = cp.asnumpy(res3_gpu)
#    print(res3.shape)
 #   print(res3.shape, end = '\r')

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2
#    print('res3_sq.shape',res3_sq.shape)
    

    res4_abs = np.sum(res3_abs,axis=0)
    res4_sq = np.sum(res3_sq,axis=0)
 #   print(res4_sq.shape)
    
#    res4_abs = res4_abs / (phi_num * theta_num) / natom
#    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
#    print(res4_sq)
    res4_abs = res4_abs / (points_num) / natom
    res4_sq = res4_sq / (points_num) / natom / natom
#    print(res4_sq)

#    print(res4_sq.shape)
    res_abs_arr.append(res4_abs)
    res_sq_arr.append(res4_sq)
    res_qs_arr.append(r_my)
    
   
    printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

  res_abs_out = np.concatenate(res_abs_arr,axis=None)
  res_sq_out = np.concatenate(res_sq_arr,axis=None)
  res_qs_out = np.concatenate(res_qs_arr,axis=None)

 # print(res_abs_out.shape, res_sq_out.shape, res_qs_out.shape)

  return res_abs_out,res_sq_out,res_qs_out

def structure_factor_cuda_better_wrap2_progress_fibonacci_cupy_directions(rf,qs,points_num,fmean):
  
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

  # print(r.shape,r_mys[0].shape,xyz.shape) 
  
  #r_my = qs
  res_qs_arr = []
  res_abs_arr = []
  res_sq_arr = []
  res_xyz_arr = []
  l = len(r_mys)
  printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
  for ir_my,r_my in enumerate(r_mys):
    xyz2 = np.einsum('i,jk',r_my,xyz)
    
#    print(xyz2.shape)
    
    xyz2_gpu = cp.asarray(xyz2)
    r_gpu = cp.asarray(r)
#    print(xyz2_gpu.shape,r_gpu.shape)

    res_gpu = cp.einsum('ik,zlk',r_gpu,xyz2_gpu)
#    print(res_gpu.shape)
#    res2_gpu = cp.exp(-2*np.pi*1j*res_gpu)
    res2_gpu = cp.exp(1j*res_gpu)
#    print(res2_gpu.shape)
    
    
    f_gpu = cp.asarray(f)
    res3_gpu = cp.einsum('ikz,i',res2_gpu,f_gpu)
#    print(res3_gpu.shape)
    
    res3_gpu = res3_gpu * np.conjugate(res3_gpu)
#    print(res3_gpu.shape)
    
    res3 = cp.asnumpy(res3_gpu)
#    print(res3.shape)
 #   print(res3.shape, end = '\r')

    res3_abs = np.abs(res3)
    res3_sq = res3.real**2 + res3.imag**2
#    print('res3_sq.shape',res3_sq.shape)
    

#    res4_abs = np.sum(res3_abs,axis=0)
#    res4_sq = np.sum(res3_sq,axis=0)
 #   print(res4_sq.shape)
    
#    res4_abs = res4_abs / (phi_num * theta_num) / natom
#    res4_sq = res4_sq / (phi_num * theta_num) / natom / natom
#    print(res4_sq)
#    res4_abs = res4_abs / (points_num) / natom
#    res4_sq = res4_sq / (points_num) / natom / natom
    res3_abs = res3_abs / natom
    res3_sq = res3_sq / natom / natom
#    print(res4_sq)

#    print(res4_sq.shape)
#    res_abs_arr.append(res4_abs)
#    res_sq_arr.append(res4_sq)
#    res_qs_arr.append(r_my)
    res_abs_arr.append(res3_abs)
    res_sq_arr.append(res3_sq)
    res_qs_arr.append(r_my)
    res_xyz_arr.append(xyz2)
   
    printProgressBar(ir_my + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


  res_abs_out = np.squeeze(np.stack(res_abs_arr,axis=0))
  res_sq_out  = np.squeeze(np.stack(res_sq_arr,axis=0))
  res_qs_out  = np.squeeze(np.stack(res_qs_arr,axis=0))
  res_xyz_out = np.squeeze(np.stack(res_xyz_arr,axis=0))

 # print(res_abs_out.shape, res_sq_out.shape, res_qs_out.shape)

  return res_abs_out,res_sq_out,res_qs_out,res_xyz_out,xyz,phi_theta


def calculateStructureFactor(x, y, z, natom, box,  g, ikkmax):
    """ Calculate S values for a range of k^2 values
    Output: S(k^2), number_of_samples(k^2) """
    tbox = 2. * np.pi / float(box)
    sk = np.zeros(ikkmax + 1, dtype = np.longdouble)
    ns = np.zeros(ikkmax + 1, dtype = int)
    # npAtoms = createNumpyAtomsArrayFromConfig(config)
    el = np.cos(tbox * x, dtype = np.longdouble) + 1j * np.sin(tbox * x, dtype = np.longdouble)
    em = np.cos(tbox * y, dtype = np.longdouble) + 1j * np.sin(tbox * y, dtype = np.longdouble)
    en = np.cos(tbox * z, dtype = np.longdouble) + 1j * np.sin(tbox * z, dtype = np.longdouble)
    els, ems, ens, ems_neg, ens_neg = [], [], [], [], []
    eln, emn, enn = el[:], em[:], en[:]
    els.append(np.array([1.] * natom, dtype = np.clongdouble))
    ems.append(np.array([1.] * natom, dtype = np.clongdouble))
    ens.append(np.array([1.] * natom, dtype = np.clongdouble))
    for l in range(1, g + 1):
        els.append(eln)
        eln = eln * el
    for m in range(1, g + 1):
        ems.append(emn)
        emn = emn * em
    for n in range(1, g + 1):
        ens.append(enn)
        enn = enn * en
    for m in range(len(ems)):
        ems_neg.append(np.conjugate(ems[m]))
    for n in range(len(ens)):
        ens_neg.append(np.conjugate(ens[n]))
    for l in range(0, g + 1):
        for m in range(-g, g + 1):
            for n in range(-g, g + 1):
                ikk = l * l + m * m + n * n
                if ikk <= ikkmax:
                    if m < 0: em_value = ems_neg[-m]
                    else: em_value = ems[m]
                    if n < 0: en_value = ens_neg[-n]
                    else: en_value = ens[n]
                    expikr = np.sum(els[l] * em_value * en_value)
                    sk[ikk] += np.real(expikr * np.conjugate(expikr))
                    ns[ikk] += 1
    for ikk in range(1, ikkmax+1):
        if ns[ikk] > 0:
            sk[ikk] = sk[ikk] / natom
    return [1.] + list(sk[1:]), [1] + list(ns[1:])

def PGStructureFactor(x,y,z,box,g=50,ksqmax=2500):
    sk,ns = calculateStructureFactor(x, y, z, len(x), box,  g, ksqmax)
    k_out=[]
    sk_out=[]
    for i in range(len(sk)):
        if ns[i] > 0:
            k_out.append(np.sqrt(i) / box)
            sk_out.append(sk[i]/float(ns[i]))
                	
    return np.array(sk_out), np.array(k_out)

def cell_duplicate(rf,lx,ly,lz):
    dr = np.array([[lx, 0, 0,0],
                   [ 0,ly, 0,0],
                   [ 0, 0,lz,0],
                   [lx,ly, 0,0],
                   [lx, 0,lz,0],
                   [ 0,ly,lz,0],
                   [lx,ly,lz,0],
                   ])
    
    rf_new = rf
    for i in dr:
        # print(i)
        rf2 = np.add(rf,i)
        rf_new = np.concatenate((rf_new,rf2),axis=0)
    
    # print(rf_new)
    return rf_new

def cell_sphere(rf,xc,yc,zc,r):
    rf2 = rf[ (rf[:,0]-xc)*(rf[:,0]-xc) + (rf[:,1]-yc)*(rf[:,1]-yc) + (rf[:,2]-zc)*(rf[:,2]-zc) < (r*r) ]
    return rf2

# Print iterations progress
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


