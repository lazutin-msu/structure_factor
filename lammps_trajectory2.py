import numpy as np
# import cupy as cp
import re
# import time

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



def write_trj(filename,frames):
    f = open(filename, 'w')
    lines = []
    for frame in frames:
        lines.append('ITEM: TIMESTEP\n{}\n'.format(frame['timestep']))
        lines.append('ITEM: NUMBER OF ATOMS\n{}\n'.format(frame['natoms']))
        lines.append('ITEM: BOX BOUNDS pp pp pp\n')
        lines.append('{} {}\n'.format(frame['xlo'],frame['xhi']))
        lines.append('{} {}\n'.format(frame['ylo'],frame['yhi']))
        lines.append('{} {}\n'.format(frame['zlo'],frame['zhi']))
        lines.append('ITEM: ATOMS id type xs ys zs\n')
        for atom in frame['atoms']:
            lines.append('{} {} {} {} {}\n'.format(atom['id'],atom['type'],atom['xs'],atom['ys'],atom['zs']))
    f.writelines(lines)
    f.close()

def write_trj_opt(filename,frames):
    f = open(filename, 'w')
    # lines = []
    for frame in frames:
        f.write('ITEM: TIMESTEP\n{}\n'.format(frame['timestep']))
        f.write('ITEM: NUMBER OF ATOMS\n{}\n'.format(frame['natoms']))
        f.write('ITEM: BOX BOUNDS pp pp pp\n')
        f.write('{} {}\n'.format(frame['xlo'],frame['xhi']))
        f.write('{} {}\n'.format(frame['ylo'],frame['yhi']))
        f.write('{} {}\n'.format(frame['zlo'],frame['zhi']))
        f.write('ITEM: ATOMS id type xs ys zs\n')
        for atom in frame['atoms']:
            f.write('{} {} {} {} {}\n'.format(atom['id'],atom['type'],atom['xs'],atom['ys'],atom['zs']))
        printCounter(frame['timestep'],'Write: timestep','')
    print()
    f.close()

def readtrj(filename,every=0):

    f = open(filename, 'r')

    lines = f.read().splitlines()

    f.close()
    
    
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x),'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x)}
    
    iline = 0
   # with open(filename, 'r') as f: 
   #  for line
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
            if every > 0:
                if(framenum % every == 0):
                    frames.append(d1)
            else:
                frames.append(d1)
            printCounter(framenum,'Read: timestep','')
    print()
    return frames

def readtrj_opt(filename,every=0):

    f = open(filename, 'r')

    # lines = f.read().splitlines()

    # f.close()
    
    
    frames = []
    conv = {'id': lambda x: int(x), 'type': lambda x: int(x), 'mol': lambda x: int(x), 'xu': lambda x: float(x), 'yu': lambda x: float(x), 'zu': lambda x: float(x), 'xs': lambda x: float(x), 'ys': lambda x: float(x), 'zs': lambda x: float(x),'x': lambda x: float(x), 'y': lambda x: float(x), 'z': lambda x: float(x), 'ix': lambda x: int(x),'iy': lambda x: int(x),'iz': lambda x: int(x),'c_poten': lambda x: float(x),'c_bonen': lambda x: float(x)}
    
    iline = 0
    lines = iter(f.readlines())
   # with open(filename, 'r') as f: 
   #  for line
    # while iline<len(lines):
    eof = False
    while ~eof:
        # line = lines[iline]
        try:
            line = next(lines)
        except StopIteration:
            eof = True
            break
        line = line.lstrip() 
        if not line.startswith('ITEM: TIMESTEP'):
            print("should be TIMESTEP")
            return 0
        else:
            try:
                line = next(lines)
            except StopIteration:
                eof = True
                break
            line = line.lstrip() 
            framenum = int(line.lstrip())
            # framenum = int(lines[iline+1].lstrip())
            iline += 2
            try:
                line = next(lines)
            except StopIteration:
                eof = True
                break
            # line = lines[iline]
            line = line.lstrip() 
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                print("should be NUMBER OF ATOMS")
                quit
            else:
                try:
                    line = next(lines)
                except StopIteration:
                    eof = True
                    break
                # atomnum = int(lines[iline+1].lstrip())
                atomnum = int(line.lstrip())
                iline += 2
                try:
                    line = next(lines)
                except StopIteration:
                    eof = True
                    break                
                # line = lines[iline]
                line = line.lstrip() 
                if not line.startswith('ITEM: BOX BOUNDS pp pp pp'):
                    print("should be BOX BOUNDS pp pp pp")
                    quit
                else:
                    try:
                        line = next(lines)
                    except StopIteration:
                        eof = True
                        break
                    # (xlo,xhi) = (lines[iline+1].lstrip()).split()
                    (xlo,xhi) = (line.lstrip()).split()
                    try:
                        line = next(lines)
                    except StopIteration:
                        eof = True
                        break
                    # (ylo,yhi) = (lines[iline+2].lstrip()).split()
                    (ylo,yhi) = (line.lstrip()).split()
                    try:
                        line = next(lines)
                    except StopIteration:
                        eof = True
                        break
                    # (zlo,zhi) = (lines[iline+3].lstrip()).split()
                    (zlo,zhi) = (line.lstrip()).split()
                    iline += 4
                    try:
                        line = next(lines)
                    except StopIteration:
                        eof = True
                        break
                    # line = lines[iline]
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
                            try:
                                line = next(lines)
                            except StopIteration:
                                eof = True
                                break
                            # arr = lines[iline].lstrip().split()
                            arr = line.lstrip().split()
                            
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
            if every > 0:
                if(framenum % every == 0):
                    frames.append(d1)
            else:
                frames.append(d1)
            printCounter(framenum,'Read: timestep','')
    print()
    f.close()
    return frames

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


