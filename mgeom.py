#!filename: mgeom.py

import numpy as np
import numpy.linalg as la
import sys, os, copy
np.set_printoptions(linewidth=400)

at_list = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']
at_dict = {'X':0, 'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18, 'K':19, 'Ca':20}
# relative atomic masses of elements (in atomic mass units [amu]) from
# "CRC Handbook" 84th ed, ed Lide, pgs 1-12 - 1-14
at_masses = {    'H' : 1.00794, 'C' : 12.0107, 'O' : 15.9994, 'N' : 14.0067,
  'F' : 18.9984, 'P' : 30.9738, 'S' : 32.0650, 'Cl': 35.4530, 'Br': 79.9040,
  'I' : 126.904, 'He': 4.00260, 'Ne': 20.1797, 'Ar': 39.9480, 'Li': 6.94100,
  'Be': 9.01218, 'B' : 10.8110, 'Na': 22.9898, 'Mg': 24.3050, 'Al': 26.9815,
  'Si': 28.0855, 'K' : 39.0983, 'Ca': 40.0780, 'Sc': 44.9559, 'Ti': 47.8670,
  'V' : 50.9415, 'Cr': 51.9961, 'Mn': 54.9380, 'Fe': 55.8450, 'Co': 58.9332,
  'Ni': 58.6934, 'Cu': 63.5460, 'Zn': 65.4090, 'Ga': 69.7230, 'Ge': 72.6400,
  'As': 74.9216, 'Se': 78.9600, 'Kr': 83.7980, 'X' :  0.0000}

## CONSTANTS ##
bondcut = 2.0 # Anstrong
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0

# read file data into a 2-d array
def get_file_string_array(file):
    try:
        file = open(file, "r")
    except IOError:
        print('Error: file (%s) not found!\n' % (file))
        sys.exit()
    lines = file.readlines()
    file.close()
    array = []
    for line in lines:
        array.append(line.split())
    return array


## MATH FUNCTIONS ##
def vec3d_outer(V1, V2):
    """ numpy array
    V1 ~ (3) or (:,3)
    V2 ~ (3) or (:,3)
    """
    if(len(np.shape(V1))== 1):
        A = V1.reshape((len(V1)//3, 3))
        B = V2.reshape((len(V2)//3, 3))
    elif(len(np.shape(V1))== 2):
        A = V1; B = V2
    C = np.zeros(np.shape(A))
    C[:,0] = A[:,1]*B[:,2] - A[:,2]*B[:,1]
    C[:,1] = A[:,2]*B[:,0] - A[:,0]*B[:,2]
    C[:,2] = A[:,0]*B[:,1] - A[:,1]*B[:,0]
    return C.reshape(np.shape(V1))


def vec3d_inner(V1, V2):
    """ numpy array
    V1 ~ (3) or (:,3)
    V2 ~ (3) or (:,3)
    """
    if(len(np.shape(V1))== 1):
        A = V1.reshape((len(V1)//3, 3))
        B = V2.reshape((len(V2)//3, 3))
        C = np.zeros((len(V1)//3))
    elif(len(np.shape(V1))== 2):
        A = V1; B = V2
        C = np.zeros((len(V1[:,0])))
    C[:] = A[:,0]*B[:,0] + A[:,1]*B[:,1] + A[:,2]*B[:,2]
    if(len(np.shape(V1))== 1):
        return C[0]
    elif(len(np.shape(V1))== 2):
        return C

def vec3d_bond(V1, V2):
    return np.sqrt( vec3d_inner(V1-V2, V1-V2) )


def vec3d_unit(V):
    """ numpy array
    V ~ (3) or (:,3)
    """
    if(len(np.shape(V))== 1):
        return V / vec3d_bond(V, np.zeros(np.shape(V)))
    elif(len(np.shape(V))== 2):
        return V / np.outer( vec3d_bond(V, np.zeros(np.shape(V))), np.ones((3)) )

def vec3d_cosangle(V1, V2, V3):
    b1 = vec3d_bond(V1, V2)
    b2 = vec3d_bond(V3, V2)
    return vec3d_inner(V1-V2, V3-V2) / (b1*b2)

def vec3d_angle(V1, V2, V3):
    return np.arccos(vec3d_cosangle(V1, V2, V3))

def vec3d_cosdihedral(V1, V2, V3, V4):
    """ numpy array
    V1 ~ (3) or (:,3)
    V2 ~ (3) or (:,3)
    V3 ~ (3) or (:,3)
    V4 ~ (3) or (:,3)
    """
    X1 = V2 - V1; X2 = V3 - V2; X3 = V4 - V3
    Y1 = vec3d_outer(X1, X2)
    Y2 = vec3d_outer(X2, X3)
    Z = np.zeros(np.shape(Y1))
    return vec3d_cosangle(Y1+Z, Z, Y2+Z)

def vec3d_dihedral(V1, V2, V3, V4):
    """ numpy array
    V1 ~ (3) or (:,3)
    V2 ~ (3) or (:,3)
    V3 ~ (3) or (:,3)
    V4 ~ (3) or (:,3)
    """
    X1 = V2 - V1; X2 = V3 - V2; X3 = V4 - V3
    Y1 = vec3d_outer(X1, X2)
    Y2 = vec3d_outer(X2, X3)
    s = np.sign( vec3d_inner(vec3d_outer(Y1, Y2), X2) )
    Z = np.zeros(np.shape(Y1))
    return np.arccos( vec3d_cosangle(Y1+Z, Z, Y2+Z) ) * s

# get local axis system from 3 coordinates
def vec3d_axis(V2, V3, V4):
    """ numpy array
    V2 ~ (3) or (:,3), bond
    V3 ~ (3) or (:,3), angle
    V4 ~ (3) or (:,3), torsion
    """
    # left hand local axis
    Z = vec3d_unit( V3-V2 )                      # z axis
    Y = vec3d_unit( - vec3d_outer(V3-V2,V4-V3) ) # y axis
    X = vec3d_unit( vec3d_outer(Z, Y)  )         # x axis
    return [X, Y, Z]

# calculate vector of bond in local axes of internal coordinates
def vec3d_zmat(r, a, t):
    x = r * np.sin(a) * np.cos(t)
    y = r * np.sin(a) * np.sin(t)
    z = r * np.cos(a)
    return [x, y, z]


# molecule class for molecular data
class mol_class:
    _before = ''
    _after  = ''
    _suc_zmat = False
    _suc_cart = False

    # constructor
    def __init__(self, file, type='zmat', fmt='mndo'):
        self.file = file
        self.type = type
        self.fmt  = fmt
        self.name = self.file.split('/')[-1].split('.')[0]
        try:
            if self.type == 'zmat':
                self.get_zmat(self.file)
                self._suc_zmat = True
            elif self.type == 'cart':
                self.get_cart(self.file)
                self._suc_cart = True
            else:
                raise ValueError('type: zmat or cart')
        except ValueError as e:
            print(repr(e))

    def get_cart(self, file):
        if self.fmt == 'mndo':
            array = get_file_string_array(file)
            isbefore = True; isafter = False
            self.data = []
            for i in range(len(array)):
                if array[i] == []:
                    if isbefore:
                        self._before += '\n'
                    if isafter:
                        self._after += '\n'
                    continue
                if array[i][0] == 'OM2':
                    self._before += ' '.join(array[i])
                    isbefore = False
                    continue
                if(array[i][0] == '0'):
                    self._after += ' '.join(array[i]) + '\n'
                    isafter = True
                    continue
                if isbefore:
                    self._before += ' '.join(array[i]) + '\n'
                elif isafter:
                    self._after += ' '.join(array[i]) + '\n'
                else:
                    self.data.append(array[i])
            tmp = np.array(self.data)
            self.at_idx = tmp[:,0].astype(np.int)
            self.cart = tmp[:,1:6:2].astype(np.float)
            self.cart_fix = tmp[:,2::2].astype(np.int)
            self.natom = len(self.at_idx)
        elif self.fmt == 'gauss':
            array = get_file_string_array(file)
            isbefore = True; isafter = False
            self.data = []
            for i in range(len(array)):
                if array[i] == []:
                    if isbefore:
                        self._before += '\n'
                    elif isafter:
                        self._after += '\n'
                    else:
                        isafter = True
                        self._after += '\n'
                    continue
                if array[i][0] == '0' and array[i][1] == '1':
                    self._before += ' '.join(array[i])
                    isbefore = False
                    continue
                if isbefore:
                    self._before += ' '.join(array[i]) + '\n'
                elif isafter:
                    self._after += ' '.join(array[i]) + '\n'
                else:
                    self.data.append(array[i])
            tmp = np.array(self.data)
            self.at_flag = tmp[:,0].astype(np.str); self.at_idx = np.zeros(np.shape(self.at_flag), dtype=np.int)
            self.cart = tmp[:,1:].astype(np.float)
            self.cart_fix = np.ones(np.shape(self.cart))
            self.natom = len(self.at_flag)
            for i in range(self.natom):
                self.at_idx[i] = at_dict.get(self.at_flag[i], 0)

    # read in z-matrix from zmat file
    def get_zmat(self, file):
        if self.fmt == 'mndo':
            array = get_file_string_array(file)
            isbefore = True; isafter = False
            self.data = []
            for i in range(len(array)):
                if array[i] == []:
                    if isbefore:
                        self._before += '\n'
                    if isafter:
                        self._after += '\n'
                    continue
                if array[i][0] == 'OM2':
                    self._before += ' '.join(array[i])
                    isbefore = False
                    continue
                if(array[i][0] == '0'):
                    self._after += ' '.join(array[i]) + '\n'
                    isafter = True
                    continue
                if isbefore:
                    self._before += ' '.join(array[i]) + '\n'
                elif isafter:
                    self._after += ' '.join(array[i]) + '\n'
                else:
                    self.data.append(array[i])
            tmp = np.array(self.data)
            self.at_idx = tmp[:,0].astype(np.int)
            self.zmat = tmp[:,1:6:2].astype(np.float)
            self.zmat[:,1:] *= deg2rad # deg2rad
            self.zmat_fix = tmp[:,2:7:2].astype(np.int)
            self.zmat_ref = tmp[:,7:].astype(np.int) - 1
            self.natom = len(self.at_idx)
        elif self.fmt=='gauss':
            array = get_file_string_array(file)
            isbefore = True; isafter = False
            self.data = []
            for i in range(len(array)):
                if array[i] == []:
                    if isbefore:
                        self._before += '\n'
                    elif isafter:
                        self._after += '\n'
                    else:
                        isafter = True
                        self._after += '\n'
                    continue
                if array[i][0] == '0' and array[i][1] == '1':
                    self._before += ' '.join(array[i])
                    isbefore = False
                    continue
                if isbefore:
                    self._before += ' '.join(array[i]) + '\n'
                elif isafter:
                    self._after += ' '.join(array[i]) + '\n'
                else:
                    self.data.append(array[i])
            self.data[0] += ['0']*7
            self.data[1] += ['0']*5
            self.data[2] += ['0']*3
            tmp = np.array(self.data)
            self.at_flag = tmp[:,0].astype(np.str); self.at_idx = np.zeros(np.shape(self.at_flag), dtype=np.int)
            self.zmat = tmp[:,2::2].astype(np.float); self.zmat[:,1:] *= deg2rad # deg2rad
            self.zmat_ref = tmp[:,1:-1:2].astype(np.int) - 1
            self.zmat_fix = np.zeros(np.shape(self.zmat))
            self.natom = len(self.at_flag)
            for i in range(self.natom):
                self.at_idx[i] = int( at_dict.get(self.at_flag[i], 0) )

    def bondlist(self):
        self.dist = np.zeros((self.natom, self.natom))
        for i in range(self.natom):
            for j in range(i):
                self.dist[i,j] = vec3d_bond(self.cart[i,:], self.cart[j,:])
                self.dist[j,i] = self.dist[i,j]
        self.blist = []
        for i in range(self.natom):
            self.blist.append( np.where( (self.dist[i,:]-0.01)*(self.dist[i,:]-bondcut) < 0 )[0] )
        for i in range(self.natom):
            print(self.blist[i], self.dist[i,self.blist[i]])

    # obtain cartesian xyz-cartinates from z-matrix values
    def zmat2cart(self):
        self.cart = np.zeros((self.natom, 3))
        if self.natom >= 1:
            self.cart[0,:] = 0
        if self.natom >= 2:
            self.cart[1,0] = self.zmat[1,0]
        if self.natom >= 3:
            if self.zmat_ref[2,0] == 1:
                self.cart[2,0] = self.zmat[1,0] - self.zmat[2,0] * np.cos(self.zmat[2,1])
                self.cart[2,1] = self.zmat[2,0] * np.sin(self.zmat[2,1])
            elif self.zmat_ref[2,0] == 0:
                self.cart[2,0] = self.zmat[2,0] * np.cos(self.zmat[2,1])
                self.cart[2,1] = self.zmat[2,0] * np.sin(self.zmat[2,1])
        for i in range(3, self.natom):
            carts2 = self.cart[self.zmat_ref[i,0],:]
            carts3 = self.cart[self.zmat_ref[i,1],:]
            carts4 = self.cart[self.zmat_ref[i,2],:]
            [X, Y, Z] = vec3d_axis(carts2, carts3, carts4)
            disp = ( X * self.zmat[i,0] * np.sin(self.zmat[i,1]) * np.cos(self.zmat[i,2])
                   + Y * self.zmat[i,0] * np.sin(self.zmat[i,1]) * np.sin(self.zmat[i,2])
                   + Z * self.zmat[i,0] * np.cos(self.zmat[i,1]) )
            carts1 = carts2 + disp
            self.cart[i,:] = carts1
        self.cart_fix = np.ones(np.shape(self.cart)).astype(np.int)
        self._suc_cart = True

    # obtain cartesian xyz-cartinates from z-matrix values
    def cart2zmat(self):
        self.bondlist()
        self.zmat = np.zeros((self.natom, 3))
        if self.natom >= 1:
            self.zmat[0,:] = 0
        if self.natom >= 2:
            self.zmat_ref[1,0] = 0
            self.zmat[1,0] = self.dist[1,0]
        if self.natom >= 3:
            if self.dist[2,0] < self.dist[2,1]:
                self.zmat_ref[2,0] = 0; self.zmat_ref[2,1] = 1;
                self.zmat[2,0] = self.dist[2,0]
                self.zmat[2,1] = vec3d_angle(self.cart[2,:], self.cart[0,:], self.cart[1,:])
            else:
                self.zmat_ref[2,0] = 1; self.zmat_ref[2,1] = 0;
                self.zmat[2,0] = self.dist[2,1]
                self.zmat[2,1] = vec3d_angle(self.cart[2,:], self.cart[1,:], self.cart[0,:])
        for i in range(3, self.natom):
            bone = [0,0,0,0]; rcdtmp = 999
            for s2 in self.blist[i]:
                if s2>=i:
                    continue
                for s3 in self.blist[s2]:
                    if s3>=i or s3==s2:
                        continue
                    for s4 in self.blist[s3]:
                        if s4>=i or s4==s2 or s4==s3:
                            continue
                        sumbond = self.dist[i,s2]+self.dist[s2,s3]+self.dist[s3,s4]
                        if(sumbond < rcdtmp):
                            bone = [i,s2,s3,s4]
                            rcdtmp = sumbond
            if bone[0] == i:
                self.zmat_ref[i,:] = bone[1:]
                carts1 = self.cart[i,:]
                carts2 = self.cart[self.zmat_ref[i,0],:]
                carts3 = self.cart[self.zmat_ref[i,1],:]
                carts4 = self.cart[self.zmat_ref[i,2],:]
                self.zmat[i,0] = vec3d_bond(carts1, carts2)
                self.zmat[i,1] = vec3d_angle(carts1, carts2, carts3)
                self.zmat[i,2] = vec3d_dihedral(carts1, carts2, carts3, carts4)

        self.zmat[:,1:] = self.zmat[:,1:]%(2*np.pi)
        self.zmat_fix = np.ones(np.shape(self.zmat)).astype(np.int)
        self.zmat_fix[0,:] = 0; self.zmat_fix[1,1:] = 0; self.zmat_fix[2,2:] = 0
        self._suc_cart = True

    def print(self, fmt='xyz', type='cart'):
        if type=='zmat':
            if fmt=='mndo':
                if(self.fmt=='mndo'):
                    print(self._before)
                else:
                    print('IOP=-6 JOP=0\n\nOM2')
                for i in range(self.natom):
                    print(' %d\t%12.8f\t%d\t%12.8f\t%d\t%12.8f\t%d\t%d\t%d\t%d'%
                        (self.at_idx[i], self.zmat[i,0], self.zmat_fix[i,0],
                        self.zmat[i,1]*rad2deg, self.zmat_fix[i,1],
                        self.zmat[i,2]*rad2deg, self.zmat_fix[i,2],
                        self.zmat_ref[i,0]+1, self.zmat_ref[i,1]+1, self.zmat_ref[i,2]+1) )
                if(self.fmt=='mndo'):
                    print(self._after)
                else:
                    print(' %d\t%12.8f\t%d\t%12.8f\t%d\t%12.8f\t%d\t%d\t%d\t%d'%(0,0,0,0,0,0,0,0,0,0))
            if fmt=='gauss':
                if(self.fmt=='gauss'):
                    print(self._before)
                else:
                    print('# sp hf/STO-3G\n\nTitle Card Required\n\n0 1')
                if self.natom >= 1:
                    print(' %s'%at_list[self.at_idx[0]])
                if self.natom >= 2:
                    print(' %s\t%d\t%12.8f'%
                        (at_list[self.at_idx[1]], self.zmat_ref[1,0]+1, self.zmat[1,0]) )
                if self.natom >= 3:
                    print(' %s\t%d\t%12.8f\t%d\t%12.8f'%
                        (at_list[self.at_idx[2]], self.zmat_ref[2,0]+1, self.zmat[2,0],
                        self.zmat_ref[2,1]+1, self.zmat[2,1]*rad2deg ) )
                for i in range(3,self.natom):
                    print(' %s\t%d\t%12.8f\t%d\t%12.8f\t%d\t%12.8f\t%d'%
                        (at_list[self.at_idx[0]], self.zmat_ref[i,0]+1, self.zmat[i,0],
                        self.zmat_ref[i,1]+1, self.zmat[i,1]*rad2deg,
                        self.zmat_ref[i,2]+1, self.zmat[i,2]*rad2deg, 0) )
                if(self.fmt=='gauss'):
                    print(self._after)
                else:
                    print('\n\n\n')
        elif type=='cart':
            if fmt == 'mndo':
                if(self.fmt=='mndo'):
                    print(self._before)
                else:
                    print('IOP=-6 JOP=0\n\nOM2')
                for i in range(self.natom):
                    print(' %d\t%12.8f\t%d\t%12.8f\t%d\t%12.8f\t%d'%
                        (self.at_idx[i], self.cart[i,0], self.cart_fix[i,0],
                        self.cart[i,1], self.cart_fix[i,1],
                        self.cart[i,2], self.cart_fix[i,2] ) )
                if(self.fmt=='mndo'):
                    print(self._after)
                else:
                    print(' %d\t%12.8f\t%d\t%12.8f\t%d\t%12.8f\t%d'%(0,0,0,0,0,0,0))
            elif fmt == 'gauss':
                if(self.fmt=='gauss'):
                    print(self._before)
                else:
                    print('# sp hf/STO-3G\n\nTitle Card Required\n\n0 1')
                for i in range(self.natom):
                    print(' %s\t%12.8f\t%12.8f\t%12.8f'%
                        (at_list[self.at_idx[i]], self.cart[i,0], self.cart[i,1],
                        self.cart[i,2] ) )
                if(self.fmt=='gauss'):
                    print(self._after)
                else:
                    print('\n\n\n')
            elif fmt == 'xyz':
                print(self.natom)
                print(self.name)
                for i in range(self.natom):
                    print(' %s\t%12.8f\t%12.8f\t%12.8f'%
                        (at_list[self.at_idx[i]], self.cart[i,0], self.cart[i,1],
                        self.cart[i,2]) )

    def mix(self, other, per):
        obj = copy.copy(other)
        obj.zmat = other.zmat * per + (1.0-per)*self.zmat
        #print((np.abs(other.zmat[:,2] - self.zmat[:,2]) > np.pi).astype(np.int))

        #for i in
        shift = (np.abs(other.zmat[:,2] - self.zmat[:,2]) > np.pi).astype(np.int)
        shift *= (2*(other.zmat[:,2] < 0).astype(np.int)-1)
        obj.zmat[:,2] += per*shift * 2*np.pi
        obj.zmat2cart()
        #print(obj.zmat[:,1:]*rad2deg)
        return obj


if __name__ == '__main__':
    n = len(sys.argv)
    v = sys.argv
    dd = {}
    if n%2==0:
        flag_value = False
        dd['target'] = []
        for i in range(1,n):
            if v[i][0] == '-':
                flag_value = True
                continue
            if flag_value:
                dd[v[i-1]] = v[i]
                flag_value = False
            else:
                dd['target'].append(v[i])

        fmti = dd.get('-i', 'mndo-zmat')
        fmto = dd.get('-o', 'xyz-cart')
        task = dd.get('-t', '')
        mixf = dd.get('-m', '')

        for tar in dd['target']:
            if task =='':
                ssi = fmti.split('-')
                sso = fmto.split('-')
                mol = mol_class(file=tar, fmt=ssi[0], type=ssi[1])
                if(ssi[1]=='zmat' and sso[1]=='cart'):
                    mol.zmat2cart()
                if(ssi[1]=='cart' and sso[1]=='zmat'):
                    mol.cart2zmat()
                mol.print(fmt=sso[0],type=sso[1])
            elif task =='mix' and fmti=='mndo-zmat' and mixf!='':
                ssi = fmti.split('-')
                sso = fmto.split('-')
                mol = mol_class(file=tar, fmt=ssi[0], type=ssi[1])
                mol2 = mol_class(file=mixf, fmt=ssi[0], type=ssi[1])
                for i in np.linspace(0,1,50):
                    molx = mol.mix(mol2,per=i)
                    molx.print(fmt=sso[0], type=sso[1])


