#!/usr/env/python
''' py-iCSD toolbox!
    Translation of the core functionality of the CSDplotter MATLAB package
    to python.
    
    Most of the comments got lost in the process. Sorry!
    
    The method themselves are implemented as callable subclasses of the base
    Icsd class object, which incorporate the initialization of some variables,
    and a basic function for calculating the iCSD, and a general filter
    implementation.
    
    The raw- and filtered CSD estimates are stored as arrays, after calling the
    classes;
    subclass.csd
    subclass.csd_filtered
    
    The CSD estimations are purely spatial processes, and doesn't care about
    the temporal resolution of the input data.
    
    Requires pylab environment to work, i.e numpy+scipy+matplotlib
    
    Adapted from CSDplotter-0.1.1, copyrighted under General Public License,
    Klas. H. Pettersen 2005,
    by Espen.Hagen@umb.no, Nov. 2010.
    
    Basic usage script:
    ############################################################################
    #!/usr/env/python
    
    import pylab as pl
    import icsd
    from scipy import io
    
    #loading test data
    test_data = io.loadmat('test_data.mat')
    
    #using one of the datasets, corresponding electrode coordinates
    lfp_data = test_data['pot1']                #[mV] -> [V]
    z_data = np.linspace(100E-6, 2300E-6, 23)   #[m]
    
    # Input dictionaries for each method
    delta_input = {
        'lfp' : lfp_data,
        'coord_electrode' : z_data,
        'diam' : 500E-6,        # source diameter
        'cond' : 0.3,           # extracellular conductivity
        'cond_top' : 0.3,       # conductivity on top of cortex
        'f_type' : 'gaussian',  # gaussian filter
        'f_order' : (3, 1),     # 3-point filter, sigma = 1.
    }
    step_input = {
        'lfp' : lfp_data,
        'coord_electrode' : z_data,
        'diam' : 500E-6,
        'cond' : 0.3,
        'cond_top' : 0.3,
        'tol' : 1E-12,          # Tolerance in numerical integration
        'f_type' : 'gaussian',
        'f_order' : (3, 1),
    }
    spline_input = {
        'lfp' : lfp_data,
        'coord_electrode' : z_data,
        'diam' : 500E-6,
        'cond' : 0.3,
        'cond_top' : 0.3,
        'num_steps' : 200,      # Spatial CSD upsampling to N steps
        'tol' : 1E-12,
        'f_type' : 'gaussian',
        'f_order' : (20, 5),
    }
    std_input = {
        'lfp' : lfp_data,
        'coord_electrode' : z_data,
        'f_type' : 'gaussian',
        'f_order' : (3, 1),
    }
    
    #Calling the different subclasses, with respective inputs.
    delta_icsd = icsd.DeltaiCSD(**delta_input)
    step_icsd = icsd.StepiCSD(**step_input)
    spline_icsd = icsd.SplineiCSD(**spline_input)
    std_csd = icsd.StandardCSD(**std_input)
    ############################################################################
    '''

import numpy as np
import scipy.integrate as si
import scipy.signal as ss
import quantities as pq
import neo

class Icsd(object):
    '''Base iCSD class'''
    def __init__(self):
        '''Initialize class iCSD'''
        self.name = 'iCSD Toolbox'
        self.lfp = None
        self.csd = None
        self.csd_filtered = None
        self.f_matrix = None
        self.f_type = None
        self.f_order = None
    
    def calc_csd(self, ):
        '''Perform the iCSD calculation, i.e: iCSD=F**-1*LFP'''
        #self.csd = np.array(np.matrix(self.f_matrix)**-1 * np.matrix(self.lfp))
        self.csd = np.linalg.solve(self.f_matrix, self.lfp)
        self.csd = self.csd * (self.f_matrix.units**-1*self.lfp.units).simplified
    
    def filter_csd(self):
        '''Spatial filtering of the CSD estimate, using an N-point filter'''
        if not self.f_order > 0 and type(self.f_order) == type(3):
            raise Exception, 'Filter order must be int > 0!'
        
        if self.f_type == 'boxcar':
            num = ss.boxcar(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == 'hamming':
            num = ss.hamming(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == 'triangular':
            num = ss.triang(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == 'gaussian':
            num = ss.gaussian(self.f_order[0], self.f_order[1])
            denom = np.array([num.sum()])
        elif self.f_type == 'identity':
            num = np.array([1.])
            denom = np.array([1.])
        else:
            raise Exception, '%s Wrong filter type!' % self.f_type
        
        num_string = '[ '
        for i in num:
            num_string = num_string + '%.3f ' % i
        num_string = num_string + ']'
        denom_string = '[ '
        for i in denom:
            denom_string = denom_string + '%.3f ' % i
        denom_string = denom_string + ']'
        
        print 'discrete filter coefficients: \nb = %s, \na = %s' % \
                                                     (num_string, denom_string) 
        self.csd_filtered = np.empty(self.csd.shape)
        for i in xrange(self.csd.shape[1]):
            self.csd_filtered[:, i] = ss.filtfilt(num, denom, self.csd[:, i])
        
class StandardCSD(Icsd):
    '''Standard CSD method with Vaknin electrodes'''
    def __init__(self, lfp, coord_electrode=np.linspace(-700E-6, 700E-6, 15),
                 cond=0.3, vaknin_el=True, f_type='gaussian', f_order=(3, 1)):
        Icsd.__init__(self)

        self.lfp = lfp
        self.coord_electrode = coord_electrode
        self.cond = cond
        self.f_type = f_type
        self.f_order = f_order
        
        if vaknin_el:
            self.lfp = np.empty((lfp.shape[0]+2, lfp.shape[1]))
            self.lfp[0, ] = lfp[0, ]
            self.lfp[1:-1, ] = lfp
            self.lfp[-1, ] = lfp[-1, ]
            self.f_inv_matrix = np.zeros((lfp.shape[0]+2, lfp.shape[0]+2)) * pq.A / pq.m
        else:
            self.lfp = lfp
            self.f_inv_matrix = np.zeros((lfp.shape[0], lfp.shape[0])) * pq.A / pq.m
        
        self.calc_f_inv_matrix()
        self.calc_csd()
        self.filter_csd()
    
    def calc_f_inv_matrix(self):
        '''Calculate the inverse F-matrix for the standard CSD method'''
        h_val = abs(np.diff(self.coord_electrode)[0])
        
        #Inner matrix elements  is just the discrete laplacian coefficients
        self.f_inv_matrix[0, 0] = -1 * pq.A / pq.m
        for j in xrange(1, self.f_inv_matrix.shape[0]-1):
            self.f_inv_matrix[j, j-1:j+2] = np.array([1., -2., 1.]) * pq.A / pq.m
        self.f_inv_matrix[-1, -1] = -1 * pq.A / pq.m
        
        self.f_inv_matrix = self.f_inv_matrix * -self.cond / h_val**2
    
    def calc_csd(self):
        '''Perform the iCSD calculation, i.e: iCSD=F_inv*LFP'''
        #self.csd = np.array(np.matrix(self.f_inv_matrix) * \
        #                    np.matrix(self.lfp))[1:-1, 1:-1]
        self.csd = np.dot(self.f_inv_matrix, self.lfp)[1:-1, ]
        self.lfp = self.lfp[1:-1, ]
     
class DeltaiCSD(Icsd):
    '''delta-iCSD method'''
    def __init__(self, lfp,
                 coord_electrode=np.linspace(-700E-6, 700E-6, 15)*pq.m,
                 diam=500E-6*pq.m,
                 cond=0.3*pq.S/pq.m,
                 cond_top=0.3*pq.S/pq.m,
                 f_type='gaussian', f_order=(3, 1)):
        '''Initialize delta-iCSD method'''
        Icsd.__init__(self)
        
        self.lfp = lfp
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.cond = cond
        self.cond_top = cond_top
        self.f_type = f_type
        self.f_order = f_order
        
        #initialize F- and iCSD-matrices
        self.f_matrix = np.empty((self.coord_electrode.size, \
                                 self.coord_electrode.size))
        self.csd = np.empty(lfp.shape)
        
        self.calc_f_matrix()
        self.calc_csd()
        self.filter_csd()
    
    def calc_f_matrix(self):
        '''Calculate the F-matrix'''
        h_val = abs(np.diff(self.coord_electrode)[0])
        
        for j in xrange(self.coord_electrode.size):
            for i in xrange(self.coord_electrode.size):
                self.f_matrix[j, i] = h_val / (2 * self.cond) * \
                    ((np.sqrt((self.coord_electrode[j] - \
                    self.coord_electrode[i])**2 + (self.diam / 2)**2) - \
                    abs(self.coord_electrode[j] - self.coord_electrode[i])) +\
                    (self.cond - self.cond_top) / (self.cond + self.cond_top) *\
                    (np.sqrt((self.coord_electrode[j] + \
                    self.coord_electrode[i])**2 + (self.diam / 2)**2) - \
                    abs(self.coord_electrode[j] + self.coord_electrode[i])))

        self.f_matrix = self.f_matrix / self.cond.units * h_val.units**2


class StepiCSD(Icsd):
    '''Step-iCSD method'''
    def __init__(self, lfp, coord_electrode=np.linspace(-700E-6, 700E-6, 15),
                 diam=500E-6, cond=0.3, cond_top=0.3, tol=1E-6,
                 f_type = 'gaussian', f_order = (3, 1)):
        '''Initialize Step-iCSD method'''
        Icsd.__init__(self)
        
        self.lfp = lfp
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.cond = cond
        self.cond_top = cond_top
        self.tol = tol
        self.f_type = f_type
        self.f_order = f_order
        
        # compute stuff
        self.calc_f_matrix()
        self.calc_csd()
        self.filter_csd()
        
    def calc_f_matrix(self):
        '''Calculate F-matrix for step iCSD method'''
        el_len = self.coord_electrode.size
        h_val = abs(np.diff(self.coord_electrode)[0])
        self.f_matrix = np.zeros((el_len, el_len))
        for j in xrange(el_len):
            for i in xrange(el_len):
                if i != 0:
                    lower_int = self.coord_electrode[i] - \
                        (self.coord_electrode[i] - \
                         self.coord_electrode[i - 1]) / 2
                else:
                    if self.coord_electrode[i] - h_val/2 > 0:
                        lower_int = self.coord_electrode[i] - h_val/2
                    else:
                        lower_int = h_val.unit
                    #lower_int = np.max([0*h_val, self.coord_electrode[i] - h_val/2])
                if i != el_len-1:
                    upper_int = self.coord_electrode[i] + \
                        (self.coord_electrode[i + 1] - \
                         self.coord_electrode[i]) / 2
                else:
                    upper_int = self.coord_electrode[i] + h_val / 2
                
                self.f_matrix[j, i] = si.quad(self.f_cylinder, a=lower_int, b=upper_int,
                                              args=(self.coord_electrode[j]),
                                              epsabs=self.tol)[0] + (self.cond - self.cond_top) / (self.cond + self.cond_top) * \
                    si.quad(self.f_cylinder, a=lower_int, b=upper_int, 
                            args=(-self.coord_electrode[j]), 
                            epsabs=self.tol)[0]
        #assume si.quad trash the units
        self.f_matrix = self.f_matrix * h_val.units**2 / self.cond.units
        
    
    def f_cylinder(self, zeta, z_val):
        '''function used by class method'''
        return 1. / (2. * self.cond) * (np.sqrt((self.diam / 2)**2 + \
            ((z_val - zeta*z_val.units))**2) - abs(z_val - zeta*z_val.units))

class SplineiCSD(Icsd):
    '''Spline iCSD method'''
    def __init__(self, lfp, coord_electrode=np.linspace(-700E-6, 700E-6, 15),
                 diam=500E-6, cond=0.3, cond_top=0.3, tol=1E-6,
                 f_type='gaussian', f_order=(3, 1), num_steps=200):
        '''Initialize Spline iCSD method'''
        Icsd.__init__(self)
        
        self.lfp = lfp
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.cond = cond
        self.cond_top = cond_top
        self.tol = tol
        self.f_type = f_type
        self.f_order = f_order
        self.num_steps = num_steps
        
        # compute stuff
        self.calc_f_matrix()
        self.calc_csd()
        self.filter_csd()
        
        
    def calc_f_matrix(self):
        '''Calculate the F-matrix for cubic spline iCSD method'''
        el_len = self.coord_electrode.size
        z_js = np.zeros(el_len+2) * self.coord_electrode.units
        z_js[1:-1] = self.coord_electrode
        z_js[-1] = z_js[-2] + np.diff(self.coord_electrode).mean()
        
        # Define integration matrixes
        f_mat0 = np.zeros((el_len, el_len+1))
        f_mat1 = np.zeros((el_len, el_len+1))
        f_mat2 = np.zeros((el_len, el_len+1))
        f_mat3 = np.zeros((el_len, el_len+1))
        
        # Calc. elements
        for j in xrange(el_len):
            for i in xrange(el_len):
                f_mat0[j, i] = si.quad(self.f_mat0, a=z_js[i], b=z_js[i+1], \
                    args=(z_js[j+1]), epsabs=self.tol)[0]
                f_mat1[j, i] = si.quad(self.f_mat1, a=z_js[i], b=z_js[i+1], \
                                   args=(z_js[j+1], z_js[i]), \
                                        epsabs=self.tol)[0]
                f_mat2[j, i] = si.quad(self.f_mat2, a=z_js[i], b=z_js[i+1], \
                                   args=(z_js[j+1], z_js[i]), \
                                        epsabs=self.tol)[0]
                f_mat3[j, i] = si.quad(self.f_mat3, a=z_js[i], b=z_js[i+1], \
                                   args=(z_js[j+1], z_js[i]), \
                                        epsabs=self.tol)[0]
                # image technique if conductivity not constant:
                if self.cond != self.cond_top:    
                    f_mat0[j, i] = f_mat0[j, i] + (self.cond-self.cond_top) / \
                        (self.cond + self.cond_top) * \
                            si.quad(self.f_mat0, a=z_js[i], b=z_js[i+1], \
                                args=(-z_js[j+1]), \
                                    epsabs=self.tol)[0]
                    f_mat1[j, i] = f_mat1[j, i] + (self.cond-self.cond_top) / \
                        (self.cond + self.cond_top) * \
                            si.quad(self.f_mat1, a=z_js[i], b=z_js[i+1], \
                                args=(-z_js[j+1], z_js[i]), epsabs=self.tol)[0]
                    f_mat2[j, i] = f_mat2[j, i] + (self.cond-self.cond_top) / \
                        (self.cond + self.cond_top) * \
                            si.quad(self.f_mat2, a=z_js[i], b=z_js[i+1], \
                                args=(-z_js[j+1], z_js[i]), epsabs=self.tol)[0]
                    f_mat3[j, i] = f_mat3[j, i] + (self.cond-self.cond_top) / \
                        (self.cond + self.cond_top) * \
                            si.quad(self.f_mat3, a=z_js[i], b=z_js[i+1], \
                                args=(-z_js[j+1], z_js[i]), epsabs=self.tol)[0]
        
        e_mat0, e_mat1, e_mat2, e_mat3 = self.calc_e_matrices()
        
        # Calculate the F-matrix
        self.f_matrix = np.zeros((el_len+2, el_len+2))
        self.f_matrix[1:-1, :] = np.dot(f_mat0, e_mat0) + np.dot(f_mat1, e_mat1) + \
                                 np.dot(f_mat2, e_mat2) + np.dot(f_mat3, e_mat3)
        self.f_matrix[0, 0] = 1
        self.f_matrix[-1, -1] = 1
        
        self.f_matrix = self.f_matrix * self.coord_electrode.units**2 / self.cond.units

        
    def calc_csd(self):
        '''Calculate the iCSD using the spline iCSD method'''
        #e_mat0, e_mat1, e_mat2, e_mat3 = self.calc_e_matrices()
        e_mat = self.calc_e_matrices()
        
        [el_len, n_tsteps] = self.lfp.shape
        
        # padding the lfp with zeros on top/bottom
        cs_lfp = np.zeros((el_len+2, n_tsteps)) * self.lfp.units
        cs_lfp[1:-1, :] = self.lfp
    
        # CSD coefficients
        csd_coeff = np.linalg.solve(self.f_matrix, cs_lfp)
        
        # The cubic spline polynomial coefficients
        a_mat0 = np.dot(e_mat[0], csd_coeff)
        a_mat1 = np.dot(e_mat[1], csd_coeff)
        a_mat2 = np.dot(e_mat[2], csd_coeff)
        a_mat3 = np.dot(e_mat[3], csd_coeff)
        
        
        # Extend electrode coordinates in both end by mean interdistance
        coord_ext = np.zeros(el_len + 2)
        coord_ext[0] = 0
        coord_ext[1:-1] = self.coord_electrode
        coord_ext[-1] = self.coord_electrode[-1] + \
            np.diff(self.coord_electrode).mean()
        
        # create high res spatial grid
        out_zs = np.linspace(coord_ext[0], coord_ext[-1], self.num_steps)
        self.csd = np.empty((self.num_steps, self.lfp.shape[1]))
        
        # Calculate iCSD estimate on grid from polynomial coefficients. 
        i = 0
        for j in xrange(self.num_steps):
            if out_zs[j] > coord_ext[i+1]:
                i += 1
            self.csd[j, :] = a_mat0[i, :] + a_mat1[i, :] * \
                            (out_zs[j] - coord_ext[i]) +\
                a_mat2[i, :] * (out_zs[j] - coord_ext[i])**2 + \
                a_mat3[i, :] * (out_zs[j] - coord_ext[i])**3
        
        self.csd = (self.csd * self.f_matrix.units**-1 * self.lfp.units).simplified

    
    def f_mat0(self, zeta, z_val):
        '''0'th order potential function'''
        if type(zeta) is float:
            zeta = zeta*z_val.units
        return 1./(2.*self.cond) * (np.sqrt((self.diam/2)**2 + ((z_val-zeta))**2) - abs(z_val-zeta))
    
    def f_mat1(self, zeta, z_val, zi_val):
        '''1'th order potential function'''
        if type(zeta) is float:
            zeta = zeta*z_val.units
        return (zeta-zi_val) * self.f_mat0(zeta, z_val)
    
    def f_mat2(self, zeta, z_val, zi_val):
        '''2'nd order potential function'''
        if type(zeta) is float:
            zeta = zeta*z_val.units
        return (zeta-zi_val)**2 * self.f_mat0(zeta, z_val)
    
    def f_mat3(self, zeta, z_val, zi_val):
        '''3'rd order potential function'''
        if type(zeta) is float:
            zeta = zeta*z_val.units
        return (zeta-zi_val)**3 * self.f_mat0(zeta, z_val)
    
    def calc_k_matrix(self):
        '''Calculate the K-matrix used by to calculate E-matrices'''
        el_len = self.coord_electrode.size
        # expanding electrode grid
        z_js = np.zeros(el_len+2)
        z_js[1:-1] = self.coord_electrode
        z_js[-1] = self.coord_electrode[-1] + \
            np.diff(self.coord_electrode).mean()
        
        c_vec = 1./np.diff(z_js)
        # Define transformation matrices
        c_jm1 = np.matrix(np.zeros((el_len+2, el_len+2)))
        c_j0 = np.matrix(np.zeros((el_len+2, el_len+2)))
        c_jall = np.matrix(np.zeros((el_len+2, el_len+2)))
        c_mat3 = np.matrix(np.zeros((el_len+1, el_len+1)))
        
        for i in xrange(el_len+1):
            for j in xrange(el_len+1):
                if i == j:
                    c_jm1[i+1, j+1] = c_vec[i]
                    c_j0[i, j] = c_jm1[i+1, j+1]
                    c_mat3[i, j] = c_vec[i]
        
        c_jm1[-1, -1] = 0
        
        c_jall = c_j0
        c_jall[0, 0] = 1
        c_jall[-1, -1] = 1
        
        c_j0 = 0
        
        tjp1 = np.matrix(np.zeros((el_len+2, el_len+2)))
        tjm1 = np.matrix(np.zeros((el_len+2, el_len+2)))
        tj0 = np.matrix(np.eye(el_len+2))
        tj0[0, 0] = 0
        tj0[-1, -1] = 0

        for i in xrange(1, el_len+2):
            for j in xrange(el_len+2):
                if i == j-1:
                    tjp1[i, j] = 1
                elif i == j+1:
                    tjm1[i, j] = 1
        
        # Defining K-matrix used to calculate e_mat1-3
        return np.array((c_jm1*tjm1 + 2*c_jm1*tj0 + 2*c_jall + c_j0*tjp1)**-1 * 3 * 
            (c_jm1**2 * tj0 - c_jm1**2 * tjm1 + c_j0**2 * tjp1 - c_j0**2 * tj0))
        
    def calc_e_matrices(self):
        '''Calculate the E-matrices used by cubic spline iCSD method'''
        el_len = self.coord_electrode.size
        ## expanding electrode grid
        z_js = np.zeros(el_len+2)
        z_js[1:-1] = self.coord_electrode
        z_js[-1] = self.coord_electrode[-1] + \
            np.diff(self.coord_electrode).mean()
        
        ## Define transformation matrices
        c_mat3 = np.matrix(np.zeros((el_len+1, el_len+1)))
        
        for i in xrange(el_len+1):
            for j in xrange(el_len+1):
                if i == j:
                    c_mat3[i, j] = 1./np.diff(z_js)[i]

        # Get K-matrix
        k_matrix = self.calc_k_matrix()
        
        # Define matrixes for C to A transformation:
        # identity matrix except that it cuts off last element:
        tja = np.matrix(np.zeros((el_len+1, el_len+2)))
        # converting k_j to k_j+1 and cutting off last element:
        tjp1a = np.matrix(np.zeros((el_len+1, el_len+2))) 

        # C to A
        for i in xrange(el_len+1):
            for j in xrange(el_len+2):
                if i == j-1:
                    tjp1a[i, j] = 1
                elif i == j:
                    tja[i, j] = 1
        
        # Define spline coeffiscients
        e_mat0 = tja    
        e_mat1 = tja*k_matrix
        e_mat2 = 3 * c_mat3**2 * (tjp1a-tja) - c_mat3 * \
                (tjp1a + 2 * tja) * k_matrix
        e_mat3 = 2 * c_mat3**3 * (tja-tjp1a) + c_mat3**2 * \
                (tjp1a + tja) * k_matrix
        
        return np.array(e_mat0), np.array(e_mat1), np.array(e_mat2), np.array(e_mat3)

    
def estimate_csd(lfp_ansigarr, coord_electrode,
                 diam=500E-6, cond=0.3, cond_top=0.3, tol=1E-6,
                 f_type='gaussian', f_order=(3, 1), num_steps=200,
                 method='standard'):

    if not method in ['standard', 'delta', 'step', 'spline']:
        raise ValueError("method must be either 'standard', 'delta', 'step', 'spline'")

    if not isinstance(lfp_ansigarr, neo.AnalogSignalArray):
        raise TypeError('LFP is not an neo.AnalogSignalArray!')

    lfp = lfp_ansigarr.magnitude.T * lfp_ansigarr.units

    arg_dict = {'lfp': lfp,
                'coord_electrode': coord_electrode,
                'diam': diam,
                'cond': cond,
                'cond_top': cond_top,
                'tol': tol,
                'f_type': f_type,
                'f_order': f_order,
                'num_steps': num_steps}

    if method == 'standard':
        csd = StandardCSD(**arg_dict)
    elif method == 'delta':
        csd = DeltaiCSD(**arg_dict)
    elif method == 'step':
        csd = StepiCSD(**arg_dict)
    elif method == 'spline':
        csd = SplineiCSD(**arg_dict)

    csd_ansigarr = neo.AnalogSignalArray(csd.T, t_start=lfp_ansigarr.t_start,
                                        sampling_rate=lfp_ansigarr.sampling_rate)

    return csd_ansigarr
