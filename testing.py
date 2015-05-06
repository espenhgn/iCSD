#!/usr/bin/env python

import os
import numpy as np
import quantities as pq
import scipy.integrate as si
import icsd
import unittest
import matplotlib.pyplot as plt
    


def potential_of_plane(z_j, z_i=0.*pq.m,
                      C_i=1*pq.A/pq.m**2,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of infinite horizontal plane with constant
    current source density at a vertical offset z_j. 
    
    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to source layer
    z_i : float*pq.m
        z-position of source layer
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m

    Notes
    -----
    The potential is 0 at the plane, as the potential goes to infinity for
    large distances
    
    '''
    if z_j.units != z_i.units:
        raise ValueError, 'units of z_j ({}) and z_i ({}) not equal'.format(z_j.units, z_i.units)
    
    return -C_i/(2*sigma)*abs(z_j-z_i).simplified


def potential_of_disk(z_j,
                      z_i=0.*pq.m,
                      C_i=1*pq.A/pq.m**2,
                      R_i=1E-3*pq.m,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of circular disk in horizontal plane with constant
    current source density at a vertical offset z_j. 
    
    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z_j-position of source disk
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    '''
    if z_j.units != z_i.units or z_j.units != R_i.units or R_i.units != z_i.units:
        raise ValueError, 'units of z_j ({}), z_i ({}) and R_i ({}) not equal'.format(z_j.units, z_i.units, R_i.units)
    
    return C_i/(2*sigma)*(np.sqrt((z_j-z_i)**2 + R_i**2) - abs(z_j-z_i)).simplified


def potential_of_cylinder(z_j,
                          z_i=0.*pq.m,
                          h_i=0.1*pq.m,
                          C_i=1*pq.A/pq.m**3,
                          R_i=1E-3*pq.m,
                          sigma=0.3*pq.S/pq.m,
                          ):
    '''
    Return potential of cylinder in horizontal plane with constant homogeneous
    current source density at a vertical offset z_j. 
    

    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z-position of center of source cylinder
    h_i : float*pq.m
        thickness of cylinder
    C_i : float*pq.A/pq.m**3
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m

    Notes
    -----
    Sympy can't deal with eq. 11 in Pettersen et al 2006, J neurosci Meth,
    so we numerically evaluate it in this function.
    
    Tested with
    
    >>>from sympy import *
    >>>C_i, z_i, h, z_j, z_j, sigma, R = symbols('C_i z_i h z z_j sigma R')
    >>>C_i*integrate(1/(2*sigma)*(sqrt((z-z_j)**2 + R**2) - abs(z-z_j)), (z, z_i-h/2, z_i+h/2))


    '''
    if not z_j.units == z_i.units == R_i.units == h_i.units:
        raise ValueError, 'units of z_j ({}), z_i ({}), R_i ({}) and h ({}) not equal'.format(z_j.units, z_i.units, R_i.units, h_i.units)


    #evaluate integrand using quad
    def integrand(z):
        z *= z_i.units
        return 1/(2*sigma)*(np.sqrt((z-z_j)**2 + R_i**2) - abs(z-z_j))
        
    phi_j, abserr = C_i*si.quad(integrand, z_i-h_i/2, z_i+h_i/2)
    print('quad absolute error {}'.format(abserr))
    
    return (phi_j * z_i.units**2 / sigma.units)



def get_lfp_of_planes(z_j=np.arange(21)*1E-4*pq.m,
                      z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                      C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**2,
                      sigma=0.3*pq.S/pq.m):
    '''
    Compute the lfp of spatially separated planes with given current source
    density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, zi in enumerate(z_i):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_plane(zj, zi, C_i[i], sigma)
    
    #test plot
    plt.figure()
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(np.zeros(z_j.size), z_j, 'r-o')
    for i, C in enumerate(C_i):
        ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_ylabel('z_j ({})'.format(z_j.units))
    ax.set_xlabel('C_i ({})'.format(C_i.units))
    ax.set_title('planar CSD')

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(phi_j, z_j, 'r-o')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_xlabel('phi_j ({})'.format(phi_j.units))
    ax.set_title('LFP')
    
    return phi_j, C_i


def get_lfp_of_disks(z_j=np.arange(21)*1E-4*pq.m,
                         z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                         C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**2,
                         R_i = np.array([1, 1, 1])*1E-3*pq.m,
                         sigma=0.3*pq.S/pq.m):
    '''
    Compute the lfp of spatially separated disks with a given current source density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, zi in enumerate(z_i):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_disk(zj, zi, C_i[i], R_i[i], sigma)
    
    #test plot
    plt.figure()
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(np.zeros(z_j.size), z_j, 'r-o')
    for i, C in enumerate(C_i):
        ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_ylabel('z_j ({})'.format(z_j.units))
    ax.set_xlabel('C_i ({})'.format(C_i.units))
    ax.set_title('disk CSD\nR={}'.format(R_i))

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(phi_j, z_j, 'r-o')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_xlabel('phi_j ({})'.format(phi_j.units))
    ax.set_title('LFP')
    
    return phi_j, C_i
    

def get_lfp_of_cylinders(z_j=np.arange(21)*1E-4*pq.m,
                         z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                         h_i=np.array([1, 1, 1])*1E-4*pq.m,
                         C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**3,
                         R_i = np.array([1, 1, 1])*1E-3*pq.m,
                         sigma=0.3*pq.S/pq.m):
    '''
    Compute the lfp of spatially separated disks with a given current source density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, zi in enumerate(z_i):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_cylinder(zj, zi, h_i[i], C_i[i], R_i[i], sigma)
    
    #test plot
    plt.figure()
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(np.zeros(z_j.size), z_j, 'r-o')
    ax.barh(np.asarray(z_i-h_i/2),
            np.asarray(C_i),
            np.asarray(h_i), color='r')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_ylabel('z_j ({})'.format(z_j.units))
    ax.set_xlabel('C_i ({})'.format(C_i.units))
    ax.set_title('cylinder CSD\nR={}'.format(R_i))

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(phi_j, z_j, 'r-o')
    ax.set_ylim(z_j.min(), z_j.max())
    ax.set_xlabel('phi_j ({})'.format(phi_j.units))
    ax.set_title('LFP')
    
    return phi_j, C_i
    


class TestICSD(unittest.TestCase):
    '''
    Set of test functions for each CSD estimation method comparing
    estimate to LFPs calculated with known ground truth CSD
    '''
    
    def test_StandardCSD(self):
        raise NotImplementedError
    
    def test_DeltaiCSD(self):
        raise NotImplementedError
    
    def test_StepiCSD(self):
        raise NotImplementedError
    
    def test_SplineiCSD(self):
        raise NotImplementedError






def test(verbosity=2):
    '''
    Run unittests for the CSD toolbox
    
    
    Arguments
    ---------
    verbosity : int
        verbosity level
        
    '''
    suite = unittest.TestLoader().loadTestsFromTestCase(TestICSD)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
    


if __name__ == '__main__':
    #run test function
    test()
    
    #print potential_of_cylinder(0*pq.m)
    #print potential_of_disk(0*pq.m)
    #print potential_of_plane(0*pq.m)
    
    plt.close('all')

    phi_j, C_i = get_lfp_of_planes()
    get_lfp_of_disks()
    get_lfp_of_cylinders()
    
    plt.show()