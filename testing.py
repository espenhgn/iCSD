#!/usr/bin/env python

import os
import numpy as np
import quantities as pq
import icsd
import unittest


def potential_of_plane(z,
                      rho=1*pq.A/pq.m**2,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of infinite horizontal plane with constant
    current source density at a vertical offset z. 
    
    Arguments
    ---------
    z : float*pq.m
        distance perpendicular to center of disk
    rho : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    '''
    
    return rho/(2*sigma)*abs(z)



def potential_of_disk(z,
                      rho=1*pq.A/pq.m**2,
                      R=1E-3*pq.m,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of circular disk in horizontal plane with constant
    current source density at a vertical offset z. 
    
    Arguments
    ---------
    z : float*pq.m
        distance perpendicular to center of disk
    rho : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    R : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    '''
    
    return rho/(2*sigma)*(np.sqrt(z**2 + R**2) - abs(z))


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

