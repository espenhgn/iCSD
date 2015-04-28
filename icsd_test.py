#!/usr/env/python

import matplotlib.pyplot as plt
import numpy as np
import icsd
from scipy import io
import quantities as pq

#loading test data
test_data = io.loadmat('test_data.mat')

# Using one of the datasets, corresponding electrode coordinates
lfp_data = test_data['pot1'] * 1E-3  *pq.V       # [mV] -> [V]
z_data = np.linspace(100E-6, 2300E-6, 23) *pq.m  # [m]
diam = 500E-6 * pq.m
sigma = 0.3 * pq.S / pq.m


# Input dictionaries for each method
delta_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,        # source diameter
    'cond' : sigma,           # extracellular conductivity
    'cond_top' : sigma,       # conductivity on top of cortex
    'f_type' : 'gaussian',  # gaussian filter
    'f_order' : (3, 1),     # 3-point filter, sigma = 1.
}
step_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,
    'cond' : sigma,
    'cond_top' : sigma,
    'tol' : 1E-12,          # Tolerance in numerical integration
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
}
spline_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,
    'cond' : sigma,
    'cond_top' : sigma,
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



plt.figure()
plt.imshow(np.array(lfp_data), origin='upper', vmin=-abs(lfp_data).max(), \
          vmax=abs(lfp_data).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
cb = plt.colorbar()
cb.set_label('LFP (%s)' % lfp_data.dimensionality.string)
plt.title('LFP')

plt.figure()
plt.subplot(211)
plt.imshow(np.array(std_csd.csd), origin='upper', vmin=-abs(std_csd.csd).max(), \
          vmax=abs(std_csd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('standard CSD')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % std_csd.csd.dimensionality.string)

plt.subplot(212)
plt.imshow(np.array(std_csd.csd_filtered), origin='upper', vmin=-abs(std_csd.csd_filtered).max(), \
          vmax=abs(std_csd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('standard CSD, filtered')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % std_csd.csd.dimensionality.string)

plt.figure()
plt.subplot(211)
plt.imshow(np.array(delta_icsd.csd), origin='upper', vmin=-abs(delta_icsd.csd).max(), \
          vmax=abs(delta_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('delta iCSD')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % delta_icsd.csd.dimensionality.string)

plt.subplot(212)
plt.imshow(np.array(delta_icsd.csd_filtered), origin='upper', vmin=-abs(delta_icsd.csd_filtered).max(), \
          vmax=abs(delta_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('delta iCSD, filtered')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % delta_icsd.csd.dimensionality.string)

plt.figure()
plt.subplot(211)
plt.imshow(np.array(step_icsd.csd), origin='upper', vmin=-abs(step_icsd.csd).max(), \
          vmax=abs(step_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('step iCSD')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % step_icsd.csd.dimensionality.string)

plt.subplot(212)
plt.imshow(np.array(step_icsd.csd_filtered), origin='upper', vmin=-abs(step_icsd.csd_filtered).max(), \
          vmax=abs(step_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('step iCSD, filtered')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % step_icsd.csd.dimensionality.string)

plt.figure()
plt.subplot(211)
plt.imshow(np.array(spline_icsd.csd), origin='upper', vmin=-abs(spline_icsd.csd).max(), \
          vmax=abs(spline_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('spline iCSD')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % spline_icsd.csd.dimensionality.string)

plt.subplot(212)
plt.imshow(np.array(spline_icsd.csd_filtered), origin='upper', vmin=-abs(spline_icsd.csd_filtered).max(), \
          vmax=abs(spline_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('spline iCSD')
cb = plt.colorbar()
cb.set_label('CSD (%s)' % spline_icsd.csd.dimensionality.string)
plt.show()