#!/usr/env/python

import matplotlib.pyplot as plt
import numpy as np
import icsd
from scipy import io

#loading test data
test_data = io.loadmat('test_data.mat')

# Using one of the datasets, corresponding electrode coordinates
lfp_data = test_data['pot1'] * 1E-3         # [mV] -> [V]
z_data = np.linspace(100E-6, 2300E-6, 23)   # [m]

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



plt.figure()
plt.imshow(lfp_data, origin='upper', vmin=-abs(lfp_data).max(), \
          vmax=abs(lfp_data).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.colorbar()
plt.title('LFP')

plt.figure()
plt.subplot(211)
plt.imshow(std_csd.csd, origin='upper', vmin=-abs(std_csd.csd).max(), \
          vmax=abs(std_csd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('standard CSD')
plt.colorbar()

plt.subplot(212)
plt.imshow(std_csd.csd_filtered, origin='upper', vmin=-abs(std_csd.csd_filtered).max(), \
          vmax=abs(std_csd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('standard CSD, filtered')
plt.colorbar()

plt.figure()
plt.subplot(211)
plt.imshow(delta_icsd.csd, origin='upper', vmin=-abs(delta_icsd.csd).max(), \
          vmax=abs(delta_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('delta iCSD')
plt.colorbar()

plt.subplot(212)
plt.imshow(delta_icsd.csd_filtered, origin='upper', vmin=-abs(delta_icsd.csd_filtered).max(), \
          vmax=abs(delta_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('delta iCSD, filtered')
plt.colorbar()

plt.figure()
plt.subplot(211)
plt.imshow(step_icsd.csd, origin='upper', vmin=-abs(step_icsd.csd).max(), \
          vmax=abs(step_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('step iCSD')
plt.colorbar()

plt.subplot(212)
plt.imshow(step_icsd.csd_filtered, origin='upper', vmin=-abs(step_icsd.csd_filtered).max(), \
          vmax=abs(step_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('step iCSD, filtered')
plt.colorbar()

plt.figure()
plt.subplot(211)
plt.imshow(spline_icsd.csd, origin='upper', vmin=-abs(spline_icsd.csd).max(), \
          vmax=abs(spline_icsd.csd).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('spline iCSD')
plt.colorbar()

plt.subplot(212)
plt.imshow(spline_icsd.csd_filtered, origin='upper', vmin=-abs(spline_icsd.csd_filtered).max(), \
          vmax=abs(spline_icsd.csd_filtered).max(), cmap='jet_r', interpolation='nearest')
plt.axis('tight')
plt.title('spline iCSD')
plt.colorbar()
plt.show()