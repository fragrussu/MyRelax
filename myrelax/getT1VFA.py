### Voxel-wise fitting of T1 on Variable Flip Angle data
#
# Author: Francesco Grussu, University College London
#		    CDSQuaMRI Project 
#		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>
#
# Code released under BSD Two-Clause license
#
# Copyright (c) 2019 University College London. 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

### Load useful modules
import argparse, os, sys
import multiprocessing
import numpy as np
from scipy.optimize import minimize
import nibabel as nib


def VFAsignal(mri_tr,mri_fa,tissue_par):
	''' Generate the signal for a variable flip angle (VFA) experiment at fixed TR
		
		
	    INTERFACE
	    signal = VFAsignal(mri_tr,mri_fa,tissue_par)
	    
	    PARAMETERS
	    - mri_tr: scalar value, indicating the TR (repetition time, in ms) used throughout the experiment
	    - mri_fa: list/array indicating the FAs (flip angles, in deg) used for the experiment (one measurement per FA)
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2*w-weighted proton density)
             		  tissue_par[1] = T1 (longitudinal relaxation time, in ms)
		
	    RETURNS
	    - signal: a numpy array of measurements generated according to the VFA signal model,
			
		         signal  =  S0 * ( (1 - exp(-TR/T1)) / (1 - cos(FA)*exp(-TR/T1)) )*sin(FA)
		
		      where TR and FA are the sequence parameters (repetition time and flip angle)
		      and where S0 and T1 are the tissue parameters (S0 is the T2*-weighted proton 
                      density, and T1 is the longitudinal relaxation time).

	    
	    References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al,
		        Magnetic Resonance in Medicine (2015), 73:514-522
 
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Handle inputs
	fa_values = np.array(mri_fa,'float64')  # Make sure flip angles are stored as a numpy array
	tr_value = float(mri_tr)         # Repetition time
	s0_value = tissue_par[0]         # S0
	t1_value = tissue_par[1]         # T1

	### Calculate signal
	signal = s0_value * np.sin(np.deg2rad(fa_values)) * ( ( 1 - np.exp((-1.0)*tr_value/t1_value) )  / ( 1 - (np.cos(np.deg2rad(fa_values)))*(np.exp((-1.0)*tr_value/t1_value))   )  )

	### Output signal
	return signal
	

def VFAFobj(tissue_par,mri_tr,mri_fa,meas):
	''' Fitting objective function for variable flip angle (VFA) signal model		
		
	    INTERFACE
	    fobj = VFAFobj(tissue_par,mri_tr,mri_fa,meas)
	    
	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2*w-weighted proton density)
             		  tissue_par[1] = T1 (longitudinal relaxation time, in ms)
	    - mri_tr: scalar value, indicating the TR (repetition time, in ms) used throughout the experiment
	    - mri_fa: list/array indicating the FAs (flip angles, in deg) used for the experiment (one measurement per FA)
	    - meas: list/array of measurements
		
	    RETURNS
	    - fobj: objective function measured as sum of squared errors between measurements and VFA predictions, i.e.
			
				 fobj = SUM_OVER_n( (prediction - measurement)^2 )
		
		     Above, the prediction are obtained using the VFA model implemented by function VFAsignal(). 		      
	    
	    References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al,
		        Magnetic Resonance in Medicine (2015), 73:514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	
	### Predict signals given tissue and sequence parameters
	pred = VFAsignal(mri_tr,mri_fa,tissue_par)

	### Calculate objective function and return
	fobj = np.sum( (np.array(pred) - np.array(meas))**2 )
	return fobj


def VFAGridSearch(mri_tr,mri_fa,meas):
	''' Grid search for non-linear fitting of the variable flip angle (VFA) signal model		
		
	    INTERFACE
	    tissue_estimate, fobj_grid = VFAGridSearch(mri_tr,mri_fa,meas)
	    
	    PARAMETERS
	    - mri_tr: scalar value, indicating the TR (repetition time, in ms) used throughout the experiment
	    - mri_fa: list/array indicating the FAs (flip angles, in deg) used for the experiment (one measurement per FA)
	    - meas: list/array of measurements
		
	    RETURNS
	    - tissue_estimate: estimate of tissue parameters that explain the measurements reasonably well. The parameters are
			       estimated sampling the fitting objective function VFAFobj() over a grid; the output is
                               tissue_estimate[0] = S0 (T2*w-weighted proton density)
             		       tissue_estimate[1] = T1 (longitudinal relaxation time, in ms)
	    - fobj_grid:       value of the objective function when the tissue parameters equal tissue_estimate
	    
	    References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al,
		        Magnetic Resonance in Medicine (2015), 73:514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Prepare grid for grid search
	t1_grid = np.array([100.0,300.0,600.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2200.0,2400.0,2600.0,3000.0,3500.0,4000.0,4500.0])  # Grid of T1 values
	s0_grid = np.linspace(0.0,40*np.max(meas),num=24)    # Grid of S0 values: from 0 up to 40 times the maximum signal taken as input

	### Initialise objective function to infinity and parameters for grid search
	fobj_best = float('inf')
	s0_best = 0.0
	t1_best = 0.0
	
	### Run grid search
	for ii in range(0, len(t1_grid)):

		t1_ii =  t1_grid[ii]   		
		for jj in range(0, len(s0_grid)):

			s0_jj =  s0_grid[jj]
			params = np.array([s0_jj,t1_ii])
			
			# Objective function
			fval = VFAFobj(params,mri_tr,mri_fa,meas)

			# Check if objective function is smaller than previous value
			if fval<fobj_best:
				fobj_best = fval
				s0_best = s0_jj
				t1_best = t1_ii

	### Return output
	paramsgrid = np.array([s0_best, t1_best])
	fobjgrid = fobj_best
	return paramsgrid, fobjgrid



def T1FitVFAslice(data):
	''' Fit T1 for a variable flip angle (VFA) experiment on one MRI slice stored as a 2D numpy array  
	    

	    INTERFACE
	    data_out = T1FitVFAslice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
		            are the slice first and second dimensions, whereas the third dimension of data[0] stores
                            measurements obtained with different flip angles
		    data[1] is a numpy monodimensional array storing the nominal flip angle values (deg) 
		    data[2] is a scalar describing the TR of the VFA experiment (ms)
		    data[3] is a string describing the fitting algorithm ("linear" or "nonlinear", see T1FitVFA())
		    data[4] is a 2D numpy array contaning the fitting mask within the MRI slice (see T1FitVFA())
                    data[5] is a 2D numpy array containing the B1 map within the MRI slice (see T1FitVFA())
		    data[6] is a scalar containing the index of the MRI slice in the 3D volume
	    
	    RETURNS
	    - data_out: a list of 4 elements, such that
		    data_out[0] is the parameter S0 (see T1FitVFA()) within the MRI slice
	            data_out[1] is the parameter T1 (see T1FitVFA()) within the MRI slice
                    data_out[2] is the exit code of the fitting (see T1FitVFA()) within the MRI slice
		    data_out[3] is the fitting sum of squared errors withint the MRI slice
                    data_out[4] equals data[6]
	
		    Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
	    
	    References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al,
		        Magnetic Resonance in Medicine (2015), 73:514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''

	
	### Extract signals and sequence information from the input list
	vfa_slice = data[0]         # VFA signal
	fa_value = data[1]          # flip angles (FA) (nominal; in deg)
	TR_value = data[2]          # TR (ms)
	fit_algo = data[3]          # fitting algorithm
	mask_slice = data[4]        # fitting mask
	b1map_slice = data[5]       # B1 map
	idx_slice = data[6]         # Slice index
	slicesize = vfa_slice.shape # Get number of voxels of current MRI slice along each dimension
	fa_value = np.array(fa_value)     # Make sure the FA is an array
	
	### Check whether a sensible algorithm has been requested
	if fit_algo!="linear" and fit_algo!="nonlinear":
		print('')
		print('ERROR: unrecognised fitting algorithm. Exiting with 1.')
		print('')
		sys.exit(1)

	### Allocate output variables
	s0_slice = np.zeros(slicesize[0:2],'float64')
	t1_slice = np.zeros(slicesize[0:2],'float64')
	exit_slice = np.zeros(slicesize[0:2],'float64')
	mse_slice = np.zeros(slicesize[0:2],'float64')
	Nmeas = slicesize[2]   # Number of measurements


	### Fit VFA model in the voxels within the current slice
	for xx in range(0, slicesize[0]):
			for yy in range(0, slicesize[1]):
		
				# Get mask for current voxel
				mask_voxel = mask_slice[xx,yy]           # Fitting mask for current voxel

				# The voxel is not background: fit VFA model				
				if(mask_voxel==1):

					# Get signal, fitting mask and B1 map
					vfa_voxel = vfa_slice[xx,yy,:]           # Extract signals for current voxel
					vfa_voxel = np.array(vfa_voxel)  # Convert to array
					b1map_voxel = b1map_slice[xx,yy]         # B1 map for current voxel
						
					## Simplest case: there are only two flip angles --> get the solution analytically
					if(Nmeas==2):
						sig1 = vfa_voxel[0]                            # Signal for first flip angle
						sig2 = vfa_voxel[1] 		               # Signal for seconds flip angle
						fa1_nominal = fa_value[0]              # First flip angle (nominal value)
						fa2_nominal = fa_value[1]              # Second flip angle (nominal value)
						fa1_actual = fa1_nominal*b1map_voxel   # First flip angle: use B1 map to correct nominal value
						fa2_actual = fa2_nominal*b1map_voxel   # Second flip angle: use B1 map to correct nominal value	
						
						# Calculate maps analytically, handling warnings
						with np.errstate(divide='raise',invalid='raise'):	
							try:
								A_voxel = ( sig1*np.sin( np.deg2rad(fa2_actual) ) ) / ( sig2*np.sin( np.deg2rad(fa1_actual) ) )
								t1_voxel = (-1.0)*TR_value / np.log( ( A_voxel - 1 ) / ( A_voxel*np.cos( np.deg2rad(fa1_actual) )  - np.cos( np.deg2rad(fa2_actual) ) )  )
								s0_voxel = sig1 / ( ( ( 1 - np.exp((-1)*TR_value/t1_voxel) ) / ( 1 - ( np.cos( np.deg2rad(fa1_actual) ) )*(np.exp((-1)*TR_value/t1_voxel)) ) )*np.sin( np.deg2rad(fa1_actual) ) )
								
								# Check whether the solution is plausible or whether any measurement was negative
								if s0_voxel<0 or t1_voxel<0:
									s0_voxel = 0.0
									t1_voxel = 0.0
									exit_voxel = -1
									mse_voxel = 0.0
								else:
									exit_voxel = 1
									mse_voxel = 0.0   # The analytical solution provides predictions that are exactly equal to the measurements
								
							except FloatingPointError:
								s0_voxel = 0.0
								t1_voxel = 0.0
								exit_voxel = -1
								mse_voxel = 0.0

					## General case: there are more than two flip angles --> get the solution minimising an objective function
					else:

						# Perform linear fitting as first thing - if non-linear fitting is required, the linear fitting will be used to initialise the non-linear optimisation afterwards
						fa_actual = b1map_voxel*fa_value    # Calculate actual flip angles
						fa_actual_column = np.reshape(fa_actual,(Nmeas,1))  # Store actual flip angles as a column array						
						vfa_voxel_column = np.reshape(vfa_voxel,(Nmeas,1))   # Reshape measurements as column array

						# Calculate linear regression coefficients as ( W * Q )^-1 * (W * m), while handling warnings
						with np.errstate(divide='raise',invalid='raise'):
							try:
								# Create matrices and arrays to be combinted via matrix multiplication
								Yvals = vfa_voxel_column / np.sin(np.deg2rad(fa_actual_column))    # Independent variable of linearised model
								Xvals = vfa_voxel_column / np.tan(np.deg2rad(fa_actual_column))    # Dependent variable of linearised model
								allones = np.ones([Nmeas,1])        # Column of ones						
								Qmat = np.concatenate((allones,Xvals),axis=1)    # Design matrix Q
								Wmat = np.diag(vfa_voxel)  # Matrix of weights W
									
								# Calculate coefficients via matrix multiplication
								coeffs = np.matmul( np.linalg.pinv( np.matmul(Wmat,Qmat) ) , np.matmul(Wmat,Yvals) )
									
								# Retrieve signal model parameters from linear regression coefficients
								acoeff = coeffs[0]
								bcoeff = coeffs[1]
								t1_voxel = (-1.0)*TR_value / np.log(bcoeff)
								s0_voxel = acoeff / ( 1 - np.exp((-1)*TR_value/t1_voxel) )
									
								# Check whether the solution is plausible: if not, declare fitting failed
								if s0_voxel<0 or t1_voxel<0:
									s0_voxel = 0.0
									t1_voxel = 0.0
									exit_voxel = -1
									mse_voxel = 0.0
								else:
									exit_voxel = 1
									mse_voxel = VFAFobj([s0_voxel,t1_voxel],TR_value,fa_actual,vfa_voxel)   # Fitting was successful
								
							except FloatingPointError:
								s0_voxel = 0.0
								t1_voxel = 0.0
								exit_voxel = -1
								mse_voxel = 0.0
						
						# Refine the results from linear with non-linear optimisation if the selected algorithm is "nonlinear"
						if fit_algo=="nonlinear":

							# Check whether linear fitting has failed
							if exit_voxel==-1:
								param_init, fobj_init = VFAGridSearch(TR_value,fa_actual,vfa_voxel)   # Linear fitting has failed: run a grid search
							else:
								param_init = [s0_voxel,t1_voxel]   # Linear fitting did not fail: use linear fitting output to initialise non-linear optimisation
								fobj_init = mse_voxel             
							
							# Minimise the objective function numerically
							param_bound = ((0,2*s0_voxel),(0,6000),)                      # Range for S0 and T1 (T1 limited to be < 6000)						
							modelfit = minimize(VFAFobj, param_init, method='L-BFGS-B', args=tuple([TR_value,fa_actual,vfa_voxel]), bounds=param_bound)
							fit_exit = modelfit.success
							fobj_fit = modelfit.fun

							# Get fitting output if non-linear optimisation was successful and if succeeded in providing a smaller value of the objective function as compared to the grid search
							if fit_exit==True and fobj_fit<fobj_init:
								param_fit = modelfit.x
								s0_voxel = param_fit[0]
								t1_voxel = param_fit[1]
								exit_voxel = 1
								mse_voxel = fobj_fit

							# Otherwise, output the best we could find with linear fitting or, when linear fitting fails, with grid search (note that grid search cannot fail by implementation)
							else:
								s0_voxel = param_init[0]
								t1_voxel = param_init[1]
								exit_voxel = -1
								mse_voxel = fobj_init
							
						
							
				# The voxel is background
				else:
					s0_voxel = 0.0
					t1_voxel = 0.0
					exit_voxel = 0
					mse_voxel = 0.0
				
				# Store fitting results for current voxel
				s0_slice[xx,yy] = s0_voxel
				t1_slice[xx,yy] = t1_voxel
				exit_slice[xx,yy] = exit_voxel
				mse_slice[xx,yy] = mse_voxel

	### Create output list storing the fitted VFA parameters and then return
	data_out = [s0_slice, t1_slice, exit_slice, mse_slice, idx_slice]
	return data_out
	



def T1FitVFA(*argv, **kwargs):
	''' Fit T1 for a variable flip angle (VFA) experiment  
	    

	    INTERFACES
	    T1FitVFA(vfa_nifti, vfa_text, tr_text, output_basename, algo, ncpu)
	    T1FitVFA(vfa_nifti, vfa_text, tr_text, output_basename, algo, ncpu, b1map_nifti=</path/to/file>)
	    T1FitVFA(vfa_nifti, vfa_text, tr_text, output_basename, algo, ncpu, mask_nifti)
	    T1FitVFA(vfa_nifti, vfa_text, tr_text, output_basename, algo, ncpu, mask_nifti, b1map_nifti=</path/to/file>)
	     
	    PARAMETERS
	    - vfa_nifti: path of a Nifti file storing the VFA data as 4D data.
	    - vfa_text: path of a text file storing the flip angles (deg) used to acquire the VFA data.
	    - tr_text: path of a text file storing the numerical value of the TR used to acquire the data (ms) 
	    - output_basename: base name of output files. Output files will end in 
                            "_S0VFA.nii"   --> T2*-weighted proton density, with receiver coil field bias
		            "_T1VFA.nii"   --> T1 map (ms)
			    "_ExitVFA.nii" --> exit code (1: successful fitting; -1 fail, 0 background)
			    "_SSEVFA.nii"  --> fitting sum of squared errors
			    
			    Note that in the background and where fitting fails, S0, T1 and MSE are set to 0.0
			    Output files will be stored as double-precision floating point (FLOAT64)
			
	    - algo: fitting algorithm ("linear" or "nonlinear")
	    - ncpu: number of processors to be used for computation
	    - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the VFA
		        signal model needs to be fitted, and 0 otherwise
	    - b1map_nifti: path of a Nifti file storing the deviation of the actual flip angle from the
		         nominal flip angle in each voxel, with ACTUAL_FA = B1MAP x NOMINAL_FA
	    
	    References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al,
		        Magnetic Resonance in Medicine (2015), 73:514-522
	    
            Dependencies: numpy, nibabel, scipy (other than standard library)
 
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Get input parametrs
	Nargv = len(argv)
	Nkwargs = len(kwargs)
	vfa_nifti = argv[0]
	vfa_text = argv[1]
	TR_text = argv[2]
	output_rootname = argv[3]
	algo = argv[4]
	ncpu = argv[5]
	ncpu_physical = multiprocessing.cpu_count()
	if ncpu>ncpu_physical:
		print('')
		print('WARNING: {} CPUs were requested. Using {} instead (all available CPUs)...'.format(ncpu,ncpu_physical))					 
		print('')
		ncpu = ncpu_physical     # Do not open more workers than the physical number of CPUs

	### Check whether the requested fitting algorithm makes sense or not
	if algo!="linear" and algo!="nonlinear":
		print('')
		print('ERROR: unrecognised fitting algorithm. Exiting with 1.')
		print('')
		sys.exit(1)
	
	### Get python version (some parts of code are different in python 2 and python 3)
	py_version = sys.version_info

	### Load VFA data
	print('    ... loading input data')
	
	# Make sure VFA data exists
	try:
		vfa_obj = nib.load(vfa_nifti)
	except:
		print('')
		print('ERROR: the 4D input NIFTI file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(vfa_nifti))					 
		print('')
		sys.exit(1)
	
	# Get image dimensions and convert to float64
	vfa_data = vfa_obj.get_data()
	imgsize = vfa_data.shape
	vfa_data = np.array(vfa_data,'float64')
	imgsize = np.array(imgsize)
	
	# Make sure flip angle data exists and makes sense
	try:
		faarray = np.loadtxt(vfa_text)
		faarray = np.array(faarray,'float64')
		faarray_size = faarray.size
	except:
		print('')
		print('ERROR: the flip angle file {} does not exist or is not a numeric text file. Exiting with 1.'.format(vfa_text))					 
		print('')
		sys.exit(1)

	# Make sure TR data exists and makes sense
	try:
		TRarray = np.loadtxt(TR_text)
		TRarray = np.array(TRarray,'float64')
		TRarray_size = TRarray.size
	except:
		print('')
		print('ERROR: the TR file {} does not exist or is not a numeric text file. Exiting with 1.'.format(TR_text))					 
		print('')
		sys.exit(1)
	

	# Check consistency of TR file (one scalar value is expected)
	if TRarray_size!=1:
		print('')
		print('ERROR: the TR file {} contains more than one entry or is empty. Exiting with 1.'.format(TR_text))					 
		print('')
		sys.exit(1)
	else:
		TR = TRarray
			
	# Check consistency of FA file and number of measurements
	if imgsize.size!=4:
		print('')
		print('ERROR: the input file {} is not a 4D nifti. Exiting with 1.'.format(vfa_nifti))					 
		print('')
		sys.exit(1)
	if faarray_size!=imgsize[3]:
		print('')
		print('ERROR: the number of measurements in {} does not match the number of flip angles in {}. Exiting with 1.'.format(vfa_nifti,vfa_text))					 
		print('')
		sys.exit(1)
	fa = faarray

	### Deal with optional arguments: mask
	if Nargv==7:
		mask_nifti = argv[6]
		try:
			mask_obj = nib.load(mask_nifti)
		except:
			print('')
			print('ERROR: the mask file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mask_nifti))					 
			print('')
			sys.exit(1)
		
		# Make sure that the mask has header information that is consistent with the input data containing the VFA measurements
		vfa_header = vfa_obj.header
		vfa_affine = vfa_header.get_best_affine()
		vfa_dims = vfa_obj.shape
		mask_dims = mask_obj.shape		
		mask_header = mask_obj.header
		mask_affine = mask_header.get_best_affine()			
		# Make sure the mask is a 3D file
		mask_data = mask_obj.get_data()
		masksize = mask_data.shape
		masksize = np.array(masksize)
		if masksize.size!=3:
			print('')
			print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(vfa_affine==mask_affine)!=16) or (vfa_dims[0]!=mask_dims[0]) or (vfa_dims[1]!=mask_dims[1]) or (vfa_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the mask file {} does not match that of the input data. Ignoring mask...'.format(mask_nifti))					 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask data is a numpy array
			mask_data[mask_data>0] = 1
			mask_data[mask_data<=0] = 0
	else:
		mask_data = np.ones(imgsize[0:3],'float64')
	
	### Deal with optional arguments: B1 map
	if Nkwargs!=0:

		kwitems = kwargs.items()
		
		# Python version 2
		if(py_version.major<3):
			kwitems_list = kwitems
		# Python version 3
		else:
			kwitems_list = list(kwitems)
		
		b1map_nifti = kwitems_list[0][1]
		try:
			b1map_obj = nib.load(b1map_nifti)
		except:
			print('')
			print('ERROR: the B1 map file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(b1map_nifti))					 
			print('')
			sys.exit(1)
		
		# Make sure that the B1 map has header information that is consistent with the input data containing the VFA measurements
		vfa_header = vfa_obj.header
		vfa_affine = vfa_header.get_best_affine()
		vfa_dims = vfa_obj.shape
		b1_dims = b1map_obj.shape	
		b1_header = b1map_obj.header
		b1_affine = b1_header.get_best_affine()
		# Make sure the B1 map is a 3D file
		b1map_data = b1map_obj.get_data()
		b1size = b1map_data.shape
		b1size = np.array(b1size)
		if b1size.size!=3:
			print('')
			print('WARNING: the B1 file {} is not a 3D Nifti file. Ignoring B1 map...'.format(b1map_nifti))				 
			print('')
			b1map_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(vfa_affine==b1_affine)!=16) or (vfa_dims[0]!=b1_dims[0]) or (vfa_dims[1]!=b1_dims[1]) or (vfa_dims[2]!=b1_dims[2]) ):
			print('')
			print('WARNING: the geometry of the B1 map file {} does not match that of the input data. Ignoring B1 map...'.format(b1map_nifti))					 
			print('')
			b1map_data = np.ones(imgsize[0:3],'float64')
		else:
			b1map_data = np.array(b1map_data,'float64')
				
		
	else:
		b1map_data = np.ones(imgsize[0:3],'float64')
	

	### Allocate memory for outputs
	s0_data = np.zeros(imgsize[0:3],'float64')	       # T2*-weighted proton density with receiver field bias (double-precision floating point)
	t1_data = np.zeros(imgsize[0:3],'float64')	       # T1 (double-precision floating point)
	exit_data = np.zeros(imgsize[0:3],'float64')           # Exit code (double-precision floating point)
	mse_data = np.zeros(imgsize[0:3],'float64')            # Fitting sum of squared errors (MSE) (double-precision floating point)

	#### Fitting
	print('    ... T1 estimation')
	# Create the list of input data
	inputlist = [] 
	for zz in range(0, imgsize[2]):
		sliceinfo = [vfa_data[:,:,zz,:],fa,TR,algo,mask_data[:,:,zz],b1map_data[:,:,zz],zz]  # List of information relative to the zz-th MRI slice
		inputlist.append(sliceinfo)     # Append each slice list and create a longer list of MRI slices whose processing will run in parallel

	# Clear some memory
	del vfa_data, mask_data, b1map_data 
	
	# Call a pool of workers to run the fitting in parallel if parallel processing is required (and if the the number of slices is > 1)
	if ncpu>1 and imgsize[2]>1:

		# Create the parallel pool and give jobs to the workers
		fitpool = multiprocessing.Pool(processes=ncpu)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get initial process identifications (PIDs)
		fitresults = fitpool.map_async(T1FitVFAslice,inputlist)      # Give jobs to the parallel processes
		
		# Busy-waiting: until work is done, check whether any worker dies (in that case, PIDs would change!)
		while not fitresults.ready():
			fitpool_pids_new = [proc.pid for proc in fitpool._pool]  # Get process IDs again
			if fitpool_pids_new!=fitpool_pids_initial:               # Check whether the IDs have changed from the initial values
				print('')					 # Yes, they changed: at least one worker has died! Exit with error
				print('ERROR: some processes died during parallel fitting. Exiting with 1.')					 
				print('')
				sys.exit(1)
		
		# Work done: get results
		fitlist = fitresults.get()

		# Collect fitting output and re-assemble MRI slices		
		for kk in range(0, imgsize[2]):					
			fitslice = fitlist[kk]    # Fitting output relative to kk-th element in the list
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of VFA model
			t1_data[:,:,slicepos] = fitslice[1]    # Parameter T1 of VFA model
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors
	
	# Run serial fitting as no parallel processing is required (it can take up to 1 hour per brain)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = T1FitVFAslice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of VFA model
			t1_data[:,:,slicepos] = fitslice[1]    # Parameter T1 of VFA model
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors


	### Save the output maps
	print('    ... saving output files')
	buffer_string=''
	seq_string = (output_rootname,'_T1VFA.nii')
	t1_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_S0VFA.nii')
	s0_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_ExitVFA.nii')
	exit_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_SSEVFA.nii')
	mse_outfile = buffer_string.join(seq_string)
	buffer_header = vfa_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type
	t1_obj = nib.Nifti1Image(t1_data,vfa_obj.affine,buffer_header)
	nib.save(t1_obj, t1_outfile)
	s0_obj = nib.Nifti1Image(s0_data,vfa_obj.affine,buffer_header)
	nib.save(s0_obj, s0_outfile)
	exit_obj = nib.Nifti1Image(exit_data,vfa_obj.affine,buffer_header)
	nib.save(exit_obj, exit_outfile)
	mse_obj = nib.Nifti1Image(mse_data,vfa_obj.affine,buffer_header)
	nib.save(mse_obj, mse_outfile)

	### Done
	print('')




# Run the module as a script when required
if __name__ == "__main__":


	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise fitting of T1 from Variable Flip Angle (VFA) MRI magnitude data already corrected for motion. Dependencies (Python packages): numpy, nibabel, scipy (other than standard library). References: "On the accuracy of T1 mapping: searching for common ground", Stikov N et al, Magnetic Resonance in Medicine (2015), 73:514-522. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('vfa_file', help='4D Nifti file of magnitude images from a VFA experiment')
	parser.add_argument('fa_file', help='text file of nominal flip angles (FA) used to acquire the VFA images (FA in degrees; FA separated by spaces)')
	parser.add_argument('TR_file', help='text file storing the TR value used to acquire the VFA images as a single number (in ms)')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0VFA.nii" (T2*-weighted proton density, with receiver coil field bias); "_T1VFA.nii" (T1 map in ms); "_ExitVFA.nii" (exit code: 1 for successful fitting; -1 fail or warning, 0 background); "_SSEVFA.nii" (fitting sum of squared errors).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	parser.add_argument('--b1', metavar='<file>', help='Nifti file containing a B1 map for actual flip angle calculation; the actual and nominal flip angles in each voxels will be related as ACTUAL = B1 x NOMINAL')
	parser.add_argument('--algo', metavar='<type>', default='linear', help='fitting algorithm; choose among "linear" and "nonlinear" (default: "linear")')
	parser.add_argument('--ncpu', metavar='<N>', help='number of CPUs to be used for computation (default: half of available CPUs)')
	args = parser.parse_args()

	### Get input arguments
	vfafile = args.vfa_file
	fafile = args.fa_file
	tr = args.TR_file
	outroot = args.out_root
	maskfile = args.mask
	b1file = args.b1
	fittype = args.algo
	nprocess = args.ncpu

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for fitting has been provided
	    maskrequest = True
	else:
	    # A mask for fitting has not been provided
	    maskrequest = False

	if isinstance(b1file, str)==1:
	    # A B1 map has been provided
	    b1request = True
	else:
	    # A B1 map has not been provided
	    b1request = False

	if isinstance(nprocess, str)==1:
	    # A precise number of CPUs has been requested
	    nprocess = int(float(nprocess))
	else:
	    # No precise number of CPUs has been requested: use 50% of available CPUs
	    nprocess = multiprocessing.cpu_count()
	    nprocess = int(float(nprocess)/2)

	### Sort out things to print
	buffer_str=''
	seq_str = (outroot,'_T1VFA.nii')
	t1_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_S0VFA.nii')
	s0_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_ExitVFA.nii')
	exit_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_SSEVFA.nii')
	mse_out = buffer_str.join(seq_str)


	print('')
	print('********************************************************************')
	print('   Fitting of Variable Flip Angle signal model for T1 estimation    ')
	print('********************************************************************')
	print('')
	print('Called on 4D Nifti file: {}'.format(vfafile))
	print('Flip angle file: {}'.format(fafile))
	print('TR file: {}'.format(tr))
	print('Output files: {}, {}, {}, {}'.format(t1_out,s0_out,exit_out,mse_out))
	print('')


	### Call fitting routine
	# The entry point of the parallel pool has to be protected with if(__name__=='__main__') (for Windows): 
	if(__name__=='__main__'):
		if (maskrequest==False) and (b1request==False):
			T1FitVFA(vfafile, fafile, tr, outroot, fittype, nprocess)
		elif (maskrequest==True) and (b1request==False):
			T1FitVFA(vfafile, fafile, tr, outroot, fittype, nprocess, maskfile)
		elif (maskrequest==False) and (b1request==True):
			T1FitVFA(vfafile, fafile, tr, outroot, fittype, nprocess, b1map_nifti=b1file)
		elif (maskrequest==True) and (b1request==True):
			T1FitVFA(vfafile, fafile, tr, outroot, fittype, nprocess, maskfile, b1map_nifti=b1file)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)


