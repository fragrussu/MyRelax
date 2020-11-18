### Voxel-wise mono-exponential T2 or T2star fitting on multi-echo spin/gradient echo data
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


def MEsignal(mri_te,tissue_par):
	''' Generate the signal for a multi-echo experiment at fixed TR
		
		
	    INTERFACE
	    signal = MEsignal(mri_te,tissue_par)
	    
	    PARAMETERS
	    - mri_te: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T1-weighted proton density)
             		  tissue_par[1] = T2 or T2star (transvere relaxation time, in ms)
		
	    RETURNS
	    - signal: a numpy array of measurements generated according to a multi-echo signal model,
			
		         signal  =  S0 * exp(-TE/Txy) 
		
		      where TE is the echo time and where S0 and Txy are the tissue parameters (S0 is the T1-weighted proton 
                      density, and Txy is the transverse relaxation time, i.e. T2 or T2*).
		
		
	    Dependencies (Python packages): numpy
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Handle inputs
	te_values = np.array(mri_te,'float64')  # Make sure TE values are stored as a numpy array
	s0_value = tissue_par[0]         # S0
	txy_value = tissue_par[1]        # T2 or T2star

	### Calculate signal
	with np.errstate(divide='raise',invalid='raise'):
		try:
			signal = s0_value * np.exp((-1.0)*te_values/txy_value)
		except FloatingPointError:
			signal = 0.0 * te_values      # Just output zeros when txy_value is 0.0			

	### Output signal
	return signal
	

def MEFobj(tissue_par,mri_te,meas):
	''' Fitting objective function for exponential decay signal model		
		
	    INTERFACE
	    fobj = MEFobj(tissue_par,mri_te,meas)
	    
	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T1-weighted proton density)
             		  tissue_par[1] = T2 or T2star (transverse relaxation time, in ms)
	    - mri_te: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - meas: list/array of measurements
		
	    RETURNS
	    - fobj: objective function measured as sum of squared errors between measurements and predictions, i.e.
			
				 fobj = SUM_OVER_n( (prediction - measurement)^2 )
		
		     Above, the prediction are obtained using the multi-echo signal model implemented by function MEsignal().
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	
	### Predict signals given tissue and sequence parameters
	pred = MEsignal(mri_te,tissue_par)

	### Calculate objective function and return
	fobj = np.sum( (np.array(pred) - np.array(meas))**2 )
	return fobj


def MEGridSearch(mri_te,meas):
	''' Grid search for non-linear fitting of exponential decay signal models		
		
	    INTERFACE
	    tissue_estimate, fobj_grid = MEGridSearch(mri_te,meas)
	    
	    PARAMETERS
	    - mri_te: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - meas: list/array of measurements
		
	    RETURNS
	    - tissue_estimate: estimate of tissue parameters that explain the measurements reasonably well. The parameters are
			       estimated sampling the fitting objective function MEFobj() over a grid; the output is
                               tissue_estimate[0] = S0 (T1-weighted proton density)
             		       tissue_estimate[1] = T2 or T2star (transverse relaxation time, in ms)
	    - fobj_grid:       value of the objective function when the tissue parameters equal tissue_estimate
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Prepare grid for grid search
	txy_grid = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0])  # Grid of T2 or T2star values
	s0_grid = np.linspace(0.0,10*np.max(meas),num=24)    # Grid of S0 values: from 0 up to 10 times the maximum signal taken as input

	### Initialise objective function to infinity and parameters for grid search
	fobj_best = float('inf')
	s0_best = 0.0
	txy_best = 0.0
	
	### Run grid search
	for ii in range(0, len(txy_grid)):

		txy_ii =  txy_grid[ii]   		
		for jj in range(0, len(s0_grid)):

			s0_jj =  s0_grid[jj]
			params = np.array([s0_jj,txy_ii])
			
			# Objective function
			fval = MEFobj(params,mri_te,meas)

			# Check if objective function is smaller than previous value
			if fval<fobj_best:
				fobj_best = fval
				s0_best = s0_jj
				txy_best = txy_ii

	### Return output
	paramsgrid = np.array([s0_best, txy_best])
	fobjgrid = fobj_best
	return paramsgrid, fobjgrid



def TxyFitMEslice(data):
	''' Fit T2 or T2star for a multi-echo experiment on one MRI slice stored as a 2D numpy array  
	    

	    INTERFACE
	    data_out = T1TxyFitMEslice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
		            are the slice first and second dimensions, whereas the third dimension of data[0] stores
                            measurements obtained with different flip angles
		    data[1] is a numpy monodimensional array storing the TE values (ms) 
		    data[2] is a string describing the fitting algorithm ("linear" or "nonlinear", see TxyFitME())
		    data[3] is a 2D numpy array contaning the fitting mask within the MRI slice (see TxyFitME())
		    data[4] is a scalar containing the index of the MRI slice in the 3D volume
	    
	    RETURNS
	    - data_out: a list of 4 elements, such that
		    data_out[0] is the parameter S0 (see TxyFitME()) within the MRI slice
	            data_out[1] is the parameter T2 or T2star (see TxyFitME()) within the MRI slice
                    data_out[2] is the exit code of the fitting (see TxyFitME()) within the MRI slice
		    data_out[3] is the fitting sum of squared errors withint the MRI slice
                    data_out[4] equals data[4]
	
		    Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''

	
	### Extract signals and sequence information from the input list
	signal_slice = data[0]      # Signal
	te_value = data[1]          # TE values (in ms)
	fit_algo = data[2]          # fitting algorithm
	mask_slice = data[3]        # fitting mask
	idx_slice = data[4]         # Slice index
	slicesize = signal_slice.shape    # Get number of voxels of current MRI slice along each dimension
	te_value = np.array(te_value)     # Make sure the TE is an array
	
	### Check whether a sensible algorithm has been requested
	if fit_algo!="linear" and fit_algo!="nonlinear":
		print('')
		print('ERROR: unrecognised fitting algorithm. Exiting with 1.')
		print('')
		sys.exit(1)

	### Allocate output variables
	s0_slice = np.zeros(slicesize[0:2],'float64')
	txy_slice = np.zeros(slicesize[0:2],'float64')
	exit_slice = np.zeros(slicesize[0:2],'float64')
	mse_slice = np.zeros(slicesize[0:2],'float64')
	Nmeas = slicesize[2]   # Number of measurements


	### Fit monoexponential decay model in the voxels within the current slice
	for xx in range(0, slicesize[0]):
			for yy in range(0, slicesize[1]):
		
				# Get mask for current voxel
				mask_voxel = mask_slice[xx,yy]           # Fitting mask for current voxel

				# The voxel is not background: fit the signal model				
				if(mask_voxel==1):

					# Get signal and fitting mask
					sig_voxel = signal_slice[xx,yy,:]           # Extract signals for current voxel
					sig_voxel = np.array(sig_voxel)           # Convert to array
						
					## Simplest case: there are only two echo times --> get the solution analytically
					if(Nmeas==2):
						sig1 = sig_voxel[0]                            # Signal for first TE
						sig2 = sig_voxel[1] 		               # Signal for second TE
						te1 = te_value[0]                              # First TE
						te2 = te_value[1]                              # Second TE
						
						# Calculate maps analytically, handling warnings
						with np.errstate(divide='raise',invalid='raise'):	
							try:
								txy_voxel = ( te2 - te1 ) / np.log( sig1/sig2 )
								s0_voxel = sig1 / np.exp( (-1.0)*te1 / txy_voxel )
								exit_voxel = 1

								# Check whether the solution is plausible
								if txy_voxel<0:
									s0_voxel = np.mean(sig_voxel)
									txy_voxel = 1200.0    # We fix the maximum possible T2star to 1200
									exit_voxel = -1
								if s0_voxel<0:
									s0_voxel = 0.0
									exit_voxel = -1

								mse_voxel = MEFobj([s0_voxel,txy_voxel],te_value,sig_voxel)   # Error (0 when fitting provides txy > 0 ad s0 > 0 at the first attempt)
								
								
							except FloatingPointError:
								s0_voxel = 0.0
								txy_voxel = 0.0
								exit_voxel = -1
								mse_voxel = 0.0

					## General case: there are more than two echo times --> get the solution minimising an objective function
					else:

						# Perform linear fitting as first thing - if non-linear fitting is required, the linear fitting will be used to initialise the non-linear optimisation afterwards
						te_column = np.reshape(te_value,(Nmeas,1))    # Store TE values as a column array						
						sig_voxel_column = np.reshape(sig_voxel,(Nmeas,1))   # Reshape measurements as column array

						# Calculate linear regression coefficients as ( W * Q )^-1 * (W * m), while handling warnings
						with np.errstate(divide='raise',invalid='raise'):
							try:
								# Create matrices and arrays to be combinted via matrix multiplication
								Yvals = np.log(sig_voxel)           # Independent variable of linearised model
								Xvals = (-1.0)*te_column            # Dependent variable of linearised model
								allones = np.ones([Nmeas,1])        # Column of ones						
								Qmat = np.concatenate((allones,Xvals),axis=1)    # Design matrix Q
								Wmat = np.diag(sig_voxel)                        # Matrix of weights W
									
								# Calculate coefficients via matrix multiplication
								coeffs = np.matmul( np.linalg.pinv( np.matmul(Wmat,Qmat) ) , np.matmul(Wmat,Yvals) )
									
								# Retrieve signal model parameters from linear regression coefficients
								s0_voxel = np.exp(coeffs[0])
								txy_voxel = 1.0 / coeffs[1]
								exit_voxel = 1	
								
								# Check whether the solution is plausible: if not, declare fitting failed
								if txy_voxel<0:
									s0_voxel = np.mean(sig_voxel)
									txy_voxel = 1200.0    # We fix the maximum possible T2star to 1200
									exit_voxel = -1
								if s0_voxel<0:
									s0_voxel = 0.0
									exit_voxel = -1	

								mse_voxel = MEFobj([s0_voxel,txy_voxel],te_value,sig_voxel)   # Measure of quality of fit
								
								
							except FloatingPointError:
								s0_voxel = 0.0
								txy_voxel = 0.0
								exit_voxel = -1
								mse_voxel = 0.0
								
						
						# Refine the results from linear with non-linear optimisation if the selected algorithm is "nonlinear"
						if fit_algo=="nonlinear":

							# Check whether linear fitting has failed
							if exit_voxel==-1:
								param_init, fobj_init = MEGridSearch(te_value,sig_voxel)   # Linear fitting has failed: run a grid search
							else:
								param_init = [s0_voxel,txy_voxel]   # Linear fitting did not fail: use linear fitting output to initialise non-linear optimisation
								fobj_init = mse_voxel               
							
							# Minimise the objective function numerically
							param_bound = ((0,2*s0_voxel),(0,1200),)                      # Range for S0 and T2 or T2star (T2/T2star limited to be < 1800)						
							modelfit = minimize(MEFobj, param_init, method='L-BFGS-B', args=tuple([te_value,sig_voxel]), bounds=param_bound)
							fit_exit = modelfit.success
							fobj_fit = modelfit.fun

							# Get fitting output if non-linear optimisation was successful and if succeeded in providing a smaller value of the objective function as compared to the grid search
							if fit_exit==True and fobj_fit<fobj_init:
								param_fit = modelfit.x
								s0_voxel = param_fit[0]
								txy_voxel = param_fit[1]
								exit_voxel = 1
								mse_voxel = fobj_fit

							# Otherwise, output the best we could find with linear fitting or, when linear fitting fails, with grid search (note that grid search cannot fail by implementation)
							else:
								s0_voxel = param_init[0]
								txy_voxel = param_init[1]
								exit_voxel = -1
								mse_voxel = fobj_init
							
						
							
				# The voxel is background
				else:
					s0_voxel = 0.0
					txy_voxel = 0.0
					exit_voxel = 0
					mse_voxel = 0.0
				
				# Store fitting results for current voxel
				s0_slice[xx,yy] = s0_voxel
				txy_slice[xx,yy] = txy_voxel
				exit_slice[xx,yy] = exit_voxel
				mse_slice[xx,yy] = mse_voxel

	### Create output list storing the fitted parameters and then return
	data_out = [s0_slice, txy_slice, exit_slice, mse_slice, idx_slice]
	return data_out
	



def TxyFitME(*argv):
	''' Fit T2 or T2star for multi-echo experiment  
	    

	    INTERFACES
	    TxyFitME(me_nifti, te_text, output_basename, algo, ncpu)
	    TxyFitME(me_nifti, te_text, output_basename, algo, ncpu, mask_nifti)
	     
	    PARAMETERS
	    - me_nifti: path of a Nifti file storing the multi-echo data as 4D data.
	    - te_text: path of a text file storing the echo times (ms) used to acquire the data.
	    - output_basename: base name of output files. Output files will end in 
                            "_S0ME.nii"   --> T1-weighted proton density, with receiver coil field bias
		            "_TxyME.nii"  --> T2 or T2star map (ms)
			    "_ExitME.nii" --> exit code (1: successful fitting; 0 background; -1: unsuccessful fitting)
			    "_SSEME.nii"  --> fitting sum of squared errors
			    
			    Note that in the background and where fitting fails, S0, Txy and MSE are set to 0.0
			    Output files will be stored as double-precision floating point (FLOAT64)
			
	    - algo: fitting algorithm ("linear" or "nonlinear")
	    - ncpu: number of processors to be used for computation
	    - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the 
			  signal model needs to be fitted, and 0 otherwise
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Dependencies: numpy, nibabel, scipy (other than standard library)

	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Get input parametrs
	Nargv = len(argv)
	sig_nifti = argv[0]
	seq_text = argv[1]
	output_rootname = argv[2]
	algo = argv[3]
	ncpu = argv[4]
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

	### Load MRI data
	print('    ... loading input data')
	
	# Make sure MRI data exists
	try:
		sig_obj = nib.load(sig_nifti)
	except:
		print('')
		print('ERROR: the 4D input NIFTI file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(me_nifti))					 
		print('')
		sys.exit(1)
	
	# Get image dimensions and convert to float64
	sig_data = sig_obj.get_fdata()
	imgsize = sig_data.shape
	sig_data = np.array(sig_data,'float64')
	imgsize = np.array(imgsize)
	
	# Make sure that the text file with sequence parameters exists and makes sense
	try:
		seqarray = np.loadtxt(seq_text)
		seqarray = np.array(seqarray,'float64')
		seqarray_size = seqarray.size
	except:
		print('')
		print('ERROR: the echo time file {} does not exist or is not a numeric text file. Exiting with 1.'.format(seq_text))					 
		print('')
		sys.exit(1)
			
	# Check consistency of sequence parameter file and number of measurements
	if imgsize.size!=4:
		print('')
		print('ERROR: the input file {} is not a 4D nifti. Exiting with 1.'.format(sig_nifti))					 
		print('')
		sys.exit(1)
	if seqarray_size!=imgsize[3]:
		print('')
		print('ERROR: the number of measurements in {} does not match the number of echo times in {}. Exiting with 1.'.format(sig_nifti,seq_text))					 
		print('')
		sys.exit(1)
	seq = seqarray

	### Deal with optional arguments: mask
	if Nargv==6:
		mask_nifti = argv[5]
		try:
			mask_obj = nib.load(mask_nifti)
		except:
			print('')
			print('ERROR: the mask file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mask_nifti))					 
			print('')
			sys.exit(1)
		
		# Make sure that the mask has header information that is consistent with the input data containing the VFA measurements
		sig_header = sig_obj.header
		sig_affine = sig_header.get_best_affine()
		sig_dims = sig_obj.shape
		mask_dims = mask_obj.shape		
		mask_header = mask_obj.header
		mask_affine = mask_header.get_best_affine()			
		# Make sure the mask is a 3D file
		mask_data = mask_obj.get_fdata()
		masksize = mask_data.shape
		masksize = np.array(masksize)
		if masksize.size!=3:
			print('')
			print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(sig_affine==mask_affine)!=16) or (sig_dims[0]!=mask_dims[0]) or (sig_dims[1]!=mask_dims[1]) or (sig_dims[2]!=mask_dims[2]) ):
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
	

	### Allocate memory for outputs
	s0_data = np.zeros(imgsize[0:3],'float64')	       # T1-weighted proton density with receiver field bias (double-precision floating point)
	txy_data = np.zeros(imgsize[0:3],'float64')	       # T1 (double-precision floating point)
	exit_data = np.zeros(imgsize[0:3],'float64')           # Exit code (double-precision floating point)
	mse_data = np.zeros(imgsize[0:3],'float64')            # Fitting sum of squared errors (MSE) (double-precision floating point)

	#### Fitting
	print('    ... transverse relaxation time estimation')
	# Create the list of input data
	inputlist = [] 
	for zz in range(0, imgsize[2]):
		sliceinfo = [sig_data[:,:,zz,:],seq,algo,mask_data[:,:,zz],zz]  # List of information relative to the zz-th MRI slice
		inputlist.append(sliceinfo)     # Append each slice list and create a longer list of MRI slices whose processing will run in parallel

	# Clear some memory
	del sig_data, mask_data 
	
	# Call a pool of workers to run the fitting in parallel if parallel processing is required (and if the the number of slices is > 1)
	if ncpu>1 and imgsize[2]>1:

		# Create the parallel pool and give jobs to the workers
		fitpool = multiprocessing.Pool(processes=ncpu)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get initial process identifications (PIDs)
		fitresults = fitpool.map_async(TxyFitMEslice,inputlist)      # Give jobs to the parallel processes
		
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
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of mono-exponential decay model
			txy_data[:,:,slicepos] = fitslice[1]   # Parameter T2 or T2star of mono-exponential decay model
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors	


	# Run serial fitting as no parallel processing is required (it can take up to 1 hour per brain)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = TxyFitMEslice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of VFA model
			txy_data[:,:,slicepos] = fitslice[1]   # Parameter T2 or T2star of mono-exponential decay model
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors


	### Save the output maps
	print('    ... saving output files')
	buffer_string=''
	seq_string = (output_rootname,'_TxyME.nii')
	txy_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_S0ME.nii')
	s0_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_ExitME.nii')
	exit_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_SSEME.nii')
	mse_outfile = buffer_string.join(seq_string)
	buffer_header = sig_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type
	txy_obj = nib.Nifti1Image(txy_data,sig_obj.affine,buffer_header)
	nib.save(txy_obj, txy_outfile)
	s0_obj = nib.Nifti1Image(s0_data,sig_obj.affine,buffer_header)
	nib.save(s0_obj, s0_outfile)
	exit_obj = nib.Nifti1Image(exit_data,sig_obj.affine,buffer_header)
	nib.save(exit_obj, exit_outfile)
	mse_obj = nib.Nifti1Image(mse_data,sig_obj.affine,buffer_header)
	nib.save(mse_obj, mse_outfile)

	### Done
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise fitting of T2 or T2star from multi-echo MRI magnitude data already corrected for motion. Dependencies (Python packages): numpy, nibabel, scipy (other than standard library). References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('me_file', help='4D Nifti file of multi-echo magnitude images from a spin echo or gradient echo experiment')
	parser.add_argument('te_file', help='text file of echo times (TEs) used to acquire the images (TEs in ms; TEs separated by spaces)')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0ME.nii" (T1-weighted proton density, with receiver coil field bias); "_TxyME.nii" (T2 or T2star map in ms); "_ExitME.nii" (exit code: 1 for successful non-linear fitting; 0 background; -1 for failing of non-linear fitting, with results from a grid search/linear fitting provided instead); "_SSEME.nii" (fitting sum of squared errors).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	parser.add_argument('--algo', metavar='<type>', default='linear', help='fitting algorithm; choose among "linear" and "nonlinear" (default: "linear")')
	parser.add_argument('--ncpu', metavar='<N>', help='number of CPUs to be used for computation (default: half of available CPUs)')
	args = parser.parse_args()

	### Get input arguments
	sigfile = args.me_file
	seqfile = args.te_file
	outroot = args.out_root
	maskfile = args.mask
	fittype = args.algo
	nprocess = args.ncpu

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for fitting has been provided
	    maskrequest = True
	else:
	    # A mask for fitting has not been provided
	    maskrequest = False
	
	if isinstance(nprocess, str)==1:
	    # A precise number of CPUs has been requested
	    nprocess = int(float(nprocess))
	else:
	    # No precise number of CPUs has been requested: use 50% of available CPUs
	    nprocess = multiprocessing.cpu_count()
	    nprocess = int(float(nprocess)/2)

	### Sort out things to print
	buffer_str=''
	seq_str = (outroot,'_TxyME.nii')
	txy_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_S0ME.nii')
	s0_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_ExitME.nii')
	exit_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_SSEME.nii')
	mse_out = buffer_str.join(seq_str)


	print('')
	print('********************************************************************************')
	print('   Fitting of mono-exponetial signal decay model for T2 or T2star estimation    ')
	print('********************************************************************************')
	print('')
	print('Called on 4D Nifti file: {}'.format(sigfile))
	print('Echo time file: {}'.format(seqfile))
	print('Output files: {}, {}, {}, {}'.format(txy_out,s0_out,exit_out,mse_out))
	print('')


	### Call fitting routine
	# The entry point of the parallel pool has to be protected with if(__name__=='__main__') (for Windows): 
	if(__name__=='__main__'):
		if (maskrequest==False):
			TxyFitME(sigfile, seqfile, outroot, fittype, nprocess)
		else:
			TxyFitME(sigfile, seqfile, outroot, fittype, nprocess, maskfile)
	
	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)


