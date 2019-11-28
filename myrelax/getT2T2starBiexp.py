### Voxel-wise bi-exponential T2 or T2star fitting on multi-echo spin/gradient echo data
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


def signal(mri_par,tissue_par):
	''' Generate the signal for an inversion recovery experiment at fixed TR
		
		
	    INTERFACE
	    signal = signal(mri_par,mri_par3,tissue_par)
	    
	    PARAMETERS
	    - mri_par: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T1-weighted proton density)
             		  tissue_par[1] = Txya (transverse longitudinal relaxation time of slowly-decaying component, in ms)
			  tissue_par[2] = va (signal fraction of the slow-decaying component, between 0 and 1)
			  tissue_par[3] = Txyb (transverse longitudinal relaxation time of fast-decaying component, in ms, such that Txyb <= Txya) 
		
	    RETURNS
	    - signal: a numpy array of measurements generated according to a bi-exponential multi-echo signal model,
			
		         signal  =  S0 * va * exp(-TE/Txya)  +   S0 * (1 - va) * exp(-TE/Txyb)
		
		      where TE is the echo time, where S0 is the signal for zero TE, Txya and va are the T2 or T2star and signal fraction of the slowly-decaying water, and Txyb <= Txya is the T2 or T2star of the fast-decaying water .
		
		
	    Dependencies (Python packages): numpy, nibabel, scipy (other than standard library)
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Handle inputs
	time_values = np.array(mri_par,'float64')  # Make sure sequence values are stored as a numpy array
	s0_value = tissue_par[0]        # S0
	ta_value = tissue_par[1]        # Txya
	va_value = tissue_par[2]        # va
	tb_value = tissue_par[3]        # Txyb

	### Calculate signal
	with np.errstate(divide='raise',invalid='raise'):
		try:
			signal = s0_value * va_value * np.exp(-time_values/ta_value) + s0_value * (1 - va_value) * np.exp(-time_values/tb_value)
		except FloatingPointError:
			signal = 0.0 * np.exp(-time_values)   # Output zeros when T2/T2star are 0		

	### Output signal
	return signal
	

def Fobj(tissue_par,mri_par,meas,wtikh):
	''' Fitting objective function for bi-exponential T2 or T2star (Txy) decay 		
		 
	    INTERFACE
	    fobj = Fobj(tissue_par,mri_par,meas,wtikh)
	    
	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T1-weighted proton density)
             		  tissue_par[1] = Txya (relaxation time of slowly-decaying water, in ms)
             		  tissue_par[2] = va (signal fracto slowly-decaying water, in [0;1])
             		  tissue_par[3] = kb (proportionality factor decsribing the relaxation time of slowly-decaying water Txyb, s.t. Txyb = kb*Txya, with kb in [0; 1])
	    - mri_par: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - meas: list/array of measurements
            - wtikh: Tikhonov regularisation weight
		
	    RETURNS
	    - fobj: objective function measured as sum of squared errors between measurements and predictions, i.e.
			
				 fobj = SUM_OVER_n( (prediction(parameters) - measurement)^2 + wtikh*L2norm(parameters) )
		
		     Above, the prediction are obtained using the signal model implemented by function signal().
	    
		     References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	
	### Predict signals given tissue and sequence parameters
	pred = signal(mri_par,np.array([tissue_par[0],tissue_par[1],tissue_par[2],tissue_par[3]*tissue_par[1]]))

	### Calculate objective function and return
	fobj = np.sum( (np.array(pred) - np.array(meas))**2 )    # Data fidelity term
	fobj = fobj + wtikh*( tissue_par[0]*tissue_par[0] +  tissue_par[1]*tissue_par[1] +  tissue_par[2]*tissue_par[2] + tissue_par[3]*tissue_par[1]*tissue_par[3]*tissue_par[1] ) # Regularisation term
	return fobj



def GridSearch(mri_par,meas,regw):
	''' Grid search for non-linear fitting of bi-exponential decay signal models		
		
	    INTERFACE
	    tissue_estimate, fobj_grid = GridSearch(mri_par,meas,regw)
	    
	    PARAMETERS
	    - mri_par: list/array indicating the TEs (echo times, in ms) used for the experiment (one measurement per TE)
	    - meas: list/array of measurements
	    - regw: Tikhonov regularisation weight
		
	    RETURNS
	    - tissue_estimate: estimate of tissue parameters that explain the measurements reasonably well. The parameters are
			       estimated sampling the fitting objective function Fobj() over a grid; the output is
                               tissue_estimate[0] = S0 (T1-weighted proton density)
             		       tissue_estimate[1] = Txya (transverse relaxation time of slowly-decaying component, in ms)
             		       tissue_estimate[2] = va (signal fraction of slowly-decaying component, in ms)
             		       tissue_estimate[3] = kb (proportionality factor describing the transverse relaxation time of the fast-decaying component, defined in [0;1], s.t. Txyb = kb*Txya)
	    - fobj_grid:       value of the objective function when the tissue parameters equal tissue_estimate
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Prepare grid for grid search
	time_grid = np.concatenate( (np.linspace(20.0,120.0,7), np.logspace(np.log(180.0),np.log(1400.0),6, base=np.exp(1.0)) ) )
	v_grid = np.linspace(0.0,1.0,num=5)  # Grid for va values 
	k_grid = np.linspace(0.0,1.0,num=5)  # Grid for kb value, such that Txyb = kb*Txya
	s0_grid = np.linspace(0.0,6*np.max(meas),num=5)    # Grid of S0 values: from 0 up to a multiple times the maximum signal taken as input

	### Initialise objective function to infinity and parameters for grid search
	fobj_best = float('inf')
	s0_best = 0.0
	time_best = 0.0
	k_best = 0.0
	v_best = 0.0
	
	### Run grid search
	for ii in range(0, len(time_grid)):

		time_ii =  time_grid[ii]   		
		for jj in range(0, len(s0_grid)):

			s0_jj =  s0_grid[jj]
			for mm in range(0, len(v_grid)):

				v_mm = v_grid[mm]
				for nn in range(0, len(k_grid)):

					k_nn = k_grid[nn]
				
					# Tissue parameters
					params = np.array([s0_jj,time_ii,v_mm,k_nn])
			
					# Objective function
					fval = Fobj(params,mri_par,meas,regw)

					# Check if objective function has decreased
					if fval<fobj_best:
						fobj_best = fval
						s0_best = s0_jj
						time_best = time_ii
						k_best = k_nn
						v_best = v_mm

	### Return output
	paramsgrid = np.array([s0_best, time_best, v_best, k_best])
	fobjgrid = fobj_best
	return paramsgrid, fobjgrid



def FitSlice(data):
	''' Fit bi-exponential T2 or T2star decay on one MRI slice stored as a 2D numpy array  
	    

	    INTERFACE
	    data_out = FitSlice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0] are the 			            slice first and second dimensions, whereas the third dimension of data[0] stores measurements
		    data[1] is a numpy monodimensional array storing the TE values (ms)
		    data[2] is a 2D numpy array contaning the fitting mask within the MRI slice (see FitVox())
		    data[3] is a scalar containing the index of the MRI slice in the 3D volume
		    data[4] is a the weight for Tikhonov regularisation
	    
	    RETURNS
	    - data_out: a list of 4 elements, such that
		    data_out[0] is the parameter S0 (see FitVox()) within the MRI slice
	            data_out[1] is the parameter Txya (see FitVox()) within the MRI slice
	            data_out[2] is the parameter va (see FitVox()) within the MRI slice
	            data_out[3] is the parameter Txyb (see FitVox()) within the MRI slice
                    data_out[4] is the exit code of the fitting (see FitVox()) within the MRI slice
		    data_out[5] is the fitting sum of squared errors within the MRI slice
                    data_out[6] equals data[3]
	
		    Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
	    
	     References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''


	
	### Extract signals and sequence information from the input list
	signal_slice = data[0]      # Signal
	time_value = data[1]        # TE values (in ms)
	mask_slice = data[2]        # fitting mask
	idx_slice = data[3]         # Slice index
	tikh_w = data[4]            # Tikhonov regularisation weight
	slicesize = signal_slice.shape    # Get number of voxels of current MRI slice along each dimension
	time_value = np.array(time_value)     # Make sure the TE is an array

	### Allocate output variables
	s0_slice = np.zeros(slicesize[0:2],'float64')
	ta_slice = np.zeros(slicesize[0:2],'float64')
	va_slice = np.zeros(slicesize[0:2],'float64')
	tb_slice = np.zeros(slicesize[0:2],'float64')
	exit_slice = np.zeros(slicesize[0:2],'float64')
	mse_slice = np.zeros(slicesize[0:2],'float64')
	Nmeas = slicesize[2]   # Number of measurements


	### Fit bi-exponential T2 or T2star decay model in the voxels within the current slice
	for xx in range(0, slicesize[0]):
			for yy in range(0, slicesize[1]):
		
				# Get mask for current voxel
				mask_voxel = mask_slice[xx,yy]           # Fitting mask for current voxel

				# The voxel is not background: fit the signal model				
				if(mask_voxel==1):

					# Get signal and fitting mask
					sig_voxel = signal_slice[xx,yy,:]          # Extract signals for current voxel
					sig_voxel = np.array(sig_voxel)           # Convert to array
						
					# Grid search
					param_init, fobj_init = GridSearch(time_value,sig_voxel,tikh_w)   # Grid search (no regularisation used in in grid search
					             						
					# Minimise the objective function numerically
					param_bound = ((0.0,5.0*param_init[0]),(0.0,3250.0),(0.0,1.0),(0.0,1.0),)  # Range for S0, Txya, va, ka s.t. Txyb = kb*Txya		
					modelfit = minimize(Fobj, param_init, method='L-BFGS-B', args=tuple([time_value,sig_voxel,tikh_w]), bounds=param_bound)
					fit_exit = modelfit.success
					fobj_fit = modelfit.fun

					# Non-linear optimisation was successful and provided a smaller objective function as compared to the grid search
					if fit_exit==True and fobj_fit<fobj_init:
						param_fit = modelfit.x
						s0_voxel = param_fit[0]
						ta_voxel = param_fit[1]
						va_voxel = param_fit[2]
						tb_voxel = param_fit[3]*param_fit[1]
						exit_voxel = 1
						mse_voxel = fobj_fit
						
					# Grid search provided a better minimisation of objective function (non-linear fitting probably stuck in local minimum)
					else:
						s0_voxel = param_init[0]
						ta_voxel = param_init[1]
						va_voxel = param_init[2]
						tb_voxel = param_init[3]*param_init[1]
						exit_voxel = -1
						mse_voxel = fobj_init
							
						
							
				# The voxel is background
				else:
					s0_voxel = 0.0
					ta_voxel = 0.0
					va_voxel = 0.0
					tb_voxel = 0.0
					exit_voxel = 0
					mse_voxel = 0.0
				
				# Store fitting results for current voxel
				s0_slice[xx,yy] = s0_voxel
				ta_slice[xx,yy] = ta_voxel
				va_slice[xx,yy] = va_voxel
				tb_slice[xx,yy] = tb_voxel
				exit_slice[xx,yy] = exit_voxel
				mse_slice[xx,yy] = mse_voxel

	### Create output list storing the fitted parameters and then return
	data_out = [s0_slice, ta_slice, va_slice, tb_slice, exit_slice, mse_slice, idx_slice]
	return data_out
	



def FitVox(*argv):
	''' Fit bo-exponential T2 or T2star decay from multi-echo data
	    

	    INTERFACES
	    FitVox(input_nifti, te_text, output_basename, ncpu)
	    FitVox(input_nifti, te_text, output_basename, ncpu, mask_nifti)
	     
	    PARAMETERS
	    - input_nifti: path of a Nifti file storing the multi-echo time data as 4D data.
	    - te_text: path of a text file storing the echo times (ms) used to acquire the data.
	    - output_basename: base name of output files. Output files will end in 
                            "_S0.nii"   --> T1-weighted proton density, with receiver coil field bias
		            "_Txylong.nii" --> T2 or T2star map (ms) of the slowly decaying water (long T2 or T2star)
		            "_Vlong.nii"   --> signal fraction of the slowly decaying water
		            "_Txyshort.nii" --> T2 or T2star map (ms) of the fast decaying water (short T2 or T2star)
			    "_ExitME.nii" --> exit code (1: successful fitting; 0 background)
			    "_SSEME.nii"  --> fitting sum of squared errors
			    
			    Note that in the background and where fitting fails, S0, Txya, Va, Txyb and SSE are 0.0
			    Output files will be stored as double-precision floating point (FLOAT64)

	    - ncpu: number of processors to be used for computation
	    - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the 
			  signal model needs to be fitted, and 0 otherwise
	    
	     References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Get input parametrs
	Nargv = len(argv)
	sig_nifti = argv[0]
	seq_text = argv[1]
	output_rootname = argv[2]
	ncpu = argv[3]
	ncpu_physical = multiprocessing.cpu_count()
	if ncpu>ncpu_physical:
		print('')
		print('WARNING: {} CPUs were requested. Using {} instead (all available CPUs)...'.format(ncpu,ncpu_physical))					 
		print('')
		ncpu = ncpu_physical - 1     # Do not open more workers than the physical number of CPUs - 1 (one will be used to wait for the results)
		if(ncpu==0):
			ncpu = 1      # Worst case when the system has only one CPU
	lreg = argv[4]   # Regularisation weight
	if(lreg<0):
		print('')
		print('WARNING: the Tikhonov regularisation weight must be grater or equal to 0, while it is set to {}. No regularisation performed...'.format(lreg))				 
		print('')
		lreg = 0.0
		


	### Load data
	print('    ... loading input data')
	
	# Make sure data exists
	try:
		sig_obj = nib.load(sig_nifti)
	except:
		print('')
		print('ERROR: the 4D input NIFTI file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(input_nifti))
		print('')
		sys.exit(1)
	
	# Get image dimensions and convert to float64
	sig_data = sig_obj.get_data()
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
		print('ERROR: the number of measurements in {} does not match the number of inversion times in {}. Exiting with 1.'.format(sig_nifti,seq_text))	
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
		
		# Make sure that the mask has header information that is consistent with the input data containing the measurements
		sig_header = sig_obj.header
		sig_affine = sig_header.get_best_affine()
		sig_dims = sig_obj.shape
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
	ta_data = np.zeros(imgsize[0:3],'float64')	       # T2(star) of slowly decaying water (double-precision floating point)
	va_data = np.zeros(imgsize[0:3],'float64')	       # signal fraction of slowly decaying water (double-precision floating point)
	tb_data = np.zeros(imgsize[0:3],'float64')	       # T2(star) of fast decaying water (double-precision floating point)
	exit_data = np.zeros(imgsize[0:3],'float64')           # Exit code (double-precision floating point)
	mse_data = np.zeros(imgsize[0:3],'float64')            # Fitting sum of squared errors (double-precision floating point)

	#### Fitting
	print('    ... bi-exponential T2 or T2star fitting')
	# Create the list of input data
	inputlist = [] 
	for zz in range(0, imgsize[2]):
		sliceinfo = [sig_data[:,:,zz,:],seq,mask_data[:,:,zz],zz,lreg]  # List of information relative to the zz-th MRI slice
		inputlist.append(sliceinfo)     # Append each slice list and create a longer list of MRI slices whose processing will run in parallel

	# Clear some memory
	del sig_data, mask_data 
	
	# Call a pool of workers to run the fitting in parallel if parallel processing is required (and if the the number of slices is > 1)
	if ncpu>1 and imgsize[2]>1:

		# Create the parallel pool and give jobs to the workers
		fitpool = multiprocessing.Pool(processes=ncpu)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get initial process identifications (PIDs)
		fitresults = fitpool.map_async(FitSlice,inputlist)      # Give jobs to the parallel processes
		
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
			slicepos = fitslice[6]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of mono-exponential model
			ta_data[:,:,slicepos] = fitslice[1]     # Relaxation constant 1
			va_data[:,:,slicepos] = fitslice[2]     # Signal fraction
			tb_data[:,:,slicepos] = fitslice[3]     # Relaxation constant 1
			exit_data[:,:,slicepos] = fitslice[4]  # Exit code
			mse_data[:,:,slicepos] = fitslice[5]   # Sum of Squared Errors	


	# Run serial fitting as no parallel processing is required (it can take up to 1 hour per brain)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = FitSlice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[6]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of mono-exponential model
			ta_data[:,:,slicepos] = fitslice[1]     # Relaxation constant 1
			va_data[:,:,slicepos] = fitslice[2]     # Signal fraction
			tb_data[:,:,slicepos] = fitslice[3]     # Relaxation constant 1
			exit_data[:,:,slicepos] = fitslice[4]  # Exit code
			mse_data[:,:,slicepos] = fitslice[5]   # Sum of Squared Errors


	### Save the output maps
	print('    ... saving output files')

	buffer_string=''
	seq_string = (output_rootname,'_Txylong.nii')
	ta_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (output_rootname,'_Vlong.nii')
	va_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (output_rootname,'_Txyshort.nii')
	tb_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (output_rootname,'_S0.nii')
	s0_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (output_rootname,'_ExitME.nii')
	exit_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (output_rootname,'_SSEME.nii')
	mse_outfile = buffer_string.join(seq_string)

	buffer_header = sig_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type

	ta_obj = nib.Nifti1Image(ta_data,sig_obj.affine,buffer_header)
	nib.save(ta_obj, ta_outfile)

	va_obj = nib.Nifti1Image(va_data,sig_obj.affine,buffer_header)
	nib.save(va_obj, va_outfile)

	tb_obj = nib.Nifti1Image(tb_data,sig_obj.affine,buffer_header)
	nib.save(tb_obj, tb_outfile)

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
	parser = argparse.ArgumentParser(description='Voxel-wise fitting of bi-exponential T2 or T2star multi-echo spin echo/gradient echo decay from  MRI magnitude data already corrected for motion. Dependencies (Python packages): numpy, nibabel, scipy (other than standard library). References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('me_file', help='4D Nifti file of multi-echo magnitude images from a spin echo or gradient echo experiment')
	parser.add_argument('te_file', help='text file of echo times (TEs) used to acquire the images (TEs in ms; TEs separated by spaces)')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0.nii" (T1-weighted proton density, with receiver coil field bias); "_Txylong.nii" (T2 or T2star map in ms of slowly decaying water, i.e. long T2 or T2star); "_Vlong.nii" (signal fraction of slowly decaying water, between 0 and 1); "_Txyshort.nii" (T2 or T2star map in ms of fast decaying water, s.t. Txyshort <= Txylong, i.e. short T2 or T2star); "_ExitME.nii" (exit code: 1 for successful non-linear fitting; 0 background; -1 for failing of non-linear fitting, with results from a grid search provided instead); "_SSEME.nii" (fitting sum of squared errors).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	parser.add_argument('--ncpu', metavar='<N>', help='number of CPUs to be used for computation (default: half of available CPUs)')
	parser.add_argument('--reg', metavar='<N>', default='0.0', help='weight of Tikhonov regularisation (default: 0.0, i.e. no regularisation)')
	args = parser.parse_args()

	### Get input arguments
	sigfile = args.me_file
	seqfile = args.te_file
	outroot = args.out_root
	maskfile = args.mask
	nprocess = args.ncpu
	regval = float(args.reg)

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
	seq_str = (outroot,'_Txylong.nii')
	ta_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_Vlong.nii')
	va_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_Txyshort.nii')
	tb_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_S0.nii')
	s0_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_ExitME.nii')
	exit_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_SSEME.nii')
	mse_out = buffer_str.join(seq_str)


	print('')
	print('********************************************************************************')
	print('                  Fitting of bi-exponetial T2 or T2star decay                   ')
	print('********************************************************************************')
	print('')
	print('** Called on 4D Nifti file: {}'.format(sigfile))
	print('** Echo time file: {}'.format(seqfile))
	print('** Tikhonov regularisation weight: {}'.format(args.reg))
	print('** Output files: {}, {}, {}, {}, {}, {}'.format(ta_out,va_out,tb_out,s0_out,exit_out,mse_out))
	print('')


	### Call fitting routine
	# The entry point of the parallel pool has to be protected with if(__name__=='__main__') (for Windows): 
	if(__name__=='__main__'):
		if (maskrequest==False):
			FitVox(sigfile, seqfile, outroot, nprocess,regval)
		else:
			FitVox(sigfile, seqfile, outroot, nprocess, regval, maskfile)
	
	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)


