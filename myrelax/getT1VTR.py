### Voxel-wise mono-exponential T1 fitting on spin echo data at variable TR
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
	''' Generate the signal for a spin echo experiment at variable TR
		
		
	    INTERFACE
	    signal = MEsignal(mri_tr,tissue_par)
	    
	    PARAMETERS
	    - mri_tr: list/array indicating the TRs (repetition times, in ms) used for the experiment (one measurement per TR).
	              For the most accurate modelling, these should be "corrected" TR values (TR - TE/2)
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2-weighted proton density)
             		 tissue_par[1] = T1 (longitudinal relaxation time, in ms)
		
	    RETURNS
	    - signal: a numpy array of measurements generated according to a multi-echo signal model,
			
		         signal  =  S0 * (1 - exp(-TR/T1) ) 
		
		      where TR is the repetition time and where S0 and T1 are the tissue parameters (S0 is the T2-weighted proton 
                      density, and T1 is the longitudinal relaxation time).
		
		
	    Dependencies (Python packages): numpy
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Handle inputs
	te_values = np.array(mri_te,'float64')  # Make sure TR values are stored as a numpy array
	s0_value = tissue_par[0]         # S0
	txy_value = tissue_par[1]        # T1

	### Calculate signal
	with np.errstate(divide='raise',invalid='raise'):
		try:
			signal = s0_value * (1.0 - np.exp((-1.0)*te_values/txy_value) )
		except FloatingPointError:
			signal = 0.0 * te_values      # Just output zeros when txy_value is 0.0			

	### Output signal
	return signal
	

def MEFobj(tissue_par,mri_te,meas):
	''' Fitting objective function for exponential decay signal model		
		
	    INTERFACE
	    fobj = MEFobj(tissue_par,mri_tr,meas)
	    
	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2-weighted proton density)
             		 tissue_par[1] = T1 (longitudinal relaxation time, in ms)
	    - mri_tr: list/array indicating the TRs (repetition times, in ms) used for the experiment (one measurement per TR).
	              For the most accurate modelling, these should be "corrected" TR values (TR - TE/2)
	    - meas: list/array of measurements
		
	    RETURNS
	    - fobj: objective function measured as sum of squared errors between measurements and predictions, i.e.
			
				 fobj = SUM_OVER_n( (prediction - measurement)^2 )
		
		     Above, the prediction are obtained using the signal model implemented by function MEsignal().
	    
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
	    - mri_te: list/array indicating the TRs (repetition times, in ms) used for the experiment (one measurement per TR).
	              For the most accurate modelling, these should be "corrected" TR values (TR - TE/2)
	    - meas: list/array of measurements
		
	    RETURNS
	    - tissue_estimate: estimate of tissue parameters that explain the measurements reasonably well. The parameters are
			       estimated sampling the fitting objective function MEFobj() over a grid; the output is
                               tissue_estimate[0] = S0 (T2-weighted proton density)
             		       tissue_estimate[1] = T1 (transverse relaxation time, in ms)
	    - fobj_grid:       value of the objective function when the tissue parameters equal tissue_estimate
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Prepare grid for grid search
	txy_grid = np.linspace(100.0,4500.0,num=12)       # Grid of T1 values
	s0_grid = np.linspace(0.0,10*np.max(meas),num=12)    # Grid of S0 values: from 0 up to 10 times the maximum signal taken as input

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
	''' Fit T1 for a variable TR experiment on one MRI slice stored as a 2D numpy array  
	    

	    INTERFACE
	    data_out = TxyFitMEslice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
		            are the slice first and second dimensions, whereas the third dimension of data[0] stores
                            measurements obtained with different repetition times
		    data[1] is a numpy monodimensional array storing the TR values (ms) 
		            (for the most accurate modelling, these should be "corrected" TR values (TR - TE/2))
		    data[2] is a 2D numpy array contaning the fitting mask within the MRI slice (see TxyFitME())
		    data[3] is a scalar containing the index of the MRI slice in the 3D volume
	    
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
### fit_algo
	
	### Extract signals and sequence information from the input list
	signal_slice = data[0]      # Signal
	te_value = data[1]          # TR values (in ms)
	mask_slice = data[2]        # fitting mask
	idx_slice = data[3]         # Slice index
	slicesize = signal_slice.shape    # Get number of voxels of current MRI slice along each dimension
	te_value = np.array(te_value)     # Make sure the TR is an array
	
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

					# Grid search
					param_init, fobj_init = MEGridSearch(te_value,sig_voxel)          
							
					# Minimise the objective function numerically
					param_bound = ((0,2*param_init[0]),(0,4500),)                      # Range for S0 and T1
					modelfit = minimize(MEFobj, param_init, method='L-BFGS-B', args=tuple([te_value,sig_voxel]), bounds=param_bound)
					fit_exit = modelfit.success
					fobj_fit = modelfit.fun

					# Get fitting output if non-linear optimisation was successful and if succeeded in providing a smaller value of the objective function as compared to the grid search
					if fit_exit==True:
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
	''' Fit T1 for multi-TR spin echo experiment  
	    

	    INTERFACES
	    TxyFitME(nifti, text, output_basename, ncpu, mask_nifti)
	     
	    PARAMETERS
	    - nifti: path of a Nifti file storing the variable TR data as 4D data.
	    - text: path of a text file storing the repetition times (ms) used to acquire the data. For the most
	            accurate T1 estimation, this should contain a list of "corrected" TR values (i.e., TR - TE/2)
	    - output_basename: base name of output files. Output files will end in 
                            "_S0ME.nii"   --> T2-weighted proton density, with receiver coil field bias
		            "_T1ME.nii"   --> T1 (ms)
			    "_ExitME.nii" --> exit code (1: successful fitting; 0 background; -1: unsuccessful fitting)
			    "_SSEME.nii"  --> fitting sum of squared errors
			    
			    Note that in the background and where fitting fails, S0, Txy and MSE are set to 0.0
			    Output files will be stored as double-precision floating point (FLOAT64)
			
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
	ncpu = argv[3]
	ncpu_physical = multiprocessing.cpu_count()
	if ncpu>ncpu_physical:
		print('')
		print('WARNING: {} CPUs were requested. Using {} instead (all available CPUs)...'.format(ncpu,ncpu_physical))					 
		print('')
		ncpu = ncpu_physical     # Do not open more workers than the physical number of CPUs


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
		print('ERROR: the repetition time file {} does not exist or is not a numeric text file. Exiting with 1.'.format(seq_text))					 
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
		print('ERROR: the number of measurements in {} does not match the number of repetition times in {}. Exiting with 1.'.format(sig_nifti,seq_text))					 
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
		sliceinfo = [sig_data[:,:,zz,:],seq,mask_data[:,:,zz],zz]  # List of information relative to the zz-th MRI slice
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
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0
			txy_data[:,:,slicepos] = fitslice[1]   # Parameter T1
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors	


	# Run serial fitting as no parallel processing is required (it can take up to 1 hour per brain)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = TxyFitMEslice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0
			txy_data[:,:,slicepos] = fitslice[1]   # Parameter T1
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors


	### Save the output maps
	print('    ... saving output files')
	buffer_string=''
	seq_string = (output_rootname,'_T1ME.nii')
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
	parser = argparse.ArgumentParser(description='Voxel-wise fitting of T1 from variable TR spin echo MRI magnitude data already corrected for motion. Dependencies (Python packages): numpy, nibabel, scipy (other than standard library). References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('vtr_file', help='4D Nifti file of multi-echo magnitude images from a spin echo or gradient echo experiment')
	parser.add_argument('tr_file', help='text file of repetition times (TEs) used to acquire the images (TRs in ms; TRs separated by spaces. For the most accurate T1 estimation, this should contain a list of "corrected" TR values (i.e., TR - TE/2))')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0ME.nii" (T2-weighted proton density, with receiver coil field bias); "_T1ME.nii" (T1 map in ms); "_ExitME.nii" (exit code: 1 for successful non-linear fitting; 0 background; -1 for failing of non-linear fitting, with results from grid searchfitting provided instead); "_SSEME.nii" (fitting sum of squared errors).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	parser.add_argument('--ncpu', metavar='<N>', help='number of CPUs to be used for computation (default: half of available CPUs)')
	args = parser.parse_args()

	### Get input arguments
	sigfile = args.vtr_file
	seqfile = args.tr_file
	outroot = args.out_root
	maskfile = args.mask
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
	seq_str = (outroot,'_T1ME.nii')
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
	print('   Fitting of a mono-exponetial variable TR spin echo signal model')
	print('********************************************************************************')
	print('')
	print('Called on 4D Nifti file: {}'.format(sigfile))
	print('Repetition time file: {}'.format(seqfile))
	print('Output files: {}, {}, {}, {}'.format(txy_out,s0_out,exit_out,mse_out))
	print('')


	### Call fitting routine
	# The entry point of the parallel pool has to be protected with if(__name__=='__main__') (for Windows): 
	if(__name__=='__main__'):
		if (maskrequest==False):
			TxyFitME(sigfile, seqfile, outroot, nprocess)
		else:
			TxyFitME(sigfile, seqfile, outroot, nprocess, maskfile)
	
	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)


