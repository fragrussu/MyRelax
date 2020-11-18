### Voxel-wise fitting of T1 from inversion recovery magnitude data
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
	    signal = signal(mri_par,tissue_par)
	    
	    PARAMETERS
	    - mri_par: list/array indicating the TIs (inversion times, in ms) used for the experiment (one measurement per TI)
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2/T2*-weighted proton density)
             		  tissue_par[1] = T1 (longitudinal relaxation time, in ms)
		
	    RETURNS
	    - signal: a numpy array of measurements generated according to a inversion recovery signal model (for long TR),
			
		         signal  =  S0 * | 1 - 2*exp(-TI/T1) |
		
		      where TI is the inversion time and where S0 and T1 are the tissue parameters (S0 is the T2/T2*-weighted proton 
                      density, and T1 is the longitudinal relaxation time).
		
		
	    Dependencies (Python packages): numpy, nibabel, scipy (other than standard library)
	    
	    References: "Quantitative MRI of the brain", 2nd edition, Tofts, Cercignani and Dowell editors, Taylor and Francis Group
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	

	### Handle inputs
	time_values = np.array(mri_par,'float64')  # Make sure sequence values are stored as a numpy array
	s0_value = tissue_par[0]        # S0
	t_value = tissue_par[1]        # T1

	### Calculate signal
	with np.errstate(divide='raise',invalid='raise'):
		try:
			signal = s0_value * np.abs( 1 - 2*np.exp(-time_values/t_value) )
		except FloatingPointError:
			signal = 0.0 * t_value      # Just output zeros when T1 is 0.0			

	### Output signal
	return signal
	

def Fobj(tissue_par,mri_par,meas):
	''' Fitting objective function for inversion recovery signal model		
		
	    INTERFACE
	    fobj = Fobj(tissue_par,mri_par,meas)
	    
	    PARAMETERS
	    - tissue_par: list/array of tissue parameters, in the following order:
                          tissue_par[0] = S0 (T2/T2*-weighted proton density)
             		  tissue_par[1] = T1 (longitudinal relaxation time, in ms)
	    - mri_par: list/array indicating the TIs (echo times, in ms) used for the experiment (one measurement per TI)
	    - meas: list/array of measurements
		
	    RETURNS
	    - fobj: objective function measured as sum of squared errors between measurements and predictions, i.e.
			
				 fobj = SUM_OVER_n( (prediction - measurement)^2 )
		
		     Above, the prediction are obtained using the signal model implemented by function signal().
	    
	    References: "On the accuracy of T1 mapping: Searching for common ground". 
                         Stikov N et al, Magnetic Resonance in Medicine (2015), 73(2): 514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	
	### Predict signals given tissue and sequence parameters
	pred = signal(mri_par,tissue_par)

	### Calculate objective function and return
	fobj = np.sum( (np.array(pred) - np.array(meas))**2 )
	return fobj


def GridSearch(mri_par,meas):
	''' Grid search for non-linear fitting of inversion recovery for long TR	
		
	    INTERFACE
	    tissue_estimate, fobj_grid = GridSearch(mri_par,meas)
	    
	    PARAMETERS
	    - mri_par: list/array indicating the TIs (inversion times, in ms) used for the experiment (one measurement per TI)
	    - meas: list/array of measurements
		
	    RETURNS
	    - tissue_estimate: estimate of tissue parameters that explain the measurements reasonably well. The parameters are
			       estimated sampling the fitting objective function Fobj() over a grid; the output is
                               tissue_estimate[0] = S0 (T2/T2*-weighted proton density)
             		       tissue_estimate[1] = T1 (longitudinal relaxation time, in ms)
	    - fobj_grid:       value of the objective function when the tissue parameters equal tissue_estimate
	    
	    References: "On the accuracy of T1 mapping: Searching for common ground". 
                         Stikov N et al, Magnetic Resonance in Medicine (2015), 73(2): 514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Prepare grid for grid search
	time_grid = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2500.0, 3000.0, 3500.0, 3750.0, 4000.0, 4250.0, 4500.0])     # Grid of T1 values
	s0_grid = np.linspace(0.0,9*np.max(meas),num=10)    # Grid of S0 values: from 0 up to 10 times the maximum signal taken as input

	### Initialise objective function to infinity and parameters for grid search
	fobj_best = float('inf')
	s0_best = 0.0
	time_best = 0.0
	
	### Run grid search
	for ii in range(0, len(time_grid)):

		time_ii =  time_grid[ii]   		
		for jj in range(0, len(s0_grid)):

			s0_jj =  s0_grid[jj]
			params = np.array([s0_jj,time_ii])
			
			# Objective function
			fval = Fobj(params,mri_par,meas)

			# Check if objective function is smaller than previous value
			if fval<fobj_best:
				fobj_best = fval
				s0_best = s0_jj
				time_best = time_ii

	### Return output
	paramsgrid = np.array([s0_best, time_best])
	fobjgrid = fobj_best
	return paramsgrid, fobjgrid



def FitSlice(data):
	''' Fit T1 from an inversion-recovery experiment on one MRI slice stored as a 2D numpy array  
	    

	    INTERFACE
	    data_out = FitSlice(data)
	     
	    PARAMETERS
	    - data: a list of 7 elements, such that
	            data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
		            are the slice first and second dimensions, whereas the third dimension of data[0] stores measurements 
		    data[1] is a numpy monodimensional array storing the TI values (ms)
		    data[2] is a 2D numpy array contaning the fitting mask within the MRI slice (see FitVox())
		    data[3] is a scalar containing the index of the MRI slice in the 3D volume
	    
	    RETURNS
	    - data_out: a list of 4 elements, such that
		    data_out[0] is the parameter S0 (see FitVox()) within the MRI slice
	            data_out[1] is the parameter T1 (see FitVox()) within the MRI slice
                    data_out[2] is the exit code of the fitting (see FitVox()) within the MRI slice
		    data_out[3] is the fitting sum of squared errors withint the MRI slice
                    data_out[4] equals data[3]
	
		    Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
	    
	    References: "On the accuracy of T1 mapping: Searching for common ground". 
                         Stikov N et al, Magnetic Resonance in Medicine (2015), 73(2): 514-522
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''

	
	### Extract signals and sequence information from the input list
	signal_slice = data[0]      # Signal
	time_value = data[1]        # TI values (in ms)
	mask_slice = data[2]        # fitting mask
	idx_slice = data[3]         # Slice index
	slicesize = signal_slice.shape    # Get number of voxels of current MRI slice along each dimension
	time_value = np.array(time_value)     # Make sure the TI is an array

	### Allocate output variables
	s0_slice = np.zeros(slicesize[0:2],'float64')
	t_slice = np.zeros(slicesize[0:2],'float64')
	exit_slice = np.zeros(slicesize[0:2],'float64')
	mse_slice = np.zeros(slicesize[0:2],'float64')
	Nmeas = slicesize[2]   # Number of measurements


	### Fit monoexponential inversion recovery model in the voxels within the current slice
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
					param_init, fobj_init = GridSearch(time_value,sig_voxel) 
					             
						
					# Minimise the objective function numerically
					param_bound = ((0,5*param_init[0]),(0,5000),)  # Range for S0 and T1 (T1 limited to be < 5000)		
					modelfit = minimize(Fobj, param_init, method='L-BFGS-B', args=tuple([time_value,sig_voxel]), bounds=param_bound)
					fit_exit = modelfit.success
					fobj_fit = modelfit.fun

					# Non-linear optimisation was successful and provided a smaller objective function as compared to the grid search
					if fit_exit==True and fobj_fit<fobj_init:
						param_fit = modelfit.x
						s0_voxel = param_fit[0]
						t_voxel = param_fit[1]
						exit_voxel = 1
						mse_voxel = fobj_fit
						
					# Grid search provided a better minimisation of objective function (non-linear fitting probably stuck in local minimum)
					else:
						s0_voxel = param_init[0]
						t_voxel = param_init[1]
						exit_voxel = -1
						mse_voxel = fobj_init
							
						
							
				# The voxel is background
				else:
					s0_voxel = 0.0
					t_voxel = 0.0
					exit_voxel = 0
					mse_voxel = 0.0
				
				# Store fitting results for current voxel
				s0_slice[xx,yy] = s0_voxel
				t_slice[xx,yy] = t_voxel
				exit_slice[xx,yy] = exit_voxel
				mse_slice[xx,yy] = mse_voxel

	### Create output list storing the fitted parameters and then return
	data_out = [s0_slice, t_slice, exit_slice, mse_slice, idx_slice]
	return data_out
	



def FitVox(*argv):
	''' Fit T1 from inversion recovery experiments
	    

	    INTERFACES
	    FitVox(input_nifti, ti_text, output_basename, ncpu)
	    FitVox(input_nifti, ti_text, output_basename, ncpu, mask_nifti)
	     
	    PARAMETERS
	    - input_nifti: path of a Nifti file storing the multi-inversion time data as 4D data.
	    - ti_text: path of a text file storing the inversion times (ms) used to acquire the data.
	    - output_basename: base name of output files. Output files will end in 
                            "_S0IR.nii"   --> T2/T2*-weighted proton density, with receiver coil field bias
		            "_T1IR.nii"   --> T1 map (ms)
			    "_ExitIR.nii" --> exit code (1: successful fitting; 0 background)
			    "_SSEIR.nii"  --> fitting sum of squared errors
			    
			    Note that in the background and where fitting fails, S0, T1 and MSE are set to 0.0
			    Output files will be stored as double-precision floating point (FLOAT64)

	    - ncpu: number of processors to be used for computation
	    - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the 
			  signal model needs to be fitted, and 0 otherwise
	    
	    References: "On the accuracy of T1 mapping: Searching for common ground". 
                        Stikov N et al, Magnetic Resonance in Medicine (2015), 73(2): 514-522
	     
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
		print('ERROR: the inversion time file {} does not exist or is not a numeric text file. Exiting with 1.'.format(seq_text))
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
	if Nargv==5:
		mask_nifti = argv[4]
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
	s0_data = np.zeros(imgsize[0:3],'float64')	       # T2/T2*-weighted proton density with receiver field bias (double-precision floating point)
	t_data = np.zeros(imgsize[0:3],'float64')	       # T1 (double-precision floating point)
	exit_data = np.zeros(imgsize[0:3],'float64')           # Exit code (double-precision floating point)
	mse_data = np.zeros(imgsize[0:3],'float64')            # Fitting sum of squared errors (MSE) (double-precision floating point)

	#### Fitting
	print('    ... T1 estimation')
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
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of mono-exponential model
			t_data[:,:,slicepos] = fitslice[1]     # Relaxation constant
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors	


	# Run serial fitting as no parallel processing is required (it can take up to 1 hour per brain)
	else:
		for kk in range(0, imgsize[2]):
			fitslice = FitSlice(inputlist[kk])   # Fitting output relative to kk-th element in the list
			slicepos = fitslice[4]    # Spatial position of kk-th MRI slice
			s0_data[:,:,slicepos] = fitslice[0]    # Parameter S0 of mono-exponential model
			t_data[:,:,slicepos] = fitslice[1]     # Relaxation constant
			exit_data[:,:,slicepos] = fitslice[2]  # Exit code
			mse_data[:,:,slicepos] = fitslice[3]   # Sum of Squared Errors


	### Save the output maps
	print('    ... saving output files')
	buffer_string=''
	seq_string = (output_rootname,'_T1IR.nii')
	t_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_S0IR.nii')
	s0_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_ExitIR.nii')
	exit_outfile = buffer_string.join(seq_string)
	buffer_string=''
	seq_string = (output_rootname,'_SSEIR.nii')
	mse_outfile = buffer_string.join(seq_string)
	buffer_header = sig_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type
	t_obj = nib.Nifti1Image(t_data,sig_obj.affine,buffer_header)
	nib.save(t_obj, t_outfile)
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
	parser = argparse.ArgumentParser(description='Voxel-wise fitting of T1 from inversion recovery MRI magnitude data already corrected for motion. Dependencies (Python packages): numpy, nibabel, scipy (other than standard library). References: "On the accuracy of T1 mapping: Searching for common ground". Stikov N et al, Magnetic Resonance in Medicine (2015), 73(2): 514-522. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('ir_file', help='4D Nifti file of multi-inversion time magnitude images from an inversion recovery experiment')
	parser.add_argument('ti_file', help='text file of inversion times (TIs) used to acquire the images (TIs in ms; TIs separated by spaces)')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0IR.nii" (T2/T2*-weighted proton density, with receiver coil field bias); "_T1IR.nii" (T1 map in ms); "_ExitIR.nii" (exit code: 1 for successful non-linear fitting; 0 background; -1 for failing of non-linear fitting, with results from a grid search provided instead); "_SSEIR.nii" (fitting sum of squared errors).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	parser.add_argument('--ncpu', metavar='<N>', help='number of CPUs to be used for computation (default: half of available CPUs)')
	args = parser.parse_args()

	### Get input arguments
	sigfile = args.ir_file
	seqfile = args.ti_file
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
	seq_str = (outroot,'_T1IR.nii')
	t_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_S0IR.nii')
	s0_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_ExitIR.nii')
	exit_out = buffer_str.join(seq_str)
	buffer_str=''
	seq_str = (outroot,'_SSEIR.nii')
	mse_out = buffer_str.join(seq_str)


	print('')
	print('********************************************************************************')
	print('     Fitting of mono-exponetial inversion recovery model for T1 estimation      ')
	print('********************************************************************************')
	print('')
	print('Called on 4D Nifti file: {}'.format(sigfile))
	print('Inversion time file: {}'.format(seqfile))
	print('Output files: {}, {}, {}, {}'.format(t_out,s0_out,exit_out,mse_out))
	print('')


	### Call fitting routine
	# The entry point of the parallel pool has to be protected with if(__name__=='__main__') (for Windows): 
	if(__name__=='__main__'):
		if (maskrequest==False):
			FitVox(sigfile, seqfile, outroot, nprocess)
		else:
			FitVox(sigfile, seqfile, outroot, nprocess, maskfile)
	
	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)


