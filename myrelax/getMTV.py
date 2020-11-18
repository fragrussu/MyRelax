### Voxel-wise calculation of MTV
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
import argparse
import nibabel as nib
import numpy as np
import sys
import warnings
from sklearn import linear_model
from skimage import transform
from scipy import ndimage
from scipy import interpolate

def MTVmap(sig_file, t1_file, txy_file, tissue_file, water_file, te_file, out_base, std_val, niter_val):
	''' Calculate MTV from PD with the pseudo-T1 method of Volz S et al, NeuroImage 2012  
	    

	    INTERFACE
	    MTVmap(sig_file, t1_file, txy_file, tissue_file, water_file, te_file, out_base, std_val, npoly_val, niter_val)

	     
	    PARAMETERS
	    - sig_file: 3D Nifti file storing the T2/T2*-weighted signal map, 
			obtained from inversion recovery or variable flip angle imaging
	    - t1_file: 3D Nifti file storing the voxel-wise longitudinal relaxation time map, in ms
	    - txy_file: 3D Nifti file storing the voxel-wise transverse relaxation time map 
		       (T2 for spin echo, T2* for gradient echo), in ms
	    - tissue_file: 3D Nifti file storing the tissue binary mask, containing ONLY grey and white matter 
		          (note that for patients it should NOT contain focal lesions, ans should ONLY contain 
			  normal-appearing grey and white matter)
	    - water_file: 3D Nifti file storing a binary masks that indicates the voxels containing pure water,
			  to be used to normalise MTV (e.g. brain ventricles or a water phantom within the field-of-view)
	    - te_file: text file storing the TE (in ms) used for inversion recovery or variable flip angle imaging 
		      (from which input files sig_file and t1_file were obtained)
	    - out_base: root of output file names, to which file-specific strings will be added; 
                        output files will be double-precision floating point (FLOAT64) and will end in "*_MTV.nii" 
                        (voxel-wise macromolecular tissue volume or MTV map); "*_qPD.nii" (voxel-wise proton density or PD map, 
                        s.t. PD = 1 - MTV); "*_RX.nii" (voxel-wise receiver bias field; note that here we use radial basis function
                        interpolation, rather than polynomial interpolation as done by Volz et al); "*_A.dat" and "*_B.dat" 
                        (text files storing the coefficients of the pseudo-T1 relationship 1/PD = A + B/T1 over the 
                        iterations of the algorithm, where PD is the proton density -- see Volz S et al, NeuroImage 2012; note that
                        here we use RANSAC robust linear regression).
                        To estimate the smooth receiver field, radial basis function interpolation is used.
	    - std_val: standard deviation of Gaussian kernels used to smooth the estimated receiver bias field, in mm 
                       (5.31 mm in Volz S et al, NeuroImage 2012)
	    - niter: number of algorithm iterations (suggested value: 7)
	    
	    
	    Dependencies (Python packages): nibabel, numpy, sys, warnings, argparse, scipy (ndimage, interpolate), 
                                            sklearn (linear_model), skimage (transform)
	    
	    Reference: "Quantitative proton density mapping: correcting the receiver sensitivity bias via pseudo proton densities", 
                        Volz S et al, NeuroImage (2012): 63(1): 540-552
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''


	
	### Load input data check for consistency
	print('    ... loading input data')

	### Load NIFTIs

	## T2/T2* weighted signal intensity
	try:
		sig_obj = nib.load(sig_file)
	except:
		print('')
		print('ERROR: the 3D T2/T2*-weighted signal file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(sig_file))
		print('')
		sys.exit(1)
	sig_data = sig_obj.get_fdata()
	imgsize = sig_data.shape
	imgsize = np.array(imgsize)
	sig_header = sig_obj.header
	sig_affine = sig_header.get_best_affine()
	sig_dims = sig_obj.shape
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D T2/T2*-weighted signal file {} is not a 3D NIFTI. Exiting with 1.'.format(sig_file))
		print('')
		sys.exit(1)
	# Make sure signal is a numpy array
	sig_data = np.array(sig_data)

	# Header that will be copied to the output NIFTI maps
	buffer_header = sig_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type

	## Quantitative T1
	try:
		t1_obj = nib.load(t1_file)
	except:
		print('')
		print('ERROR: the 3D T1 file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(t1_file))
		print('')
		sys.exit(1)
	t1_data = t1_obj.get_fdata()
	t1_header = t1_obj.header
	t1_affine = t1_header.get_best_affine()
	t1_dims = t1_obj.shape
	t1_size = t1_data.shape
	t1_size = np.array(t1_size)
	if t1_size.size!=3:
		print('')
		print('ERROR: the 3D T1 file {} is not a 3D NIFTI. Exiting with 1.'.format(t1_file))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(sig_affine==t1_affine)!=16) or (sig_dims[0]!=t1_dims[0]) or (sig_dims[1]!=t1_dims[1]) or (sig_dims[2]!=t1_dims[2]) ):
		print('')
		print('ERROR: the geometry of the T1 file {} does not match that of the signal file {}. Exiting with 1.'.format(t1_file,sig_file))
		print('')
		sys.exit(1)
	# Make sure T1 is a numpy array
	t1_data = np.array(t1_data)

	## Quantitative T2 (or T2*)
	try:
		txy_obj = nib.load(txy_file)
	except:
		print('')
		print('ERROR: the 3D T2/T2* file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(txy_file))
		print('')
		sys.exit(1)
	txy_data = txy_obj.get_fdata()
	txy_header = txy_obj.header
	txy_affine = txy_header.get_best_affine()
	txy_dims = txy_obj.shape
	txy_size = txy_data.shape
	txy_size = np.array(txy_size)
	if txy_size.size!=3:
		print('')
		print('ERROR: the 3D T2/T2* file {} is not a 3D NIFTI. Exiting with 1.'.format(txy_file))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(sig_affine==txy_affine)!=16) or (sig_dims[0]!=txy_dims[0]) or (sig_dims[1]!=txy_dims[1]) or (sig_dims[2]!=txy_dims[2]) ):
		print('')
		print('ERROR: the geometry of the T2/T2* file {} does not match that of the signal file {}. Exiting with 1.'.format(txy_file,sig_file))
		print('')
		sys.exit(1)
	# Make sure transverse relaxation is a numpy array
	txy_data = np.array(txy_data)

	## Tissue mask
	try:
		tissue_obj = nib.load(tissue_file)
	except:
		print('')
		print('ERROR: the 3D tissue mask {} does not exist or is not in NIFTI format. Exiting with 1.'.format(tissue_file))
		print('')
		sys.exit(1)
	tissue_data = tissue_obj.get_fdata()
	tissue_header = tissue_obj.header
	tissue_affine = tissue_header.get_best_affine()
	tissue_dims = tissue_obj.shape
	tissue_size = tissue_data.shape
	tissue_size = np.array(tissue_size)
	if tissue_size.size!=3:
		print('')
		print('ERROR: the 3D tissue mask {} is not a 3D NIFTI. Exiting with 1.'.format(tissue_file))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(sig_affine==tissue_affine)!=16) or (sig_dims[0]!=tissue_dims[0]) or (sig_dims[1]!=tissue_dims[1]) or (sig_dims[2]!=tissue_dims[2]) ):
		print('')
		print('ERROR: the geometry of the tissue mask {} does not match that of the signal file {}. Exiting with 1.'.format(tissue_file,sig_file))
		print('')
		sys.exit(1)
	# Make sure the water mask is a binary numpy array
	tissue_data = np.array(tissue_data)
	tissue_data[tissue_data>0] = 1
	tissue_data[tissue_data<=0] = 0

	## Water mask
	try:
		water_obj = nib.load(water_file)
	except:
		print('')
		print('ERROR: the 3D water mask {} does not exist or is not in NIFTI format. Exiting with 1.'.format(water_file))
		print('')
		sys.exit(1)
	water_data = water_obj.get_fdata()
	water_header = water_obj.header
	water_affine = water_header.get_best_affine()
	water_dims = water_obj.shape
	water_size = water_data.shape
	water_size = np.array(water_size)
	if water_size.size!=3:
		print('')
		print('ERROR: the 3D water mask {} is not a 3D NIFTI. Exiting with 1.'.format(water_file))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(sig_affine==water_affine)!=16) or (sig_dims[0]!=water_dims[0]) or (sig_dims[1]!=water_dims[1]) or (sig_dims[2]!=water_dims[2]) ):
		print('')
		print('ERROR: the geometry of the water mask {} does not match that of the signal file {}. Exiting with 1.'.format(water_file,sig_file))
		print('')
		sys.exit(1)
	# Make sure the water mask is a binary numpy array
	water_data = np.array(water_data)
	water_data[water_data>0] = 1
	water_data[water_data<=0] = 0

	### Load TE file
	
	# Make sure TE data exists and makes sense
	try:
		TEarray = np.loadtxt(te_file)
		TEarray = np.array(TEarray,'float64')
		TEarray_size = TEarray.size
	except:
		print('')
		print('ERROR: the TE file {} does not exist or is not a numeric text file. Exiting with 1.'.format(te_file))					 
		print('')
		sys.exit(1)
	
	# Check consistency of TE file (one scalar value is expected)
	if TEarray_size!=1:
		print('')
		print('ERROR: the TE file {} contains more than one entry or is empty. Exiting with 1.'.format(te_file))					 
		print('')
		sys.exit(1)
	else:
		TE = TEarray

	## Check scalar inputs std_val, niter_val

	# Kernel standard deviation
	try:
		stdval = np.array(std_val,'float')
	except:
		print('')
		print('ERROR: the kernel standard deviation {} is not a numeric value. Exiting with 1.'.format(std_val))					 
		print('')
		sys.exit(1)

	if stdval.size!=1:
		print('')
		print('ERROR: the kernel standard deviation {} contains more than one entry or is empty. Exiting with 1.'.format(std_val))
		print('')
		sys.exit(1)

	# Number of iterations
	try:
		niterval = np.array(niter_val,'float')
	except:
		print('')
		print('ERROR: the number of iterations {} is not a numeric value. Exiting with 1.'.format(niter_val))					 
		print('')
		sys.exit(1)

	if niterval.size!=1:
		print('')
		print('ERROR: the number of iterations {} contains more than one entry or is empty. Exiting with 1.'.format(niter_val))
		print('')
		sys.exit(1)

	if np.round(niterval)!=niterval:
		print('')
		print('ERROR: the number of iterations {} is not an integer. Exiting with 1.'.format(niter_val))
		print('')
		sys.exit(1)

	if niterval<1:
		print('')
		print('ERROR: the number of iterations is {} but must be at least 1. Exiting with 1.'.format(niter_val))
		print('')
		sys.exit(1)
	
	niterval = np.int(niterval)

	### Check that the kernel standard deviation is compatible with image resolution
	sig_header = sig_obj.header
	sig_header_raw = sig_header.structarr
	pixdim = sig_header_raw['pixdim']
	pixdim = np.array(pixdim[1:4])
	stdvalpix = stdval/pixdim
	fwhmpix = np.round(2*np.sqrt(2*np.log(2))*stdvalpix)
	if( (fwhmpix[0]==0) or (fwhmpix[1]==0) or (fwhmpix[2]==0) ):
		print('')
		print('ERROR: choose a bigger standard deviation for the Gaussian kernels. Exiting with 1.')
		print('')
		sys.exit(1)

	### Rescale signals to deal with numerical tractable numbers (i.e. most of the signal contained between 0 and 100)
	SIGRESCALE = 100.0;
	shigh = np.percentile(sig_data[tissue_data==1],97.5)  # Get an estimate of high signal level in tissue: get 97.5 percentile of signal distribution
	sig_data = SIGRESCALE*sig_data/shigh      # Rescale signal so that it is expected to vary in a range where interpolation is numerically stable

	### Cope with outliers, NaNs, infinities and negative values in signal intensity
	NFACT = 3.5;   # Maximum signal: NFACT times SIGRESCALE
	tissue_data[np.isnan(sig_data)] = 0   # Remove voxels with NaN signal from the analysis
	tissue_data[np.isinf(sig_data)] = 0   # Remove voxels with Inf signal from the analysis
	tissue_data[sig_data<0] = 0           # Remove voxels with negative signals from the analysis
	tissue_data[tissue_data>0] = 1        # Binarise mask
	sig_data[sig_data>NFACT*SIGRESCALE] = NFACT*SIGRESCALE
	sig_data[np.isinf(sig_data)] = NFACT*SIGRESCALE
	sig_data[np.isnan(sig_data)] = 0
	sig_data[sig_data<0] = 0

	### Cope with non-plausible T1 values (Infs, NaNs, negative and eccesively high)
	T1MAX = 10000                          # Maximum T1 allowed: 10 000 ms
	T1MIN = 0.01                           # Minimum T1 allowed: 0.01 ms
	tissue_data[np.isnan(t1_data)] = 0     # Remove voxels with NaN T1 from the analysis
	tissue_data[np.isinf(t1_data)] = 0     # Remove voxels with Inf T1 from the analysis
	tissue_data[t1_data<0] = 0             # Remove voxels with negative T1 from the analysis
	t1_data[t1_data>T1MAX] = T1MAX 
	t1_data[np.isinf(t1_data)] = T1MAX 
	t1_data[np.isnan(t1_data)] = T1MIN
	
	### Cope with non-plausible T2/T2* values (Infs, NaNs, negative and eccesively high)
	TXYMAX = 2000                          # Maximum T2 or T2* allowed: 2000 ms
	TXYMIN = 0.01                          # Minimum T2 or T2* allowed: 0.01 ms
	tissue_data[np.isnan(txy_data)] = 0;   # Remove voxels with NaN T2/T2* from the analysis
	tissue_data[np.isinf(txy_data)] = 0;   # Remove voxels with Inf T2/T2* from the analysis
	tissue_data[txy_data<0] = 0;           # Remove voxels with negative T2/T2* from the analysis
	txy_data[np.isnan(txy_data)] = TXYMIN;
	txy_data[np.isinf(txy_data)] = TXYMAX;
	txy_data[txy_data>TXYMAX] = TXYMAX;

	### Remove T2 or T2* weighting from the signal intensity and obtain apparent proton density (aPD) and cope with NaN and Inf
	warnings.filterwarnings('ignore')    # Ignore warnings - these are going to happen in the background for sure
	apd_data = sig_data/np.exp((-1.0)*TE/txy_data)
	tissue_data[np.isnan(apd_data)] = 0          # Remove NaN from the analysis
	tissue_data[np.isinf(apd_data)] = 0          # Remove Inf from the analysis
	apd_data[np.isnan(apd_data)] = 0
	apd_data[np.isinf(apd_data)] = 0
 
	### Iterative calculation of quantitative proton density (qPD) and receiver field (RF) 
	print('    ... iterative MTV calculation:')

	# Allocate variables to store the intermediate iterations
	A_array = np.zeros((1,niterval+1))
	B_array = np.zeros((1,niterval+1))

	# Initialise the coefficients A and B used to fit 1/PD = A + B/T1 as in Volz et al, NeuroImage 2012
	A = 0.916  # Dimensionless number
	B = 436.0    # Units of ms
	A_array[0,0] = A
	B_array[0,0] = B
	
	# Iterative estimation of the receiver field (RF)

	for ii in range(1,niterval+1):
		print('                iteration {} out of {}'.format(ii,niterval))
		
		# Step 1: get pseudo PD and approximate RF values
		pseudopd_data = 1.0 / ( A + B/t1_data )    # Use A and B to get an estimate of pseudo PD from T1 in tissue (grey and white matter)
		rf_data = apd_data/pseudopd_data;         # Initialisation for the receiver field (RF) map  
		tissue_data[np.isnan(pseudopd_data)] = 0  # Exclude bad voxels
		tissue_data[np.isinf(pseudopd_data)] = 0  # Exclude bad voxels
		tissue_data[np.isnan(rf_data)] = 0  # Exclude bad voxels
		tissue_data[np.isinf(rf_data)] = 0  # Exclude bad voxels
		pseudopd_data[tissue_data==0] = np.nan;   # Set NaN outside tissue mask
				   
		# Step 2: interpolate the RF map with a smooth function (radial basis function interpolation)
		rf_data_size = rf_data.shape    # Size of data
		resample_size = np.round(rf_data_size/fwhmpix)   # Size to which data will be downsampled for the estimation of the smooth RX field
		rf_smooth = InterpolateField(rf_data,tissue_data,stdvalpix,resample_size)    # Interpolated RF field

		# Step 3: get a guess PD using the smooth interpolation of RF, and normalise to the PD of water  
		pdguess_data = apd_data/rf_smooth;
		pdarray = pdguess_data[water_data==1]
		pdarray[np.isinf(pdarray)] = np.nan
		pdguess_data = pdguess_data/np.nanmean(pdarray)   # Make sure PDguess equals 1 in CSF/free water
				    
		# Step 4: re-estimate A and B
		A,B = FitPDversusT1(pdguess_data,t1_data,tissue_data)
				 
		# Keep track of A and B
		A_array[0,ii] = A    
		B_array[0,ii] = B



	# Get output quantitative PD (qPD) and RF values from the last iteration
	qpd_data = pdguess_data;      # Quantitative proton density qPD
	rf_data = rf_smooth*shigh/SIGRESCALE;   # Receiver coil bias field -- scale back to original range after rescaling to the easily tractable range (i.e. most of signal contained between 0 and 100) 
	mtv_data = 1.0 - qpd_data;    # Macromolecular tissue volume: 1 - qPD

	# Remove NaN as they can be quite annoying for viewers and statistics
	rf_data[np.isnan(rf_data)] = 0.0
	qpd_data[np.isnan(qpd_data)] = 0.0
	mtv_data[np.isnan(mtv_data)] = 0.0

	# Remove Infs as they can be quite annoying for viewers and statistics
	rf_data[np.isinf(rf_data)] = 0.0
	qpd_data[np.isinf(qpd_data)] = 0.0
	mtv_data[np.isinf(mtv_data)] = 0.0

	### Save output files
	print('    ... saving output files')
	buffer_string=''
	seq_string = (out_base,'_MTV.nii')
	mtv_outfile = buffer_string.join(seq_string)
	
	buffer_string=''
	seq_string = (out_base,'_qPD.nii')
	pd_outfile = buffer_string.join(seq_string)
	
	buffer_string=''
	seq_string = (out_base,'_RX.nii')
	rx_outfile = buffer_string.join(seq_string)
	
	buffer_string=''
	seq_string = (out_base,'_A.dat')
	a_outfile = buffer_string.join(seq_string)
	
	buffer_string=''
	seq_string = (out_base,'_B.dat')
	b_outfile = buffer_string.join(seq_string)
	

	# MTV
	mtv_obj = nib.Nifti1Image(mtv_data,sig_obj.affine,buffer_header)
	nib.save(mtv_obj, mtv_outfile)

	# PD
	pd_obj = nib.Nifti1Image(qpd_data,sig_obj.affine,buffer_header)
	nib.save(pd_obj, pd_outfile)

	# RX field
	rx_obj = nib.Nifti1Image(rf_data,sig_obj.affine,buffer_header)
	nib.save(rx_obj, rx_outfile)

	# Text files with A and B coefficients over the iterations
	np.savetxt(a_outfile,A_array)
	np.savetxt(b_outfile,B_array)

	### Done
	print('')



def InterpolateField(data,mask,STDvoxel,down_size):

	### Check data

	# Remove NaN and Inf from data
	mask = np.array(mask,'float64')
	mask[np.isnan(data)] = 0
	mask[np.isinf(data)] = 0
	data[np.isnan(data)] = 0
	data[np.isinf(data)] = 0
	mask[mask<=0] = 0    # Binarise data
	mask[mask>0] = 1     # Binarise data
	data = np.array(data)*np.array(mask)

	### Deichmann filtering

	# Filter input data with Gaussian filter of given standard deviation (in voxels)
	data_filt = ndimage.gaussian_filter(data, STDvoxel, order=0, mode='constant', cval=0.0, truncate=6.0)

	# Filter tissue mask with Gaussian filter of given standard deviation (in voxels)
	mask_filt = ndimage.gaussian_filter(mask, STDvoxel, order=0, mode='constant', cval=0.0, truncate=6.0)

	# Use smoothed mask to correct partial volume (filtering algorithm as Deichmann R, MRM 2005, 54:20-27)
	data_filt = data_filt/mask_filt

	### Downsample data
	data_filt_down = transform.resize(data_filt,down_size)      # Downsample the field map before interpolation
	mask_small = transform.resize(mask,down_size)               # Downsample mask
	mask_small[mask_small<=0.5] = 0                             # Binarise mask
	mask_small[mask_small>0.5] = 1                              # Binarise mask
	data_filt_down = np.array(data_filt_down)                   # Make sure we deal with numpy arrays
	mask_small = np.array(mask_small)                           # Make sure we deal with numpy arrays


	### Perform actual interpolation

	# Get voxel positions
	data_filt_down_size = data_filt_down.shape
	data_filt_size = data_filt.shape
	xpos = np.zeros(data_filt_down_size,'float64')
	ypos = np.zeros(data_filt_down_size,'float64')
	zpos = np.zeros(data_filt_down_size,'float64')
	totvox = data_filt.size   # Total number of voxels
	
	grid_pred = np.zeros((totvox,3))    # Grid where python interpolator will store the interpolated field
	vox_count=0
	for ii in range(0,data_filt_down_size[0]):
		for jj in range(0,data_filt_down_size[1]):
			for kk in range(0,data_filt_down_size[2]):

				# Extract spatial position: for estimating the interpolating coefficients
				xpos[ii,jj,kk] = 1.0*ii
				ypos[ii,jj,kk] = 1.0*jj
				zpos[ii,jj,kk] = 1.0*kk			

				# Extract spatial position: for the actual interpolation on a regular grid
				grid_pred[vox_count,0] = 1.0*ii
				grid_pred[vox_count,1] = 1.0*jj
				grid_pred[vox_count,2] = 1.0*kk
				vox_count = vox_count + 1


	# Extract field value and voxel position for all voxels within the tissue mask
	xpos_array = xpos[mask_small==1]
	ypos_array = ypos[mask_small==1]
	zpos_array = zpos[mask_small==1]
	values_array = data_filt_down[mask_small==1]

	# Interpolate field with radial basis functions
	rbfi = interpolate.Rbf(xpos_array, ypos_array, zpos_array, values_array)

	# Predict receiver field in all voxels
	field_predicted_lowres = np.zeros(data_filt_down_size,'float64')
	for ii in range(0,data_filt_down_size[0]):
		for jj in range(0,data_filt_down_size[1]):
			for kk in range(0,data_filt_down_size[2]):
				field_predicted_lowres[ii,jj,kk] = rbfi(1.0*ii,1.0*jj,1.0*kk)

	# Upsample the estimated 3D matrix storting the smooth field
	field_predicted = transform.resize(field_predicted_lowres,data_filt_size)

	## Return predicted smooth field
	return field_predicted



def FitPDversusT1(pd,t1,tissues):

	### Binarise tissue mask
	tissues[tissues>0] = 1
	tissues[tissues<=0] = 0
	
	### Extract PD values within tissue and store them as a column array for robust linear fitting
	pdvals = pd[tissues==1]
	t1vals = t1[tissues==1]
	totvox = np.int(pdvals.size)             # Total number of voxels (as an integer)
	pdvals = np.reshape(pdvals,(totvox,1))   # Make sure it is a column array
	t1vals = np.reshape(t1vals,(totvox,1))   # Make sure it is a column array

	### Fit the relationship 1/PD = A + B/T1: use robust fitting on nfits bootstrap replicates of the values of PD and T1
	nfits = 1000                                    # Repeat robust linear regression 1000 times...
	A_array = np.zeros((1,np.int(nfits)))
	B_array = np.zeros((1,np.int(nfits)))

	# Loop over the different bootstrap replicates of the voxels
	for qq in range(0,nfits):

		# Get the bootstrap replicate of the T1 and PD voxels
		idx = np.random.choice(totvox, size=(totvox,1))      # Get the indices of voxels selected for the current bootstrap sample (sampling with replacement)
		pdvalsrep = pdvals[idx,0]
		t1valsrep = t1vals[idx,0]
		pdvalsrep = np.reshape(pdvalsrep,(totvox,1))
		t1valsrep = np.reshape(t1valsrep,(totvox,1))
		
		# Get 1/PD and 1/T1 and make sure they are column arrays
		oneOverPD = 1.0 / pdvalsrep
		oneOverT1 = 1.0 / t1valsrep
		oneOverPD = np.reshape(oneOverPD,(totvox,1))
		oneOverT1 = np.reshape(oneOverT1,(totvox,1))
		
		# Perform robust linear fitting of 1/PD = A + B/T1
		ransac = linear_model.RANSACRegressor()    # Robust linear regressor (RANSAC algoritum from sklearn)
		ransac.fit(oneOverT1, oneOverPD)           # Fit robustly 1/PD = A + B/T1
		A_array[0,qq] = ransac.estimator_.intercept_   # Store parameter A (intercept of 1/PD = A + B/T1) for current loop iteration
		B_array[0,qq] = ransac.estimator_.coef_        # Store parameter B (slope of 1/PD = A + B/T1) for current loop iteration

	# Return the median A and B over the nfits loop iterations
	A_array[np.isinf(A_array)] = np.nan    # Remove and infinite values and replace with NaN
	B_array[np.isinf(B_array)] = np.nan    # Remove and infinite values and replace with NaN
	A_out = np.nanmedian(A_array)
	B_out = np.nanmedian(B_array)

	# Return
	return A_out, B_out





# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of Macromolecular Tissue Volume (MTV) from Proton Density (PD) with the pseudo-T1 method of Volz S et al, NeuroImage 2012. Dependencies (Python packages): nibabel, numpy, sys, warnings, argparse, scipy, sklearn, skimage. Reference: "Quantitative proton density mapping: correcting the receiver sensitivity bias via pseudo proton densities", Volz S et al, NeuroImage (2012): 63(1): 540-552. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('sig_file', help='3D Nifti file storing S0, i.e. the T2/T2*-weighted apparent proton density map, obtained from inversion recovery or variable flip angle imaging')
	parser.add_argument('t1_file', help='3D Nifti file storing the voxel-wise longitudinal relaxation time map, in ms')
	parser.add_argument('txy_file', help='3D Nifti file storing the voxel-wise transverse relaxation time map (T2 for spin echo, T2* for gradient echo), in ms ')
	parser.add_argument('tissue_file', help='3D Nifti file storing the tissue binary mask (note that for patients it should NOT contain focal lesions)')
	parser.add_argument('water_file', help='3D Nifti file storing a binary masks that indicates the voxels containing pure water, to be used to normalise MTV (e.g. brain ventricles or a water phantom within the field-of-view) ')
	parser.add_argument('te_file', help='text file storing the TE (in ms) used for inversion recovery or variable flip angle imaging (from which input files sig_file and t1_file were obtained)')
	parser.add_argument('out_base', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "*_MTV.nii" (voxel-wise MTV map; water has MTV = 0); "*_qPD.nii" (voxel-wise PD map; PD = 1 - MTV, such that water has PD = 1); "*_RX.nii" (voxel-wise receiver bias field; note that here we used radial basis function interpolation, rather than polynomial interpolation as done by Volz et al); "*_A.dat" and "*_B.dat" (text files storing the coefficients of the pseudo-T1 relationship 1/PD = A + B/T1 over the iterations of the algorithm, where PD is the proton density -- see Volz S et al, NeuroImage 2012; note that here we use RANSAC robust linear regression)')
	parser.add_argument('--std', metavar='<value>', default='5.31', help='standard deviation of Gaussian kernels (in mm) to be used to smooth the estimated receiver bias field (default 5.31 mm, as Volz S et al, NeuroImage 2012)')
	parser.add_argument('--niter', metavar='<value>', default='7', help='number of algorithm iterations (default: 7)')
	args = parser.parse_args()

	### Get input arguments
	sigfile = args.sig_file
	t1file = args.t1_file
	txyfile = args.txy_file
	tissuefile = args.tissue_file
	waterfile = args.water_file
	tefile = args.te_file
	outbase = args.out_base
	stdval = args.std
	niterval = args.niter

	print('')
	print('********************************************************************')
	print('                           MTV calculation                          ')
	print('********************************************************************')
	print('')
	print('3D Nifti storing S0, the T2- (or T2*)-weighted signal intensity from inversion recovery or variable flip angle imaging: {}'.format(sigfile))
	print('3D Nifti storing the T1 map (ms): {}'.format(t1file))
	print('3D Nifti storing the T2 or T2* map (ms): {}'.format(txyfile))
	print('3D Nifti storing the tissue mask: {}'.format(tissuefile))
	print('3D Nifti storing the water mask: {}'.format(waterfile))
	print('Text file storing the TE used in inversion recovery or variable flip angle imaging: {}'.format(tefile))
	print('Standard deviation of smoothing kernels: {} mm'.format(stdval))
	print('Number of algorithm iterations: {}'.format(niterval))
	print('')
	print('Output files: {}_MTV.nii, {}_qPD.nii, {}_RX.nii, {}_A.dat, {}_B.dat'.format(outbase,outbase,outbase,outbase,outbase))
	print('')

	MTVmap(sigfile, t1file, txyfile, tissuefile, waterfile, tefile, outbase, stdval, niterval)
	
	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

