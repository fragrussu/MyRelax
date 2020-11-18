### Voxel-wise calculation of T2prime
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
import nibabel as nib
import numpy as np
import warnings


def T2Primemap(*argv):
	''' Calculate T2prime from quantitative T2 and T2star maps  
	    

	    INTERFACES
	    T2Primemap(t2_nifti,t2star_nifti,t2prime_output)
	    T2Primemap(t2_nifti,t2star_nifti,t2prime_output,mask_nifti)

	     
	    PARAMETERS
	    - t2_nifti: path of a Nifti file storing the quantitative T2 map
	    - t2star: path of a Nifti file storing the quantitative T2star map
	    - t2prime_output: path of the Nifti file that will store the 3D output T2prime map (saved as a 
			  double-precision floating point image FLOAT64); such an output map is
			  calculated as
			
				T2prime = 1 / ( (1/T2star) - (1/T2) )

	    - mask_nifti: path of a Nifti file storting a mask (T2prime will be calculated only where
			  mask_nifti equals 1; 0 will be set in the T2prime output map otherwise)
	    
	    Dependencies (Python packages): nibabel, numpy (other than standard library).
	    
	    Reference: "Age-dependent normal values of T2* and T2prime in brain parenchyma" 
                      Siemonsen S et al, American Journal of Neuroradiology (2008), 29(5):950-955
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	t2_nifti = argv[0]
	t2star_nifti = argv[1]
	t2prime_output = argv[2]
	
	### Load T2 check for consistency
	print('    ... loading input data')
	try:
		t2_obj = nib.load(t2_nifti)
	except:
		print('')
		print('ERROR: the 3D input T2 map {} does not exist or is not in NIFTI format. Exiting with 1.'.format(t2_nifti))	 			 
		print('')
		sys.exit(1)
	t2_data = t2_obj.get_fdata()
	imgsize = t2_data.shape
	imgsize = np.array(imgsize)
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D input T2 map {} is not a 3D NIFTI. Exiting with 1.'.format(t2_nifti))					 
		print('')
		sys.exit(1)
	
	### Load T2star and check for consistency
	try:
		t2star_obj = nib.load(t2star_nifti)
	except:
		print('')
		print('ERROR: the 3D input T2star map {} does not exist or is not in NIFTI format. Exiting with 1.'.format(t2star_nifti))	   			 
		print('')
		sys.exit(1)
	t2star_data = t2star_obj.get_fdata()
	t2_header = t2_obj.header
	t2_affine = t2_header.get_best_affine()
	t2_dims = t2_obj.shape
	t2star_header = t2star_obj.header
	t2star_affine = t2star_header.get_best_affine()
	t2star_dims = t2star_obj.shape
	t2star_size = t2star_data.shape
	t2star_size = np.array(t2star_size)
	if t2star_size.size!=3:
		print('')
		print('ERROR: the 3D input T2star map {} is not a 3D NIFTI. Exiting with 1.'.format(t2star_nifti))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(t2_affine==t2star_affine)!=16) or (t2_dims[0]!=t2star_dims[0]) or (t2_dims[1]!=t2star_dims[1]) or (t2_dims[2]!=t2star_dims[2]) ):
		print('')
		print('ERROR: the geometry of the T2 file {} and of the T2star file {} do not match. Exiting with 1.'.format(t2_nifti,t2star_nifti))					 
		print('')
		sys.exit(1)
	
	### Deal with optional arguments: mask
	if Nargv==4:
		got_mask = True
		mask_nifti = argv[3]
		try:
			mask_obj = nib.load(mask_nifti)
		except:
			print('')
			print('ERROR: the mask file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mask_nifti))	  			 
			print('')
			sys.exit(1)
		# Make sure that the mask geometry matches that of the other files
		mask_data = mask_obj.get_fdata()
		mask_size = mask_data.shape
		mask_size = np.array(mask_size) 
		mask_header = mask_obj.header
		mask_affine = mask_header.get_best_affine()
		mask_dims = mask_obj.shape
		if mask_size.size!=3:
			print('')
			print('WARNING: the mask file {} is not a 3D NIFTI file. Ignoring mask...'.format(mask_nifti))				 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(t2_affine==mask_affine)!=16) or (t2_dims[0]!=mask_dims[0]) or (t2_dims[1]!=mask_dims[1]) or (t2_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the the mask file {} does not match that of the T2 and T2star data. Ignoring mask...'.format(mask_nifti))					 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask is a binary file
			mask_data[mask_data>1] = 1
			mask_data[mask_data<0] = 0	
		
	else:
		got_mask = False

	### Calculate T2prime map
	print('    ... calculating T2prime')
	t2_data = np.array(t2_data,'float64')
	t2star_data = np.array(t2star_data,'float64')
	warnings.filterwarnings('ignore')    # Ignore warnings - these are going to happen in the background for sure
	t2prime_map = 1 / ( (1/t2star_data) - (1/t2_data)  )     # Calculate T2prime
	t2prime_map[np.isnan(t2prime_map)] = 0.0   # Remove NaNs
	t2prime_map[np.isinf(t2prime_map)] = 0.0   # Remove Inf
	t2prime_map[t2prime_map<0.0] = 0.0   # Remove negative values
	print('    ... substituting any nan, inf and negative values with 0')
	# Use mask if required
	if got_mask==True:
		t2prime_map[mask_data==0]=0.0

	### Save output
	print('    ... saving output file')
	buffer_header = t2_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative map as a float64
	t2prime_obj = nib.Nifti1Image(t2prime_map,t2_obj.affine,buffer_header)
	nib.save(t2prime_obj, t2prime_output)

	### Done
	print('')


# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of quantitative T2prime from T2 and T2star, which must be defined in the same MRI space (i.e. they must be co-registered). Dependencies (Python packages): nibabel, numpy (other than standard library). References: "Age-dependent normal values of T2* and T2prime in brain parenchyma", Siemonsen S et al, American Journal of Neuroradiology (2008), 29(5):950-955. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('t2_file', help='3D Nifti file storing the T2 image (in ms)')
	parser.add_argument('t2star_file', help='3D Nifti file storing the T2star image (in ms)')
	parser.add_argument('out_file', help='3D Nifti file that will store the T2prime map, i.e. T2prime = 1 / ( (1/T2star) - (1/T2) ), as double-precision floating point (FLOAT64)')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where T2prime is required, 0 where is not')
	args = parser.parse_args()

	### Get input arguments
	t2file = args.t2_file
	t2starfile = args.t2star_file
	outfile = args.out_file
	maskfile = args.mask

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for T2' calculation has been provided
	    maskrequest = True
	else:
	    # A mask for T2' calculation has not been provided
	    maskrequest = False



	print('')
	print('********************************************************************')
	print('                       T2 prime calculation                         ')
	print('********************************************************************')
	print('')
	print('Called on 3D Nifti files: {} (T2) and {} (T2 star)'.format(t2file,t2starfile))
	print('Output T2 prime file: {}'.format(outfile))
	print('')


	if (maskrequest==False):
		T2Primemap(t2file, t2starfile, outfile)
	else:
		T2Primemap(t2file, t2starfile, outfile, maskfile)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

