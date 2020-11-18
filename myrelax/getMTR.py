### Voxel-wise calculation of MTR
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


def MTRmap(*argv):
	''' Calculate MTR from a MT "on" and a MT "off" acquisition  
	    

	    INTERFACES
	    MTRmap(mton_nifti,mtoff_nifti,mtr_output)
	    MTRmap(mton_nifti,mtoff_nifti,mtr_output,mask_nifti)

	     
	    PARAMETERS
	    - mton_nifti: path of a Nifti file storing the 3D MT "on" image (with off-res. pulse)
	    - mtoff_nifti: path of a Nifti file storing the 3D MT "off" image (without off-res. pulse)
	    - mtr_output: path of the Nifti file that will store the 3D output MTR image (saved as a 
			  double-precision floating point image FLOAT64); such an output map is
			  calculated as
			
				MTR = 100 * (MToff - MTon)/MToff
		
			  above, MTon is the image where the off-resonance pulse is played (so is "on")
			  while MToff is the image where the off-resonance pulse is not played (so is "off")

	    - mask_nifti: path of a Nifti file storting a mask (MTR will be calculated only where
			  mask_nifti equals 1; 0 will be set in the MTR output map otherwise)
	    
	    Dependencies (Python packages): nibabel, numpy (other than standard library).
	    
	    References: "T1, T2 relaxation and magnetization transfer in tissue at 3T", Stanisz GJ,
		        Magnetic Resonance in Medicine (2005), 54:507-512
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	mton_nifti = argv[0]
	mtoff_nifti = argv[1]
	mtr_output = argv[2]
	
	### Load MT "on" data and check for consistency
	print('    ... loading input data')
	try:
		mton_obj = nib.load(mton_nifti)
	except:
		print('')
		print('ERROR: the 3D input MT "on" file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mton_nifti))	 			 
		print('')
		sys.exit(1)
	mton_data = mton_obj.get_fdata()
	imgsize = mton_data.shape
	imgsize = np.array(imgsize)
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D input MT "on" file {} is not a 3D NIFTI. Exiting with 1.'.format(mton_nifti))					 
		print('')
		sys.exit(1)
	
	### Load MT off data and check for consistency
	try:
		mtoff_obj = nib.load(mtoff_nifti)
	except:
		print('')
		print('ERROR: the 3D input MT "off" file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mtoff_nifti))	   			 
		print('')
		sys.exit(1)
	mtoff_data = mtoff_obj.get_fdata()
	mton_header = mton_obj.header
	mton_affine = mton_header.get_best_affine()
	mton_dims = mton_obj.shape
	mtoff_header = mtoff_obj.header
	mtoff_affine = mtoff_header.get_best_affine()
	mtoff_dims = mtoff_obj.shape
	mtoff_size = mtoff_data.shape
	mtoff_size = np.array(mtoff_size)
	if mtoff_size.size!=3:
		print('')
		print('ERROR: the 3D input MT "off" file {} is not a 3D NIFTI. Exiting with 1.'.format(mtoff_nifti))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(mton_affine==mtoff_affine)!=16) or (mton_dims[0]!=mtoff_dims[0]) or (mton_dims[1]!=mtoff_dims[1]) or (mton_dims[2]!=mtoff_dims[2]) ):
		print('')
		print('ERROR: the geometry of the MT on file {} and the MT off file {} do not match. Exiting with 1.'.format(mton_nifti,mtoff_nifti))					 
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
		elif ( (np.sum(mton_affine==mask_affine)!=16) or (mton_dims[0]!=mask_dims[0]) or (mton_dims[1]!=mask_dims[1]) or (mton_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the the mask file {} does not match that of the MT data. Ignoring mask...'.format(mask_nifti))					 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask is a binary file
			mask_data[mask_data>1] = 1
			mask_data[mask_data<0] = 0	
		
	else:
		got_mask = False

	### Calculate MTR map
	print('    ... calculating MTR')
	mton_data = np.array(mton_data,'float64')
	mtoff_data = np.array(mtoff_data,'float64')
	warnings.filterwarnings('ignore')    # Ignore warnings - these are going to happen in the background for sure
	mtr_map = 100*(mtoff_data - mton_data) / mtoff_data     # Calculate MTR
	mtr_map[np.isnan(mtr_map)] = 0.0   # Remove NaNs
	mtr_map[np.isinf(mtr_map)] = 0.0   # Remove Inf
	print('    ... substituting any nan and inf values with 0')
	# Use mask if required
	if got_mask==True:
		mtr_map[mask_data==0]=0

	### Save output
	print('    ... saving output file')
	buffer_header = mton_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative map as a float64
	mtr_obj = nib.Nifti1Image(mtr_map,mton_obj.affine,buffer_header)
	nib.save(mtr_obj, mtr_output)

	### Done
	print('')


# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of Magnetisation Transfer Ratio (MTR) from MT "on" and MT "off" images, which have already been co-registered. Dependencies (Python packages): nibabel, numpy (other than standard library). References: "T1, T2 relaxation and magnetization transfer in tissue at 3T", Stanisz GJ, Magnetic Resonance in Medicine (2005), 54:507-512. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('mton_file', help='3D Nifti file storing the MT "on" image (with off-resonance pulse)')
	parser.add_argument('mtoff_file', help='3D Nifti file storing the MT "off" image (without off-resonance pulse)')
	parser.add_argument('out_file', help='3D Nifti file that will store the MTR map (100 x (MToff - MTon)/MToff), as double-precision floating point (FLOAT64)')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where MTR is required, 0 where is not')
	args = parser.parse_args()

	### Get input arguments
	mtonfile = args.mton_file
	mtofffile = args.mtoff_file
	outfile = args.out_file
	maskfile = args.mask

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for MT calculation has been provided
	    maskrequest = True
	else:
	    # A mask for MT calculation has not been provided
	    maskrequest = False



	print('')
	print('********************************************************************')
	print('                           MTR calculation                          ')
	print('********************************************************************')
	print('')
	print('Called on 3D Nifti files: {} (MT "on", with off-resonance pulse) and {} (MT "off", without off-resonance pulse)'.format(mtonfile,mtofffile))
	print('Output MTR file: {}'.format(outfile))
	print('')


	if (maskrequest==False):
		MTRmap(mtonfile, mtofffile, outfile)
	else:
		MTRmap(mtonfile, mtofffile, outfile, maskfile)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

