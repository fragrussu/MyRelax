### Voxel-wise calculation of g-ratio
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


def GRatiomap(*argv):
	''' Calculate g-ratio from myelin volume fraction and axonal water fraction maps, already in the same space  
	    

	    INTERFACES
	    GRatiomap(mvf_nifti,awf_nifti,gratio_output)
	    GRatiomap(mvf_nifti,awf_nifti,gratio_output,mask_nifti)

	     
	    PARAMETERS
	    - mvf_nifti: path of a Nifti file storing the 3D myelin volume fraction map
	    - awf_nifti: path of a Nifti file storing the 3D axonal water fraction map (from diffusion)
	    - gratio_output: path of the Nifti file that will store the 3D output g-ratio image (saved as a 
			  double-precision floating point image FLOAT64); such an output map is
			  calculated as
			
				g = sqrt(1 - MWF/FVF)

                          where FVF is the fibre volume fraction:

			        FVF = MVF + (1 - MVF)xAWF
		
			  Above, MWF is the myelin volume fraction and AWF is the axonal water fraction.

	    - mask_nifti: path of a Nifti file storting a mask (g-ratio will be calculated only where
			  mask_nifti equals 1; 0 will be set in the g-ratio output map otherwise; the mask
                          should indicate white matter voxels)
	    
	    Dependencies (Python packages): nibabel, numpy (other than standard library).
	    
	    References: "g-Ratio weighted imaging of the human spinal cord in vivo", Duval T et al,
		        NeuroImage (2017), 145: 11-23
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	mvf_nifti = argv[0]
	awf_nifti = argv[1]
	g_output = argv[2]
	
	### Load MVF data and check for consistency
	print('    ... loading input data')
	try:
		mvf_obj = nib.load(mvf_nifti)
	except:
		print('')
		print('ERROR: the 3D input MVF file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(mvf_nifti))	 			 
		print('')
		sys.exit(1)
	mvf_data = mvf_obj.get_data()
	imgsize = mvf_data.shape
	imgsize = np.array(imgsize)
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D input MVF file {} is not a 3D NIFTI. Exiting with 1.'.format(mvf_nifti))					 
		print('')
		sys.exit(1)
	
	### Load AWF data and check for consistency
	try:
		awf_obj = nib.load(awf_nifti)
	except:
		print('')
		print('ERROR: the 3D input AWF file {} does not exist or is not in NIFTI format. Exiting with 1.'.format(awf_nifti))	   			 
		print('')
		sys.exit(1)
	awf_data = awf_obj.get_data()
	mvf_header = mvf_obj.header
	mvf_affine = mvf_header.get_best_affine()
	mvf_dims = mvf_obj.shape
	awf_header = awf_obj.header
	awf_affine = awf_header.get_best_affine()
	awf_dims = awf_obj.shape
	awf_size = awf_data.shape
	awf_size = np.array(awf_size)
	if awf_size.size!=3:
		print('')
		print('ERROR: the 3D input AWF file {} is not a 3D NIFTI. Exiting with 1.'.format(awf_nifti))					 
		print('')
		sys.exit(1)
	elif ( (np.sum(mvf_affine==awf_affine)!=16) or (mvf_dims[0]!=awf_dims[0]) or (mvf_dims[1]!=awf_dims[1]) or (mvf_dims[2]!=awf_dims[2]) ):
		print('')
		print('ERROR: the geometry of the the MVF file {} and the AWF file {} do not match. Exiting with 1.'.format(mvf_nifti,awf_nifti))					 
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
		mask_data = mask_obj.get_data()
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
		elif ( (np.sum(mvf_affine==mask_affine)!=16) or (mvf_dims[0]!=mask_dims[0]) or (mvf_dims[1]!=mask_dims[1]) or (mvf_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the the mask file {} does not match that of the MVF and AWF data. Ignoring mask...'.format(mask_nifti))					 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask is a binary file
			mask_data[mask_data>1] = 1
			mask_data[mask_data<0] = 0	
		
	else:
		got_mask = False

	### Calculate g-ratio map
	print('    ... calculating g-ratio')
	mvf_data = np.array(mvf_data,'float64')
	awf_data = np.array(awf_data,'float64')
	warnings.filterwarnings('ignore')    # Ignore warnings - these are going to happen in the background for sure
	fvf_data = mvf_data + (1.0 - mvf_data)*awf_data     # Calculate fibre volume fraction FVF from myelin volume fraction MVF and axonal volume fraction AWF
	g_map = np.sqrt(1.0 - mvf_data/fvf_data)            # Calculate g-ratio from myelin volume fraction MVF and fibre volume fraction FVF
	g_map[np.isnan(g_map)] = 0.0   # Remove NaNs
	g_map[np.isinf(g_map)] = 0.0   # Remove Inf
	print('    ... substituting any nan and inf values with 0')
	# Use mask if required
	if got_mask==True:
		g_map[mask_data==0]=0

	### Save output
	print('    ... saving output file')
	buffer_header = mvf_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative map as a float64
	g_obj = nib.Nifti1Image(g_map,mvf_obj.affine,buffer_header)
	nib.save(g_obj, g_output)

	### Done
	print('')


# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of g-ratio from myelin volume fraction and axonal water fraction, which have already been co-registered. Dependencies (Python packages): nibabel, numpy (other than standard library). Reference: "g-Ratio weighted imaging of the human spinal cord in vivo", Duval T et al, NeuroImage (2017), 145: 11-23. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('mvf_file', help='3D Nifti file storing the myelin volume fraction (MVF) map (ratio between myelin volume and the sum of myelin, intra-axonal space and extra-axonal volumes)')
	parser.add_argument('awf_file', help='3D Nifti file storing the axonal water fraction (AWF) from diffusion MRI (ratio between intra-axonal volume and the sum of intra-axonal plus extra-axonal volume, excluding myelin)')
	parser.add_argument('out_file', help='3D Nifti file that will store the g-ratio map (g = sqrt(1 - MVF/FVF), where FVF = MVF + (1 - MVF) x AWF, with FVF being the fibre volume fraction map), as double-precision floating point (FLOAT64)')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where g-ratio is required, 0 where is not (usually it should indicate white matter)')
	args = parser.parse_args()

	### Get input arguments
	mvffile = args.mvf_file
	awffile = args.awf_file
	outfile = args.out_file
	maskfile = args.mask

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for g-ratio calculation has been provided
	    maskrequest = True
	else:
	    # A mask for g-ratio calculation has not been provided
	    maskrequest = False



	print('')
	print('********************************************************************')
	print('                        G-ratio calculation                         ')
	print('********************************************************************')
	print('')
	print('Called on 3D Nifti files: {} (MVF map) and {} (AWF map)'.format(mvffile,awffile))
	print('Output g-ratio file: {}'.format(outfile))
	print('')


	if (maskrequest==False):
		GRatiomap(mvffile, awffile, outfile)
	else:
		GRatiomap(mvffile, awffile, outfile, maskfile)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

