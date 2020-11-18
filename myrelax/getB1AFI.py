### Voxel-wise calculation of flip angle deviation with the actual flip angle method by Yarnykh VL
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


def AFI(*argv):
	''' Calculate B1 map from the two-pulse spoiled gradient echo method known as "actual flip angle imaging"
	    

	    INTERFACES
	    AFI(gre1_nifti,gre2_nifti,tr1,tr2,b1_output)
	    AFI(gre1_nifti,gre2_nifti,tr1,tr2,b1_output,mask_nifti)

	     
	    PARAMETERS
	    - gre1_nifti: path of a Nifti file storing the image acquired with short TR (TR1)
	    - gre2_nifti: path of a Nifti file storing the image acquired with long TR (TR2), which should have less signal
			  than gre1_nifti (i.e. it should be darker)
	    - tr1: path of a text file storing the short TR or TR1 (in ms)
	    - tr2: path of a text file storting the long TR or TR2 (in ms), such that TR2 > TR1
	    - b1_output:  path of the Nifti file that will store the 3D B1 map (saved as a 
			  double-precision floating point image FLOAT64); such an output map is such that
			
				AF = B1 x NF
		
			  above, NF is the nominal flip angle prescribed to the scanner; 
			  AF is the actual flip angle (true flip angle played out by the scanner in each voxel); 
			  B1 is the voxel-wise B1 map (deviation between actual and nominal flip angle)

	    - mask_nifti: path of a Nifti file storting a mask (B1 map will be calculated only where
			  mask_nifti equals 1; 0 will be set in the B1 output map otherwise)
	    
	    Dependencies (Python packages): nibabel, numpy (other than standard library).
	    
	    Reference: "Actual flip-angle imaging in the pulsed steady state: a method for rapid three-dimensional 
		        mapping of the transmitted radiofrequency field", Yarnykh VL, Magnetic Resonance in Medicine 57:192-200 (2007)
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	scan1_nifti = argv[0]
	scan2_nifti = argv[1]
	tr1 = argv[2]
	tr2 = argv[3]
	b1_output = argv[4]
	
	### Load scan with short TR and check for consistency
	print('    ... loading input data')
	try:
		scan1_obj = nib.load(scan1_nifti)
	except:
		print('')
		print('ERROR: the 3D gradient echo NIFTI {} does not exist or is not in NIFTI format. Exiting with 1.'.format(scan1_nifti))
		print('')
		sys.exit(1)
	scan1_data = scan1_obj.get_fdata()
	imgsize = scan1_data.shape
	imgsize = np.array(imgsize)
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D gradient echo NIFTI {} is not a 3D NIFTI. Exiting with 1.'.format(scan1_nifti))
		print('')
		sys.exit(1)
	
	### Load scan with long TR and check for consistency
	try:
		scan2_obj = nib.load(scan2_nifti)
	except:
		print('')
		print('ERROR: the 3D gradient echo NIFTI {} does not exist or is not in NIFTI format. Exiting with 1.'.format(scan2_nifti))
		print('')
		sys.exit(1)
	scan2_data = scan2_obj.get_fdata()
	scan1_header = scan1_obj.header
	scan1_affine = scan1_header.get_best_affine()
	scan1_dims = scan1_obj.shape
	scan2_header = scan2_obj.header
	scan2_affine = scan2_header.get_best_affine()
	scan2_dims = scan2_obj.shape
	scan2_size = scan2_data.shape
	scan2_size = np.array(scan2_size)
	if scan2_size.size!=3:
		print('')
		print('ERROR: the 3D gradient echo NIFTI {} is not a 3D NIFTI. Exiting with 1.'.format(scan2_nifti))
		print('')
		sys.exit(1)
	elif ( (np.sum(scan1_affine==scan2_affine)!=16) or (scan1_dims[0]!=scan2_dims[0]) or (scan1_dims[1]!=scan2_dims[1]) or (scan1_dims[2]!=scan2_dims[2]) ):
		print('')
		print('ERROR: the geometries of the the two gradient echo files {} and {} do not match. Exiting with 1.'.format(scan1_nifti,scan2_nifti))
		print('')
		sys.exit(1)

	### Load short TR (TR1) and check for consistency

	# Make sure data exists and makes sense
	try:
		TR1value = np.loadtxt(tr1)
		TR1value = np.array(TR1value,'float64')
		TR1value_size = TR1value.size
	except:
		print('')
		print('ERROR: the TR1 file {} does not exist or is not a numeric text file. Exiting with 1.'.format(tr1))
		print('')
		sys.exit(1)
	
	# Make sure we get only one scalar value
	if TR1value_size!=1:
		print('')
		print('ERROR: the TR1 file {} contains more than one entry or is empty. Exiting with 1.'.format(tr1))
		print('')
		sys.exit(1)
	else:
		TR1 = TR1value

	### Load long TR (TR2) and check for consistency

	# Make sure data exists and makes sense
	try:
		TR2value = np.loadtxt(tr2)
		TR2value = np.array(TR2value,'float64')
		TR2value_size = TR2value.size
	except:
		print('')
		print('ERROR: the TR2 file {} does not exist or is not a numeric text file. Exiting with 1.'.format(tr2))
		print('')
		sys.exit(1)
	
	# Make sure we get only one scalar value
	if TR2value_size!=1:
		print('')
		print('ERROR: the TR2 file {} contains more than one entry or is empty. Exiting with 1.'.format(tr2))
		print('')
		sys.exit(1)
	else:
		TR2 = TR2value

	# Make sure TR2 is bigger than TR1
	if TR2<=TR1:
		print('')
		print('ERROR: the TR2 file {} must contain a TR that is bigger than the TR1 file {}. Exiting with 1.'.format(tr2,tr1))
		print('')
		sys.exit(1)
	
	### Deal with optional arguments: mask
	if Nargv==6:
		got_mask = True
		mask_nifti = argv[5]
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
		elif ( (np.sum(scan1_affine==mask_affine)!=16) or (scan1_dims[0]!=mask_dims[0]) or (scan1_dims[1]!=mask_dims[1]) or (scan1_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the the mask file {} does not match that of the gradient echo data. Ignoring mask...'.format(mask_nifti))
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask is a binary file
			mask_data[mask_data>1] = 1
			mask_data[mask_data<0] = 0	
		
	else:
		got_mask = False

	### Calculate B1 map
	print('    ... calculating B1 map')
	scan1_data = np.array(scan1_data,'float64')
	scan2_data = np.array(scan2_data,'float64')
	warnings.filterwarnings('ignore')    # Ignore warnings - these are going to happen in the background for sure
	Nratio = TR2/TR1
	Sratio = scan2_data/scan1_data
	b1_map =   np.arccos((Sratio*Nratio - 1)/(Nratio - Sratio)) # Calculate B1 map
	b1_map[np.isnan(b1_map)] = 0.0   # Remove NaNs
	b1_map[np.isinf(b1_map)] = 0.0   # Remove Inf
	print('    ... substituting any nan and inf values with 0')
	# Use mask if required
	if got_mask==True:
		b1_map[mask_data==0]=0

	### Save output
	print('    ... saving output file')
	buffer_header = scan1_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative map as a float64
	b1_obj = nib.Nifti1Image(b1_map,scan1_obj.affine,buffer_header)
	nib.save(b1_obj, b1_output)

	### Done
	print('')


# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of B1 map (flip angle deviation) using "actual flip angle imaging" based on two-pulse spoiled gradient echo imaging, by Yarnykh VL. Dependencies (Python packages): nibabel, numpy (other than standard library). Reference: "Actual flip-angle imaging in the pulsed steady state: a method for rapid three-dimensional mapping of the transmitted radiofrequency field", Yarnykh VL, Magnetic Resonance in Medicine 57:192-200 (2007). Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('nifti1_file', help='3D Nifti file storing the image acquired at short TR (TR1)')
	parser.add_argument('nifti2_file', help='3D Nifti file storing the image acquired at long TR (TR2, such that TR2 > TR1), which should exhibit less signal than the image in nifti1_file (i.e. it should be darker)')
	parser.add_argument('TR1_file', help='path of a text file storting the short TR or TR1 (in ms)')
	parser.add_argument('TR2_file', help='path of a text file storting the long TR or TR2 (in ms), such that TR2 > TR1')
	parser.add_argument('out_file', help='3D Nifti file that will store the B1 map as double-precision floating point (FLOAT64), such that the actual flip angle (AF) is related to a nominal flip angle (NF) as AF = B1 x NF')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where the flip angle map is required, 0 where is not')
	args = parser.parse_args()

	### Get input arguments
	gretr1file = args.nifti1_file
	gretr2file = args.nifti2_file
	tr1file = args.TR1_file
	tr2file = args.TR2_file
	outfile = args.out_file
	maskfile = args.mask

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for flip angle calculation has been provided
	    maskrequest = True
	else:
	    # A mask for flip angle calculation has not been provided
	    maskrequest = False



	print('')
	print('********************************************************************')
	print('            B1 mapping with actual flip angle imaging               ')
	print('********************************************************************')
	print('')
	print('Called on 3D Nifti files: {} (gradient echo with short TR, TR1) and {} (gradient echo with long TR, TR2)'.format(gretr1file,gretr2file))
	print('TR1 or short TR file: {}; TR2 or long TR file: {}'.format(tr1file,tr2file))	
	print('Output B1 map file: {}'.format(outfile))
	print('')

	if (maskrequest==False):
		AFI(gretr1file, gretr2file, tr1file, tr2file, outfile)
	else:
		AFI(gretr1file, gretr2file, tr1file, tr2file, outfile, maskfile)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

