### Voxel-wise calculation of flip angle deviation with the double angle method on spin echo data
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


def DASE(*argv):
	''' Calculate B1 map from two spin echo images using the "double angle" method
	    

	    INTERFACES
	    DASE(se1_nifti,se2_nifti,theta,se_type,b1_output)
	    DASE(se1_nifti,se2_nifti,theta,se_type,b1_output,mask_nifti)

	     
	    PARAMETERS
	    - se1_nifti: path of a Nifti file storing the spin echo image acquired with excitation flip angle theta
	    - se2_nifti: path of a Nifti file storing the spin echo image acquired with excitation flip angle 2 x theta
	    - theta: path of a text file storing the flip angle (in degrees)
	    - se_type: a string specifying the type of spin echo sequence used (set it to "theta_2theta" for a spin echo 
                       sequence like "theta-TE/2-2theta-TE/2-acquisition", as in Insko et al, JMR 1993; 
                       or set it to "theta_180" for a spin echo sequence like "theta-TE/2-180-TE/2-acquisition", 
                       as in Stollberger et al, MRM 1996) 
	    - b1_output:  path of the Nifti file that will store the 3D B1 map (saved as a 
			  double-precision floating point image FLOAT64); such an output map is such that
			
				AF = B1 x NF
		
			  above, NF is the nominal flip angle prescribed to the scanner; 
			  AF is the actual flip angle (true flip angle played out by the scanner in each voxel); 
			  B1 is the voxel-wise B1 map (deviation between actual and nominal flip angle)

	    - mask_nifti: path of a Nifti file storting a mask (B1 map will be calculated only where
			  mask_nifti equals 1; 0 will be set in the B1 output map otherwise)
	    
	    Dependencies (Python packages): nibabel, numpy (other than standard library)
	    
	    References: 
		        "Mapping of the radiofrequency field", 
		        Insko EK and Bolinger L, Journal of Magnetic Resonance Series A, 103:82-85 (1993)

			"Imaging of the active B1 field in vivo", 
                        Stollberger R and Wach P, Magnetic Resonance in Medicine, 35(2): 246-251 (1996)
	     
	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	scan1_nifti = argv[0]
	scan2_nifti = argv[1]
	theta_file = argv[2]
	se_type = argv[3]
	b1_output = argv[4]
	
	### Load scan with original flip angle and check for consistency
	print('    ... loading input data')
	try:
		scan1_obj = nib.load(scan1_nifti)
	except:
		print('')
		print('ERROR: the 3D spin echo NIFTI {} does not exist or is not in NIFTI format. Exiting with 1.'.format(scan1_nifti))
		print('')
		sys.exit(1)
	scan1_data = scan1_obj.get_fdata()
	imgsize = scan1_data.shape
	imgsize = np.array(imgsize)
	if imgsize.size!=3:
		print('')
		print('ERROR: the 3D spin echo NIFTI {} is not a 3D NIFTI. Exiting with 1.'.format(scan1_nifti))
		print('')
		sys.exit(1)
	
	### Load scan acquired with doubled flip angle and check for consistency
	try:
		scan2_obj = nib.load(scan2_nifti)
	except:
		print('')
		print('ERROR: the 3D spin echo NIFTI {} does not exist or is not in NIFTI format. Exiting with 1.'.format(scan2_nifti))
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
		print('ERROR: the 3D spin echo NIFTI {} is not a 3D NIFTI. Exiting with 1.'.format(scan2_nifti))
		print('')
		sys.exit(1)
	elif ( (np.sum(scan1_affine==scan2_affine)!=16) or (scan1_dims[0]!=scan2_dims[0]) or (scan1_dims[1]!=scan2_dims[1]) or (scan1_dims[2]!=scan2_dims[2]) ):
		print('')
		print('ERROR: the geometries of the the two spin echo files {} and {} do not match. Exiting with 1.'.format(scan1_nifti,scan2_nifti))
		print('')
		sys.exit(1)

	### Load the flip angle file

	# Make sure data exists and makes sense
	try:
		thetavalue = np.loadtxt(theta_file)
		thetavalue = np.array(thetavalue,'float64')
		thetavalue_size = thetavalue.size
	except:
		print('')
		print('ERROR: the flip angle file {} does not exist or is not a numeric text file. Exiting with 1.'.format(theta_file))
		print('')
		sys.exit(1)
	
	# Make sure we get only one scalar value
	if thetavalue_size!=1:
		print('')
		print('ERROR: the flip angle file {} contains more than one entry or is empty. Exiting with 1.'.format(theta_file))
		print('')
		sys.exit(1)
	else:
		theta_nominal = thetavalue

	## Select the sequence type
	if se_type=='theta_2theta':
		isseq_theta180 = 0

	elif se_type=='theta_180':
		isseq_theta180 = 1

	else:
		print('')
		print('ERROR: the sequence type "{}" is not recognised. Choose either "theta_2theta" or "theta_180". Exiting with 1.'.format(se_type))
		print('')			
		sys.exit(1)

	### Deal with optional arguments: mask
	if Nargv==6:
		got_mask = True
		mask_nifti = argv[4]
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
			print('WARNING: the geometry of the the mask file {} does not match that of the spin echo data. Ignoring mask...'.format(mask_nifti))
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
	Sratio = scan1_data/scan2_data
	if isseq_theta180==0:
		theta_measured = np.arccos( 0.5*( (1.0/Sratio)**(1/3) ) )  # Sequence is theta - TE/2 - 2 x theta - TE/2, as Insko EK and Bolinger L
	elif isseq_theta180==1:
		theta_measured = np.arccos( 0.5*(1.0/Sratio) ) # Sequence is theta - TE/2 - 180 - TE/2, as Insko EK and Bolinger L
	else:
		print('')
		print('ERROR: the sequence type "{}" is not recognised. Choose either "theta_2theta" or "theta_180". Exiting with 1.'.format(se_type))
		print('')			
		sys.exit(1)

	b1_map = theta_measured/(np.pi*theta_nominal/180.0)
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
	parser = argparse.ArgumentParser(description='Voxel-wise calculation of B1 map (flip angle deviation) using the "double angle" method based on two spin echo acquisitions. Dependencies (Python packages): nibabel, numpy (other than standard library). References: "Mapping of the radiofrequency field", Insko EK and Bolinger L, Journal of Magnetic Resonance Series A, 103:82-85 (1993); "Imaging of the active B1 field in vivo", Stollberger R and Wach P, Magnetic Resonance in Medicine, 35(2): 246-251 (1996). Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('nifti1_file', help='3D Nifti file storing the spin echo image acquired with flip angle theta')
	parser.add_argument('nifti2_file', help='3D Nifti file storing the spin echo image acquired with flip angle 2 x theta')
	parser.add_argument('flipangle_file', help='path of a text file storting the flip angle theta (in degrees)')
	parser.add_argument('out_file', help='3D Nifti file that will store the B1 map as double-precision floating point (FLOAT64), such that the actual flip angle (AF) is related to a nominal flip angle (NF) as AF = B1 x NF')
	parser.add_argument('--refocus', metavar='<type>', default="theta_180", help='type of refocussing pulse of the spin echo sequence (set it to "theta_2theta" for a spin echo sequence like "theta-TE/2-2theta-TE/2-acquisition", as in Insko et al, JMR 1993; or set it to "theta_180" for a spin echo sequence like "theta-TE/2-180-TE/2-acquisition", as in Stollberger et al, MRM 1996; default "theta_180")')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where the flip angle map is required, 0 where is not')
	args = parser.parse_args()

	### Get input arguments
	spinecho1file = args.nifti1_file
	spinecho2file = args.nifti2_file
	fafile = args.flipangle_file
	outfile = args.out_file
	refocustype = args.refocus
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
	print('     B1 mapping with the double angle method on spin echo data      ')
	print('********************************************************************')
	print('')
	print('Called on 3D Nifti files: {} (spin echo images for flip angle theta) and {} (spin echo images for flip angle 2 x theta)'.format(spinecho1file,spinecho2file))
	print('Flip angle file: {}'.format(fafile))
	print('Refocusing pulse thype: {}'.format(refocustype))	
	print('Output B1 map file: {}'.format(outfile))
	print('')


	if (maskrequest==False):
		DASE(spinecho1file, spinecho2file, fafile, refocustype, outfile)
	else:
		DASE(spinecho1file, spinecho2file, fafile, refocustype, outfile, maskfile)

	### Done
	print('Processing completed.')
	print('')
	sys.exit(0)

