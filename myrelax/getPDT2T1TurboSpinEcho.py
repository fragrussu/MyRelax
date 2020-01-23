### Voxel-wise estimation of relaxometry metrics from 3 TSE or RARE scans
#
# Author: Francesco Grussu, University College London
#		    CDSQuaMRI Project 
#		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>
#
# Code released under BSD Two-Clause license
#
# Copyright (c) 2019, 2020 University College London. 
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

def AnalysePDT1T2(*argv):
	'''	Analyse three TSE or RARE scans with various TR and TE to estimate apparent proton density, T1 and T2
		
		The function expects two scans with different TE and the same TR (scans 1 and 2), as well as a third scan
               with different TR and, potentially, even with a third different TE
		
		INTERFACES:
		AnalysePDT1T2(Scan1_file, Scan1_TE, Scan1_TR, Scan2_file, Scan2_TE, Scan3_file, Scan3_TE, Scan3_TR, out_root, maskfile)
		AnalysePDT1T2(Scan1_file, Scan1_TE, Scan1_TR, Scan2_file, Scan2_TE, Scan3_file, Scan3_TE, Scan3_TR, out_root)
				     
		Author: Francesco Grussu, University College London
		<f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''


	# Get input parametrs
	print('    ... loading input data')
	Nargv = len(argv)

	Scan1_file = argv[0] 
	TE1 = argv[1] 
	TR1 = argv[2]

	Scan2_file = argv[3] 
	TE2 = argv[4]

	Scan3_file = argv[5] 
	TE3 = argv[6] 
	TR3 = argv[7] 
	out_root = argv[8]


	# Load data
	scan1 = nib.load(Scan1_file)
	scan1_data = scan1.get_data()
	imgsize = scan1_data.shape
	scan1_data = np.array(scan1_data,'float64')
	imgsize = np.array(imgsize)

	scan2 = nib.load(Scan2_file)
	scan2_data = scan2.get_data()
	scan2_data = np.array(scan2_data,'float64')

	scan3 = nib.load(Scan3_file)
	scan3_data = scan3.get_data()
	scan3_data = np.array(scan3_data,'float64')

	if Nargv==10:
		mask_file = argv[9]
		mask = nib.load(mask_file)
		mask_data = mask.get_data()
		mask_data = np.array(mask_data,'float64')
	else:
		mask_data = np.ones(imgsize[0:3],'float64')

	# Create empty outputs
	s0_map = np.zeros(imgsize[0:3],'float64')
	t2_map = np.zeros(imgsize[0:3],'float64')
	t1_map = np.zeros(imgsize[0:3],'float64')
	flag_map = np.zeros(imgsize[0:3],'float64')

	# Voxel-wise analysis
	print('    ... performing voxel-wise analysis:')
	t1_array_init = np.linspace(300.0,5500.0,num=5201)   # Array used for initial grid search with resolution of 1 ms
	for zz in range(0, imgsize[2]):
		print('             axial slice {} out of {}'.format(zz+1,imgsize[2]))

		for yy in range(0, imgsize[1]):
			for xx in range(0, imgsize[0]):

				# Get mask
				mask_val = mask_data[xx,yy,zz]

				# Mask flags that the voxel should be analysed
				if mask_val==1:
					
					S1 = scan1_data[xx,yy,zz]
					S2 = scan2_data[xx,yy,zz]
					S3 = scan3_data[xx,yy,zz]

					# Check that measurements make sense
					if( (S1<0) or (S2<0) or (S3<0) or (np.isnan(S1)==1) or (np.isnan(S2)==1) or (np.isnan(S3)==1) or (np.isinf(S1)==1) or (np.isinf(S2)==1) or (np.isinf(S3)==1) ):
						s0_voxel = 0.0
						t1_voxel = 0.0
						t2_voxel = 0.0
						flag_voxel = -1.0

					# If so, estimate T2
					else:
						with np.errstate(divide='raise',invalid='raise'):
							try:
								t2_voxel = (TE2 - TE1) / np.log(S1/S2)
							except FloatingPointError:
								s0_voxel = 0.0
								t1_voxel = 0.0
								t2_voxel = 0.0
								flag_voxel = -1.0

						# Check that T2 makes sense
						if( (t2_voxel<=0) or (np.isnan(t2_voxel)) or (np.isinf(t2_voxel)) ):
							s0_voxel = 0.0
							t1_voxel = 0.0
							t2_voxel = 0.0
							flag_voxel = -1.0
						else:
							flag_voxel = 1.0

						# If so, estimate T1 iteratively
						if(flag_voxel==1):
							with np.errstate(divide='raise',invalid='raise'):	
								try:
									R_const = S1/S3
									beta_const = np.exp((TE3 - TE1)/t2_voxel)

									# First gross grid search with resolution of 1 ms													
									foo_val_init = ( R_const  - beta_const*( 1.0 - np.exp( (-1.0)*TR1/t1_array_init ) )/( 1.0 - np.exp( (-1.0)*TR3/t1_array_init ) ) )**2
									t1_idx_init = np.argmin(foo_val_init)
									t1_voxel_init = t1_array_init[t1_idx_init]
							
									# Second, much finer grid search in the range T1gross_search +/- 25 ms with resolution 0.01 ms
									t1_array_fine = np.linspace(t1_voxel_init - 25.0,t1_voxel_init + 25.0,num=5000)													
									foo_val = ( R_const  - beta_const*( 1.0 - np.exp( (-1.0)*TR1/t1_array_fine ) )/( 1.0 - np.exp( (-1.0)*TR3/t1_array_fine ) ) )**2
									t1_idx = np.argmin(foo_val)
									t1_voxel = t1_array_fine[t1_idx]

									# Apparent proton density and fitting flag
									s0_voxel = S3 / ( ( 1.0 - np.exp( (-1.0)*TR3/t1_voxel )  )*np.exp( (-1.0)*TE3/t2_voxel ) )
									flag_voxel = 1.0

								except FloatingPointError:
									s0_voxel = 0.0
									t1_voxel = 0.0
									t2_voxel = 0.0
									flag_voxel = -1.0


				# Background voxel
				else:
					s0_voxel = 0.0
					t1_voxel = 0.0
					t2_voxel = 0.0
					flag_voxel = 0.0

				# Store quantitative metrics and flags for current voxels
				s0_map[xx,yy,zz] = s0_voxel
				t2_map[xx,yy,zz] = t2_voxel 
				t1_map[xx,yy,zz] = t1_voxel
				flag_map[xx,yy,zz] = flag_voxel

	### Save the output maps
	print('    ... saving output files')
	buffer_string=''
	seq_string = (out_root,'_S0.nii')
	s0_outfile = buffer_string.join(seq_string)
	
	buffer_string=''
	seq_string = (out_root,'_T2.nii')
	t2_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (out_root,'_T1.nii')
	t1_outfile = buffer_string.join(seq_string)

	buffer_string=''
	seq_string = (out_root,'_outflag.nii')
	flag_outfile = buffer_string.join(seq_string)


	buffer_header = scan1.header
	buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type
	
	s0_obj = nib.Nifti1Image(s0_map,scan1.affine,buffer_header)
	nib.save(s0_obj, s0_outfile)
	
	t2_obj = nib.Nifti1Image(t2_map,scan1.affine,buffer_header)
	nib.save(t2_obj, t2_outfile)

	t1_obj = nib.Nifti1Image(t1_map,scan1.affine,buffer_header)
	nib.save(t1_obj, t1_outfile)

	flag_obj = nib.Nifti1Image(flag_map,scan1.affine,buffer_header)
	nib.save(flag_obj, flag_outfile)


	### Done
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Analyse three Spin Echo scans (e.g. TSE or RARE) jointly to perform a voxel-wise three-point estimation of three tissue parameters: apparent proton density; quantitative value of T1; quantitative value of T2. Scan 1 and scan 2 ought to have different TE but the same TR; scan 3 ought to have different TR from scan 1 and 2. Scan 3 can have the same or different TE as scan 1 and 2. Note that depending on the readout design, TR may need be corrected to account for long echo trains befoire being passed to this code (e.g. TR = TRnominal - TF x ES, where TF is the readout turbo factor and ES is the echo spacing). Similarly, nominal echo time should be replaced by effective echo time (i.e. time at which the centre of k-space is filled). Author: Francesco Grussu, UCL Queen Square Institute of Neurology and Department of Computer Science, University College London <f.grussu@ucl.ac.uk>')
	parser.add_argument('Scan1_file', help='3D Nifti file storing scan 1 (short TR and short TE)')
	parser.add_argument('Scan1_te', help='Numeric value of the equivalent TE for scan 1 (ms)')
	parser.add_argument('Scan1_tr', help='Numeric value of the TR for scan 1 (ms)')
	parser.add_argument('Scan2_file', help='3D Nifti file storing scan 2 (short TR and long TE; same TR as scan 1)')
	parser.add_argument('Scan2_te', help='Numeric value of the equivalent TE for scan 2 (ms)')
	parser.add_argument('Scan3_file', help='3D Nifti file storing scan 1 (different TR from scan 1 and 2)')
	parser.add_argument('Scan3_te', help='Numeric value of the equivalent TE for scan 3 (ms)')
	parser.add_argument('Scan3_tr', help='Numeric value of the TR for scan 3 (ms)')
	parser.add_argument('out_root', help='root of output file names, to which file-specific strings will be added; output files will be double-precision floating point (FLOAT64) and will end in "_S0.nii" (apparent proton density); "_T1.nii" (T1 map in ms); "_T2.nii" (T2 map in ms); "_outflag.nii" (0: background; 1: successful analysis; -1: error, voxel to be discarded due to invalid T2/T1/PD).')
	parser.add_argument('--mask', metavar='<file>', help='mask in Nifti format where 1 flags voxels where fitting is required, 0 where is not')
	args = parser.parse_args()


	maskfile = args.mask

	### Deal with optional arguments
	if isinstance(maskfile, str)==1:
	    # A mask for fitting has been provided
	    maskrequest = True
	else:
	    # A mask for fitting has not been provided
	    maskrequest = False

	print('')
	print('********************************************************************************')
	print('   Analysis of three TSE or RARE scans to estimate proton density, T1 and T2    ')
	print('********************************************************************************')
	print('')

	if (maskrequest==False):
		AnalysePDT1T2(args.Scan1_file, np.float(args.Scan1_te), np.float(args.Scan1_tr), args.Scan2_file, np.float(args.Scan2_te), args.Scan3_file, np.float(args.Scan3_te), np.float(args.Scan3_tr), args.out_root)
	else:
		AnalysePDT1T2(args.Scan1_file, np.float(args.Scan1_te), np.float(args.Scan1_tr), args.Scan2_file, np.float(args.Scan2_te), args.Scan3_file, np.float(args.Scan3_te), np.float(args.Scan3_tr), args.out_root, maskfile)

	print('Processing completed.')
	print('')
	sys.exit(0)



