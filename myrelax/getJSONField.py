### Get MRI sequence parameters from JSON fields in BIDS data structures 
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
import json
import numpy as np
import warnings


def getField(jsonfile,jsonfield,*argv):
	''' Get value of JSON field and save it to file if required  
	    

	    INTERFACES
	    myfield = getField(jsonfile,jsonfield)
	    myfield = getField(jsonfile,jsonfield,outfile)

	     
	    PARAMETERS
	    - jsonfile: path of a JSON file
	    - jsonfield: string containing the JSON field of interest
	    - outfile (optional): path of an output file that will store only
                                  the JSON field of interest 

	    OUTPUT
	    - myfield: value of the JSON field of interest

	    Author: Francesco Grussu, University College London
		    CDSQuaMRI Project 
		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>'''
	
	### Get input parametrs
	Nargv = len(argv)
	if(Nargv>1):
		print('')
		print('ERROR: too many input arguments. Exiting with 1.')	 			 
		print('')
		sys.exit(1)	
	
	### Load input JSON
	try:
		myhandle = open(jsonfile)
		myjson = json.load(myhandle)
		myhandle.close()
	except:
		print('')
		print('ERROR: the input JSON file {} does not exist or is not in JSON format. Exiting with 1.'.format(jsonfile))	  
		print('')
		sys.exit(1)

	### Read field and print result
	myfield = myjson.get(jsonfield)

	### If field was EchoTime or RepetitionTime, make sure value is reported in ms
	if(jsonfield=='EchoTime'):
		myfield = 1000.0*myfield
	if(jsonfield=='RepetitionTime'):
		myfield = 1000.0*myfield

	### Save field if required
	if(Nargv==1):
		mywriter = open(argv[0],'w')
		mywriter.write('{}'.format(myfield))
		mywriter.close()

	### Return field
	return myfield


# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Print field of interest from a JSON file, as those used in NIFTI-JSON pairs in BIDS, and optionally write field value to text file. Author: Francesco Grussu, University College London, CDSQuaMRI Project. Email: <francegrussu@gmail.com> <f.grussu@ucl.ac.uk>.')
	parser.add_argument('json_file', help='input JSON file')
	parser.add_argument('json_field', help='JSON field of interest (note: if field is "EchoTime" or "RepetitionTime", the value will be converted, displayed and saved to text file in ms)')
	parser.add_argument('--out', metavar='<file>', help='optional output file to which the JSON field will be written.')
	args = parser.parse_args()

	### Get input arguments
	json_file = args.json_file
	json_field = args.json_field
	out_file = args.out

	### Deal with optional arguments
	if isinstance(out_file, str)==1:
	    # An output file has been provided
	    outrequest = True
	else:
	    # An output file has not been provided
	    outrequest = False



	print('')
	print('********************************************************************')
	print('                        JSON field evaluation                       ')
	print('********************************************************************')
	print('')
	print('** Called on JSON file: {}'.format(json_file))
	print('** JSON field of interest (if RepetitionTime or EchoTime, it will be converted to ms): {}'.format(json_field))
	if(outrequest==True):
		print('** Output file: {}'.format(out_file))	
	print('')

	### Get field
	if (outrequest==False):
		fieldval = getField(json_file, json_field)
	else:
		fieldval = getField(json_file, json_field, out_file)

	print('** Value of JSON field of interest: {}'.format(fieldval))

	### Done
	print('')
	sys.exit(0)



