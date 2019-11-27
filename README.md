# MyRelax
"Myelin and Relaxation" (MyRelax) is a collection of command line scripts written in Python 3 for myelin and relaxometry MRI. MyRelax was developed as part of the [CDS-QuaMRI project](http://cds-quamri.eu), funded under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541.

MyRelax tools require MRI scans in NIFTI format, as well as other MRI sequence parameters that are passed to MyRelax routines as text files. These can be readily obtained from DICOM fields or from the JSON files associated to NIFTIs when the Brain Imaging Data Structure is adopted ([BIDS](http://bids.neuroimaging.io)).

# Installation
Gettins MyRelax is extremely easy.

1. Clone this repository:
```
git clone https://github.com/fragrussu/MyRelax.git 
```
2. This will download MyRelax to the `./MyRelax` folder. MyRelax scripts are in: 
```
./MyRelax/myrelax
```
3. You should now be able to use the code. Try to print the manual of `getMTV.py` to make sure:
```
python ./MyRelax/myrelax/getMTV.py --help
```
