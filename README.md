# MyRelax overview
"Myelin and Relaxation" (MyRelax) is a collection of command line scripts written in Python 3 for myelin and relaxometry MRI. MyRelax was developed as part of the [CDS-QuaMRI project](http://cds-quamri.eu), funded under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541.

![CDSQuaMRI](https://github.com/fragrussu/MyRelax/blob/master/cdsquamri_logo.png)

MyRelax tools process MRI scans in NIFTI format, as well as other MRI sequence parameters that are passed in the form of text files. These can be readily obtained from DICOM fields or from the JSON files associated to NIFTIs according to the Brain Imaging Data Structure ([BIDS](http://bids.neuroimaging.io)).

# MyRelax dependencies
To run MyRelax you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [SciPy](http://www.scipy.org)
* [NumPy](https://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [Scikit-learn](http://scikit-learn.org/stable)


# MyRelax installation (Linux and MacOS)
Getting MyRelax is extremely easy.

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone MyRelax:
```
git clone https://github.com/fragrussu/MyRelax.git 
```
4. MyRelax is ready for you in `./MyRelax` and MyRelax scripts are in: 
```
./MyRelax/myrelax
```
5. You should now be able to use the code. Try to print the manual of a script, for instance of `getMTV.py`, to make sure this is really the case:
```
python ./MyRelax/myrelax/getMTV.py --help
```

# MyRelax tools
The following command line scripts are available.
* `getB1AFI.py`: to calculate a B1 map with the Actual Flip Angle Imaging method;
* `getB1DAGE.py`: to calculate a B1 map with the Double Angle method (gradient echo readout);
* `getB1DASE.py`: to calculate a B1 map with the Double Angle method (spin echo readout);
* `getGratio.py`: to calculate a weighted g-ratio map combining indices of myelin and axonal fraction;
* `getJSONField.py`: to extract the value of a field of interest from a JSON file (this command is useful to create text files storing TR, TE, flip angles, etc that can be passed to other MyRelax fitting routines);
* `getMTR.py`: to calculate the magnetisation transfer ratio;
* `getMTV.py`: to calculate the macromolecular tissue volume with the method of the pseudo-proton densities;
* `getT1IR.py`: to fit a mono-exponential inversion recovery model to magnitude MRI data for T1 estimation;
* `getT1VFA.py`: to fit a mono-exponential variable flip angle model to magnitude MRI data for T1 estimation;
* `getT2Prime.py`: to calculate T2' from T2 and T2*;
* `getT2T2star.py`: to fit a mono-exponential decay model to magnitude MRI data for T2 (T2*) estimation;
* `getT2T2starBiexp.py`: to fit a bi-exponential decay model to magnitude MRI data, obtaining estimates of 2 distinct T2 (or T2*) constants and of their relative signal fractions.

You can run MyRelax scripts from command line, for instance using a Bash or C shell. Some scripts support multi-core analyses (option `--ncpu`; no GPU at the moment though). Importantly, each tool has a manual: to print it, simply type in your terminal
```
python </PATH/TO/SCRIPT> --help
```
(for example, `python ./MyRelax/myrelax/getMTR.py --help`).

# If you use MyRelax
If you use MyRelax in your research, please remember to cite our [preprint](http://www.biorxiv.org/content/10.1101/859538v1):
"Multi-parametric quantitative spinal cord MRI with unified signal readout and image denoising", Grussu F, Battiston M, Veraart J, Schneider T, Cohen-Adad J, Shepherd TM, Alexander DC, Novikov DS, Fieremenas E, Gandini Wheeler-Kingshott CAM, biorxiv 2019 (DOI: 10.1101/859538).

# License
MyRelax is distributed under the BSD 2-Clause License, Copyright (c) 2019, University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/MyRelax/blob/master/LICENSE).

# Acknowledgements
The development of MyRelax was enabled by the [CDS-QuaMRI project](http://cds-quamri.eu), funded under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541. Support from the United Kingdom Engineering and Physical Sciences Research Council (EPSRC R006032/1 and M020533/1) is also acknowledged.
