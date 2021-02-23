# MyRelax overview
"Myelin and Relaxation" (MyRelax) is a collection of command line scripts written in Python 3 for myelin and relaxometry MRI. MyRelax was developed as part of the [CDS-QuaMRI project](https://cordis.europa.eu/project/id/634541), funded under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541.

![CDSQuaMRI](https://github.com/fragrussu/MyRelax/blob/master/cdsquamri_logo.png)

MyRelax tools process MRI scans in NIFTI format, as well as other MRI sequence parameters that are passed in the form of text files. These can be readily obtained from DICOM fields or from the JSON files associated to NIFTIs according to the Brain Imaging Data Structure ([BIDS](http://bids.neuroimaging.io)).

# MyRelax dependencies
To run MyRelax you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [SciPy](http://www.scipy.org)
* [NumPy](https://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [Scikit-learn](http://scikit-learn.org/stable)
* [Scikit-image](http://scikit-image.org)


# MyRelax download
Getting MyRelax is extremely easy: cloning this repository is all you need to do.


If you use Linux or MacOs:

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
The following command line scripts are available within the [`myrelax`](http://github.com/fragrussu/MyRelax/tree/master/myrelax) folder.
* [`getB1AFI.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getB1AFI.py): to calculate a B1 map with the [Actual Flip Angle Imaging method](http://doi.org/10.1002/mrm.21120);
* [`getB1DAGE.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getB1DAGE.py): to calculate a B1 map with the [Double Angle method](https://doi.org/10.1006/jmra.1993.1133) (gradient echo readout);
* [`getB1DASE.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getB1DASE.py): to calculate a B1 map with the [Double Angle method](https://doi.org/10.1006/jmra.1993.1133) (spin echo readout);
* [`getGratio.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getGratio.py): to calculate a [weighted g-ratio](http://doi.org/10.1016/j.neuroimage.2015.05.023) map combining indices of myelin and axonal fraction;
* [`getJSONField.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getJSONField.py): to extract the value of a field of interest from a JSON file (this command is useful to create text files storing TR, TE, flip angles, etc that can be passed to other MyRelax fitting routines);
* [`getMTR.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getMTR.py): to calculate the [magnetisation transfer ratio](http://doi.org/10.1002/ana.20202);
* [`getMTV.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getMTV.py): to calculate the macromolecular tissue volume with the method of the [pseudo-proton densities](http://doi.org/10.1016/j.neuroimage.2012.06.076);
* [`getT1IR.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getT1IR.py): to fit a mono-exponential inversion recovery model to magnitude MRI data for [T1 estimation](http://doi.org/10.1002/mrm.25135);
* [`getT1VFA.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getT1VFA.py): to fit a mono-exponential variable flip angle model to magnitude MRI data for [T1 estimation](http://doi.org/10.1002/mrm.25135);
* [`getT2Prime.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getT2Prime.py): to calculate [T2'](http://doi.org/10.1148/radiol.2483071602) from [T2 and T2*](http://doi.org/10.1097/RMR.0b013e31821e56d8);
* [`getT2T2star.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getT2T2star.py): to fit a mono-exponential decay model to magnitude MRI data for [T2 or T2*](http://doi.org/10.1097/RMR.0b013e31821e56d8) estimation;
* [`getT2T2starBiexp.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getT2T2starBiexp.py): to fit a [bi-exponential](http://doi.org/10.1002/mrm.1910350212) decay model to magnitude MRI data, obtaining estimates of 2 distinct T2 (or T2*) constants and of their relative signal fractions;
* [`getPDT2T1TurboSpinEcho.py`](https://github.com/fragrussu/MyRelax/blob/master/myrelax/getPDT2T1TurboSpinEcho.py): to estimate quantitative proton density, T1 and T2 from 3 spin echo scans performed at two different TEs and one different TR - see Ricciardi A, Grussu F et al, Proceedings of ISMRM 2018, p. 2846.

You can run MyRelax scripts from command line, for instance using a Bash or C shell. Some scripts support multi-core analyses (option `--ncpu`; no GPU at the moment though). Importantly, each tool has a manual: to print it, simply type in your terminal
```
python </PATH/TO/SCRIPT> --help
```
(for example, `python ./MyRelax/myrelax/getMTR.py --help`).

# If you use MyRelax
If you use MyRelax in your research, please remember to cite our paper:

"Multi-parametric quantitative in vivo spinal cord MRI with unified signal readout and image denoising". Grussu F, Battiston M, Veraart J, Schneider T, Cohen-Adad J, Shepherd TM, Alexander DC, Fieremans E, Novikov DS, Gandini Wheeler-Kingshott CAM; [NeuroImage 2020, 217: 116884](http://doi.org/10.1016/j.neuroimage.2020.116884) (DOI: 10.1016/j.neuroimage.2020.116884).

# License
MyRelax is distributed under the BSD 2-Clause License, Copyright (c) 2019 and 2020, University College London. All rights reserved.

Link to license [here](http://github.com/fragrussu/MyRelax/blob/master/LICENSE).

# Acknowledgements
The development of MyRelax was enabled by the [CDS-QuaMRI project](https://cordis.europa.eu/project/id/634541), funded under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541. Support from the United Kingdom Engineering and Physical Sciences Research Council (EPSRC R006032/1 and M020533/1) is also acknowledged.
