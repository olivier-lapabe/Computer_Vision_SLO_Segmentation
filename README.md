## MSIA - Retina blood vessel segmentation
A comprehensive solution employing morphological image processing techniques to segment blood vessels within retinal images, aimed at improving diagnostic procedures and educational understanding of medical image analysis.


## Getting Started

### Dependencies

Refer to `requirements.txt` for a full list of dependencies.

### Installing

#### For Users:

* To install our project: 

```
git clone https://github.com/olivier-lapabe/Computer_Vision_SLO_Segmentation.git
cd Computer_Vision_SLO_Segmentation
pip install .
```

#### For Developers/Contributors:

If you're planning to contribute or test the latest changes, you should first set up a virtual environment and then install the package in "editable" mode. This allows any changes you make to the source files to immediately affect the installed package without requiring a reinstall.

* Clone the repository:

```
git clone https://github.com/JPGodTier/Computer_Vision_SLO_Segmentation.git
cd Computer_Vision_SLO_Segmentation
```

* Set up a virtual environment:

```
python3 -m venv seg_env
source seg_env/bin/activate  # On Windows, use: seg_env\Scripts\activate
```

* Install the required dependencies:

```
pip install -r requirements.txt
```

* Install the project in editable mode:

```
pip install -e . 
```

### Executing program

Launch the program:  
```
python3 bin/SegmentationRunner.py
```

## Authors

* Paul Aristidou
* Olivier Lapabe-Goastat

## Version History

* **1.0.0** - Initial release