# Fluorescence Dataset Loader
This repository contains code to load and process the Fluorescence Dataset, which includes fluorescence and histology images derived from experimental analyses of mammary gland adenocarcinomas in mice. The main functionality is provided via a Python script that returns the dataset as a `scikit-learn` `Bunch` object, making it convenient to use within the Python ecosystem, especially with tools like scikit-learn and pandas.

The dataset was first conceived with the aim of automating the histological analysis using the elemental composition of tissues given by micro-X-ray fluorescence. Since originally there were not any labels available, clustering techniques were used to identify the classes, hence the name `hist_via_cluster`.

This dataset was assembled during 2024 by Jerónimo Fotinós, part of the Condensed Matter Theory group (GTMC, FaMAF) at the National University of Córdoba (UNC), and Constanza Bencharski, part of the Atomic and Nuclear Spectroscopy group (FaMAF, UNC), with the assistance of Dr. Elio Soria from the Faculty of Medical Sciences (UNC).

![Example Image](pics/case__diet_0_mouse_4_take_0.png)


## Repository Structure

```
.
├── data/                 # Original dataset files
├── src/
│   ├── anotate.ipynb     # Notebook illustrating the annotation process
│   └── load_dataset.py   # Script with the primary function `load_fluorescence` to load the dataset
└── README.md             # This file
```

## About the Dataset

The Fluorescence Dataset comprises fluorescence images (and corresponding histology images) collected from a study on murine mammary gland adenocarcinomas. The dataset supports analysis of elemental composition through micro-XRF imaging, along with conventional histological analysis. Specific features of the dataset include:

- **Multiple Measurements:** Each mouse sample is measured one or more times, with measurements distinguished by a `take` identifier.
- **Dietary Groups:** Mice are divided into three dietary groups:
  - **0:** Control
  - **1:** Omega-3 rich
  - **2:** Omega-6 rich
- **Data Formats:** The dataset is structured as a `scikit-learn` `Bunch` object, and it can optionally include a pandas DataFrame that provides pixel-level data for the images.

### Key Attributes

The loaded dataset provides several key attributes, including (but not limited to):

- **DESCR:** A textual description of the dataset.
- **diet_names & diet_map:** Information for mapping dietary group encodings to their corresponding names.
- **element_order & element_map:** Details on the order and mapping of elements measured in the fluorescence images (e.g., Ca, Cu, Fe, etc.).
- **diet, mouse, take:** Arrays indicating the dietary group, mouse number, and measurement count respectively for each image.
- **images:** A collection (list or dictionary) of 3D NumPy arrays representing the fluorescence images. The structure depends on whether the `as_dict` parameter is set.
- **hist_img & hist_img_labels:** Histology images along with their corresponding labels.
- **img_labels:** Resized label images that match the dimensions of the fluorescence images.
- **label_map:** A mapping from integer encoding to the actual label names used for histological analysis.
- **frame:** Optionally, a pandas DataFrame with detailed pixel data that includes columns for diet, mouse, take, coordinates, element fluorescence values, and labels.

### Experimental Methodology

- **In Vivo Study:** Conducted on BALB/C mice with subcutaneous inoculation of transplantable mammary gland adenocarcinoma cells.
- **Dietary Intervention:** Mice were divided into three groups based on diet (control, omega-3, and omega-6).
- **Histological Analysis:** Tumors were processed (fixed, embedded, sectioned) and stained using hematoxylin-eosin, with images captured using an optical microscope.
- **Micro-XRF Analysis:** Conducted at the Brazilian Synchrotron Light Laboratory (LNLS) to obtain spatial distributions of elements such as S, P, Ca, Mn, Fe, Cu, and Zn.

For further details about the experimental design and analytical methods, please refer to the references provided in the dataset description below.

## Loading the Dataset

The main entry point for loading the dataset is the `load_fluorescence` function in `src/load_dataset.py`. Below is a quick example of how to use this function, assuming you have cloned the repository and that you are in its main directory:

```python
from src.load_dataset import load_fluorescence

# Load the dataset into a scikit-learn Bunch object
data = load_fluorescence(as_dict=False, as_frame=False)

# Access the dataset description
print(data.DESCR)

# Access fluorescence images:
images = data.images
```

Parameters such as `as_dict` or `as_frame` can be toggled to load the images and additional data in the desired format.

## Example Notebook

For an interactive demonstration, refer to 

## Annotation of the Data
Histological images were annotated with the aid of Dr. Elio Soria, using the `anotate.ipynb` notebook in the `src` directory. This notebook walks you through the annotation process and provides examples of how to manipulate and visualize the data.

## References

1. **Bencharski, C., Soria, E. A., Falchini, G. E., Pasqualini, M. E., & Perez, R. D. (2023).** Study of anti-tumorigenic actions of essential fatty acids in a murine mammary gland adenocarcinoma by micro-XRF. *Analytical Methods, 15(16), 2044–2051.*

2. **Falchini, G. E., Malezan, A., Poletti, M. E., Soria, E., Pasqualini, M., & Perez, R. D. (2021).** Analysis of phosphorous content in cancer tissue by synchrotron micro-XRF. *Radiation Physics and Chemistry, 179, 109157.*


---

Happy data exploring!
