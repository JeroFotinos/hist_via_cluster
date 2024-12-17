import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import Bunch

# Define the order of elements in lowercase
element_order = ["Ca", "Cu", "Fe", "K", "Mn", "P", "S", "Zn"]

# Mapping for diet categories
diet_mapping = {"control": 0, "omega3": 1, "omega6": 2}
diet_names = ["control", "omega3", "omega6"]

# Maps for Bunch objects
diet_map = {0: "control", 1: "omega3", 2: "omega6"}
element_map = {i: elem for i, elem in enumerate(element_order)}
label_map = {
    0: "no label",
    1: "necrotic tissue",
    2: "tumoral A",
    3: "tumoral B",
    4: "tumoral C",
    5: "artifacts",
    6: "blood",
    7: "loose connective tissue",
    8: "no sample",
    9: "dense connective tissue",
    10: "paraffin",
}


# Function to extract metadata from the filename
def extract_metadata(filename: str) -> Tuple[int, int, int, str]:
    """
    Extracts metadata from the filename.

    Names are of the form 'dieta_control_raton_1_toma_0_element_Ca.dat'.

    Parameters
    ----------
    filename : str
        The name of the file from which to extract metadata.

    Returns
    -------
    diet : int
        Integer corresponding to the diet category (0 for 'control',
        1 for 'omega3', 2 for 'omega6').
    mouse : int
        The mouse identifier number.
    take : int
        The take number.
    elem : str
        The element symbol.
    """
    # Extract just the file name, ignoring directory
    basename = os.path.basename(filename)
    parts = basename.split("_")
    diet = diet_mapping[parts[1]]
    mouse = int(parts[3])
    take = int(parts[5])
    elem = parts[7].replace(".dat", "")
    return diet, mouse, take, elem


# Function to identify image type and extract metadata
def identify_and_extract_metadata(
    filename: str,
) -> Tuple[str, Optional[Tuple[int, int, int, Optional[str]]]]:
    """
    Identifies the type of image and extracts metadata from the filename.

    Names are of the form
    - Fluorescence images (element):
        'dieta_control_raton_1_toma_0_element_Ca.dat';
    - Histological images (hist_img - recort):
        'dieta_control_raton_7_toma_0_hist-recort.png';
    - Labels for histological images (hist_img_labels - labels):
        'dieta_control_raton_7_toma_0_hist-labels.tif'.

    Parameters
    ----------
    filename : str
        The name of the file to analyze.

    Returns
    -------
    img_type : str
        The type of the image. Possible values are:
        - 'fluorescence': Fluorescence images with 'element' in the name.
        - 'hist_img': Histology images with 'recort' in the name.
        - 'hist_img_labels': Labels for the histology images with 'labeles' in
            the name.
    metadata : Tuple[int, int, int, Optional[str]]
        A tuple containing metadata:
        - diet : int
        - mouse : int
        - take : int
        - elem : str (only for fluorescence images, otherwise None)
    """
    basename = os.path.basename(filename)
    if "element" in basename:
        parts = basename.split("_")
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        elem = parts[7].replace(".dat", "")
        return "fluorescence", (diet, mouse, take, elem)
    elif "recort" in basename:
        parts = basename.split("_")
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        return "hist_img", (diet, mouse, take, None)
    elif "labels" in basename:
        parts = basename.split("_")
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        return "hist_img_labels", (diet, mouse, take, None)
    else:
        return "unknown", None


# Function to load fluorescence data from a file
def load_fluo_image(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the fluorescence image data from a file.

    Parameters
    ----------
    filepath : str
        The path to the file containing the image data.

    Returns
    -------
    pixels : np.ndarray
        A 2D array where each row contains the row and column indices of a
        pixel.
    fluorescence : np.ndarray
        A 1D array containing the fluorescence values for each pixel.
    """
    data = np.loadtxt(filepath, skiprows=1)  # Assuming first row is header
    return data[:, :2], data[:, 2]  # Pixel coordinates and fluorescence


def load_hist_image(filepath: str) -> np.ndarray:
    """
    Loads a histology image (e.g., PNG format) into a NumPy array.

    Parameters
    ----------
    filepath : str
        The path to the histology image file.

    Returns
    -------
    hist_image : np.ndarray
        A NumPy array representation of the histology image.
        The shape will depend on the image (e.g., (height, width, channels)).
    """
    # Open the image file
    with Image.open(filepath) as img:
        # Convert to a NumPy array
        hist_image = np.array(img)
    return hist_image


def load_labels_image(filepath: str) -> np.ndarray:
    """
    Loads the labels for the histology image (e.g., TIFF format) into a NumPy
    array.

    Parameters
    ----------
    filepath : str
        The path to the histology labels image file.

    Returns
    -------
    labels_image : np.ndarray
        A NumPy array representation of the labels for the histology image.
        The shape will depend on the image (e.g., (height, width) for
        single-channel).
    """
    # Open the image file
    with Image.open(filepath) as img:
        # Convert to a NumPy array
        labels_image = np.array(img)
    return labels_image


def find_files(directory: str) -> List[str]:
    """
    Recursively finds all files with specific extensions in the specified
    directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The root directory in which to search for files.

    Returns
    -------
    files : list of str
        A list of paths to the found files.
    """
    valid_extensions = {".dat", ".tif", ".tiff", ".jpg", ".png"}
    found_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                found_files.append(os.path.join(root, file))

    return found_files


def resize_with_majority_rule(label_image, target_shape):
    """
    Resizes a labeled image to the target shape using a majority rule.

    Parameters
    ----------
    label_image : np.ndarray
        The original labeled image.
    target_shape : tuple
        The desired output shape (height, width).

    Returns
    -------
    resized_image : np.ndarray
        The resized labeled image.
    """
    # Determine the scaling factors
    scale_h = label_image.shape[0] / target_shape[0]
    scale_w = label_image.shape[1] / target_shape[1]

    # Create an output array for the resized image
    resized_image = np.zeros(target_shape, dtype=int)

    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            # Map the output pixel back to the original image's patch
            start_h = int(i * scale_h)
            end_h = int((i + 1) * scale_h)
            start_w = int(j * scale_w)
            end_w = int((j + 1) * scale_w)

            # Ensure the indices are within bounds
            end_h = min(end_h, label_image.shape[0])
            end_w = min(end_w, label_image.shape[1])

            # Extract the patch from the original image
            patch = label_image[start_h:end_h, start_w:end_w]

            # Apply the majority rule to determine the new pixel's value
            if patch.size > 0:
                labels, counts = np.unique(patch, return_counts=True)
                majority_label = labels[np.argmax(counts)]
                resized_image[i, j] = majority_label

    return resized_image


# Main function to load fluorescence data
def load_fluorescence(
    directory: str, as_frame: bool = False, as_dict: bool = False
) -> Bunch:
    """
    Loads fluorescence data and histological images and labels from the
    specified directory and returns it as a Bunch object.

    Parameters
    ----------
    directory : str
        The root directory containing the .dat files with the fluorescence
        data.
    as_frame : bool, default=False
        If True, Bunch includes data as a pandas DataFrame
    as_dict : bool, default=False
        If True, Bunch includes images in a dictionary of 3D NumPy arrays,
        with keys corresponding to diet, mouse, and take.

    Returns
    -------
    dataset : Bunch
        A Bunch object containing:
            - DESCR: A string description of the dataset.
            - diet_names: A list of diet names where the index of each element
                was used in `diet` instead of the string, i.e., '0' was used
                instead of diet_names[0], which is 'control'.
            - diet_map: A dictionary mapping integer encoding of diet names to
                the actual diet names.
            - element_order: A list of strings indicating the order of
                elements in the 3D images. E.g., if `as_dict` is False, that
                means that `images[n][0]` is the 2D image for the fluorescence
                of `element_order[0]`, which is `'Ca'`, of `mouse[n]`,
                `take[n]` with `diet[n]`. If `as_dict` is True,
                `images[(diet, mouse, take)][0]` is the 2D image for the
                fluorescence of `element_order[0]`, which is `'Ca'`, of take
                `take` of mouse `mouse` with diet `diet`.
            - element_map: A dictionary mapping integer encoding of element
                names to the actual element names.
            - diet: A 1D array of diet categories corresponding to each 3D
                image. Provided if `as_dict` is False.
            - mouse: A 1D array of mouse numbers corresponding to each 3D
                image. Provided if `as_dict` is False.
            - take: A 1D array of take numbers corresponding to each 3D image.
                Take is just an int that enumerates, starting from zero, the
                different measurements done for the same mouse. Provided if
                `as_dict` is False.
            - images: if `as_dict` is False, list of 3D NumPy arrays, one for
                each combination of diet, mouse, and take. If `as_dict` is
                True, a dictionary of 3D NumPy arrays, with keys corresponding
                to diet, mouse, and take, given as a tuple
                `(diet, mouse, take)`.
            - hist_img: list of histology images, either as a list of NumPy
                arrays if `as_dict` is False or as a dictionary of NumPy
                arrays with the usual `(diet, mouse, take)` key, if `as_dict`
                is True.
            - hist_img_labels: list of labeles for the histology images, with
                the same dimensions and data structure of `hist_img`.
            - img_labels: list of resized labeles images to match the
                dimensions of the fluorescence images. The transformation is a
                downscaling with majority rule. The data structure is the same
                as `hist_img` and `hist_img_labels`.
            - label_map: A dictionary mapping integer encoding of label names
                to the actual label names.
            - frame: A pandas DataFrame containing the pixel data for all
                images, with correct dtypes. Provided if `as_frame` is True.

    """
    # Initialize dictionaries to store metadata and images
    images_dict = {}
    hist_img_dict = {}
    hist_img_labels_dict = {}
    img_labels_dict = {}

    # Find all files
    all_files = find_files(directory)  # ex-find_dat_files

    # Process each file
    for filename in all_files:
        filepath = filename
        img_type, metadata = identify_and_extract_metadata(filename)

        if img_type == "fluorescence" and metadata:
            diet, mouse, take, elem = metadata
            pixels, fluorescence = load_fluo_image(filepath)  # ex-load_image

            # Organize by diet, mouse, take
            key = (diet, mouse, take)
            if key not in images_dict:
                # Initialize a 3D array with zeros for the first time
                height = int(np.max(pixels[:, 0]) + 1)
                width = int(np.max(pixels[:, 1]) + 1)
                images_dict[key] = np.zeros((height, width, len(element_order)))

            # Find the index of the current element in the predefined order
            elem_index = element_order.index(elem)
            for row, col, fluo in zip(
                pixels[:, 0].astype(int), pixels[:, 1].astype(int), fluorescence
            ):
                images_dict[key][row, col, elem_index] = fluo

        elif img_type == "hist_img" and metadata:
            diet, mouse, take, _ = metadata
            key = (diet, mouse, take)
            hist_img_dict[key] = load_hist_image(filepath)

        elif img_type == "hist_img_labels" and metadata:
            diet, mouse, take, _ = metadata
            key = (diet, mouse, take)
            hist_img_labels_dict[key] = load_labels_image(filepath)

    # Resize labeled images to match fluorescence image dimensions
    for key in hist_img_labels_dict:
        if key in images_dict:
            # Extract height and width
            fluorescence_shape = images_dict[key].shape[:2]
            # Resize the label image to match the fluorescence image
            img_labels_dict[key] = resize_with_majority_rule(
                hist_img_labels_dict[key], target_shape=fluorescence_shape
            )

    # Prepare the final dataset
    unique_keys = list(images_dict.keys())
    images_list = [images_dict[key] for key in unique_keys]
    diet_list = [key[0] for key in unique_keys]
    mouse_list = [key[1] for key in unique_keys]
    take_list = [key[2] for key in unique_keys]

    # Convert histology and label dictionaries to ordered lists if not as_dict
    if not as_dict:
        hist_img_list = [hist_img_dict.get(key) for key in unique_keys]
        hist_img_labels_list = [hist_img_labels_dict.get(key) for key in unique_keys]
        img_labels_list = [img_labels_dict.get(key) for key in unique_keys]

    # if as_frame is True, return the data as a DataFrame
    if as_frame:
        df = load_frame(directory)

    if as_dict:
        return Bunch(
            DESCR=get_description(),
            diet_names=diet_names,
            diet_map=diet_map,
            element_order=element_order,
            element_map=element_map,
            images=images_dict,
            hist_img=hist_img_dict,
            hist_img_labels=hist_img_labels_dict,
            img_labels=img_labels_dict,  # Rescaled labels
            label_map=label_map,
            frame=df if as_frame else None,
        )
    else:
        # Return the Bunch object with the dataset
        return Bunch(
            DESCR=get_description(),
            diet_names=diet_names,
            diet_map=diet_map,
            element_order=element_order,
            element_map=element_map,
            diet=np.array(diet_list),
            mouse=np.array(mouse_list),
            take=np.array(take_list),
            images=images_list,  # List of 3D arrays
            hist_img=hist_img_list,  # Ordered list of histology images
            hist_img_labels=hist_img_labels_list,  # Ordered list of label
            # images with the same size as the histology images
            img_labels=img_labels_list,  # Ordered list of labele images,
            # resized to match the shape of the corresponding fluorescence
            # image in `images`
            label_map=label_map,
            frame=df if as_frame else None,
        )


def load_frame(directory: str) -> pd.DataFrame:
    """
    Loads the fluorescence dataset and returns it as a pandas DataFrame.

    The DataFrame contains one row per pixel with the following columns:
    - diet: Encoded diet category (0: 'control', 1: 'omega3', 2: 'omega6')
    - mouse: Mouse number
    - take: Take number
    - row: Row index of the pixel in the image
    - col: Column index of the pixel in the image
    - Columns for each element (e.g., 'Ca', 'Cu', 'Fe', etc.) representing the
        fluorescence values.
    - label: Label for the pixel (if available)

    Parameters
    ----------
    directory : str
        The root directory containing the fluorescence dataset.

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame containing the pixel data for all images, with
        correct dtypes.
    """
    # Load the full dataset using load_fluorescence
    dataset = load_fluorescence(directory, as_dict=True)

    # Initialize a list to store pixel data
    data = []

    for key, image in dataset.images.items():
        diet, mouse, take = key
        labels_image = dataset.img_labels.get(key, None)

        # Loop through each pixel in the fluorescence image
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                # Extract fluorescence values for all elements
                fluorescence_values = image[row, col, :]

                # Extract label if available
                label = labels_image[row, col] if labels_image is not None else np.nan

                # Append the pixel data
                data.append(
                    {
                        "diet": diet,
                        "mouse": mouse,
                        "take": take,
                        "row": row,
                        "col": col,
                        **{
                            elem: fluorescence_values[i]
                            for i, elem in enumerate(dataset.element_order)
                        },
                        "label": label,
                    }
                )

    # Convert the list of pixel data to a DataFrame
    df = pd.DataFrame(data)

    # Ensure correct dtypes
    df["diet"] = df["diet"].astype("category")
    df["mouse"] = df["mouse"].astype(int)
    df["take"] = df["take"].astype(int)
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    df["label"] = df["label"].astype("category")  # Labels are categorical

    for elem in dataset.element_order:
        df[elem] = df[elem].astype(float)

    # Reorder the columns as specified
    columns_order = (
        ["diet", "mouse", "take", "row", "col"] +
        dataset.element_order +
        ["label"]
    )
    df = df[columns_order]

    return df.reset_index(drop=True)


def get_description() -> str:
    """
    Generates a description for the fluorescence dataset.

    Returns
    -------
    description : str
        A string describing the structure of the dataset.
    """
    description = """

    ====================
    Fluorescence Dataset
    ====================


    Summary
    -------

    This dataset contains fluorescence images for elemental composition
    analysis of mammary gland adenocarcinomas in mice, and images from the
    histological analysis. It contains various mouse samples, each with a
    specific diet, measured at least once. Different measurements of the same
    mouse are indexed by the 'take' attribute. The dataset is structured as a
    `scikit-learn` `Bunch` object, and can optionally include a pandas
    DataFrame with pixels as rows and their data (row, column, mouse, take,
    diet, fluorescence for each element, type of tissue) as columns.


    Experimental Methodology
    ------------------------

    An experimental model was implemented in BALB/C mice through the
    subcutaneous inoculation of transplantable mammary gland adenocarcinoma
    cells. The mice were divided into three dietary groups: one rich in
    omega-3, another rich in omega-6, and a control group without lipid
    supplementation. Tumors extracted from the animals were fixed in formalin,
    embedded in paraffin, and sectioned into thin slices of a few microns.
    Each sample underwent conventional histological analysis using
    hematoxylin-eosin staining, and images were captured with an optical
    microscope.

    Additionally, a micro X-ray fluorescence analysis with synchrotron
    radiation was performed at the Brazilian Synchrotron Light Laboratory
    (LNLS) in Campinas, Brazil. The samples were scanned in 2D, and XRF
    spectra were recorded using a Si(Li) detector (KETEK Vitus SDD) with a
    live counting time of 0.5 seconds per pixel and a step size of 100 μm in
    both orthogonal directions. This procedure enabled us to determine the
    spatial distribution of elements such as S, P, Ca, Mn, Fe, Cu, and Zn.


    Structure of the Dataset
    ------------------------

    By default, we favor the use of lists, having as a result the following
    attributes:

    - DESCR: A string description of the dataset.
    - diet_names: A list of diet names where the index of each element
        was used in `diet` instead of the string, i.e., '0' was used
        instead of diet_names[0], which is 'control'.
    - diet_map: A dictionary mapping integer encoding of diet names to
        the actual diet names.
    - element_order: A list of strings indicating the order of
        elements in the 3D images. E.g., if `as_dict` is False, that
        means that `images[n][0]` is the 2D image for the fluorescence
        of `element_order[0]`, which is `'Ca'`, of `mouse[n]`,
        `take[n]` with `diet[n]`. If `as_dict` is True,
        `images[(diet, mouse, take)][0]` is the 2D image for the
        fluorescence of `element_order[0]`, which is `'Ca'`, of take
        `take` of mouse `mouse` with diet `diet`.
    - element_map: A dictionary mapping integer encoding of element
        names to the actual element names.
    - diet: A 1D array of diet categories corresponding to each 3D
        image. Provided if `as_dict` is False.
    - mouse: A 1D array of mouse numbers corresponding to each 3D
        image. Provided if `as_dict` is False.
    - take: A 1D array of take numbers corresponding to each 3D image.
        Take is just an int that enumerates, starting from zero, the
        different measurements done for the same mouse. Provided if
        `as_dict` is False.
    - images: if `as_dict` is False, list of 3D NumPy arrays, one for
        each combination of diet, mouse, and take. If `as_dict` is
        True, a dictionary of 3D NumPy arrays, with keys corresponding
        to diet, mouse, and take, given as a tuple
        `(diet, mouse, take)`.
    - hist_img: list of histology images, either as a list of NumPy
        arrays if `as_dict` is False or as a dictionary of NumPy
        arrays with the usual `(diet, mouse, take)` key, if `as_dict`
        is True.
    - hist_img_labels: list of labeles for the histology images, with
        the same dimensions and data structure of `hist_img`.
    - img_labels: list of resized labeles images to match the
        dimensions of the fluorescence images. The transformation is a
        downscaling with majority rule. The data structure is the same as
        `hist_img` and `hist_img_labels`.
    - label_map: A dictionary mapping integer encoding of label names
        to the actual label names.
    - frame: A pandas DataFrame containing the pixel data for all
        images, with correct dtypes. Provided if `as_frame` is True.

    The DataFrame privided in `frame` if `as_frame`, contains one row per
    pixel with the following columns:
    - diet: Encoded diet category (0: 'control', 1: 'omega3', 2: 'omega6')
    - mouse: Mouse number
    - take: Take number
    - row: Row index of the pixel in the image
    - col: Column index of the pixel in the image
    - Columns for each element (e.g., 'Ca', 'Cu', 'Fe', etc.) representing the
        fluorescence values.
    - label: Label for the pixel

    Lables for histological images are also available in the dataset:
    - 0: no label
    - 1: necrtotic tissue
    - 2: tumoral A
    - 3: tumoral B
    - 4: tumoral C
    - 5: artifacts (e.g., folds, tears, etc.)
    - 6: blood (e.g., blood vessels, hemorrhage, etc.)
    - 7: loose connective tissue
    - 8: no sample
    - 9: dense connective tissue
    - 10: paraffin (matrix material)


    References
    ----------

    - [1] Bencharski, C., Soria, E. A., Falchini, G. E., Pasqualini, M. E., &
        Perez, R. D. (2023). Study of anti-tumorigenic actions of essential
        fatty acids in a murine mammary gland adenocarcinoma by micro-XRF.
        Analytical Methods, 15(16), 2044–2051.
    - [2] Falchini, G. E., Malezan, A., Poletti, M. E., Soria, E., Pasqualini,
        M., & Perez, R. D. (2021). Analysis of phosphorous content in cancer
        tissue by synchrotron micro-XRF. Radiation Physics and Chemistry, 179,
        109157.
    """
    return description  # .strip()
