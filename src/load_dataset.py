import os

from typing import Tuple, List, Optional

from PIL import Image

import numpy as np

import pandas as pd

from skimage.transform import resize

from scipy.stats import mode

from sklearn.utils import Bunch


# Define the order of elements in lowercase
element_order = ['Ca', 'Cu', 'Fe', 'K', 'Mn', 'P', 'S', 'Zn']

# Mapping for diet categories
diet_mapping = {'control': 0, 'omega3': 1, 'omega6': 2}
diet_names = ['control', 'omega3', 'omega6']


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
    parts = basename.split('_')
    diet = diet_mapping[parts[1]]
    mouse = int(parts[3])
    take = int(parts[5])
    elem = parts[7].replace('.dat', '')
    return diet, mouse, take, elem


# Function to identify image type and extract metadata
def identify_and_extract_metadata(filename: str) -> Tuple[str, Optional[Tuple[int, int, int, Optional[str]]]]:
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
    if 'element' in basename:
        parts = basename.split('_')
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        elem = parts[7].replace('.dat', '')
        return 'fluorescence', (diet, mouse, take, elem)
    elif 'recort' in basename:
        parts = basename.split('_')
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        return 'hist_img', (diet, mouse, take, None)
    elif 'labels' in basename:
        parts = basename.split('_')
        diet = diet_mapping[parts[1]]
        mouse = int(parts[3])
        take = int(parts[5])
        return 'hist_img_labels', (diet, mouse, take, None)
    else:
        return 'unknown', None


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
        The shape will depend on the image (e.g., (height, width) for single-channel).
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
    valid_extensions = {'.dat', '.tif', '.tiff', '.jpg', '.png'}
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
def load_fluorescence(directory: str, as_frame: bool = False, as_dict: bool = False) -> Bunch:
    """
    Loads fluorescence data from the specified directory and returns it as a
    Bunch object.

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
            - element_order: A list of strings indicating the order of
                elements in the 3D images. E.g., if `as_dict` is False, that
                means that `images[n][0]` is the 2D image for the fluorescence
                of `element_order[0]`, which is `'Ca'`, of `mouse[n]`,
                `take[n]` with `diet[n]`. If `as_dict` is True,
                `images[(diet, mouse, take)][0]` is the 2D image for the
                fluorescence of `element_order[0]`, which is `'Ca'`, of take
                `take` of mouse `mouse` with diet `diet`.
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
                to diet, mouse, and take.
            
    """
    # Initialize dictionaries to store metadata and images
    images_dict = {}
    hist_img_dict = {}
    hist_img_labels_dict = {}
    img_labels_dict = {}
    
    # Find all files
    all_files = find_files(directory) # ex-find_dat_files
    
    # Process each file
    for filename in all_files:
        filepath = filename
        img_type, metadata = identify_and_extract_metadata(filename)
        
        if img_type == 'fluorescence' and metadata:
            diet, mouse, take, elem = metadata
            pixels, fluorescence = load_fluo_image(filepath) # ex-load_image

            # Organize by diet, mouse, take
            key = (diet, mouse, take)
            if key not in images_dict:
                # Initialize a 3D array with zeros for the first time
                height = int(np.max(pixels[:, 0]) + 1)
                width = int(np.max(pixels[:, 1]) + 1)
                images_dict[key] = np.zeros((height, width, len(element_order)))

            # Find the index of the current element in the predefined order
            elem_index = element_order.index(elem)
            for row, col, fluo in zip(pixels[:, 0].astype(int), pixels[:, 1].astype(int), fluorescence):
                images_dict[key][row, col, elem_index] = fluo
        
        elif img_type == 'hist_img' and metadata:
            diet, mouse, take, _ = metadata
            key = (diet, mouse, take)
            hist_img_dict[key] = load_hist_image(filepath)
        
        elif img_type == 'hist_img_labels' and metadata:
            diet, mouse, take, _ = metadata
            key = (diet, mouse, take)
            hist_img_labels_dict[key] = load_labels_image(filepath)

    # Resize labeled images to match fluorescence image dimensions
    for key in hist_img_labels_dict:
        if key in images_dict:
            fluorescence_shape = images_dict[key].shape[:2]  # Extract height and width
            img_labels_dict[key] = resize_with_majority_rule(
                hist_img_labels_dict[key],
                target_shape=fluorescence_shape
            )

    # Prepare the final dataset
    unique_keys = list(images_dict.keys())
    images_list = [images_dict[key] for key in unique_keys]
    diet_list = [key[0] for key in unique_keys]
    mouse_list = [key[1] for key in unique_keys]
    take_list = [key[2] for key in unique_keys]

    # Convert histology and label dictionaries to ordered lists if as_dict is False
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
            images=images_dict,
            diet_names=diet_names,
            element_order=element_order,
            hist_img=hist_img_dict,
            hist_img_labels=hist_img_labels_dict,
            img_labels=img_labels_dict,  # Rescaled labels
            frame=df if as_frame else None
        )
    else:
        # Return the Bunch object with the dataset
        return Bunch(
            DESCR=get_description(),
            diet=np.array(diet_list),
            diet_names=diet_names,
            mouse=np.array(mouse_list),
            take=np.array(take_list),
            images=images_list,  # List of 3D arrays
            element_order=element_order,
            hist_img=hist_img_list,  # Ordered list of histology images
            hist_img_labels=hist_img_labels_list,  # Ordered list of labeled histology images
            img_labels=img_labels_list,  # Ordered list of resized labeled images
            frame=df if as_frame else None
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
    - Columns for each element (e.g., 'Ca', 'Cu', 'Fe', etc.) representing the fluorescence values.
    - label: Label for the pixel (if available)
    
    Parameters
    ----------
    directory : str
        The root directory containing the fluorescence dataset.
    
    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame containing the pixel data for all images, with correct dtypes.
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
                data.append({
                    'diet': diet,
                    'mouse': mouse,
                    'take': take,
                    'row': row,
                    'col': col,
                    **{elem: fluorescence_values[i] for i, elem in enumerate(dataset.element_order)},
                    'label': label
                })

    # Convert the list of pixel data to a DataFrame
    df = pd.DataFrame(data)

    # Ensure correct dtypes
    df['diet'] = df['diet'].astype('category')
    df['mouse'] = df['mouse'].astype(int)
    df['take'] = df['take'].astype(int)
    df['row'] = df['row'].astype(int)
    df['col'] = df['col'].astype(int)
    df['label'] = df['label'].astype('category')  # Assuming labels are categorical

    for elem in dataset.element_order:
        df[elem] = df[elem].astype(float)

    # Reorder the columns as specified
    columns_order = ['diet', 'mouse', 'take', 'row', 'col'] + dataset.element_order + ['label']
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
    Fluorescence Dataset

    ADD EXPERIMENTAL METHODOLOGY HERE.

    This dataset contains fluorescence images for elemental composition
    analysis of mammary gland adenocarcinomas in mice. It contains various
    mouse samples, each of which have been measured at least one. Different
    measurements of the same mouse are indexed by the 'take' attribute. Each
    mouse has also had a specific diet. The dataset is structured as follows:

    - DESCR: A string description of the dataset. (Chía=omega3, Cártamo=omega6)
    - diet: A 1D array where each element is an integer
        (0: 'control', 1: 'omega3', 2: 'omega6') indicating the diet category
        for each sample.
    - diet_names: A list of strings with the diet names:
        ['control', 'omega3', 'omega6'], where the index of each element
        coincides with its mapping value.
    - mouse: A 1D array containing the mouse number for each sample.
    - take: A 1D array containing the take number for each sample.
    - images: A list of 3D NumPy arrays representing the fluorescence images
        for each sample. The dimensions of each 3D array are (height, width, 8),
        where each of the 8 slices along the third dimension corresponds to
        the fluorescence of a specific element. The elements are stored in the
        following order: ['Ca', 'Cu', 'Fe', 'K', 'Mn', 'P', 'S', 'Zn'].
    - element_order: A list of strings representing the element order used to
        build the 3D images. That means that `images[n][0]` is
        the 2D image for the fluorescence of `element_order[0]`, which
        is `'Ca'`, of `mouse[n]`, `take[n]` with `diet[n]`.

    Lables for histological images are also available in the dataset:
    - 0: no label
    - 1: necrtotic tissue
    - 2: tumoral A
    - 3: tumoral B
    - 4: tumoral C
    - 5: artifacts (e.g., folds, tears, etc.)
    - 6: blood vessels
    - 7: loose connective tissue
    - 8: no sample
    - 9: dense connective tissue
    """
    return description.strip()
