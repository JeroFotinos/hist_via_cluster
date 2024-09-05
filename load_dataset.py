import numpy as np
import os
from sklearn.utils import Bunch
from typing import Tuple, List

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

# Function to load fluorescence data from a file
def load_image(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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

def find_dat_files(directory: str) -> List[str]:
    """
    Recursively finds all .dat files in the specified directory and its
    subdirectories.

    Parameters
    ----------
    directory : str
        The root directory in which to search for .dat files.

    Returns
    -------
    dat_files : list of str
        A list of paths to the found .dat files.
    """
    dat_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    return dat_files

# Main function to load fluorescence data
def load_fluorescence(directory: str) -> Bunch:
    """
    Loads fluorescence data from the specified directory and returns it as a
    Bunch object.

    Parameters
    ----------
    directory : str
        The root directory containing the .dat files with the fluorescence
        data.

    Returns
    -------
    dataset : Bunch
        A Bunch object containing:
            - DESCR: A string description of the dataset.
            - diet: A 1D array of diet categories corresponding to each 3D
                image.
            - diet_names: A list of diet names where the index of each element
                was used in `diet` instead of the string, i.e., '0' was used
                instead of diet_names[0], which is 'control'.
            - mouse: A 1D array of mouse numbers corresponding to each 3D
                image.
            - take: A 1D array of take numbers corresponding to each 3D image.
                Take is just an int that enumerates, starting from zero, the
                different measurements done for the same mouse.
            - images: A list of 3D NumPy arrays, one for each combination of
                diet, mouse, and take.
            - element_order: A list of strings indicating the order of
                elements in the 3D images. That means that `images[n][0]` is
                the 2D image for the fluorescence of `element_order[0]`, which
                is `'Ca'`, of `mouse[n]`, `take[n]` with `diet[n]`.
    """
    # Initialize dictionaries to store metadata and images
    images_dict = {}
    
    # Find all .dat files
    dat_files = find_dat_files(directory)
    
    # Process each file
    for filename in dat_files:
        filepath = filename
        diet, mouse, take, elem = extract_metadata(filename)

        pixels, fluorescence = load_image(filepath)

        # Organize by diet, mouse, take
        key = (diet, mouse, take)
        if key not in images_dict:
            # If first time for this key, initialize a 3D array with zeros
            height = int(np.max(pixels[:, 0]) + 1)  # Assuming pixel indices are 0-based
            width = int(np.max(pixels[:, 1]) + 1)
            images_dict[key] = np.zeros((height, width, len(element_order)))

        # Find the index of the current element in the predefined order
        elem_index = element_order.index(elem)
        for row, col, fluo in zip(pixels[:, 0].astype(int), pixels[:, 1].astype(int), fluorescence):
            images_dict[key][row, col, elem_index] = fluo

    # Prepare the final dataset
    unique_keys = list(images_dict.keys())
    images_list = [images_dict[key] for key in unique_keys]
    diet_list = [key[0] for key in unique_keys]
    mouse_list = [key[1] for key in unique_keys]
    take_list = [key[2] for key in unique_keys]

    # Return the Bunch object with the dataset
    return Bunch(
        DESCR=get_description(),
        diet=np.array(diet_list),
        diet_names=diet_names,
        mouse=np.array(mouse_list),
        take=np.array(take_list),
        images=images_list,  # List of 3D arrays
        element_order=element_order
    )

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

    - DESCR: A string description of the dataset.
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
    """
    return description.strip()
