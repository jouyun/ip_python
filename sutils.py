import numpy as np
import pandas as pd
import napari
import tifffile
import skimage as ski
import scipy.ndimage as ndi
import glob
import plotly.express as px
import cellpose.models as models
import matplotlib.pyplot as plt
import cv2
import dask

def backsub_2D(inp, radius=60):
    """
    This function performs background subtraction on a 2D image using a morphological operation called "top-hat".

    Parameters:
    inp (numpy.ndarray): The input 2D image.
    radius (int, optional): The radius of the structuring element used for the top-hat operation. Default is 60.

    Returns:
    rtn (numpy.ndarray): The result image after background subtraction.

    How it works:
    1. It first creates an elliptical structuring element with the specified radius.
    2. It then applies a Gaussian blur to the input image to reduce noise.
    3. It performs a top-hat operation on the blurred image. The top-hat operation is the difference between the input image and its opening. It is used to isolate small elements and details from the larger ones.
    4. It subtracts the result of the top-hat operation from the input image to get the final result.
    5. Finally, it clips the result to have a minimum value of 0 (to remove any negative values that might have resulted from the subtraction).
    """
    filterSize =(radius, radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
    blurred = cv2.GaussianBlur(inp, (5, 5), 0)
    tophat_img = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    rtn = inp.astype(np.single) - (blurred-tophat_img)
    rtn = np.clip(rtn, 0, np.inf)
    return rtn

def backsub(inp, radius=60):
    """
    Performs background subtraction on an input image or a stack of images.

    Parameters:
    inp (numpy.ndarray): Input image or stack of images. If 2D, it's treated as a single image. If nD, it's treated as a stack of images.
    radius (int, optional): Radius for the background subtraction. Default is 60.

    Returns:
    numpy.ndarray: Image or stack of images with the background subtracted.
    """

    # If the input is 2D, it's a single image. Perform background subtraction directly.
    if len(inp.shape) <= 2:
        return backsub_2D(inp, radius)

    # If the input is nD, it's a stack of images. We need to process each image separately.
    # First, we reshape the stack to a 2D array where each row is an image.
    orig_shape = inp.shape
    img = np.reshape(inp, (-1, orig_shape[-2], orig_shape[-1]))

    # We use Dask to perform the background subtraction in parallel for each image.
    # This is done by creating a list of delayed computations, one for each image.
    process = [dask.delayed(backsub_2D)(i,radius) for i in img]

    # We then compute all the delayed computations in parallel.
    rslt = dask.compute(*process)

    # The result is a list of images. We convert it to a numpy array.
    rslt = np.array(rslt)

    # Finally, we reshape the result back to the original shape of the input.
    rslt = np.reshape(rslt, orig_shape)

    return rslt


def remove_objects(label_image, area_min, area_max):
    """
    This function removes objects from a labeled image based on their area.

    Parameters:
    label_image (numpy.ndarray): A 2D array where each unique non-zero value represents a unique object/region.
    area_min (int): The minimum area threshold. Objects with area less than this will be removed.
    area_max (int): The maximum area threshold. Objects with area more than this will be removed.

    Returns:
    cleaned_label_image (numpy.ndarray): A 2D array similar to label_image but with small and large objects removed.

    How it works:
    1. It first calculates the properties of each labeled region in the input image using skimage.measure.regionprops.
    2. It then creates a boolean mask of the same size as the input image. This mask is True for pixels belonging to regions that are too small or too large.
    3. Finally, it applies this mask to the input image, setting the labels of the small and large regions to zero, effectively removing them from the image.
    """
    # Get properties of labeled regions
    props = ski.measure.regionprops(label_image)

    # Create a boolean mask where True indicates a region is too small or too large
    object_mask = np.zeros_like(label_image, dtype=bool)
    for prop in props:
        if prop.area < area_min or prop.area > area_max:
            object_mask[label_image == prop.label] = True

    # Apply the mask to the label image
    cleaned_label_image = np.where(object_mask, 0, label_image)

    return cleaned_label_image

def shrink_labels(label_image, shrinkage=3):
    """
    This function shrinks the regions in a labeled image using morphological erosion.

    Parameters:
    label_image (numpy.ndarray): A 2D array where each unique non-zero value represents a unique object/region.
    shrinkage (int, optional): The size of the structuring element used for the erosion operation. Default is 3.  This default shrinks by 1 pixel in each direction.

    Returns:
    shrunken_label_image (numpy.ndarray): A 2D array similar to label_image but with all regions shrunken.

    How it works:
    1. It first creates a circular structuring element of the specified size.
    2. It then initializes an empty image of the same size as the input image.
    3. It calculates the properties of each labeled region in the input image using skimage.measure.regionprops.
    4. For each region, it extracts the region from the original label image, erodes it using the structuring element, and then updates the corresponding region in the shrunken_label_image.
    5. The erosion operation shrinks the region by removing pixels from its boundary.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrinkage, shrinkage))

    shrunken_label_image = np.zeros_like(label_image)
    regions = ski.measure.regionprops(label_image)

    for region in regions:
        lbl = region.label
        minr, minc, maxr, maxc = region.bbox

        minr = max(minr - 1, 0)
        minc = max(minc - 1, 0)
        maxr = min(maxr + 1, label_image.shape[0])
        maxc = min(maxc + 1, label_image.shape[1])

        # Extract the region from the original label image
        region_mask = label_image[minr:maxr, minc:maxc] == lbl

        # Erode the region
        eroded_region_mask = cv2.erode(region_mask.astype(np.uint8), kernel)

        # Update the shrunken_label_image
        shrunken_label_image[minr:maxr, minc:maxc][eroded_region_mask == 1] = lbl

    return shrunken_label_image

def download_images(url, extract_to):
    '''
    From the github gist
    https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    '''
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    

#def checkdata_md5():
#    url = 
        
def setlogging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def check(gpu=True, device='mps'):
    model = models.Cellpose(gpu=gpu, device=device)
    return model
 
def checkmps():   
    d = torch.device('mps')    
    model = check(gpu=True, device=d)
    print(model.device)
    