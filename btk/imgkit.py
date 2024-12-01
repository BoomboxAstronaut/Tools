
import numpy as np
import cv2 as cv
from PIL import Image

sharpenk = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])

lagaussk = np.array([
            [ 0,  0, -1,  0,  0],
            [ 0, -1, -2, -1,  0],
            [-1, -2, 16, -2, -1],
            [ 0, -1, -2, -1,  0],
            [ 0,  0, -1,  0,  0]
        ])

gaussblurk = np.true_divide([
                            [0,  0,   0,   5,   0,  0,  0],
                            [0,  5,  18,  32,  18,  5,  0],
                            [0, 18,  64, 100,  64, 18,  0],
                            [5, 32, 100, 100, 100, 32,  5],
                            [0, 18,  64, 100,  64, 18,  0],
                            [0,  5,  18,  32,  18,  5,  0],
                            [0,  0,   0,   5,   0,  0,  0]
                        ], 256)

def grey_np(img: np.ndarray) -> np.ndarray:
    """
    Return a greyscale version of the input image

    Args:
        img (Image): Color Image

    Returns:
        np.ndarray: Greyscale Image
    """
    img = np.array(img)
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        greyscaler = [0.21, 0.72, 0.07, 0]
    elif img.shape[2] == 3:
        greyscaler = [0.21, 0.72, 0.07]
    else:
        print("Invalid Image Colors")
        return None
    return np.dot(np.array(img), greyscaler).astype('uint8')

def fit2dims(img: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """
    Resize an image slightly so that it can be sliced into a whole integer quantity without cropping based on the slice dimensions given

    Args:
        dims (tuple[int, int]): Dimensions of the image (y, x)
        img (Image): Image to be modified

    Returns:
        np.ndarray: Resized image in array format
    """
    img = Image.fromarray(img)
    img = img.resize((round((img.size[0] / dims[1]) + 0.4999) * dims[1],
                      round((img.size[1] / dims[0]) + 0.4999) * dims[0]
                    ))
    return np.array(img)

def img_slicer(img: np.ndarray, sdims: tuple[int, int], step: int or tuple[int, int], axis: int) -> tuple[np.ndarray, list]:
    """
    Slice an image into smaller pieces

    Args:
        sdims (tuple[int, int]): Dimensions of a image slice (y axis, x axis)
        img (Image): Image
        step (intortuple[int, int]): Step distance between slices (y axis, x axis)
        axis (int): String indicating slicing method
            1 to slice using only verical cuts
            0 to slice using only horizontal cuts
            2 to slice in both orientations

    Returns:
        tuple[np.ndarray, list]: Array of image slices as array objects and a list of the boundry coordinates for each slice
    """
    slices = []
    assert axis in [0, 1, 2], 'Invalid Slicing Method'
    if axis == 1:
        shape, border = img.shape, round(sdims[1] / 2)
        for x in range(border, shape[1] - border, step):
            slices.append(img[:, x - border:x + border])
    elif axis == 0:
        shape, border = img.shape, round(sdims[0] / 2)
        for y in range(border, shape[0] - border, step):
            slices.append(img[y - border:y + border, :])
    elif axis == 2:
        assert isinstance(step, tuple), 'Step must be tuple for 2 way slicing'
        shape, yborder, xborder = img.shape, round(sdims[0] / 2), round(sdims[1] / 2)
        for y in range(yborder, shape[0] - round(yborder / 2), step[0]):
            for x in range(xborder, shape[1] - round(xborder / 2), step[1]):
                slices.append(img[y - yborder:y + yborder, x - xborder:x + xborder])
    return np.array(slices)

def resize(img: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """
    Resize an image array and return an array

    Args:
        img (np.ndarray): Image in array form
        dims (tuple[int, int]): (y, x)

    Returns:
        np.ndarray: Image in array form resized
    """
    img = Image.fromarray(img)
    img = np.array(img.resize((dims[1], dims[0])))
    return img

def gen_index(dims: tuple[int, int], sdims: tuple[int, int], step: int, axis: int) -> list[tuple[int, int]]:
    """Create custom index for image slices that correspond to pixel coordinates"""
    assert axis in [0, 1, 2], 'Invalid Axis'
    if axis == 1:
        border = round(sdims[1] / 2)
        idx = [(int(x - border), int(x + border)) for x in range(border, dims[1] - border, step)]
    elif axis == 0:
        border = round(sdims[0] / 2)
        idx = [(int(x - border), int(x + border)) for x in range(border, dims[0] - border, step)]
    elif axis == 2:
        yborder, xborder = round(sdims[0] / 2), round(sdims[1] / 2)
        idx = []
        for y in range(yborder, dims[0] - round(yborder / 2), step[0]):
            for x in range(xborder, dims[1] - round(xborder / 2), step[1]):
                idx.append((
                    int(y - round(sdims[0] / 2)),
                    int(y + round(sdims[0] / 2)),
                    int(x - round(sdims[1] / 2)),
                    int(x + round(sdims[1] / 2))
                ))
    return idx

def force_dim(img: np.ndarray, dim: int, axis: int) -> np.ndarray:
    """
    Resize an image forcing one dimension to the input dimension and scaling the other dimension by the same factor

    in and out both np arrays

    Args:
        img (np.ndarray): Input image
        dim (tuple[int, int]): Dimensions to scale image to
        axis (int): Principal scaling axis

    Returns:
        np.ndarray: Rescaled image
    """
    assert axis in [1, 0], 'Invalid Axis'
    img = Image.fromarray(img)
    if axis == 1:
        img = img.resize((round(dim / img.size[1] * img.size[0]), dim))
    elif axis == 0:
        img = img.resize((dim, round(dim / img.size[1] * img.size[0])))
    return np.array(img)

def sharpen(img: np.ndarray) -> np.ndarray:
    """Apply sharpen filter to image"""
    return cv.filter2D(img, -1, sharpenk)

def img_splitter(img: np.ndarray) -> list[np.ndarray]:
    """
    Create high contrast images from the input image for OCR

    Args:
        img (Image): Input image
        orig_dims (tuple[int, int]): Dimensions of the input image
        sdims (tuple): Dimensions of the slices that the image will be cut into

    Returns:
        list[Image.Image]: A list containing 4 images in array format
    """
    area = img.shape[0] * img.shape[1]
    splits = []
    if area > 4194304:
        ksize, kdist = 25, 17
    elif area < 65536:
        ksize, kdist = 7, 5
    else:
        ksize, kdist = 11, 9
    splits.append(img)
    splits.append(np.invert(img))
    splits.append(cv.Laplacian(img, cv.CV_8U))
    splits.append(cv.adaptiveThreshold(img, 254, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ksize, kdist))
    return splits
