"""Personal toolkit functions"""

import pickle
import time
from numbers import Number
import os
import torch
import cv2 as cv
import numpy as np
import random
from PIL import Image

#Kernels
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

def pickle_set(trainx: np.ndarray, trainy: np.ndarray, valx: np.ndarray, valy: np.ndarray, datadir: str) -> str:
    """
    Create pickle files for a machine learning data set

    Args:
        trainx (np.ndarray): Training data
        trainy (np.ndarray): Training labels
        valx (np.ndarray): Validation data
        valy (np.ndarray): Validation labels
        dir (str): Target directory

    Returns:
        str: Identifier for locating files
    """
    timecode = round(time.time())
    with open(f"{os.environ['dstore']}{datadir}\\trainx-{timecode}", 'ab') as pkf:
        pickle.dump(trainx, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\trainy-{timecode}", 'ab') as pkf:
        pickle.dump(trainy, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\valx-{timecode}", 'ab') as pkf:
        pickle.dump(valx, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\valy-{timecode}", 'ab') as pkf:
        pickle.dump(valy, pkf)
    return timecode

def odder(num: int) -> int:
    """Forces a number to be odd"""
    if num % 2 == 0:
        num += 1
    return int(num)

def evener(num: int) -> int:
    """Forces a number to be even"""
    if num % 2 != 0:
        num += 1
    return int(num)

def depickler(trainx: str, trainy: str, valx: str, valy: str, datadir: str) -> tuple[str, str, str, str]:
    """
    Loads a set of pickle files for training neural networks

    Args:
        trainx (str): Training data file path
        trainy (str): Training labels file path
        valx (str): Validation data file path
        valy (str): Validation labels file path
        dir (str): Target directory

    Returns:
        tuple[str, str, str, str]: Data sets
    """
    with open(f"{os.environ['dstore']}{datadir}\\{trainx}", 'rb') as pkf:
        trainxx = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{trainy}", 'rb') as pkf:
        trainyy = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{valx}", 'rb') as pkf:
        valxx = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{valy}", 'rb') as pkf:
        valyy = pickle.load(pkf)
    return trainxx, trainyy, valxx, valyy

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

def normalize(data: np.ndarray) -> list:
    """
    Normalize input values

    (x - x.min) / (x.max - x.min)

    Args:
        data (np.ndarray): Input Data

    Returns:
        [type]: Normalized input data
    """
    return np.array((data - np.min(data)) / (np.max(data) - np.min(data)), dtype='float32')

def standardize(data: np.ndarray) -> list:
    """
    Standardize input values

    (x - x.mean) / x.standard_deviation

    Args:
        data (np.ndarray): Input data

    Returns:
        [type]: Standardized input data
    """
    return np.array((data - np.mean(data)) / np.std(data), dtype=np.float32)

def smooth_avg(data: list[Number]) -> list[Number]:
    """
    Generate a smoothed version of a data set where each point is replaced by the average of itself and immeadiately adjacent points

    Args:
        data (list[Number]): A list of continuous data points

    Returns:
        list[Number]: A smoother list of continuous data points
    """
    smoothed = []
    for i, x in enumerate(data):
        if i == 0:
            smoothed.append((x + data[i + 1]) / 2)
        elif i == len(data) - 1:
            smoothed.append((x + data[i - 1]) / 2)
        else:
            smoothed.append((data[i - 1] + x + data[i + 1]) / 3)
    return smoothed

def criticals(data: list, idx: bool = False) -> list:
    """
    Create a list of critical points of a continuous data set
    Critical Points: Maxima, Minima, Gradient Maxima, Gradient Minima, Gradient Roots

    Args:
        data (list): A list of continuous data points
        idx (bool, optional): A custom index. Defaults to False.

    Returns:
        list: A list of tuples that contains the index of a critical point and the critical point type
    """
    grads = np.gradient(data)
    grads2 = np.gradient(grads)
    crits = []
    if not idx:
        idx = range(len(data))
    """ else:
        idx = [round((x[1] - x[0]) / 2) + x[0] for x in idx] """
    for i, x in enumerate(idx, 1):
        if i > len(idx) - 2:
            break
        if data[i - 1] < data[i] and data[i + 1] < data[i]:
            crits.append((x, 'max'))
        if data[i - 1] > data[i] and data[i + 1] > data[i]:
            crits.append((x, 'min'))
        if grads[i] > 0 and grads[i + 1] < 0 or grads[i] < 0 and grads[i + 1] > 0:
            crits.append((x, 'dzero'))
        if grads[i - 1] < grads[i] and grads[i + 1] < grads[i]:
            crits.append((x, 'dmax'))
        if grads[i - 1] > grads[i] and grads[i + 1] > grads[i]:
            crits.append((x, 'dmin'))
        if grads2[i] > 0 and grads2[i + 1] < 0 or grads2[i] < 0 and grads2[i + 1] > 0:
            crits.append((x, 'ddzero'))
        if grads2[i - 1] < grads2[i] and grads2[i + 1] < grads2[i]:
            crits.append((x, 'ddmax'))
        if grads2[i - 1] > grads2[i] and grads2[i + 1] > grads2[i]:
            crits.append((x, 'ddmin'))
    return crits

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

def count(items: list) -> list:
    """
    Generate a list of counts for each item in the input list. Outputs with highest counted item at index 0

    Args:
        items (list): A list of objects to be sorted

    Returns:
        list: A list of unique items in the input and the number of occurances for each item
    """
    def row2(k):
        return k[1]
    if isinstance(items[0], list):
        items = [tuple(x) for x in items]
    uniqs = [[x, items.count(x)] for x in set(items)]
    uniqs.sort(key=row2, reverse=True)
    return uniqs

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

def bin2num(inp: list) -> int:
    """
    Convert a binary number into integer

    Args:
        inp (list): Binary number

    Returns:
        int: Integer representation of binary input
    """
    if isinstance(inp, str):
        inp = [int(x) for x in inp]
    if isinstance(inp, (int, float)):
        inp = [int(x) for x in str(round(inp))]
    if set(inp) != set(1, 0):
        return None
    bin_state = 0
    for x in inp[::-1]:
        if x == 1:
            total += 2**bin_state
            bin_state += 1
        else:
            bin_state += 1
    return total

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

def expand_data(inp: np.ndarray, top: int, dtype: type ='uint8'):
    """
    Stretch data to fit from 0 to arg(top)

    Args:
        inp (np.ndarray): Input data array
        top (int): Maximum value of the output data
        dtype (type, optional): Datatype of the output. Defaults to 'uint8'.

    Returns:
        [type]: Stretched data array
    """
    mnm = np.min(inp)
    inp = inp - mnm
    mxm = np.max(inp)
    inp = np.round(inp * (top / mxm)).astype(dtype)
    return inp

def halfpoint(num1: int or float, num2: int or float):
    """
    Gives the halfway point between input numbers

    Args:
        num1 (intorfloat): A number
        num2 (intorfloat): A number

    Returns:
        [type]: Halfway point number
    """
    if num2 > num1:
        mid = ((num2 - num1) / 2) + num1
    else:
        mid = ((num1 - num2) / 2) + num2
    return mid

def aroundpoint(num:int or float, step: int or float) -> tuple[int, int]:
    """
    Gives the points around a number seperated by the step size

    Args:
        num ([type]): number
        step ([type]): distance from input number

    Returns:
        [type]: tuple containing the points surrounding the input
    """
    return (num - step, num + step)

def timer(fnc):
    """Timing decorator"""
    def wrapper(*args, **kwargs):
        start = time.time()
        outp = fnc(*args, **kwargs)
        print(f'Execution time: {time.time() - start}')
        return outp
    return wrapper

def find_area(coords):
    """Find area of a box when given the box corners in a tuple"""
    return (coords[1] - coords[0]) * (coords[3] - coords[2])

def mlstats(nhist, measure, tm1=None, ohist=None, tm2=None):
    """Show stats from ML training runs, one or two cycles averaged"""
    score = (max(nhist.get(f"val_{measure}")) + (2.7182818**-(min(nhist.get("val_loss")))) + (sum(nhist.get(f"val_{measure}")[-7:]) / 7) + (sum(2.7182818**-(np.array(nhist.get("val_loss")[-7:]))) / 7)) / 4
    bvacc = max(nhist.get(f"val_{measure}"))
    bvloss = min(nhist.get("val_loss"))
    avacc = sum(nhist.get(f"val_{measure}")[-7:]) / 7
    avloss = sum(nhist.get("val_loss")[-7:]) / 7
    if tm1:
        samps = len(nhist.get(f"val_{measure}")) * 32 * (192 + 64) / (tm1)
    if ohist is None:
        print(f'Score: {score}')
        print(f'Best VAccuracy: {bvacc}')
        print(f'Best VLoss: {bvloss}')
        print(f'Last 7 Avg VAccuracy: {avacc}')
        print(f'Last 7 Avg VLoss: {avloss}')
        if tm1:
            print(f'Training time: {tm1}')
            print(f'Samples per second: {samps}')
    else:
        oscore = (max(ohist.get(f"val_{measure}")) + (2.7182818**-(min(ohist.get("val_loss")))) + (sum(ohist.get(f"val_{measure}")[-7:]) / 7) + (sum(2.7182818**-(np.array(ohist.get("val_loss")[-7:]))) / 7)) / 4
        obvacc = max(ohist.get(f"val_{measure}"))
        obvloss = min(ohist.get("val_loss"))
        oavacc = sum(ohist.get(f"val_{measure}")[-7:]) / 7
        oavloss = sum(ohist.get("val_loss")[-7:]) / 7
        if tm1 and tm2:
            osamps = len(ohist.get(f"val_{measure}")) * 32 * (192 + 64) / (tm2)
        print(f'Score: {(score + oscore) / 2}')
        print(f'Best VAccuracy: {(obvacc + bvacc) / 2}')
        print(f'Best VLoss: {(obvloss + bvloss) / 2}')
        print(f'Last 7 Avg VAccuracy: {(oavacc + avacc) / 2}')
        print(f'Last 7 Avg VLoss: {(oavloss + avloss) / 2}')
        if tm1 and tm2:
            print(f'Training time: {(tm1 + tm2) / 2}')
            print(f'Samples per second: {(osamps + samps) / 2}')

def zshuffle(data, labels):
    """shuffle seperate data sets while maintaining cohesion"""
    temp = list(zip(data, labels))
    random.shuffle(temp)
    data = [x[0] for x in temp]
    labels = [x[1] for x in temp]
    return np.array(data), np.array(labels)

class DataGen:
    """
    Custom data generator class for pytorch
    """
    def __init__(self, data, labels, batch_len=16):
        self.labels = labels
        self.data = data
        self.batch_len = batch_len
        self.i = 0
        self.maxi = round(len(self.data) / self.batch_len) - 2

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i >= self.maxi:
            raise StopIteration
        self.i += 1
        return self.data[(self.i - 1) * self.batch_len:self.i * self.batch_len], self.labels[(self.i - 1) * self.batch_len:self.i * self.batch_len]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[[idx]]), torch.as_tensor(self.labels[[idx]], dtype=torch.float)
