"""Personal toolkit functions"""

import pickle
import time
from numbers import Number
import torch
import cv2 as cv
import numpy as np
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

def pickle_set(trainx: np.ndarray, trainy: np.ndarray, valx: np.ndarray, valy: np.ndarray) -> str:
    """
    Create pickle files for a machine learning data set

    Args:
        trainx (np.ndarray): Training data
        trainy (np.ndarray): Training labels
        valx (np.ndarray): Validation data
        valy (np.ndarray): Validation labels

    Returns:
        str: Identifier for locating files
    """
    timecode = round(time.time())
    with open(f'trainx-{timecode}', 'ab') as pkf:
        pickle.dump(trainx, pkf)
    with open(f'trainy-{timecode}', 'ab') as pkf:
        pickle.dump(trainy, pkf)
    with open(f'valx-{timecode}', 'ab') as pkf:
        pickle.dump(valx, pkf)
    with open(f'valy-{timecode}', 'ab') as pkf:
        pickle.dump(valy, pkf)
    print(timecode)
    return timecode

def depickler(trainx: str, trainy: str, valx: str, valy: str) -> tuple[str, str, str, str]:
    """
    Loads a set of pickle files for training neural networks

    Args:
        trainx (str): Training data file path
        trainy (str): Training labels file path
        valx (str): Validation data file path
        valy (str): Validation labels file path

    Returns:
        tuple[str, str, str, str]: Data sets
    """
    with open(f'{trainx}', 'rb') as pkf:
        trainxx = pickle.load(pkf)
    with open(f'{trainy}', 'rb') as pkf:
        trainyy = pickle.load(pkf)
    with open(f'{valx}', 'rb') as pkf:
        valxx = pickle.load(pkf)
    with open(f'{valy}', 'rb') as pkf:
        valyy = pickle.load(pkf)
    return trainxx, trainyy, valxx, valyy

def grey_np(img: Image) -> np.ndarray:
    """
    Return a greyscale version of the input image

    Args:
        img (Image): Color Image

    Returns:
        np.ndarray: Greyscale Image
    """
    img = np.array(img)
    if img.shape[2] == 4:
        greyscaler = [0.21, 0.72, 0.07, 0]
    elif img.shape[2] == 3:
        greyscaler = [0.21, 0.72, 0.07]
    else:
        print("Invalid Image Colors")
        return
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
    return np.array((data - np.mean(data)) / np.std(data), dtype='float32')

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
    else:
        idx = [round((x[1] - x[0]) / 2) + x[0] for x in idx]
    for i, x in enumerate(idx, 1):
        if i > len(idx) - 2:
            break
        if data[i - 1] <= data[i] and data[i + 1] <= data[i]:
            crits.append((x, 'max'))
        if data[i - 1] >= data[i] and data[i + 1] >= data[i]:
            crits.append((x, 'min'))
        if grads[i] > 0 and grads[i + 1] < 0 or grads[i] < 0 and grads[i + 1] > 0:
            crits.append((x, 'dzero'))
        if grads[i - 1] <= grads[i] and grads[i + 1] <= grads[i]:
            crits.append((x, 'dmax'))
        if grads[i - 1] >= grads[i] and grads[i + 1] >= grads[i]:
            crits.append((x, 'dmin'))
        if grads2[i] > 0 and grads2[i + 1] < 0 or grads2[i] < 0 and grads2[i + 1] > 0:
            crits.append((x, 'ddzero'))
        if grads2[i - 1] <= grads2[i] and grads2[i + 1] <= grads2[i]:
            crits.append((x, 'ddmax'))
        if grads2[i - 1] >= grads2[i] and grads2[i + 1] >= grads2[i]:
            crits.append((x, 'ddmin'))
    return crits

def img_slicer(dims: tuple[int, int], img: Image, step: int or tuple[int, int], way: str) -> tuple[np.ndarray, list]:
    """
    Slice an image into smaller pieces

    Args:
        dims (tuple[int, int]): Dimensions of a image slice (y axis, x axis)
        img (Image): Image
        step (intortuple[int, int]): Step distance between slices (y axis, x axis)
        way (str): String indicating slicing method
            'v' to slice using only verical cuts
            'h' to slice using only horizontal cuts
            'm' to slice in both orientations

    Returns:
        tuple[np.ndarray, list]: Array of image slices as array objects and a list of the boundry coordinates for each slice
    """
    img = np.array(img)
    slices = []
    idx = []
    if way == 'v':
        img = np.array(Image.fromarray(img).resize((round(dims[0] / img.shape[0] * img.shape[1]), dims[0])))
        border = round(dims[1] / 2)
        idx = [(x - border, x + border) for x in range(border, img.shape[1] - border, step)]
        for x in range(border, img.shape[1] - border, step):
            slices.append(img[:, x - border:x + border])
    elif way == 'h':
        img = np.array(Image.fromarray(img).resize((dims[1], round(dims[1] / img.shape[1] * img.shape[0]))))
        border = round(dims[0] / 2)
        idx = [(x - border, x + border) for x in range(border, img.shape[0] - border, step)]
        for y in range(border, img.shape[0] - border, step):
            slices.append(img[y - border:y + border, :])
    else:
        assert len(step) == 2, 'Step must be tuple for 2 way slicing'
        img = np.array(Image.fromarray(img).resize((
            round((img.shape[1] / dims[1]) + 0.4999) * dims[1],
            round((img.shape[0] / dims[0]) + 0.4999) * dims[0]
        )))
        yborder, xborder = round(dims[0] / 2), round(dims[1] / 2)
        for y in range(yborder, img.shape[0] - round(yborder / 2), step[0]):
            for x in range(xborder, img.shape[1] - round(xborder / 2), step[1]):
                idx.append((
                    y - round(dims[0] / 2),
                    y + round(dims[0] / 2),
                    x - round(dims[1] / 2),
                    x + round(dims[1] / 2)
                ))
        for y in range(yborder, img.shape[0] - round(yborder / 2), step[0]):
            for x in range(xborder, img.shape[1] - round(xborder / 2), step[1]):
                slices.append(img[y - yborder:y + yborder, x - xborder:x + xborder])
    return np.array(slices), idx

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

def fit2dims(dims: tuple[int, int], img: Image) -> np.ndarray:
    """
    Resize an image slightly so that it can be sliced into a whole integer quantity without cropping based on the slice dimensions given

    Args:
        dims (tuple[int, int]): Dimensions of the image
        img (Image): Image to be modified

    Returns:
        np.ndarray: Resized image in array format
    """
    img = np.array(img)
    return np.array(Image.fromarray(img).resize((
        round((img.shape[1] / dims[1]) + 0.4999) * dims[1],
        round((img.shape[0] / dims[0]) + 0.4999) * dims[0]
    )))

def diff_compress(data: list) -> list[int]:
    """
    Create a list of the midway points when given a list of boundry points

    Input: (1, 5), (4, 12), (50, 100)
    Output: 3, 8, 75

    Args:
        data (list): A list of coordinates that define boundries around a point

    Returns:
        list[int]: A list of center points
    """
    return [round((x[1] - x[0]) / 2) + x[0] for x in data]

def diff_expand(data: list, step: int) -> list[tuple[int, int]]:
    """
    Create a list of boundary points when given a list of points and the distance from the center to the boundry

    Input: [5, 10, 50], 5
    Output: [(0, 10), (5, 15), (45, 55)]

    Args:
        data (list): List of numbers
        step (int): Distance from the input value to the boundry

    Returns:
        list[tuple[int, int]]: List of coordinates of boundry values around the input points
    """
    return [(x - step, x + step) for x in data]

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

def img_splitter(img: Image.Image, orig_dims: tuple[int, int], sdims: tuple) -> list[Image.Image]:
    """
    Create high contrast images from the input image for OCR

    Args:
        img (Image): Input image
        orig_dims (tuple[int, int]): Dimensions of the input image
        sdims (tuple): Dimensions of the slices that the image will be cut into

    Returns:
        list[Image.Image]: A list containing 4 images in array format
    """
    area = orig_dims[0] * orig_dims[1]
    splits = []
    ksize, kdist = 11, 9
    if area > 4194304:
        ksize, kdist = 25, 17
    elif area < 65536:
        ksize, kdist = 7, 5
    elif area < 8192:
        print("Image too small")
        return False
    splits.append(fit2dims(sdims, img))
    splits.append(fit2dims(sdims, Image.fromarray(np.invert(np.array(img)))))
    splits.append(fit2dims(sdims, Image.fromarray(cv.Laplacian(np.array(img), cv.CV_8U))))
    splits.append(fit2dims(sdims, Image.fromarray(cv.adaptiveThreshold(np.array(img), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ksize, kdist))))
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

class DataGen:
    """
    Custom data generator class for pytorch
    """
    def __init__(self, data, labels=False, batch_len=16):
        self.data = data
        self.labels = labels
        self.batch_len = batch_len
        self.i = 0
        self.maxi = round(self.data.shape[0] / self.batch_len) - 2

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i >= self.maxi:
            raise StopIteration
        if isinstance(self.labels, bool):
            self.i += 1
            return self.data[(self.i - 1) * self.batch_len:self.i * self.batch_len]
        self.i += 1
        return self.data[(self.i - 1) * self.batch_len:self.i * self.batch_len], self.labels[(self.i - 1) * self.batch_len:self.i * self.batch_len]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[[idx]]), torch.as_tensor(self.labels[[idx]], dtype=torch.float)
