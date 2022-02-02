"""Module for generating datasets for ML"""

import os
import logging
import random
import string
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

with open(f"{os.environ['Tools']}tfonts", encoding='utf8') as f:
    trainfonts = [x.strip('\\\n') for x in f.readlines()]

with open(f"{os.environ['Tools']}vfonts", encoding='utf8') as f:
    evalfonts = [x.strip('\\\n') for x in f.readlines()]

with open(f"{os.environ['Base']}Resources\\engwords.txt", encoding='utf8') as f:
    words = f.read().splitlines()

case_chars = list('COPSVWXZ')
icase_chars = list('copsvwxz')
hardc = list('o0OIl1Q')

chars1 = list(string.ascii_letters)
chars1.extend(string.digits)
allchars = chars1.copy()
for x in case_chars:
    chars1.remove(x)


logging.basicConfig(
    filename=f"{os.environ['base']}\\ComputerVision\\reading\\datagen.log",
    filemode='a',
    format='%(funcName)s::%(levelname)s::%(asctime)s:: %(message)s',
    level=logging.WARNING
    )

def sentence_make(length: int) -> str:
    """Generate space_index random series of real words from space_index dictionary"""
    i = length
    sentence = []
    while i > 0:
        sentence.append(random.choice(words))
        i -= 1
    return ' '.join(sentence)

def img_warper(img: Image.Image, roton: bool = True) -> Image.Image:
    """Apply random image augmentation to the input image"""
    choice = random.randint(1, 11)
    rint1 = random.choice([1, 3, 5, 7])
    rint2 = random.choice([1, 2, 3])
    rint3 = random.choice([1, 3, 5])
    if choice == 1:
        warp = 'GaussianBlur'
        img = cv.GaussianBlur(img, (rint3, rint3), rint1)
    elif choice == 2:
        warp = 'MedianBlur'
        img = cv.medianBlur(img, rint3)
    elif choice == 3:
        warp = 'Erosion'
        img = cv.erode(img, np.ones((rint2, rint2)), iterations=1)
    elif choice == 4:
        warp = 'Dilation'
        img = cv.dilate(img, np.ones((rint2, rint2)), iterations=1)
    elif choice == 5:
        warp = 'Laplacian'
        img = cv.Laplacian(img, cv.CV_8U)
    elif choice == 6 and roton:
        warp = 'Rotation'
        x_len = img.shape[1]
        y_len = img.shape[0]
        img = cv.warpAffine(img, cv.getRotationMatrix2D((x_len/2.0, y_len/2.0), random.randint(-2 * rint1, 2 * rint1), 1), (x_len, y_len))
    elif choice == 7:
        warp = 'ContrastThreshold'
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, (choice + rint1 - rint3), (2 + rint1))
    else:
        warp = 'None'
    logging.debug('Warp %s, RINTS %s %s %s', warp, rint1, rint2, rint3)
    return img

def rand_char_img(letterset: str, fnts: list, warp: bool = False) -> tuple[Image.Image, str]:
    """Create an image of space_index character

    Input
        List of letters to use
        List of fonts to use
        Bool determines whether to apply augs

    Output Tuple
        Image
        String of letter used
    """
    rint_1, rint_2, rint_3, rint_4, rint_5 = random.randint(0, 64), random.randint(0, 64), random.randint(0, 255), random.randint(1, 254), random.randint(0, 20)
    image = Image.new(mode='L', color=rint_3, size=(64, 64))
    char = random.choice(letterset)
    if rint_3 < 128:
        tfill = rint_3 + (32 * round((rint_5 + 5) / 6) - 1)
    else:
        tfill = rint_3 - (32 * round((rint_5 + 5) / 6) - 1)
    if warp:
        if rint_5 < 6:
            ImageDraw.Draw(image).rectangle((32 - round(rint_1 / 2), 0, 32 + round(rint_1 / 2), 64), fill=round((rint_3 + rint_4) / 2), outline=rint_2, width=1)
        elif 6 < rint_5 < 9:
            ImageDraw.Draw(image).line([round(rint_2 / 4), rint_1, round(rint_2 / 4), 64], rint_4, round(rint_5 / 4))
        elif 9 < rint_5 < 12:
            ImageDraw.Draw(image).line([64 -  round(rint_2 / 4), rint_1, 64 - round(rint_2 / 4), 64], rint_4, round(rint_5 / 4))
        """ elif 9 < rint_5 < 12:
            ImageDraw.Draw(image).line([0, rint_2, rint_1, rint_2], rint_4, round(rint_5 / 4))
        elif 12 < rint_5 < 15:
            ImageDraw.Draw(image).regular_polygon((32, 32, round(rint_5 * 1.5)), round(rint_5 / 2), rotation=rint_2, fill=rint_4, outline = 255 - rint_4)
        elif 15 < rint_5 < 18:
            ImageDraw.Draw(image).rounded_rectangle([64 - rint_1, 64 - rint_1, rint_1, rint_1], round(rint_3 / 2), rint_4) """
    ImageDraw.Draw(image).text(
        (random.randint(26, 38), random.randint(40, 58)),
        char,
        font=ImageFont.truetype(random.choice(fnts), random.randint(35, 65), encoding="unic"),
        fill=tfill,
        stroke_width=random.randint(0, 1),
        stroke_fill=2 * (rint_1 + rint_2),
        anchor='ms'
    )
    if rint_1 >= 48:
        image = ImageOps.invert(image)
    return image, char

def gen_chars(samples: int, letterset: str, fnts: list, warp: bool = False, superset: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset of images containing a letter and labels of the letter as a string

    Input
        Integer of the number of images to generate
        String of the letters to use
        List of fonts
        Bool that determines whether the output will have augmentations

    Output Tuple
        Images as arrays
        Booleans as arrays
    """
    tdata = []
    tlabel = []
    if not superset:
        superset = letterset
    """ lows = letterset.copy()
    for x in string.ascii_uppercase:
        lows.remove(x) """
    while samples > 0:
        samples -= 1
        if warp:
            item = rand_char_img(letterset, fnts, True)
            tdata.append(img_warper(img_warper(np.array(item[0]))))
        else:
            item = rand_char_img(letterset, fnts, False)
            tdata.append(np.array(item[0]))
        tlabel.append(np.insert(np.zeros(len(superset) - 1, np.int8), superset.index(str(item[1])), 1))
    return np.asarray(tdata, dtype='uint8'), np.asarray(tlabel)


def cropper(img: Image.Image) -> np.ndarray:
    """Trims excessive whitespace around text on an input image

    Only works with solid backgrounds
    """
    img = np.array(img)
    ylen = len(img) - 1
    xlen = len(img[0]) - 1
    msy = img[0].sum()
    msx = img[:, 0].sum()
    for x in range(0, ylen):
        if img[x].sum() != msy:
            ytop = x - 2
            break
    for x in range(ylen, 0, -1):
        if img[x].sum() != msy:
            ybot = x + 2
            break
    for x in range(0, xlen):
        if img[:, x].sum() != msx:
            xlft = x - 2
            break
    for x in range(xlen, 0, -1):
        if img[:, x].sum() != msx:
            xrgt = x + 2
            break
    return img[ytop:ybot, xlft:xrgt]

def pers_warp(image: Image.Image, fnts: list, words: str) -> np.ndarray:
    """Draw perspective shift warping to text on a background image

    Input
        Image of background
        List of fonts
        String of words

    Output
        Image as array
    """
    rand_coef, scalar = random.randint(1, 4), random.randint(0, 3)
    lyscale, ryscale, ytrans, xtrans = scalar, scalar, scalar * 5, scalar * random.randrange(-2, 2)
    ImageDraw.Draw(image).text(
        (32, 320),
        words,
        font=ImageFont.truetype(random.choice(fnts), random.randint(22, 36), encoding="unic"),
        fill=random.randint(105, 255)
    )
    if rand_coef in [1, 3]:
        ryscale = 0
    else:
        lyscale = 0
    if rand_coef in [1, 4]:
        ytrans = ytrans * -1
    image = cropper(image)
    ylen, xlen = np.array(image).shape
    image = cv.warpPerspective(
        np.array(image),
        cv.getPerspectiveTransform(
            np.float32([[0, 0], [xlen, 0], [0, ylen], [xlen, ylen]]),
            np.float32([
                [64 - xtrans, 320 - lyscale],
                [64 + xlen - xtrans, 320 - ryscale - ytrans],
                [64, 320 + ylen + lyscale],
                [64 + xlen, 320 + ylen + ryscale - ytrans]
            ])
        ),
        (640, 640)
    )
    image = cropper(image)
    return image.astype('uint16')

def tilt_words(tbg: np.ndarray, fnts: list, count: int, rinv: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """Place randomly generated strings of words on the input image

    Input
        Image as array
        List of fonts to use
        Integer for word count
        Boolean for color inversion

    Output Tuple
        Image as array
        List of coordinates for bounding box coordinates of the words
        List of strings of words
    """
    y_s = [140, 212, 284, 356, 428, 500]
    coords = []
    word_list = []
    while count > 0:
        count -= 1
        chosen = random.choice(y_s)
        y_s.remove(chosen)
        words = sentence_make(2)
        image = pers_warp(Image.new(mode='L', size=(640, 640)), fnts, words)
        xlen, ylen = len(image[0]), len(image)
        xpos, ypos = random.randint(8, 630 - xlen), random.randint(chosen - 32, chosen + 32)
        coords.append([ypos, ypos + ylen, xpos, xpos + xlen])
        word_list.append(words)
        for x in range(ylen):
            tbg[ypos][xpos:xpos + xlen] = tbg[ypos][xpos:xpos + xlen] + image[x]
            ypos += 1
    tbg[tbg > 255] = 255
    tbg = tbg.astype('uint8')
    if rinv and random.randint(0, 1) == 1:
        tbg = np.invert(tbg)
    return tbg, np.array(coords), word_list

def bg_words(samples: int, per_im: int, fnts: list) -> list:
    """Create an image set of words applied to a set of synthetic and photo backgrounds

    Input
        Integer of image samples to create
        Integer of sentences per image
        List of fonts to use

    Output List of Tuples
        Image as array
        List of coordinates for bounding box coordinates of the words
        List of strings of words
    """
    grids = []
    bglst = []
    for x in os.walk(f"{os.environ['Base']}Resources\\MiniDataSets\\ADEChallengeData2016\\images\\training"):
        for y in x[2]:
            bglst.append(os.path.join(x[0], y))
    for x in os.walk(f"{os.environ['Base']}Resources\\MiniDataSets\\naturebg"):
        for y in x[2]:
            bglst.append(os.path.join(x[0], y))
    for x in os.walk(f"{os.environ['Base']}Resources\\MiniDataSets\\grids"):
        for y in x[2]:
            grids.append(os.path.join(x[0], y))
    stored = []
    while samples > 0:
        bg_pick = random.randint(0, 9)
        samples -= 1
        if bg_pick == 1:
            tbg = np.ones((640, 640), dtype=np.uint16) * random.randint(0, 100)
        elif bg_pick == 2:
            tbg = np.random.default_rng().integers(0, random.randint(25, 225), (640, 640), dtype='uint16')
        elif bg_pick == 3:
            with Image.open(random.choice(grids)) as img:
                tbg = np.array(img.resize((640, 640)).convert("L"), dtype='uint16')
            tbg = np.array(Image.fromarray(tbg[random.randint(0, 160):random.randint(480, 640), random.randint(0, 160):random.randint(480, 640)]).resize((640, 640)))
        elif bg_pick == 4:
            tbg = Image.fromarray(np.ones((80, 640), dtype=np.uint8) * random.randint(0, 100))
            i = 7
            while i > 0:
                ypos = random.randint(0, 80)
                ImageDraw.Draw(tbg).line((0, ypos, 640, ypos), fill=random.randint(0, 120), width=random.randint(1, 5))
                i -= 1
            tbg = np.array(tbg, dtype='uint16')
            tbg = np.vstack((tbg, tbg, tbg, tbg, tbg, tbg, tbg, tbg))
        else:
            with Image.open(random.choice(bglst)) as img:
                tbg = np.array(img.resize((640, 640)).convert("L"), dtype='uint16')
        stored.append(tilt_words(tbg, fnts, per_im, True))
    return stored


def binary_text(imset: list, warp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of images containing an even split of images containing text and not containing text along with the corresponding labels

    Input
        List of Tuples
            Image as array
            List of coordinates for bounding box coordinates of the words
            List of strings of words
        Boolean for application of randomized image warping

    Output
        Tuple
            Array of images as arrays
            Array of boolean labels
    """
    binx = []
    biny = []
    for x in imset:
        inum = range((x[1][0][3] - x[1][0][2]) // 32)
        for _ in inum:
            xpick = random.randint(x[1][0][2], x[1][0][3] - 32)
            xpick = min(xpick, 575)
            xpick = max(xpick, 65)
            ypick = random.randint(((x[1][0][1] - x[1][0][0]) // 2) + x[1][0][0] - 32,
                                   ((x[1][0][1] - x[1][0][0]) // 2) + x[1][0][0])
            ypick = min(ypick, 575)
            ypick = max(ypick, 65)
            binx.append(np.array(x[0][ypick:ypick + 32, xpick:xpick + 32]))
            biny.append(1)
        if x[1][0][1] < 320:
            nypick = random.randint(320, 480)
        else:
            nypick = random.randint(1, 160)
        for _ in inum:
            nxpick = random.randint(20, 560)
            binx.append(np.array(x[0][nypick:nypick + 32, nxpick:nxpick + 32]))
            biny.append(0)
    if warp:
        binx = [img_warper(np.array(x), False) for x in binx]
    return np.array(binx, dtype='uint8'), np.array(biny)

def character_spaces(samples: int, fnts: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate labeled data for word segmentation | space detection neural network

    Args:
        samples (int): Number of samples to generate
        fnts (str): Font set to use

    Returns:
        tuple[np.ndarray, np.ndarray]: Array containing data, Array containing labels
    """
    labels = []
    data = []
    while samples > 0:
        samples -= 1
        word = sentence_make(1)
        while len(word) < 5:
            word = sentence_make(1)
        spaces = []
        fnt = ImageFont.truetype(random.choice(fnts), random.randint(18, 32), encoding="unic")
        for x in range(len(word)):
            image = Image.new(mode='L', color=random.randint(0, 100), size=(1000, 100))
            ImageDraw.Draw(image).text((30, 35), word[:x+1], font=fnt, fill=random.randint(140, 255))
            image = cropper(image)
            spaces.append(image.shape[1])
        if random.randint(0, 1) == 1:
            image = np.invert(image)
        spaces.pop(0)
        spaces.pop(-1)
        spaces.pop(-1)
        factor = 32 / image.shape[0]
        bspaces = np.round(np.array(spaces) * factor).astype('int')
        bimage = np.array(Image.fromarray(image).resize((round(image.shape[1] * factor), round(image.shape[0] * factor))))
        nons = bspaces + np.append(np.diff(bspaces) // 2, round(np.average(np.diff(bspaces) // 2)))
        bimage = np.array(img_warper(bimage, False))
        for x in range(len(spaces)):
            space_index = bspaces[x]
            non_space_index = nons[x]
            if space_index + 8 < len(bimage[0]):
                data.append(bimage[:, space_index - 6:space_index + 6])
                labels.append(1)
            if non_space_index + 8 < len(bimage[0]):
                data.append(bimage[:, non_space_index - 6:non_space_index + 6])
                labels.append(0)
    return np.array(data, dtype='uint8'), np.array(labels)

