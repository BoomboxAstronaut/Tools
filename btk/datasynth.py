"""Module for generating datasets for ML"""

import os
import logging
import random
import string

import cv2 as cv
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from btk import imgkit

synth_vars = dict()
logging.basicConfig(
    filename=f"{os.environ['Base']}\\ComputerVision\\reading\\datagen.log",
    filemode='a',
    format='%(funcName)s::%(levelname)s::%(asctime)s:: %(message)s',
    level=logging.WARNING
)

def text_init():
    with open(f"{os.environ['Base']}Resources\\engwords.txt", encoding='utf8') as f:
        synth_vars['words'] = f.read().splitlines()
    with open(f"{os.environ['Tools']}tfonts", encoding='utf8') as f:
        synth_vars['trainfonts'] = [x.strip('\\\n') for x in f.readlines()]
    with open(f"{os.environ['Tools']}vfonts", encoding='utf8') as f:
        synth_vars['evalfonts'] = [x.strip('\\\n') for x in f.readlines()]
    synth_vars['xtcls'] = list('o0OIl1Q,. ')
    synth_vars['case_chars'] = list('ZXWVSPOC')
    synth_vars['icase_chars'] = list('zxwvspoc')
    synth_vars['xtseg'] = list('aaettddhsmmonpquvwwyAWLLEMMNNNSYTVWVUHHHHH')
    synth_vars['xtsegwords'] = ['WAWAWAWA', 'kLkKkKyyYYYyy', 'ununuUNUNN', 'frfttrtrfrtfr', 'dbdbhbdhbdhbd', 'mmmwwmmww', 'THTHTHTHT', 'EEEEFFFFRRR', 'gegegegeeg']

    chars = list(string.ascii_letters)
    chars.extend(string.digits)
    chars.extend(list(' ():;.,"\'!@#$%&?+=-'))
    synth_vars['chars'] = chars
    synth_vars['chars_cased'] = [x for x in synth_vars['chars'] if x not in synth_vars['case_chars']]

def imgs_init(reload: bool = False):
    if reload:
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
    else:
        with open(f"{os.environ['Base']}Resources\\bglist", 'r', encoding='utf8') as f:
            bglst = f.readlines()
            bglst = [x.strip('\n') for x in bglst]
        with open(f"{os.environ['Base']}Resources\\gridbgs", 'r', encoding='utf8') as f:
            grids = f.readlines()
            grids = [x.strip('\n') for x in grids]
    synth_vars['grids'] = grids
    synth_vars['bglst'] = bglst

def sentence_make(length: int) -> str:
    """Generate a random series of real words from a dictionary"""
    i = length
    sentence = []
    while i > 0:
        rint = random.choice((0, 1))
        if rint == 0:
            sentence.append(random.choice(synth_vars['words']))
        elif rint == 1:
            sentence.append(str.upper(random.choice(synth_vars['words'])))
        i -= 1
    return ' '.join(sentence)

def word_make(letters):
    """Generate a random string from the input character set"""
    length = random.randint(7, 14)
    word = [random.choice(letters) for _ in range(length)]
    return ''.join(word)

def foreground_scan(img):
    """Scan an image from top to bottom to identify the y-position where the foreground begins"""
    height = len(img) - 1
    background_column_pixel_average = img[0].sum()
    for x in range(0, height):
        if img[x].sum() != background_column_pixel_average:
            foreground_start = x - 1
            break
    if not 'foreground_start' in locals():
        foreground_start = random.randint(0, 12)
    return foreground_start

def cropper(img: Image.Image, axis: int = None) -> np.ndarray:
    """ Trims excessive whitespace around text on an input image

    Only works with solid backgrounds
    """
    img = np.array(img)
    if axis == 1 or axis is None:
        img_top = foreground_scan(img)
        img_bottom = len(img) - 1 - foreground_scan(np.rot90(img, 2))
    if axis == 0 or axis is None:
        img_left = foreground_scan(np.rot90(img, 3))
        img_right = len(img[0]) - 1 - foreground_scan(np.rot90(img, 1))
    if axis == 1:
        return img[img_top:img_bottom, :]
    if axis == 0:
        return img[:, img_left:img_right]
    return img[img_top:img_bottom, img_left:img_right]

def img_warper(img: Image.Image, rotate: bool = True) -> Image.Image:
    """Apply random image augmentation to the input image"""
    choice = random.randint(1, 10)
    random_coef_1 = random.choice([1, 3, 5, 7])
    random_coef_2 = random.choice([1, 2, 3])
    random_coef_3 = random.choice([1, 3, 5])
    if choice == 1:
        img = cv.GaussianBlur(img, (random_coef_3, random_coef_3), random_coef_1)
    elif choice == 2:
        img = imgkit.sharpen(img)
    elif choice == 3:
        img = cv.erode(img, np.ones((random_coef_2, random_coef_2)), iterations=1)
    elif choice == 4:
        img = cv.dilate(img, np.ones((random_coef_2, random_coef_2)), iterations=1)
    elif choice == 5:
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, (choice + random_coef_1 - 1), (2 + random_coef_3))
    elif choice == 6 and rotate:
        x_len = img.shape[1]
        y_len = img.shape[0]
        img = cv.warpAffine(img, cv.getRotationMatrix2D((x_len/2.0, y_len/2.0), random.randint(-2 * random_coef_1, 2 * random_coef_1), 1), (x_len, y_len))
    elif choice == 7:
        img = cv.Laplacian(img, cv.CV_64F)
        img = np.uint8(np.absolute(img))
    elif choice == 8:
        img = np.invert(img)
    elif choice == 9:
        img = cv.Canny(img, 50, 200, apertureSize=3)
    return img

def get_random_picture(dimensions: tuple[int, int], d_type: str = 'uint8'):
    """Load, scale, and slice a random picture"""
    with Image.open(random.choice(synth_vars['bglst'])) as f:
        img = np.array(f.convert("L"), dtype=d_type)
    if img.shape[0] < dimensions[0]:
        img_top = 0
        img_bottom = img.shape[0]
    else:
        img_top = random.randint(0, img.shape[0] - dimensions[0])
        img_bottom = img_top + dimensions[0]
    if img.shape[1] < dimensions[1]:
        img_left = 0
        img_right = img.shape[1]
    else:
        img_left = random.randint(0, img.shape[1] - dimensions[1])
        img_right = img_left + dimensions[1]
    if img.shape[0] < dimensions[0] or img.shape[1] < dimensions[1]:
        img = np.array(Image.fromarray(img).resize((dimensions[1], dimensions[0])))
    else:
        img = img[img_top:img_bottom, img_left:img_right]
    return img

def gen_stripe_image(dimensions: tuple[int, int], d_type: str = 'uint8'):
    """Procedurally generate an image composed of various paralell lines"""
    orientation_pick = random.randint(0, 1)
    if orientation_pick == 0:
        dimensions = (dimensions[1], dimensions[0])
    img = np.ones(dimensions, dtype=d_type)
    line_seed = random.randint(25, 230)
    space_seed = random.randint(4, 64)
    for i, _ in enumerate(img):
        img[i] = (img[i] * line_seed) + random.randint(-20, 20)
        if i == space_seed:
            line_seed = random.randint(25, 230)
            space_seed = random.randint(4, 64)
    if orientation_pick == 0:
        img = img.T
    return img

def get_grid_picture(dimensions: tuple[int, int], d_type: str = 'uint8'):
    """Load, scale, and slice a random image of a grid"""
    with Image.open(random.choice(synth_vars['grids'])) as f:
        img = np.array(f.convert("L"), dtype=d_type)
    img_top = random.randint(0, img.shape[0] - 100)
    img_left = random.randint(0, img.shape[1] - 100)
    img = np.array(Image.fromarray(img[
        img_top:random.randint(img_top + 64, img.shape[0] - 1),
        img_left:random.randint(img_left + 64, img.shape[1] - 1)
        ]).resize((dimensions[1], dimensions[0])), dtype=d_type)
    return img

def get_background(dimensions: tuple[int, int], d_type: str = 'uint8', weights: tuple[int] = (1, 1, 1, 1, 1)) -> list:
    """
    Generate a random background from one of 5 types

    Blank background. Random Color
    White Noise.
    Random picture of a grid or cityscape
    Procedurally generated lines
    Random picture

    Args:
        dimensions (tuple[int, int]): Dimensions of the output image
        d_type (str, optional): Data type. Defaults to 'uint8'.
        weights (tuple[int], optional): Weights for background subtypes. Defaults to (1, 1, 1, 1, 1).

    Returns:
        list: _description_
    """
    background_pick = random.randint(0, sum(weights) - 1)
    if background_pick in range(0, sum((weights[:1]))):
        background = np.ones(dimensions, dtype=d_type) * random.randint(1, 255)
    elif background_pick in range(sum((weights[:1])), sum((weights[:2]))):
        background = np.random.default_rng().integers(0, random.randint(1, 255), dimensions, dtype=d_type)
    elif background_pick in range(sum((weights[:2])), sum((weights[:3]))):
        background = get_grid_picture(dimensions, d_type)
    elif background_pick in range(sum((weights[:3])), sum((weights[:4]))):
        background = gen_stripe_image(dimensions, d_type)
    else:
        background = get_random_picture(dimensions, d_type)
    return background

def merge_images(img1, img2):
    """Merge two images with dark pixels being transparent in one input image"""
    img2 = img2.astype('float32')
    img2 = img2 / 256
    for i, x in enumerate(img1):
        img2[i] = img2[i] + x
    img2[img2 < 1] *= 256
    img2 = img2.astype('uint8')
    return img2

def rand_char_img(letterset: str, fnts: list, augment: bool = False) -> tuple[Image.Image, str]:
    """Create an image of space_index character

    Input
        List of letters to use
        List of fonts to use
        Bool determines whether to apply augs

    Output Tuple
        Image
        String of letter used
    """
    random_coef_1, random_coef_2, random_coef_3, random_coef_4 = random.randint(0, 64), random.randint(0, 64), random.randint(1, 254), random.randint(0, 20)
    text_fill = round(abs((255 * random.randint(0, 1)) - abs(np.random.randn() * 255 / (np.pi * 2))))
    background_range = [x for x in range(0, 255) if x not in range(text_fill - 16, text_fill + 16)]
    if random.randint(0, 4) == 0:
        image = Image.fromarray(np.random.default_rng().integers(0, 255, (64, random.randint(42, 64)), dtype='uint8'))
    else:
        image = Image.new(mode='L', color=random.choice(background_range), size=(64, 64))
    char = random.choice(letterset)
    if augment:
        if random_coef_4 < 6:
            ImageDraw.Draw(image).rectangle(
                (32 - round(random_coef_1 / 2), 32 - round(random_coef_1 / 2), 32 + round(random_coef_1 / 2), 32 + round(random_coef_1 / 2)),
                fill=random.choice(background_range),
                outline=random_coef_2,
                width=1
            )
        elif 6 < random_coef_4 < 9:
            ImageDraw.Draw(image).line(
                [round(random_coef_2 / 4), random_coef_1, round(random_coef_2 / 4), 64],
                random_coef_3,
                round(random_coef_4 / 4)
            )
        elif 9 < random_coef_4 < 12:
            ImageDraw.Draw(image).line(
                [64 -  round(random_coef_2 / 4), random_coef_1, 64 - round(random_coef_2 / 4), 64],
                random_coef_3,
                round(random_coef_4 / 4)
            )
    ImageDraw.Draw(image).text(
        (random.randint(24, 40), random.randint(36, 54)),
        char,
        font=ImageFont.truetype(random.choice(fnts), random.randint(30, 66), encoding="unic"),
        fill=text_fill,
        stroke_width=random.randint(0, 1),
        stroke_fill=2 * (random_coef_1 + random_coef_2),
        anchor='ms'
    )
    image = cropper(image, 0)
    background = np.ones((64, 64), dtype='uint8') * random.choice(background_range)
    x_start_pos = round((64 - len(image[0])) / 2)
    background[:, x_start_pos:x_start_pos + len(image[0])] = image
    return background, char

def find_spaces(word, font_choice, font_color, randomize=False):
    """Find the positions of the spaces between letters in an image"""
    img = Image.new(mode='L', color=0, size=(1000, 64))
    avg_background_color = np.array(img)[:, 0].sum()
    space_positions = []
    text_position_tracker = 6
    for i, _ in enumerate(word):
        ImageDraw.Draw(img).text((text_position_tracker, 0), word[i:i+1], font=font_choice, fill=font_color, anchors='ml')
        for j, y in enumerate(np.flip(np.array(img)).T):
            if y.sum() != avg_background_color:
                if randomize:
                    if random.randint(0, 4) == 0:
                        spacing = random.randint(6, 10)
                    else:
                        spacing = random.randint(0, 5)
                else:
                    spacing = 2
                space_positions.append((1000 - j, (1000 - j) + spacing))
                text_position_tracker = (1000 - j) + spacing
                break
    space_positions.append((space_positions[-1][1] + 24, space_positions[-1][1] + 32))
    img = np.array(img)[:, :space_positions[-1][1] + 8]
    return img, space_positions

def gen_tdet_data(samples: int, fonts: str, characters: list[str], augment: bool = False):
    """
    Generate labeled data for text detection

    Args:
        samples (int): Number of samples to generate
        fonts (str): Font set to use
        characters: List of characters to use
        augment(bool): Boolean to turn img augmentation on

    Returns:
        tuple[np.ndarray, np.ndarray]: Array containing data, Array containing labels
    """
    if ' ' in characters:
        characters.remove(' ')
    finished_labels = []
    finished_images = []
    while samples > 0:
        samples -= 1
        img = Image.new(mode='L', color=0, size=(1000, 144))
        font_choice = ImageFont.truetype(random.choice(fonts), random.randint(28, 144), encoding="unic")
        font_color = abs((random.randint(0, 1) * 255) - int(np.random.triangular(1, 1, 127)))
        ImageDraw.Draw(img).text((0, 0), word_make(characters), font=font_choice, fill=font_color, anchors='ml')
        img = cropper(img)
        background = get_background(img.shape, 'uint8', (3, 1, 4, 2, 6))
        img = merge_images(img, background)
        if augment and random.randint(0, 1) == 0:
            img = img_warper(img, False)
            background = img_warper(background, False)
        if img.shape[0] < 64:
            img = imgkit.force_dim(img, 64, 1)
            background = imgkit.force_dim(background, 64, 1)
        for _ in range(7):
            ypos = random.randint(0, img.shape[0] - 64)
            xpos = random.randint(0, img.shape[1] - 64)
            finished_images.append(img[ypos:ypos + 64, xpos:xpos + 64])
            finished_labels.append(1)
            ypos = random.randint(0, img.shape[0] - 64)
            xpos = random.randint(0, img.shape[1] - 64)
            finished_images.append(background[ypos:ypos + 64, xpos:xpos + 64])
            finished_labels.append(0)
    return np.array(finished_images), np.array(finished_labels)

def gen_tseg_data(samples: int, fonts: str, characters: list[str], augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate labeled data for word segmentation | space detection neural network

    Args:
        samples (int): Number of samples to generate
        font_choices (str): Font set to use
        augment(bool): Boolean to turn img augmentation on

    Returns:
        tuple[np.ndarray, np.ndarray]: Array containing data, Array containing labels
    """
    if ' ' in characters:
        characters.remove(' ')
    finished_labels = []
    finished_images = []
    while samples > 0:
        samples -= 1
        font_choice = ImageFont.truetype(random.choice(fonts), random.randint(28, 60), encoding="unic")
        font_color = abs((random.randint(0, 1) * 255) - int(np.random.triangular(1, 1, 127)))
        img, space_positions = find_spaces(word_make(characters), font_choice, font_color, True)
        img = merge_images(img, get_background((64, space_positions[-1][1] + 8), 'uint8', (3, 1, 4, 2, 5)))
        if augment and random.randint(0, 1) == 0:
            img = img_warper(img, False)
        for i, x in enumerate(space_positions[:-1]):
            letter_left_border = space_positions[i - 1][1] + 2
            border_buffer = 3
            if i == 0:
                letter_left_border = 9
            if x[0] - letter_left_border < 7:
                border_buffer = round((x[0] - letter_left_border) / 2) - 1
            for _ in range(3):
                position = random.randint(letter_left_border + border_buffer, x[0] - border_buffer)
                finished_images.append(img[:, position - 6:position + 6])
                finished_labels.append(0)
        for x in space_positions:
            for _ in range(2):
                position = random.randint(x[0], x[1] + 1)
                finished_images.append(img[:, position - 6: position + 6])
                finished_labels.append(1)
    return np.array(finished_images), np.array(finished_labels)

def gen_tcls_data(samples: int, letterset: str, fnts: list, augment: bool = False, all_characters: list[str] = False) -> tuple[np.ndarray, np.ndarray]:
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
    if not all_characters:
        all_characters = letterset
    fixed_case_characters = [x for x in all_characters if x not in synth_vars['case_chars']]
    label_index_size = len(fixed_case_characters) - 1
    finished_images = []
    finished_labels = []
    while samples > 0:
        samples -= 1
        if augment:
            image_set = rand_char_img(letterset, fnts, True)
            finished_images.append(img_warper(np.array(image_set[0])))
        else:
            image_set = rand_char_img(letterset, fnts, False)
            finished_images.append(np.array(image_set[0]))
        if image_set[1] in synth_vars['case_chars']:
            letter_index = fixed_case_characters.index(str.lower(image_set[1]))
        else:
            letter_index = fixed_case_characters.index(image_set[1])
        finished_labels.append(np.insert(np.zeros(label_index_size, np.int8), letter_index, 1))
    return np.asarray(finished_images, dtype='uint8'), np.asarray(finished_labels)
