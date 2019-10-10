import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance

""" Reads image from disk
    Images and labels are resized to desired standard size
    Labels are also binarized based on labels
"""
def load_img(filename: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    img = cv2.imread(filename, flags)
    return img

""" Converts raw image data to array with grayscale channel
    and dtype of float32 [0,1]
    Channel data is first, so final shape is (ch, *image)
"""
def img_to_array(img: np.ndarray) -> np.ndarray:
    arr = np.array(img).reshape((1, *img.shape)).astype(np.uint8)
    arr = (arr / 255).astype(np.float32)

    return arr

""" Converts scan image to cuboid by padding with zeros
"""
def cubify_scan(data: np.ndarray, cube_dim: int) -> np.ndarray:
    pad_w = (cube_dim - data.shape[0]) // 2
    pad_h = (cube_dim - data.shape[1]) // 2
    pad_d = (cube_dim - data.shape[2]) // 2

    data = np.pad(
        data,
        [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
        mode='constant',
        constant_values=0
    )
    
    return data

""" Converts segmentation mask into distance map
"""
def calc_dist_map(seg: np.ndarray) -> np.ndarray:
    res = np.zeros_like(seg)

    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res