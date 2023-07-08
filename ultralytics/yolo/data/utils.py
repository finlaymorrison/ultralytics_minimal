# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm

from ultralytics.nn.autobackend import check_class_names
from ultralytics.yolo.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, ROOT, SETTINGS_YAML, clean_url, colorstr, emojis,
                                    yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_font, is_ascii
from ultralytics.yolo.utils.downloads import download, safe_download

HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb[:, 1:] <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(imgsz, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def check_det_dataset(dataset, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in 'train', 'val':
        if k not in data:
            raise SyntaxError(
                emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: str, split=''):
    """
    Check a classification dataset such as Imagenet.

    This function takes a `dataset` name as input and returns a dictionary containing information about the dataset.
    If the dataset is not found, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str): Name of the dataset.
        split (str, optional): Dataset split, either 'val', 'test', or ''. Defaults to ''.

    Returns:
        data (dict): A dictionary containing the following keys and values:
            'train': Path object for the directory containing the training set of the dataset
            'val': Path object for the directory containing the validation set of the dataset
            'test': Path object for the directory containing the test set of the dataset
            'nc': Number of classes in the dataset
            'names': List of class names in the dataset
    """
    data_dir = (DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
        t = time.time()
        if dataset == 'imagenet':
            subprocess.run(f"bash {ROOT / 'yolo/data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists() else None  # data/test or data/val
    test_set = data_dir / 'test' if (data_dir / 'test').exists() else None  # data/val or data/test
    if split == 'val' and not val_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == 'test' and not test_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))
    return {'train': train_set, 'val': val_set or test_set, 'test': test_set or val_set, 'nc': nc, 'names': names}

