import os
import sys
import math
import random
import functools
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image, ImageEnhance
import logging

from reader_utils import DataReader

logger = logging.getLogger(__name__)
python_ver = sys.version_info


class KineticsReader(DataReader):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """
    def __init__(self, name, phase, cfg):
        self.name = name
        self.phase = phase
        self.format = cfg['format']
        self.num_classes = cfg['num_classes']
        self.seg_num = cfg['seg_num']
        self.seglen = cfg['seglen']
        self.short_size = cfg['short_size']
        self.target_size = cfg['target_size']
        self.num_reader_threads = cfg['num_reader_threads']
        self.buf_size = cfg['buf_size']

        self.img_mean = np.array(cfg['image_mean']).reshape([3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg['image_std']).reshape([3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg['batch_size']
        self.filelist = cfg['list']


    def create_reader(self):
        _reader = _reader_creator(self.filelist, self.phase, seg_num=self.seg_num, seglen = self.seglen, \
                             short_size = self.short_size, target_size = self.target_size, \
                             img_mean = self.img_mean, img_std = self.img_std, \
                             shuffle = (self.phase == 'train'), \
                             num_threads = self.num_reader_threads, \
                             buf_size = self.buf_size, format = self.format)
        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
        return _batch_reader


def _reader_creator(pickle_list,
                    phase,
                    seg_num,
                    seglen,
                    short_size,
                    target_size,
                    img_mean,
                    img_std,
                    shuffle = False,
                    num_threads = 1, 
                    buf_size = 1024, format = 'pkl'):
    def reader():
        with open(pickle_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                pickle_path = line.strip()
                yield [pickle_path]
    if format == 'pkl':
        decode_func = decode_pickle
    elif format == 'mp4':
        decode_func = decode_mp4
    else:
        raise "Not implemented format {}".format(format)

    mapper = functools.partial(
        decode_func,
        phase=phase,
        seg_num=seg_num,
        seglen = seglen,
        short_size=short_size,
        target_size=target_size, 
        img_mean = img_mean,
        img_std = img_std)

    return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)


def decode_mp4(sample, phase, seg_num, seglen, short_size, target_size, img_mean, img_std):
    sample = sample[0].split(' ')
    mp4_path = sample[0]
    # when infer, we store vid as label
    label = int(sample[1])
    imgs = mp4_loader(mp4_path, seg_num, seglen, phase)
    imgs = group_scale(imgs, short_size)

    if phase == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (seg_num, seglen*3, target_size, target_size))

    return imgs, label


def decode_pickle(sample, phase, seg_num, seglen, short_size, target_size, img_mean, img_std):
    pickle_path = sample[0]
    try:
        if python_ver < (3, 0):
            data_loaded = pickle.load(open(pickle_path, 'rb'))
        else:
            data_loaded = pickle.load(open(pickle_path, 'rb'), encoding='bytes')

        vid, label, frames = data_loaded
        if len(frames) < 1:
            logger.info('{} frame length {} less than 1.'.format(pickle_path, len(frames)))
            raise
    except:
        logger.info('Error when loading {}'.format(pickle_path))
        return None, None


    imgs = video_loader(frames, seg_num, seglen, phase)
    imgs = group_scale(imgs, short_size)

    if phase == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (seg_num, seglen*3, target_size, target_size))

    if phase == 'train' or phase == 'valid' or phase == 'test':
        return imgs, label
    elif phase == 'infer':
        return imgs, vid


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
             "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs


def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(StringIO(buf))
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def video_loader(frames, nsample, seglen, phase):
    videolen = len(frames)
    average_dur = int(videolen / nsample)

    imgs = []
    for i in range(nsample):
        idx = 0
        if phase == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i*average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i*average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % videolen)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs


def mp4_loader(filepath, nsample, seglen, phase):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    average_dur = int(videolen / nsample)
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)

    imgs = []
    for i in range(nsample):
        idx = 0
        if phase == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = sampledFrames[int(jj % videolen)]
            img = Image.fromarray(imgbuf, mode = 'RGB')
            imgs.append(img)

    return imgs

