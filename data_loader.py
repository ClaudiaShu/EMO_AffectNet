import ast
import os.path

import h5py
import librosa
import numpy as np
import soundfile
import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import *
import pdb


class AFFECTNET_Dataset(Dataset):
    def __init__(self, df, transform, task):
        self.prefix = "/data2/yshu21/data/"
        if not isinstance(df, pd.DataFrame):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self.transforms = transform
        self.task = task
        self.list_image_id = self.df['image_id'].values
        self.arousal = self.df['labels_aro'].values
        self.valence = self.df['labels_val'].values

        self.labels_ex = self.df['labels_exp'].values
        # self.landmark = self.df['facial_landmarks'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.prefix + self.list_image_id[idx])
        # image = cv2.imread((self.list_image_id[idx])[..., ::-1]
        image = Image.open(path).convert('RGB')
        if self.task == 'va':
            labels = [self.arousal[idx], self.valence[idx]]
            # labels = [(self.arousal[idx] + 10) / 20.0,
            #           (self.valence[idx] + 10) / 20.0]
            labels = np.asarray(labels)
            # labels_conti = [(self.arousal[idx] + 10) / 20.0,
            #                 (self.valence[idx] + 10) / 20.0]

        elif self.task == 'exp':
            labels = np.asarray(self.labels_ex[idx])
        else:
            raise ValueError

        # x = self.df.face_x[idx]
        # y = self.df.face_y[idx]
        # w = self.df.face_width[idx]
        # h = self.df.face_height[idx]
        #
        # even = np.array(self.landmark[idx].split(';'), float)[0::2]
        # odd = np.array(self.landmark[idx].split(';'), float)[1::2]
        #
        # even = (even - x) / w
        # odd = (odd - y) / h
        # landmark = [[x, y] for [x, y] in zip(even, odd)]
        # landmark = np.concatenate(landmark, axis=0)

        sample = {'images': self.transforms(image),
                  # 'labels_conti': labels_conti,
                  'labels': labels}

        return sample


class AFFECTNET_hdf5(Dataset):
    def __init__(self, hdf5_file_name, transform, task):
        self.hdf5_file_name = hdf5_file_name
        self.hf = h5py.File(self.hdf5_file_name, "r")
        self.list_image_id = self.hf['imgs'][:]
        self.task = task
        self.labels_ex = self.hf['labels'][:]
        self.arousal = self.hf['arousal'][:]
        self.valence = self.hf['valence'][:]
        self.hf.close()

        self.transforms = transform

    def __len__(self):
        return len(self.list_image_id)

    def __getitem__(self, idx):
        # path = os.path.join(self.root + self.list_filepath[idx])
        # image = cv2.imread(path)[..., ::-1]
        # image = Image.open(self.list_image_id[idx]).convert('RGB')
        image = self.list_image_id[idx]
        if self.task == 'va':
            labels = [self.arousal[idx], self.valence[idx]]
            # labels = [(self.arousal[idx] + 10) / 20.0,
            #           (self.valence[idx] + 10) / 20.0]
            labels = np.asarray(labels)
        elif self.task == 'exp':
            labels = np.asarray(self.labels_ex[idx])
        else:
            raise ValueError

        sample = {'images': self.transforms(image),
                  # 'labels_conti': labels_conti,
                  'labels': labels}

        return sample


class SimCLRLoader(Dataset):
    # Takes in video and its calculated frames
    def __init__(self, data, nb_frame, transforms_vid1, transforms_vid2, path_prefix, time_aug=True):
        """
        Expected data format : list of video ?
        """

        self.data = data
        self.nb_frame = nb_frame
        self.prefix = path_prefix
        self.transforms_vid1 = transforms_vid1
        self.transforms_vid2 = transforms_vid2
        self.use_time_augmentation = time_aug

        self.mapping = self.get_idx_mapping()

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):

        vid_info = self.mapping[idx]
        if self.use_time_augmentation:
            start_frame1 = vid_info["start_frame1"]
            start_frame2 = vid_info["start_frame2"]
            end_frame1 = start_frame1 + self.nb_frame
            end_frame2 = start_frame2 + self.nb_frame
            vid1 = self.load_clip(vid_info["path"], start_frame1, end_frame1)
            vid2 = self.load_clip(vid_info["path"], start_frame2, end_frame2)
        else:
            start_frame = vid_info["start_frame"]

            end_frame = start_frame + self.nb_frame

            vid1 = self.load_clip(vid_info["path"], start_frame, end_frame)
            vid2 = vid1.copy()

        try:
            vid1 = self.transforms_vid1(vid1)
            vid2 = self.transforms_vid2(vid2)
        except:
            vid1 = np.zeros([self.nb_frame,300,300,3])
            vid2 = vid1.copy()

            vid1 = self.transforms_vid1(vid1)
            vid2 = self.transforms_vid2(vid2)

        return vid1, vid2

    def get_idx_mapping(self):
        steps = self.nb_frame // 2
        mapping = {}
        cpt = 0

        for vid_info in self.data.readlines():
            path = vid_info.split(",")[1]
            frames = int(vid_info.split(",")[2].split("\n")[0])
            vid_path = f"{self.prefix}/{path}.mp4"

            if self.use_time_augmentation:
                # augmentation in time domain
                rand_in = int(frames/2-self.nb_frame)
                if rand_in > 0:
                    offset = random.randint(0,rand_in)
                    mapping[cpt] = {
                        "path": vid_path,
                        "start_frame1": offset,
                        "start_frame2": offset+int(np.floor(frames/2))
                    }
                    cpt += 1
                else:
                    pass
            else:
                # no augmentation in time domain
                nbr_clips = (frames // steps) - 1
                for idx in range(0, nbr_clips, 30):
                    mapping[cpt] = {
                        "path": vid_path,
                        "start_frame": idx
                    }
                    cpt += 1

        return mapping

    def load_clip(self, path, start, end):

        frame_array = []

        capture = cv2.VideoCapture(path)
        # Begin at starting frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

        success, frame = capture.read()

        while success:
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            frame_array.append(frame / 255)

            success, frame = capture.read()

            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1

            if frame_number >= end:
                success = False

        return np.array(frame_array)


class SimCLRLoader_csv(Dataset):
    def __init__(self, data, nb_frame, transforms_vid1, transforms_vid2, path_prefix=""):
        """
        Expected data format : list of video ?
        """

        self.data = data
        self.nb_frame = nb_frame
        self.prefix = path_prefix
        self.transforms_vid1 = transforms_vid1
        self.transforms_vid2 = transforms_vid2
        self.mapping = self.get_idx_mapping()

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):

        vid_info = self.mapping[idx]
        start_frame = vid_info["start_frame"]

        end_frame = start_frame + self.nb_frame

        vid1 = self.load_clip(vid_info["path"], start_frame, end_frame)
        vid2 = vid1.copy()

        vid1 = self.transforms_vid1(vid1)
        vid2 = self.transforms_vid2(vid2)

        return vid1, vid2

    def get_idx_mapping(self):
        steps = self.nb_frame // 2
        mapping = {}
        cpt = 0

        for vid_info in self.data.readlines():
            path = vid_info.split(",")[1]
            frames = int(vid_info.split(",")[2].split("\n")[0])
            vid_path = f"{self.prefix}/{path}.mp4"

            nbr_clips = (frames // steps) - 1

            for idx in range(nbr_clips):
                mapping[cpt] = {
                    "path": vid_path,
                    "start_frame": idx
                }
                cpt += 1

        return mapping

    def load_clip(self, path, start, end):

        frame_array = []

        capture = cv2.VideoCapture(path)
        # Begin at starting frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

        success, frame = capture.read()

        while success:
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            frame_array.append(frame / 255)

            success, frame = capture.read()

            frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1

            if frame_number >= end:
                success = False

        return np.array(frame_array)

