import ast
import os.path

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



class Aff2_Dataset_static_shuffle(Dataset):
    def __init__(self, root, transform, df=None):
        super(Aff2_Dataset_static_shuffle, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
            self.df = pd.DataFrame()

            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.list_labels_ex = np.array(self.df['labels_ex'].values)
        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        label = self.list_labels_ex[index]

        # image = cv2.imread(self.list_image_id[index])[..., ::-1]
        image = Image.open(self.list_image_id[index]).convert('RGB')
        sample = {
            'images': self.transform(image),
            'labels': torch.tensor(label)
        }
        return sample

    def __len__(self):
        return len(self.df)


class Aff2_Dataset_audio(Dataset):
    def __init__(self, root, sec, transform_aud, df=None):
        super(Aff2_Dataset_audio, self).__init__()
        self.prefix_aff2 = '/mnt/c/Data/Yuxuan/ABAW'
        self.sec = sec
        self.fps = 30
        self.sample_rate = 22050

        if root:
            self.list_csv = glob.glob(root + '*')
            # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
            self.df = pd.DataFrame()

            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(drop=True)
        else:
            self.df = df

        self.transforms_aud = transform_aud
        self.list_labels_ex = np.array(self.df['labels_ex'].values)
        self.list_image_id = np.array(self.df['image_id'].values)

        file = "/mnt/c/Data/Yuxuan/ABAW/image_frames.txt"
        self.data = open(file, 'r')
        self.videos, self.frames = self.get_frame()
        self.data.close()

        self.mapping = self.get_mapping()

    def __getitem__(self, index):
        data_info = self.mapping[index]
        # image = Image.open(data_info["Image"]).convert('RGB')
        label = data_info["Label"]
        audio = self.load_audio(data_info["audio_path"], data_info["Audio"])
        if np.isnan(audio[0][0].item()):
            raise ValueError
        audio = self.transforms_aud(audio)
        if np.isnan(audio[0][0][0].item()):
            raise ValueError
        sample = {
            # 'images': self.transform(image),
            'labels': torch.tensor(label),
            'audios': audio
        }
        return sample

    def get_frame(self):
        videos = []
        frames = []
        for data_info in self.data.readlines():
            path = data_info.split(",")[1]
            frame = int(data_info.split(",")[2].split("\n")[0])-1
            frame_name = path.split('/')[-1]

            videos.append(frame_name)
            frames.append(frame)
        return videos, frames

    def get_real_id(self, path):
        # return the true id in video sequence
        return int(path.split('/')[-1].split('.')[0])-1

    def get_audio_id(self, ID, frames):
        '''
        length: float
            the duration of the clip(seconds)
        Sample rate:
            audio: 22050; 22050
            vid 30; 25
        '''
        hf_sec = self.sec/2
        # IDA = self.sample_rate * ID / self.fps
        ID_sec = ID/self.fps
        total_sec = frames/self.fps
        if ID_sec - hf_sec >= 0 and ID_sec + hf_sec < total_sec:
            return ID_sec - hf_sec
        # elif ID_sec - hf_sec < 0:
            # return 0
        else:
            return None
            # return total_sec - self.sec

    def _get_sample(self, path, resample=None):
        effects = [
            ["remix", "1"]
        ]
        if resample:
            effects.extend([
                ["lowpass", f"{resample // 2}"],
                ["rate", f'{resample}'],
            ])
        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

    def load_audio(self, path, offset):
        '''
        offset : float
            start reading after this time (in seconds)

        duration : float
            only load up to this much audio (in seconds)
        '''
        duration = self.sec
        resample_rate = self.sample_rate

        metadata = torchaudio.backend.sox_io_backend.info(path)
        audio_raw, sample_rate = torchaudio.backend.sox_io_backend.load(
            filepath=path,
            frame_offset=int(offset * metadata.sample_rate),
            num_frames=duration * metadata.sample_rate)
        # audio_raw[0] gets a 1 dim tendor
        # need unsqueeze here
        audio = torch.unsqueeze(audio_raw[0], dim=0)
        if audio.shape[1] < sample_rate * duration:
            audio = torch.nn.functional.pad(audio_raw, (sample_rate * duration - audio_raw.shape[1], 0))
        if sample_rate != resample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)
            audio = resampler(audio)

        return audio

    def get_mapping(self):
        mapping = {}
        cpt = 0
        for i in (pbar := tqdm(range(len(self.list_image_id)))):
            path = self.list_image_id[i]
            label = self.list_labels_ex[i]
            video = path.split('/')[-2]
            frames = self.frames[self.videos.index(video)]
            # pbar_path = path.split('/')[-1]
            pbar.set_description('Processing file %s' % video)

            # img_path = glob.glob(f"{self.prefix_aff2}/origin_faces/{path}/*.jpg")
            if '_left' in video:
                audio_name = video[:-5]
            elif '_right' in video:
                audio_name = video[:-6]
            else:
                audio_name = video
            audio_path = f"{self.prefix_aff2}/origin_audios/{audio_name}.wav"

            idx = self.get_real_id(path)
            a_idx = self.get_audio_id(idx, frames)
            if a_idx is None:
                continue

            mapping[cpt] = {
                "Image": path,
                "Label": label,
                "Audio": a_idx,
                "audio_path": audio_path
            }
            cpt += 1
        return mapping

    def __len__(self):
        return len(self.mapping)


class Aff2_Dataset_series_shuffle(Dataset):
    def __init__(self, root, transform, length_seq, df=None):
        super(Aff2_Dataset_series_shuffle, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # import pdb; pdb.set_trace()
            self.df = pd.DataFrame()
            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.one_df = pd.read_csv(i)
                self.one_df = pad_if_need(self.one_df, length_seq)
                self.df = pd.concat((self.df, self.one_df), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.list_labels_ex = np.array(self.df['labels_ex'].values)

        self.list_image_id = np.array(self.df['image_id'].values)
        self.length = length_seq

    def __getitem__(self, index):
        images = []
        labels = []
        for i in range(index * self.length, index * self.length + self.length):
            label = self.list_labels_ex[i]

            # image = cv2.imread(self.list_image_id[i])[..., ::-1]
            image = Image.open(self.list_image_id[index]).convert('RGB')
            images.append(image)
            labels.append(label)
        # import pdb; pdb.set_trace()
        images = np.stack(images)
        labels = labels[int(np.floor(self.length/2))]
        sample = {
            'images': self.transform(images),
            'labels': torch.tensor(labels)
        }
        return sample

    def __len__(self):
        return len(self.df) // self.length


class Aff2_Dataset_series_shuffle_test(Dataset):
    def __init__(self, root, transform, length_seq, df=None):
        super(Aff2_Dataset_series_shuffle_test, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # import pdb; pdb.set_trace()
            self.df = pd.DataFrame()
            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.one_df = pd.read_csv(i)
                self.one_df = pad_if_need(self.one_df, length_seq)
                self.df = pd.concat((self.df, self.one_df), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.list_labels_ex = np.array(self.df['labels_ex'].values)
        self.list_image_id = np.array(self.df['image_id'].values)
        self.length = length_seq

    def __getitem__(self, index):
        images = [] #0-N 0-N-2
        labels = []
        for i in range(index, index + self.length):
            try:
                label = self.list_labels_ex[i]
                image = Image.open(self.list_image_id[index]).convert('RGB')
                images.append(self.transform(image))
                labels.append(label)
            except:
                pass
        # import pdb; pdb.set_trace()
        images = np.stack(images)
        labels = int(np.array(labels).mean())
        if images.shape[0]==1:
            cache_image = np.zeros(self.length, images.shape[1], images.shape[2])
            cache_image[0] = images
            images = cache_image

        sample = {
            'images': torch.tensor(images),
            'labels': torch.tensor(labels)
        }
        return sample

    def __len__(self):
        return len(self.df)


class DFEW_Dataset(Dataset):
    def __init__(self, args, phase):
        # Basic info
        self.args = args
        self.phase = phase

        # File path
        label_df_path = os.path.join(self.args.data_root,
                                     "label/{data_type}_{phase}set_{fold_idx}.csv".format(data_type=self.args.data_type,
                                                                                          phase=self.phase,
                                                                                          fold_idx=str(
                                                                                              int(self.args.fold_idx))))
        label_df = pd.read_csv(label_df_path)

        # Imgs & Labels
        self.names = label_df['video_name']
        self.videos_path = [os.path.join(self.args.data_root, "data/{name}".format(name=str(ele).zfill(5)))
                            for ele in self.names]
        self.single_labels = torch.from_numpy(np.array(label_df['label']))

        # Transforms
        self.my_transforms_fun_dataAugment = self.my_transforms_fun_dataAugment()
        self.my_transforms_te = self.my_transforms_fun_te()

    def __len__(self):
        return len(self.single_labels)

    def __getitem__(self, index):
        imgs_per_video = glob.glob(self.videos_path[index] + '/*')
        imgs_per_video = sorted(imgs_per_video)
        imgs_idx = self.generate_index(nframe=self.args.nframe,
                                       idx_start=0,
                                       idx_end=len(imgs_per_video) - 1,
                                       phase=self.phase,
                                       isconsecutive=self.args.isconsecutive)
        data = torch.zeros(3, self.args.nframe, self.args.size_Resize_te, self.args.size_Resize_te)
        for i in range(self.args.nframe):
            img = Image.open(imgs_per_video[imgs_idx[i]])
            if self.phase == "train":
                if self.args.train_data_augment == True:
                    img = self.my_transforms_fun_dataAugment(img)
                else:
                    img = self.my_transforms_te(img)
            if self.phase == "test":
                img = self.my_transforms_te(img)
            data[:, i, :, :] = img

        single_label = self.single_labels[index]

        return data, single_label

    def generate_index(self, nframe, idx_start, idx_end, phase, isconsecutive):
        if (idx_end - idx_start + 1) < nframe:
            idx_list_tmp = list(range(idx_start, idx_end + 1))
            idx_list = []
            for j in range(100):
                idx_list = idx_list + idx_list_tmp
                if len(idx_list) >= nframe:
                    break
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end - nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx_tmp = list(range(idx_s, idx_s + nframe))
                idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
            if isconsecutive == False:
                if self.phase == "train":
                    idx_tmp = random.sample(range(len(idx_list)), nframe)
                    idx_tmp.sort()
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
                else:
                    idx_tmp = np.linspace(0, len(idx_list) - 1, nframe).astype(int)
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]

        if (idx_end - idx_start + 1) >= nframe:
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end - nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx = list(range(idx_s, idx_s + nframe))
            if isconsecutive == False:
                if self.phase == "train":
                    idx = random.sample(range(idx_start, idx_end + 1), nframe)
                    idx.sort()
                else:
                    idx = np.linspace(idx_start, idx_end, nframe).astype(int)

        return idx

    def my_transforms_fun_dataAugment(self):
        my_img_transforms_list = []
        if self.args.Flag_RandomRotation:        my_img_transforms_list.append(
            transforms.RandomRotation(degrees=self.args.degree_RandomRotation))
        if self.args.Flag_CenterCrop:            my_img_transforms_list.append(
            transforms.CenterCrop(self.args.size_CenterCrop))
        if self.args.Flag_RandomResizedCrop:     my_img_transforms_list.append(
            transforms.RandomResizedCrop(self.args.size_RandomResizedCrop))
        if self.args.Flag_RandomHorizontalFlip:  my_img_transforms_list.append(
            transforms.RandomHorizontalFlip(p=self.args.prob_RandomHorizontalFlip))
        if self.args.Flag_RandomVerticalFlip:    my_img_transforms_list.append(
            transforms.RandomVerticalFlip(p=self.args.prob_RandomVerticalFlip))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (
                self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        if self.args.Flag_RamdomErasing:        my_tensor_transforms_list.append(
            transforms.RandomErasing(p=self.args.prob_RandomErasing))

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms = transforms.Compose(my_transforms_list)

        return my_transforms

    def my_transforms_fun_te(self):
        my_img_transforms_list = []
        if self.args.Flag_Resize_te == True: my_img_transforms_list.append(transforms.Resize(self.args.size_Resize_te))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (
                self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms = transforms.Compose(my_transforms_list)

        return


class AFFECTNET_Dataset(Dataset):
    def __init__(self, df, transform, root):
        if not isinstance(df, pd.DataFrame):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self.transforms = transform
        self.root = root
        self.list_image_id = self.df['image_id'].values
        self.arousal = self.df['labels_aro'].values
        self.valence = self.df['labels_val'].values

        self.labels_ex = self.df['labels_exp'].values
        # self.landmark = self.df['facial_landmarks'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # path = os.path.join(self.root + self.list_filepath[idx])
        # image = cv2.imread(path)[..., ::-1]
        image = Image.open(self.list_image_id[idx]).convert('RGB')

        labels_conti = [(self.arousal[idx] + 10) / 20.0,
                        (self.valence[idx] + 10) / 20.0]
        labels_conti = np.asarray(labels_conti)
        expression = np.asarray(self.labels_ex[idx])

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
                  'labels_conti': labels_conti,
                  # 'landmark': landmark,
                  'labels': expression}

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


class AudioLoader(Dataset):
    def __init__(self, data, aud_transforms1, aud_transforms2,
                path_prefix, sec=4):
        """
        Expected data format : list of video
        """

        self.data = data
        # self.nb_frame = nb_frame
        self.prefix = path_prefix
        self.transforms_aud1 = aud_transforms1
        self.transforms_aud2 = aud_transforms2
        # self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=sec*sr)
        self.mapping = self.get_idx_mapping()
        self.sec = sec

        # self.augment_files_music = glob.glob(os.path.join(musan_path, 'music','*/*.wav'))
        # self.augment_files_noise = glob.glob(os.path.join(musan_path, 'noise', '*/*.wav'))
        # self.augment_files_speech = glob.glob(os.path.join(musan_path, 'speech', '*/*.wav'))
        # self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        aud_info = self.mapping[idx]

        aud, sr = self.load_audio(aud_info["path"], self.sec)
        # aud1 = aud[:, ::2]
        # aud2 = aud[:, 1::2]

        aud1 = torch.from_numpy(self.transforms_aud1(samples=aud.float().numpy(), sample_rate=sr))
        # aud2 = torch.from_numpy(self.transforms_aud2(samples=aud2, sample_rate=sr))
        aud2 = torch.from_numpy(self.transforms_aud1(samples=aud.float().numpy(), sample_rate=sr))

        return aud1, aud2

    def get_idx_mapping(self):
        # steps = self.nb_frame // 2
        mapping = {}
        idx = 0
        cpt = 0
        for aud_info in self.data.readlines():
            path = aud_info.split('\n')[0]
            # frames = int(aud_info.split(",")[2].split("\n")[0])
            vid_path = f"{self.prefix}/audio/{path}.wav"

            mapping[cpt] = {
                "path": vid_path,
                "start_frame": idx
            }
            cpt += 1

        return mapping

    def loadWAV(self, filename, max_sec, evalmode=True, num_eval=10):
        # Read wav file and convert to torch tensor
        audio, sample_rate = soundfile.read(filename)

        # Maximum audio length
        # max_audio = max_frames * 160 + 240
        max_audio = max_sec*sample_rate

        audiosize = audio.shape[0]

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        if evalmode:
            startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

        feats = []
        if evalmode and max_sec == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = np.stack(feats, axis=0).astype(np.float)

        return feat

    def load_audio(self, path, sec):
        audio, sample_rate = torchaudio.load(path, normalize=True)
        audio_crop = sample_rate * sec
        if audio.shape[1] >= audio_crop:
            audio = self.rand_crop(audio, audio_crop)
        else:
            audio = self.pad_audio(audio, audio_crop)

        return audio, sample_rate

    def rand_crop(self, audio, len_s):
        start_s = random.randint(0, audio.shape[1] - len_s)
        return audio[:, start_s: start_s + len_s]

    def pad_audio(self, data, N_tar):
        # Calculate target number of samples
        # N_tar = int(fs * T)
        # Calculate number of zero samples to append
        shape = data.shape
        if shape[0] > 1:
            N_pad = N_tar - shape[0]
            audio = np.vstack((np.zeros([N_pad, ]), data))
        else:
            # Create the target shape
            N_pad = N_tar - shape[1]
            audio = np.hstack((np.zeros([1, N_pad]), data))
        return torch.tensor(audio)

    # def reverberate(self, audio):
    #     rir_file = random.choice(self.rir_files)
    #     rir, sample_rate = torchaudio.load(rir_file)
    #     try:
    #         audio = torch.nn.functional.pad(audio, (rir.shape[1] - 1, 0))
    #     except:
    #         rir = torch.nn.functional.pad(rir, (audio.shape[1] - 1, 0))
    #     aud_augmented = torch.nn.functional.conv1d(audio[None, ...], rir[None, ...])[0]
    #     return aud_augmented



