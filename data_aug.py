import os.path

from torchvision import transforms as cv_tf
from torchaudio import transforms as au_tf
from utils import video_transforms as vd_tf
from utils import audio_transforms as ad_tf

from audiomentations import (
    Compose,
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    TimeStretch,
    PitchShift,
    Shift
)

from utils.gaussian_blur import GaussianBlur
# from utils.video_transforms import (
#     ChangeVideoShape,
#     ResizeVideo,
#     RandomCropVideo,
#     CenterCropVideo,
#     I3DPixelsValue,
#     RandomTrimVideo,
#     TrimVideo,
#     PadVideo,
#     RandomColorJitterVideo,
# )
from utils.image_transforms import imgRandomLandmarkMask

MUSAN = "/mnt/d/Data/Yuxuan/VoxCeleb/musan"
RIR = "/mnt/d/Data/Yuxuan/VoxCeleb/rirs_noises"
ESC = '/mnt/d/Data/Yuxuan/VoxCeleb/ESC-50'



def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = cv_tf.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = cv_tf.Compose([cv_tf.RandomResizedCrop(size=size),
                                          cv_tf.RandomHorizontalFlip(),
                                          cv_tf.RandomApply([color_jitter], p=0.8),
                                          cv_tf.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          cv_tf.ToTensor()])
    return data_transforms


def get_image_transform(size=112):
    train_transform = cv_tf.Compose(
        [
            # cv_tf.ToPILImage(),
            imgRandomLandmarkMask(),
            cv_tf.Resize(size=(112, 112)),
            # cv_tf.RandomCrop(112),
            cv_tf.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply
            # cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
            cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
            # cv_tf.GaussianBlur(3),
            cv_tf.RandomGrayscale(p=0.2),
            cv_tf.ToTensor(),
            cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    # train_transform = cv_tf.Compose([
    #     # cv_tf.ToPILImage(),
    #     cv_tf.Resize(size=(size, size)),
    #     cv_tf.RandomHorizontalFlip(),
    #     cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #     # cv_tf.GaussianBlur(3), # add
    #     cv_tf.ToTensor(),
    #     cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transform = cv_tf.Compose([
        # cv_tf.ToPILImage(),
        cv_tf.Resize(size=(size, size)),
        cv_tf.ToTensor(),
        cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return train_transform, test_transform


def get_cv_image_transform(size=112):
    train_transform = cv_tf.Compose(
        [
            cv_tf.ToPILImage(),
            imgRandomLandmarkMask(),
            # cv_tf.Resize(size=(112, 112)),
            # cv_tf.RandomCrop(112),
            cv_tf.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply
            # cv_tf.RandomApply([cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.8),
            cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            cv_tf.RandomApply([cv_tf.GaussianBlur(3)], p=0.5),
            # cv_tf.GaussianBlur(3),
            cv_tf.RandomGrayscale(p=0.2),
            cv_tf.ToTensor(),
            cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    # train_transform = cv_tf.Compose([
    #     # cv_tf.ToPILImage(),
    #     cv_tf.Resize(size=(size, size)),
    #     cv_tf.RandomHorizontalFlip(),
    #     cv_tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #     # cv_tf.GaussianBlur(3), # add
    #     cv_tf.ToTensor(),
    #     cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transform = cv_tf.Compose([
        cv_tf.ToPILImage(),
        # cv_tf.Resize(size=(size, size)),
        cv_tf.ToTensor(),
        cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return train_transform, test_transform


def get_video_transform(size):
    vid1_transform = cv_tf.Compose(
        [
            # vd_tf.ToNumpy(),
            vd_tf.RandomLandmarkMask(),
            vd_tf.PadVideo(size),
            vd_tf.ResizeVideo(112, interpolation="linear"),
            # vd_tf.RandomCropVideo((112, 112)),
            vd_tf.RandomHorizontalFlip(),
            vd_tf.RandomColorJitterVideo(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            # vd_tf.RandomGaussianBlur(),
            vd_tf.RandomGrayscale(),
            # vd_tf.I3DPixelsValue(),
            vd_tf.ToTensor(),
            vd_tf.ChangeVideoShape("CTHW"),
            vd_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Add random color drop and jitter
    vid2_transform = cv_tf.Compose(
        [
            # vd_tf.ToNumpy(),
            # vd_tf.RandomLandmarkMask(),  # p=0.5
            vd_tf.PadVideo(size),
            vd_tf.ResizeVideo(112, interpolation="linear"),
            # vd_tf.RandomCropVideo((112, 112)),
            vd_tf.RandomHorizontalFlip(),  # p=0.5
            # vd_tf.RandomColorJitterVideo(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            # vd_tf.RandomGaussianBlur(),
            # vd_tf.RandomGrayscale(),
            # vd_tf.I3DPixelsValue(),
            vd_tf.ToTensor(),
            vd_tf.ChangeVideoShape("CTHW"),
            vd_tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return vid1_transform, vid2_transform

def get_audio_transform():
    aud1_transform = ad_tf.Compose([
        ad_tf.RandomLowpass(),
        ad_tf.RandomBackgroundNoise(),
        ad_tf.RandomSpeed(),
        ad_tf.RandomReverb(),

        # ad_tf.TensorSqueeze(),
        ad_tf.ToMel(),
        ad_tf.RandomTimeMask(),
        ad_tf.RandomFreqMask(),
        cv_tf.Resize(size=(112, 112)),
        ad_tf.ToImage()
    ])
    aud2_transform = ad_tf.Compose([
        # ad_tf.RandomLowpass(),
        # ad_tf.RandomBackgroundNoise(),
        # ad_tf.RandomSpeed(),
        # ad_tf.RandomReverb(),

        # ad_tf.TensorSqueeze(),
        ad_tf.ToMel(),
        # ad_tf.RandomTimeMask(),
        # ad_tf.RandomFreqMask(),
        cv_tf.Resize(size=(112, 112)),
        ad_tf.ToImage()
    ])
    return aud1_transform, aud2_transform
