import os.path

from torchvision import transforms as cv_tf

from utils.gaussian_blur import GaussianBlur

from utils.image_transforms import imgRandomLandmarkMask

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
    test_transform = cv_tf.Compose([
        cv_tf.ToPILImage(),
        # cv_tf.Resize(size=(size, size)),
        cv_tf.ToTensor(),
        cv_tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return train_transform, test_transform
