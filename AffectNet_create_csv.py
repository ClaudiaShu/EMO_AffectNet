import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd


def take_images(data_images):
    images = glob.glob(data_images + '/*.*')
    images.sort()
    number_id = [os.path.split(image)[1].split('.')[0] for image in images]
    number_id = np.array(number_id, int)
    return images, number_id


def take_anno(dir_anno, number_id):
    aro = np.load(os.path.join(dir_anno, str(number_id)) + '_aro.npy').item()
    val = np.load(os.path.join(dir_anno, str(number_id)) + '_val.npy').item()
    exp = np.load(os.path.join(dir_anno, str(number_id)) + '_exp.npy').item()
    lnd = np.load(os.path.join(dir_anno, str(number_id)) + '_lnd.npy')
    annotation = {
        'arousal': aro,
        'valence': val,
        'expression': exp,
        'landmark': lnd
    }
    return annotation


if __name__ == '__main__':

    dir_images = '/data/AffectNet/val_set/images/' # The dir where you save the AffectNet images
    dir_anno = '/data/AffectNet/val_set/annotations/' # The dir where you save the AffectNet annotations
    save_dir = '/ys221/data/AffectNet/csv/' # Output csv file

    isFile = os.path.isfile(save_dir)
    if not isFile:
        os.makedirs(save_dir, exist_ok=True)

    images, number_id = take_images(dir_images)
    # print(images[0])

    arousal = []
    valence = []
    expression = []
    landmark = []

    for i in tqdm(number_id, total=len(number_id)):
        anno = take_anno(dir_anno, i)
        arousal.append(anno['arousal'])
        valence.append(anno['valence'])
        expression.append(anno['expression'])
        landmark.append(anno['landmark'])

    arousal = np.array(arousal)
    valence = np.array(valence)
    expression = np.array(expression, int)
    # landmark = np.array(landmark)

    df = pd.DataFrame(images, columns=['image_id'])
    df['labels_ex'] = expression
    df['labels_aro'] = arousal
    df['labels_val'] = valence
    df['facial_landmarks'] = landmark
    print()
    df.to_csv(save_dir + 'test.csv')

