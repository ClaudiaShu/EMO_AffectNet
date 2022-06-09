import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


## Loss 1 ##
def get_loss_data(file_path, col, skip_line):
    # file_path = '/data/users/ys221/software/SSL_downstream/runs/May16_15-48-52_fuhe.doc.ic.ac.uk/training.log'
    file = open(file_path, 'r')
    loss = []
    for i in range(skip_line):
        line = file.readline()

    for lines in tqdm(file.readlines()):
        loss.append(lines.split('\t')[col].split(':')[-1])
    loss = np.stack(loss).astype(float)
    file.close()
    return loss

if __name__ == '__main__':
    col_loss = 1
    col_top1 = 2
    fileb = "/mnt/d/Data/Yuxuan/logging/videoclr/vox1_frame1/runs/Jun04_23-22-43_ic_xiao/training.log"
    lossb = get_loss_data(fileb, col_top1, 9)

    filec = "/mnt/d/Data/Yuxuan/logging/videoclr/vox1_frame1/runs/videoclr_with_time_neg_img_c/training.log"
    lossc = get_loss_data(filec, col_top1, 9)

    plt.plot(lossb)
    plt.plot(lossc)
    plt.savefig('/mnt/c/Data/Yuxuan/VoxCeleb/test/acc_bc.jpg')