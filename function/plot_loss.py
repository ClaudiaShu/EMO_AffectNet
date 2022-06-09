import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


## Loss 1 ##
file_path = '/data/users/ys221/software/SSL_downstream/runs/May16_15-48-52_fuhe.doc.ic.ac.uk/training.log'
file = open(file_path, 'r')
loss1 = []
line1 = file.readline()
line2 = file.readline()
for lines in tqdm(file.readlines()):
    # print(lines)
    loss1.append(lines.split('\t')[-2].split(':')[-1])
loss1 = np.stack(loss1).astype(float)
file.close()

"D:\Data\Yuxuan\logging\videoclr\vox1_frame1\runs\Jun04_23-22-43_ic_xiao\training.log"

plt.plot(loss)
plt.show()