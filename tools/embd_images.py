import os
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2RGB
import sys
sys.path.append('..')
from backbones.resnet import ResNet_
from config import *
# from scipy.io import savemat
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

net = ResNet_(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    num_classes=NUM_FEATURES,
    repetitions=NET_CONFIG
)
# net.model.summary()
net.load_weights(WEIGHTS_FILE, by_name=True)

data_dir = IMAGE_DIR
last_id = 85741
test_classes = 20
samples_per_class = 20
imsize = IMAGE_SIZE

embds = np.zeros((test_classes * samples_per_class, NUM_FEATURES), dtype=np.float32)
cnt = 0
idx = last_id
while cnt < test_classes * samples_per_class:
    path = os.path.join(data_dir, str(idx))
    files = os.listdir(path)
    if len(files) <= 20:
        print("Skipping ID:%d." % idx)
        idx -= 1
        continue
    for file in files[:20]:
        image_path = os.path.join(path, file)
        im = imread(image_path)
        im = cvtColor(im, COLOR_BGR2RGB)
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255.
        embd = net.model.predict(im)[0]
        embd /= np.linalg.norm(embd, ord=2)
        embds[cnt, :] = embd
        cnt += 1
    print('ID:%d Embedding done.' % idx)
    idx -= 1

# savemat('embds_arcface.mat', {'embds': embds})
r = embds @ np.transpose(embds)
sup = np.max(r)
inf = np.min(r)
r -= inf
r /= (sup - inf)
plt.imshow(r)
plt.axis('off')
plt.show()