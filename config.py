IMAGE_SIZE = 112

NUM_CLASSES = 85642

# argumentation
FLIP = True
BRIGHTNESS = 0.1
HUE = 0.03
CONTRAST = 0.1
SATURATION = 0.1
# ROTATE = 30. / 360. * 2. * 3.14159
ROTATE = 0.
RESIZE = 128
CROP = 112
BLUR = 1
GRAY = True

# dataset
BATCH_SIZE = 512
IMAGE_ROOT_DIR = '/home/USER/datasets/ms1m/'
IMAGE_DIR = IMAGE_ROOT_DIR + 'imgs'
TFRECORDS_PATH = IMAGE_ROOT_DIR + 'ms1m_align_112.tfrecords'


# model
BACKBONE = 'resnet'
# BACKBONE = 'mobilenet'
HEAD = 'arcface'
NET_CONFIG = (3, 4, 6, 3)
NUM_FEATURES = 256
MODEL_NAME = 'resnet50_im112_ms1m_arcface'

# training
GPU = '0, 1'
L2 = 0.0005
INIT_LR = 0.01
MOMENTUM = 0.8
EPOCHS = 25
WORK_DIR = '/home/USER/checkpoints/arcface'
SCHEDULE = [5, 10, 15, 20]

# evaluation
WEIGHTS_FILE = '/path/to/your/weights/file/resnet50_im112_ms1m_arcface_weights.h5'
TEST_DATASET_DIR = '/path/to/your/test/dataset/dir'