import tensorflow as tf
from backbones.resnet import ResNet_
# from backbones.mobilenet_v2 import MobileNetV2_
from tools.evaluations import get_val_data, perform_val
import os
from config import *


os.environ['CUDA_VISIBLE_DEVICES'] = GPU
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

net = ResNet_(
  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
  num_classes=NUM_FEATURES,
  repetitions=NET_CONFIG
)
# net = MobileNetV2_(
#   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
#   num_classes=NUM_FEATURES,
# )

net.load_weights(WEIGHTS_FILE, by_name=True)
model = net.model

cfg = {
  'test_dataset': TEST_DATASET_DIR,
  'embd_shape': NUM_FEATURES,
  'batch_size': 8,
  'is_ccrop': True,
}

print("Loading LFW, AgeDB30 and CFP-FP...")
lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame =  get_val_data(cfg['test_dataset'])

print("Perform Evaluation on LFW...")
acc_lfw, best_th = perform_val(cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
                               is_ccrop=cfg['is_ccrop'])
print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

print("Perform Evaluation on AgeDB30...")
acc_agedb30, best_th = perform_val(cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
                                   agedb_30_issame, is_ccrop=cfg['is_ccrop'])
print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

print("Perform Evaluation on CFP-FP...")
acc_cfp_fp, best_th = perform_val(cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
                                  is_ccrop=cfg['is_ccrop'])
print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))