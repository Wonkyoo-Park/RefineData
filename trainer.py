import time, datetime, argparse, cv2, os
from models.train_model_segmentation import Model2eye
from utils.utils import *
from tfrecord_util.parse_tfrecords import read_tfrecord, read_record_validation, read_record_only_img_validation, apply_aug_train, apply_validation, apply_only_img_validation
import tensorflow as tf
import warnings , os
from models import fpn
warnings.filterwarnings(action='ignore')
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
===========================================================
                       configuration
===========================================================
"""

start = time.time()
time_now=datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",default=1,type=str)
parser.add_argument("--epoch", default=500, type=int)
parser.add_argument("--maxClsSize", default=45, type=int)
parser.add_argument("--target_size", default=250, type=list,nargs="+",help = "Image size after crop")
parser.add_argument("--batch_size", default=4, type=int,help = "Minibatch size(global)")
parser.add_argument("--val_batch_size", default=1, type=int,help = "Minibatch size(global)")
parser.add_argument("--data_root", default='/home/projects/data/train/tfrecords/', type=str,help = "Dir to data root")
parser.add_argument("--val_data_root", default='/home/projects/data/validation/val_tfrecords/', type=str,help = "Dir to val data root")
parser.add_argument("--target_width", default=512, type=int, help="target width of input image")
parser.add_argument("--target_height", default=512, type=int, help="target width of input image")
parser.add_argument("--image_file", default='./dataset/test/176039.jpg', type=str,help = "Dir to data root")
parser.add_argument("--channels", default=1, type=int,help = "Channel size")
parser.add_argument("--color_map", default="RGB", type=str,help = "Channel mode. [RGB, YCbCr]")
parser.add_argument("--model_tag", default="default", type=str,help = "Exp name to save logs/checkpoints.")
parser.add_argument("--checkpoint_dir", default="outputs/checkpoints/", type=str,help = "Dir for checkpoints")
parser.add_argument("--summary_dir", default="outputs/summaries/", type=str,help = "Dir for tensorboard logs.")
parser.add_argument("--restore_file", default=None, type=str,help = "file for restoration")
parser.add_argument("--graph_mode", default=False, type=bool,help = "use graph mode for training")
config = parser.parse_args()

def generate_expname_automatically():
    name = "alphado_eye_%s_%02d_%02d_%02d_%-2d_%02d" % (config.model_tag, time_now.month,time_now.day, time_now.hour,
                                                time_now.minute, time_now.second)
    return name

expname = generate_expname_automatically()
# config.checkpoint_dir += "falling_" + config.model_tag; check_folder(config.checkpoint_dir)
config.summary_dir += expname
check_folder(config.summary_dir)
"""
===========================================================
                      prepare dataset
===========================================================
"""
# read dataset
list_tfrecords = os.listdir(config.data_root)
list_tfrecords = [os.path.join(config.data_root,path2tfrecord) for path2tfrecord in list_tfrecords]
dataset = read_tfrecord(list_tfrecords, batch_size=config.batch_size)

list_val_tfrecords = os.listdir(config.val_data_root)
list_only_img_val_tfrecords = [os.path.join(config.val_data_root,path2tfrecord) for path2tfrecord in list_val_tfrecords if path2tfrecord.split('_')[0] =='only']
list_val_tfrecords = [os.path.join(config.val_data_root,path2tfrecord) for path2tfrecord in list_val_tfrecords if not path2tfrecord.split('_')[0] =='only']

# dataset = read_tfrecord(list_tfrecords[:-1], batch_size=config.batch_size)
validation_dataset = read_record_validation(list_val_tfrecords, batch_size=1)
only_image_validation_dataset = read_record_only_img_validation(list_only_img_val_tfrecords, batch_size=1)

"""
===========================================================
                      build model
===========================================================
"""
model = Model2eye(config)
# model = fpn.ResNet50Seg(config.maxClsSize, input_shape=(512, 512, 3), weights='imagenet')
# model.restore(50)
for e in range(config.epoch):
    for i, image_features in enumerate(dataset):
        # print(image_features),
        data = apply_aug_train(image_features, config)
        log = model.train_step(data,config.maxClsSize,e)
    print("[training : epoch {} train step {}] step : {}".format(e,i,log))
    if e % 5 == 0:
        save_path = model.save(e)
        for i, image_features in enumerate(validation_dataset):
            # print(image_features),
            data = apply_validation(image_features, config)
            log = model.validation_step(data,config.maxClsSize)

        print(" validation : epoch {} train step {}] log : {}".format(e, i, log))
        for i, image_features in enumerate(only_image_validation_dataset):
            data = apply_only_img_validation(image_features, config)
            model.only_img_validation_step(data,config.maxClsSize)
    model.reset_state()
    # model.train_loss.reset_states()
    # model.train_location_acc.reset_states()
    # model.train_falling_acc.reset_states()

config = parser.parse_args()