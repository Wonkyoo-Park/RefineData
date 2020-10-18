import sys
# from models.custom_eyeDD_segmentation import *
from models.unet import *
from utils.utils import *
from functools import partial
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
partial_resize = partial(tf.image.resize,method=tf.image.ResizeMethod.BILINEAR,antialias=True)
from utils import functional as F
from utils import metrics
import utils

from models import fpn
class Model2eye():
    def __init__(self,config,mode='normal'):
        self.config = config
        self.step = tf.Variable(0,dtype=tf.int64)
        self.build_model()
        # log_dir = os.path.join(config.summary_dir)

        log_dir = os.path.join(config.summary_dir)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir, 0o777)

        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mean_iou = tf.keras.metrics.Mean(name='train_iou')
        self.val_mean_iou = tf.keras.metrics.Mean(name='val_iou')
        self.train_mean_acc = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_mean_acc = tf.keras.metrics.Mean(name='val_accuracy')
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.val_acc = tf.keras.metrics.CategoricalAccuracy(name='train_location_accuracy')
        self.train_iou = metrics.iou_score
        self.val_iou = metrics.iou_score
        # self.val_iou = metrics.iou_score(num_classes=14,name='train_location_accuracy')
        self.train_falling_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='falling_accuracy')

    def build_model(self):
        """model"""
        # self.model = pspnet.pspnet_resnetd50b_coco(pretrained_backbone=False, classes=45, aux=False, data_format="channels_last",)
        self.model = fpn.ResNet50Seg(self.config.maxClsSize, input_shape=(512, 512, 3), weights='imagenet')
        # tf.saved_model.save(self.model,os.path.join(self.config.checkpoint_dir,"segmentation_epoch_{}/".format(5)))

            # tf2cv_get_model("pspnet", pretrained=True, data_format="channels_last")

        # self.model = build_model(batch=self.config.batch_size,maxClsSize=self.config.maxClsSize,pretrained_weights=False)
        # self.model = build_model(include_top=False,batch=self.config.batch_size,height=400, width=400, color=True, filters=64)
        learning_rate = 0.00001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # self.model.summary()

    def save(self,epoch):
        # self.model.summary()
        # self.mode.inputs[0].shape.dims[0]._value = 6
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        # self.model.summary()

        self.model.save_weights(os.path.join(self.config.checkpoint_dir,"segmentation_epoch_{}.h5".format(epoch)))

    def restore(self, N=None):
        path2load_model = os.path.join(self.config.checkpoint_dir,"segmentation_epoch_{}.h5".format(N))
        self.model.load_weights(path2load_model)

    # @tf.function
    def train_one_step(self,data,maxClsSize,e):
        image=data['img']
        targets=dict()
        targets['seg']=np.expand_dims(data['seg'][...,0],axis=3)
        # targets['falling'] = data['label_falling']
        image=tf.cast(image,tf.float32)
        image=image/255.0
        # image_dataset = oasd 1~100
        target=np.zeros([targets['seg'].shape[0], targets['seg'].shape[1], image.shape[2], maxClsSize], dtype=np.uint8)


        for i in range(maxClsSize):
            target[...,i]=np.where(targets['seg']==i,1,0)[...,0]


        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)

            # Get the error/loss using the Loss_object previously defined
            loss_location = self.loss(target,predictions)

            # loss_falling = self.loss(targets['falling'], predictions[1])
            # loss = 0*loss_location + 20*loss_falling
            loss = loss_location

        # compute the gradient with respect to the loss
        gradients = tape.gradient(loss,self.model.trainable_variables)
        # Change the weights of the model
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        # the metrics are accumulate over time. You don't need to average it yourself.
        self.train_loss(loss)

        if e % 5==0 and e>0:
            train_acc_result = self.val_acc(target, predictions)
            train_iou_result = self.train_iou(target, predictions)
            self.train_mean_acc(train_acc_result)
            self.train_mean_iou(train_iou_result)
            # self.train_location_acc(targets['location'],predictions[0])
            # self.train_falling_acc(targets['falling'], predictions[1])
            return_dicts = {'loss':self.train_loss, 'accuracy': self.val_mean_acc, 'iou':self.train_mean_iou}
            # return_dicts.update({'location_acc':self.train_location_acc})
            # return_dicts.update({'falling_acc':self.train_falling_acc})
        else:
            return_dicts = {'loss': self.train_loss}
        return return_dicts
    def reset_state(self):
        """rest state"""
        self.train_mean_iou.reset_states()
        self.train_mean_acc.reset_states()
        self.train_loss.reset_states()
        self.val_loss.reset_states()
        self.val_mean_acc.reset_states()
        self.val_mean_iou.reset_states()

    def train_step(self,data,maxClsSize,e,summary_name='train',log_interval=0):
        """training"""
        result_logs_dict = self.train_one_step(data,maxClsSize,e)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            for key, value in result_logs_dict.items():
                value = value.result().numpy()
                tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        # log = "loss:{}, location_accuracy:{}, falling_accuracy:{}".format(result_logs_dict["loss"].result().numpy(),result_logs_dict['location_acc'].result().numpy(),result_logs_dict['falling_acc'].result().numpy())
        if e%5==0 and e>0:
            log = "loss:{}, accuracy:{}, iou:{}".format(result_logs_dict["loss"].result().numpy(),result_logs_dict["accuracy"].result().numpy(), result_logs_dict["iou"].result().numpy())
        else:
            log = "loss:{}".format(result_logs_dict["loss"].result().numpy())

        return log

        # @tf.function
    def validation_one_step(self, data,maxClsSize):
        image = data['img']
        targets=dict()
        targets['seg'] = np.expand_dims(data['seg'][..., 0], axis=3)


        image = tf.cast(image, tf.float32)
        image = image / 255.0

        target = np.zeros([targets['seg'].shape[0], targets['seg'].shape[1], image.shape[2], maxClsSize], dtype=np.uint8)

        for i in range(maxClsSize):
            target[..., i] = np.where(targets['seg'] == i, 1, 0)[..., 0]

        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)

            # Get the error/loss using the Loss_object previously defined
            loss_location = self.loss(target, predictions)
            # loss_falling = self.loss(targets['falling'], predictions[1])
            # loss = 0*loss_location + 20*loss_falling
            loss = loss_location

        # compute the gradient with respect to the loss
        # Change the weights of the model
        # the metrics are accumulate over time. You don't need to average it yourself.
        # self.train_location_acc(targets['location'], predictions[0])
        # self.train_falling_acc(targets['falling'], predictions[1])
        self.val_loss(loss)
        val_acc_result = self.val_acc(target, predictions)
        val_iou_result = self.val_iou(target, predictions)
        self.val_mean_acc(val_acc_result)
        self.val_mean_iou(val_iou_result)


        return_dicts = {'loss': self.val_loss, 'accuracy': self.val_mean_acc, 'iou': self.val_mean_iou}
        # return_dicts.update({'location_acc': self.train_location_acc})
        # return_dicts.update({'falling_acc': self.train_falling_acc})
        return return_dicts

    # @tf.function
    def only_img_validation_one_step(self, data,maxClsSize):
        image = data['img']
        targets=dict()
        # targets['seg'] = np.expand_dims(data['seg'][..., 0], axis=3)


        image = tf.cast(image, tf.float32)
        image = image / 255.0

        # target = np.zeros([targets['seg'].shape[0], targets['seg'].shape[1], image.shape[2], 14], dtype=np.uint8)

        # for i in range(14):
        #     target[..., i] = np.where(targets['seg'] == i, 1, 0)[..., 0]

        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)

            # Get the error/loss using the Loss_object previously defined
            # loss_location = self.loss(target, predictions)
            # loss_falling = self.loss(targets['falling'], predictions[1])
            # loss = 0*loss_location + 20*loss_falling
            # loss = loss_location

        # compute the gradient with respect to the loss
        # Change the weights of the model
        # the metrics are accumulate over time. You don't need to average it yourself.
        # self.train_location_acc(targets['location'], predictions[0])
        # self.train_falling_acc(targets['falling'], predictions[1])
        # self.val_loss(loss)
        # return_dicts = {'loss': self.val_loss}
        # return_dicts.update({'location_acc': self.train_location_acc})
        # return_dicts.update({'falling_acc': self.train_falling_acc})
        # return return_dicts

    def validation_step(self,data,maxClsSize,summary_name='validation',log_interval=0):
        """training"""
        result_logs_dict = self.validation_one_step(data,maxClsSize)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            for key, value in result_logs_dict.items():
                value = value.result().numpy()
                tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        log = "loss:{}, accuracy:{}, iou:{}".format(result_logs_dict["loss"].result().numpy(),result_logs_dict["accuracy"].result().numpy(), result_logs_dict["iou"].result().numpy())
        return log
    def test_one_step(self, data,maxClsSize):
        image = data['img']
        targets=dict()
        targets['seg'] = np.expand_dims(data['seg'][..., 0], axis=3)


        image = tf.cast(image, tf.float32)
        image = image / 255.0

        target = np.zeros([targets['seg'].shape[0], targets['seg'].shape[1], image.shape[2], maxClsSize], dtype=np.uint8)

        for i in range(maxClsSize):
            target[..., i] = np.where(targets['seg'] == i, 1, 0)[..., 0]
            # print("no:",i, np.max(target[...,i]), len(np.where(target[...,i]>0)[0]))


        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)

        return predictions, target

    def test_single_image_one_step(self, data,maxClsSize):
        image = data
        image = np.expand_dims(image,axis=0)
        image = tf.cast(image, tf.float32)
        image = image / 255.0


        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)

        return predictions

    def only_img_validation_step(self,data,maxClsSize,summary_name='validation',log_interval=0):
        """training"""
        self.only_img_validation_one_step(data,maxClsSize)
        # result_logs_dict = self.only_img_validation_one_step(data)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        # with self.train_summary_writer.as_default():
        #     for key, value in result_logs_dict.items():
        #         value = value.result().numpy()
        #         tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        # log = "loss:{}".format(result_logs_dict["loss"].result().numpy())
        # return log


    def test_step(self,data,maxClsSize,summary_name='validation',log_interval=0):
        """training"""
        predictions, target = self.test_one_step(data,maxClsSize)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        # with self.train_summary_writer.as_default():
        #     for key, value in result_logs_dict.items():
        #         value = value.result().numpy()
        #         tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)

        return predictions, target

    def single_image_test_step(self,data,maxClsSize,summary_name='validation',log_interval=0):
        """training"""
        predictions = self.test_single_image_one_step(data,maxClsSize)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        # with self.train_summary_writer.as_default():
        #     for key, value in result_logs_dict.items():
        #         value = value.result().numpy()
        #         tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)

        return predictions
















