import sys
# from models.custom_eyeDD_segmentation import *
from models.unet import *
from utils.utils import *
from functools import partial
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
partial_resize = partial(tf.image.resize,method=tf.image.ResizeMethod.BILINEAR,antialias=True)

class Model2eye():
    def __init__(self,config):
        self.config = config
        self.step = tf.Variable(0,dtype=tf.int64)
        self.build_model()
        log_dir = os.path.join(config.summary_dir)

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir,0o777)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss = tf.keras.losses.MeanSquaredError()
        self.train_location_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_location_accuracy')
        self.train_falling_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='falling_accuracy')

    def build_model(self):
        """model"""
        self.model = build_model(batch=self.config.batch_size,pretrained_weights=False)
        # self.model = build_model(include_top=False,batch=self.config.batch_size,height=400, width=400, color=True, filters=64)
        learning_rate = 0.00001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model.summary()

    def save(self,epoch):
        # self.model.summary()
        # self.mode.inputs[0].shape.dims[0]._value = 6
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        # self.model.summary()

        # self.model.save(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(epoch)))

    def restore(self, N=None):
        self.generator = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"generator_scale.h5"),custom_objects={'InstanceNorm': InstanceNorm})


    # @tf.function
    def train_one_step(self,data):
        image=data['img']
        targets=dict()
        targets['seg']=np.expand_dims(data['seg'][...,0],axis=3)
        # targets['falling'] = data['label_falling']
        image=tf.cast(image,tf.float32)
        image=image/255.0
        # image_dataset = oasd 1~100




        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)



            # Get the error/loss using the Loss_object previously defined
            loss_location = self.loss(targets['seg'],predictions[0])
            # loss_falling = self.loss(targets['falling'], predictions[1])
            # loss = 0*loss_location + 20*loss_falling
            loss = loss_location

        # compute the gradient with respect to the loss
        gradients = tape.gradient(loss,self.model.trainable_variables)
        # Change the weights of the model
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        # the metrics are accumulate over time. You don't need to average it yourself.
        self.train_loss(loss)
        # self.train_location_acc(targets['location'],predictions[0])
        # self.train_falling_acc(targets['falling'], predictions[1])

        return_dicts = {'loss':self.train_loss}
        # return_dicts.update({'location_acc':self.train_location_acc})
        # return_dicts.update({'falling_acc':self.train_falling_acc})
        return return_dicts

    def train_step(self,data,summary_name='train',log_interval=0):
        """training"""
        result_logs_dict = self.train_one_step(data)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            for key, value in result_logs_dict.items():
                value = value.result().numpy()
                tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        # log = "loss:{}, location_accuracy:{}, falling_accuracy:{}".format(result_logs_dict["loss"].result().numpy(),result_logs_dict['location_acc'].result().numpy(),result_logs_dict['falling_acc'].result().numpy())
        log = "loss:{}".format(result_logs_dict["loss"].result().numpy())
        return log

        # @tf.function
    def validation_one_step(self, data):
        image = data['img']
        targets = dict()
        targets['location'] = data['label_location']
        targets['falling'] = data['label_falling']
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image, training=False)

        # compute the gradient with respect to the loss
        # Change the weights of the model
        # the metrics are accumulate over time. You don't need to average it yourself.
        self.train_location_acc(targets['location'], predictions[0])
        self.train_falling_acc(targets['falling'], predictions[1])

        return_dicts = {'loss': self.train_loss}
        return_dicts.update({'location_acc': self.train_location_acc})
        return_dicts.update({'falling_acc': self.train_falling_acc})
        return return_dicts

    def validation_step(self,data,summary_name='validation',log_interval=0):
        """training"""
        result_logs_dict = self.validation_one_step(data)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            for key, value in result_logs_dict.items():
                value = value.result().numpy()
                tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        log = "validation location_accuracy:{}, validation falling_accuracy:{}".format(result_logs_dict['location_acc'].result().numpy(),result_logs_dict['falling_acc'].result().numpy())
        return log
















