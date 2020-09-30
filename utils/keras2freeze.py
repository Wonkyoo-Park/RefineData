"""
this code will be updated. not yet work
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from models import *

# Create, compile and train model...
model = unet()



model.load_weights("unet_membrane.hdf5")


# frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
saver = tf.train.Saver()
saver.save(K.get_session(), 'log/', model.global_step_tensor)


tf.train.write_graph(frozen_graph, 'log/',"my_model.pb", as_text=False)
tf.train.write_graph(frozen_graph, 'log/',"my_model.pbtxt", as_text=True)