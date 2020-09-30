from tensorflow.compat.v1.keras import backend as K
import argparse
import os
import tensorflow as tf
import keras2onnx
import os

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]

        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        # f = tf.gfile.FastGFile(os.path.join('log', 'fronzen_model.pb'), "wb")
        # f.write(frozen_graph.SerializeToString())
        return frozen_graph


# src = "/root/optimization/ocrSecurity/ocr_onnx/outputs/detector/checkpoints/OCR_default/generator_scale_0.h5"
# dst = "/root/optimization/ocrSecurity/ocr_onnx/outputs/detector/checkpoints/OCR_default/generator_scale_0.pb"
src = "/root/optimization/ocrSecurity/ocr_onnx/outputs/detector/checkpoints/OCR_default/final_model_20200313_1.h5"
dst = "/root/optimization/ocrSecurity/ocr_onnx/outputs/detector/checkpoints/OCR_default/final_model_20200313_1.pb"


restored_model = tf.keras.models.load_model(src, compile=True)

onnx_model = keras2onnx.convert_keras(restored_model, restored_model.name)
keras2onnx.save_model(onnx_model, '/root/optimization/ocrSecurity/ocr_onnx/outputs/detector/checkpoints/OCR_default/final_model_20200313_1.onnx')


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in restored_model.outputs],
                             clear_devices=True)

tf.train.write_graph(frozen_graph, "/tmp", dst, as_text=False)




print("finished")