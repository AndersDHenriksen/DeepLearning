from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model


def finalize_tf2_for_ocv(model_path):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    tf_model_path = str(Path(model_path).parent / 'tf_model')
    cv_model_path = str(Path(model_path).parent / 'frozen_graph.pb')
    model = load_model(model_path)
    # Convert tf format
    model.save(tf_model_path, save_format='tf')
    loaded = tf.saved_model.load(tf_model_path)
    # Convert to function then graph
    infer = loaded.signatures['serving_default']
    input_kwargs = {model.input.name[:-2]: tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)}
    f = tf.function(infer).get_concrete_function(**input_kwargs)
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()
    # Export frozen graph
    with tf.io.gfile.GFile(cv_model_path, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == "__main__":
    finalize_tf2_for_ocv(r"\best\model\path\model.hdf5")