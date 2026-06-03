import os
import numpy as np
from PIL import Image

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "facenet.tflite")
_INPUT_SIZE = 160
_OUTPUT_DIM = 128

_interpreter = None


def _get_interpreter():
    global _interpreter
    if _interpreter is None:
        import tensorflow as tf
        _interpreter = tf.lite.Interpreter(model_path=_MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


def embed(face_img) -> np.ndarray:
    """
    face_img: PIL Image or numpy array (any size, RGB).
    Returns 128-dim L2-normalised float32 embedding — same model/normalisation as Android.
    """
    if isinstance(face_img, np.ndarray):
        face_img = Image.fromarray(face_img)
    face_img = face_img.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE))
    arr = (np.array(face_img, dtype=np.float32) - 127.5) / 128.0
    arr = arr[np.newaxis]  # [1, 160, 160, 3]

    interp = _get_interpreter()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    vec = interp.get_tensor(out_idx)[0].copy()

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
