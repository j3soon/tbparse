import numpy as np
from tensorboard.compat.tensorflow_stub import dtypes


# flake8: noqa
# pylint: skip-file
# Ref: https://github.com/tensorflow/tensorflow/blob/ad6d8cc177d0c868982e39e0823d0efbfb95f04c/tensorflow/python/framework/tensor_util.py#L633
def make_ndarray(tensor):
    """Create a numpy ndarray from a tensor.

    Create a numpy ndarray with the same shape and data as the tensor.

    For example:

    ```python
    # Tensor a has shape (2,3)
    a = tf.constant([[1,2,3],[4,5,6]])
    proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor
    tf.make_ndarray(proto_tensor) # output: array([[1, 2, 3],
    #                                              [4, 5, 6]], dtype=int32)
    # output has shape (2,3)
    ```

    Args:
        tensor: A TensorProto.

    Returns:
        A numpy array with the tensor contents.

    Raises:
        TypeError: if tensor has unsupported type.

    """
    shape = [d.size for d in tensor.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    tensor_dtype = dtypes.as_dtype(tensor.dtype)
    dtype = tensor_dtype.as_numpy_dtype

    if tensor.tensor_content:
        return (np.frombuffer(tensor.tensor_content,
                              dtype=dtype).copy().reshape(shape))

    if tensor_dtype == dtypes.string:
        # np.pad throws on these arrays of type np.object_.
        values = list(tensor.string_val)
        padding = num_elements - len(values)
        if padding > 0:
            last = values[-1] if values else ""
            values.extend([last] * padding)
        return np.array(values, dtype=dtype).reshape(shape)

    if tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
        # the half_val field of the TensorProto stores the binary representation
        # of the fp16: we need to reinterpret this as a proper float16
        values = np.fromiter(tensor.half_val, dtype=np.uint16)
        values.dtype = dtype
    # Note:
    # Does not support python3.7 since TensorBoard don't have the
    # following types before 2.12.0, which requires at least python3.8.
    # (https://github.com/tensorflow/tensorboard/blob/master/RELEASE.md#release-2120)
    # Python 3.7 EOL: 2023-06-27. (https://devguide.python.org/versions/#versions)
    # TODO: The following is a temporary fix for float8_e5m2 and float8_e4m3fn
    # Ref: https://github.com/tensorflow/tensorboard/issues/6899
    elif tensor_dtype in [
        dtypes.DType(dtypes.types_pb2.DT_FLOAT8_E5M2),
        dtypes.DType(dtypes.types_pb2.DT_FLOAT8_E4M3FN),
    ]:
        values = np.fromiter(tensor.float8_val, dtype=np.uint8)
        values.dtype = dtype
    elif tensor_dtype == dtypes.float32:
        values = np.fromiter(tensor.float_val, dtype=dtype)
    elif tensor_dtype == dtypes.float64:
        values = np.fromiter(tensor.double_val, dtype=dtype)
    elif tensor_dtype in [
        dtypes.int32,
        dtypes.uint8,
        dtypes.uint16,
        dtypes.int16,
        dtypes.int8,
        dtypes.qint32,
        dtypes.quint8,
        dtypes.qint8,
        dtypes.qint16,
        dtypes.quint16,
        dtypes.int4,
        dtypes.uint4,
    ]:
        values = np.fromiter(tensor.int_val, dtype=dtype)
    elif tensor_dtype == dtypes.int64:
        values = np.fromiter(tensor.int64_val, dtype=dtype)
    elif tensor_dtype == dtypes.uint32:
        values = np.fromiter(tensor.uint32_val, dtype=dtype)
    elif tensor_dtype == dtypes.uint64:
        values = np.fromiter(tensor.uint64_val, dtype=dtype)
    elif tensor_dtype == dtypes.complex64:
        it = iter(tensor.scomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif tensor_dtype == dtypes.complex128:
        it = iter(tensor.dcomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif tensor_dtype == dtypes.bool:
        values = np.fromiter(tensor.bool_val, dtype=dtype)
    else:
        raise TypeError(f"Unsupported tensor type: {tensor.dtype}. See "
                        "https://www.tensorflow.org/api_docs/python/tf/dtypes "
                        "for supported TF dtypes.")

    if values.size == 0:
        return np.zeros(shape, dtype)

    if values.size != num_elements:
        values = np.pad(values, (0, num_elements - values.size), "edge")

    return values.reshape(shape)
