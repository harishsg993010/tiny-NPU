#!/usr/bin/env python3
"""
Generate an ONNX model testing BatchNormalization + AveragePool.
  [1, 4, 6, 6] -> BatchNorm -> ReLU -> AvgPool(k=2,s=2) -> [1, 4, 3, 3]
BatchNorm lowers to EW_MUL + EW_ADD with pre-computed scale/offset.
AvgPool runs on dedicated avgpool2d_engine.
Saves to models/batchnorm_pool_test.onnx
"""
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    exit(1)


def main():
    np.random.seed(42)

    num_channels = 4

    # BatchNormalization parameters (per-channel)
    bn_scale = np.random.randn(num_channels).astype(np.float32) * 0.5 + 1.0
    bn_bias = np.random.randn(num_channels).astype(np.float32) * 0.1
    bn_mean = np.random.randn(num_channels).astype(np.float32) * 0.5
    bn_var = np.abs(np.random.randn(num_channels).astype(np.float32)) + 0.1  # must be positive

    H, W = 6, 6
    kh, kw, sh, sw = 2, 2, 2, 2
    out_h = (H - kh) // sh + 1  # 3
    out_w = (W - kw) // sw + 1  # 3

    # Input: [1, 4, 6, 6] (NCHW)
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, num_channels, H, W])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, num_channels, out_h, out_w])

    # Initializers for BatchNorm
    scale_init = numpy_helper.from_array(bn_scale, name='bn_scale')
    bias_init = numpy_helper.from_array(bn_bias, name='bn_bias')
    mean_init = numpy_helper.from_array(bn_mean, name='bn_mean')
    var_init = numpy_helper.from_array(bn_var, name='bn_var')

    # BatchNormalization: [1, 4, 6, 6] -> [1, 4, 6, 6]
    bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['input', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['bn_out'],
        epsilon=1e-5
    )

    # ReLU: [1, 4, 6, 6] -> [1, 4, 6, 6]
    relu_node = helper.make_node('Relu', ['bn_out'], ['relu_out'])

    # AveragePool: kernel 2x2, stride 2x2 -> [1, 4, 3, 3]
    pool_node = helper.make_node(
        'AveragePool',
        inputs=['relu_out'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )

    graph = helper.make_graph(
        [bn_node, relu_node, pool_node],
        'batchnorm_pool_test',
        [X], [Y],
        initializer=[scale_init, bias_init, mean_init, var_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'batchnorm_pool_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved BatchNorm+Pool model to {out_path}")
    print(f"  Input:     [1, {num_channels}, 6, 6]")
    print(f"  BatchNorm: epsilon=1e-5")
    print(f"  ReLU")
    print(f"  AvgPool:   kernel=2x2, stride=2x2 -> [1, {num_channels}, 3, 3]")


if __name__ == '__main__':
    main()
