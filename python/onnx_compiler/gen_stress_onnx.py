#!/usr/bin/env python3
"""
Generate a deep stress-test ONNX model with 50+ operations.
  [1, 32] -> (Gemm(32->32) -> Relu) x 10 -> ReduceSum -> Exp -> Sqrt -> [1]
Total operations: 10 Gemm + 10 Relu + ReduceSum + Exp + Abs + Sqrt = 24 nodes
  ... actually we need 50+, so we do 20 Gemm+Relu pairs + tail = 43+ nodes.
  Let's do 24 Gemm+Relu pairs (48 nodes) + ReduceSum + Exp + Abs + Sqrt = 52 nodes.
Saves to models/stress_test.onnx
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

    dim = 32
    num_gemm_relu_pairs = 24  # 48 nodes from Gemm+Relu pairs

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, dim])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

    nodes = []
    initializers = []
    prev_output = 'input'

    # 24 x (Gemm -> Relu) = 48 nodes
    for i in range(num_gemm_relu_pairs):
        # Gemm weights: [dim, dim] (transB=1, so shape is [out, in])
        W = np.random.randn(dim, dim).astype(np.float32) * 0.1
        b = np.random.randn(dim).astype(np.float32) * 0.01

        w_name = f'W_{i}'
        b_name = f'b_{i}'
        gemm_out = f'gemm_{i}_out'
        relu_out = f'relu_{i}_out'

        W_init = numpy_helper.from_array(W, name=w_name)
        b_init = numpy_helper.from_array(b, name=b_name)
        initializers.extend([W_init, b_init])

        gemm_node = helper.make_node(
            'Gemm',
            [prev_output, w_name, b_name],
            [gemm_out],
            alpha=1.0, beta=1.0, transB=1
        )
        relu_node = helper.make_node('Relu', [gemm_out], [relu_out])

        nodes.extend([gemm_node, relu_node])
        prev_output = relu_out

    # ReduceSum along axis=1: [1, 32] -> [1]  (node 49)
    reduce_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='reduce_axes'
    )
    initializers.append(reduce_axes)

    reduce_node = helper.make_node(
        'ReduceSum',
        inputs=[prev_output, 'reduce_axes'],
        outputs=['reduce_out'],
        keepdims=0
    )
    nodes.append(reduce_node)
    prev_output = 'reduce_out'

    # Reshape [1] -> [1, 1] so we can do element-wise ops cleanly
    reshape_shape = numpy_helper.from_array(
        np.array([1, 1], dtype=np.int64), name='reshape_shape'
    )
    initializers.append(reshape_shape)

    reshape_node = helper.make_node(
        'Reshape', [prev_output, 'reshape_shape'], ['reshape_out']
    )
    nodes.append(reshape_node)
    prev_output = 'reshape_out'

    # Relu to ensure non-negative before Exp (node 51)
    relu_tail = helper.make_node('Relu', [prev_output], ['relu_tail_out'])
    nodes.append(relu_tail)
    prev_output = 'relu_tail_out'

    # Exp (node 52)
    exp_node = helper.make_node('Exp', [prev_output], ['exp_out'])
    nodes.append(exp_node)
    prev_output = 'exp_out'

    # Log (node 53) - Exp output is always >= 1, safe for Log
    log_node = helper.make_node('Log', [prev_output], ['log_out'])
    nodes.append(log_node)
    prev_output = 'log_out'

    # Relu before Sqrt to ensure non-negative (node 54)
    relu_tail2 = helper.make_node('Relu', [prev_output], ['relu_tail2_out'])
    nodes.append(relu_tail2)
    prev_output = 'relu_tail2_out'

    # Sqrt (node 55)
    sqrt_node = helper.make_node('Sqrt', [prev_output], ['sqrt_out'])
    nodes.append(sqrt_node)
    prev_output = 'sqrt_out'

    # Final reshape to [1]
    final_shape = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='final_shape'
    )
    initializers.append(final_shape)

    final_reshape = helper.make_node(
        'Reshape', [prev_output, 'final_shape'], ['output']
    )
    nodes.append(final_reshape)

    total_nodes = len(nodes)

    graph = helper.make_graph(
        nodes,
        'stress_test',
        [X], [Y],
        initializer=initializers
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'stress_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Stress model to {out_path}")
    print(f"  Input:       [1, {dim}]")
    print(f"  Gemm+Relu:   {num_gemm_relu_pairs} pairs ({dim}->{dim})")
    print(f"  Tail ops:    ReduceSum, Reshape, Relu, Exp, Log, Relu, Sqrt, Reshape")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Output:      [1]")


if __name__ == '__main__':
    main()
