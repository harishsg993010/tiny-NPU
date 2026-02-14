#!/usr/bin/env python3
"""
Generate an ONNX model testing ReduceSum and ReduceMax operations.
  [1, 8, 4] -> ReduceSum(axis=2) -> [1, 8] -> ReduceMax(axis=1) -> [1]
Saves to models/reduce_test.onnx
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

    # Input: [1, 8, 4]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

    # ReduceSum axes as a constant tensor (opset 13 uses input for axes)
    reduce_sum_axes = numpy_helper.from_array(
        np.array([2], dtype=np.int64), name='reduce_sum_axes'
    )

    # ReduceMax axes as a constant tensor (opset 18 style)
    reduce_max_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='reduce_max_axes'
    )

    # ReduceSum along axis=2, keepdims=0: [1, 8, 4] -> [1, 8]
    reduce_sum_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'reduce_sum_axes'],
        outputs=['reduce_sum_out'],
        keepdims=0
    )

    # ReduceMax along axis=1, keepdims=0: [1, 8] -> [1]
    reduce_max_node = helper.make_node(
        'ReduceMax',
        inputs=['reduce_sum_out', 'reduce_max_axes'],
        outputs=['output'],
        keepdims=0
    )

    graph = helper.make_graph(
        [reduce_sum_node, reduce_max_node],
        'reduce_test',
        [X], [Y],
        initializer=[reduce_sum_axes, reduce_max_axes]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'reduce_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Reduce model to {out_path}")
    print(f"  Input:      [1, 8, 4]")
    print(f"  ReduceSum:  axis=2, keepdims=0 -> [1, 8]")
    print(f"  ReduceMax:  axis=1, keepdims=0 -> [1]")


if __name__ == '__main__':
    main()
