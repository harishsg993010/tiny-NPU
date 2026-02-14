#!/usr/bin/env python3
"""
Generate an ONNX model testing Slice and Concat operations.
  [1, 8] -> Slice[0:4] -> [1, 4]
  [1, 8] -> Slice[4:8] -> [1, 4]
  Concat([1,4], [1,4], axis=1) -> [1, 8]
Saves to models/slice_concat_test.onnx
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

    # Input: [1, 8]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8])

    # Slice 1 parameters: input[0:4] along axis 1
    slice1_starts = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name='slice1_starts'
    )
    slice1_ends = numpy_helper.from_array(
        np.array([4], dtype=np.int64), name='slice1_ends'
    )
    slice1_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='slice1_axes'
    )

    # Slice 2 parameters: input[4:8] along axis 1
    slice2_starts = numpy_helper.from_array(
        np.array([4], dtype=np.int64), name='slice2_starts'
    )
    slice2_ends = numpy_helper.from_array(
        np.array([8], dtype=np.int64), name='slice2_ends'
    )
    slice2_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='slice2_axes'
    )

    # Slice 1: [1, 8] -> [1, 4] (first half)
    slice1_node = helper.make_node(
        'Slice',
        inputs=['input', 'slice1_starts', 'slice1_ends', 'slice1_axes'],
        outputs=['slice1_out']
    )

    # Slice 2: [1, 8] -> [1, 4] (second half)
    slice2_node = helper.make_node(
        'Slice',
        inputs=['input', 'slice2_starts', 'slice2_ends', 'slice2_axes'],
        outputs=['slice2_out']
    )

    # Concat: [1, 4] + [1, 4] -> [1, 8] along axis 1
    concat_node = helper.make_node(
        'Concat',
        inputs=['slice1_out', 'slice2_out'],
        outputs=['output'],
        axis=1
    )

    graph = helper.make_graph(
        [slice1_node, slice2_node, concat_node],
        'slice_concat_test',
        [X], [Y],
        initializer=[
            slice1_starts, slice1_ends, slice1_axes,
            slice2_starts, slice2_ends, slice2_axes
        ]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'slice_concat_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Slice+Concat model to {out_path}")
    print(f"  Input:   [1, 8]")
    print(f"  Slice1:  [0:4] -> [1, 4]")
    print(f"  Slice2:  [4:8] -> [1, 4]")
    print(f"  Concat:  axis=1 -> [1, 8]")


if __name__ == '__main__':
    main()
