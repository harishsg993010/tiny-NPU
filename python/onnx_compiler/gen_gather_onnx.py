#!/usr/bin/env python3
"""
Generate an ONNX model testing the Gather operation.
  Data: [4, 8], Indices: [2] = {1, 3}, axis=0
  Output: [2, 8] (selects rows 1 and 3)
Saves to models/gather_test.onnx
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

    # Input (data): [4, 8]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 8])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 8])

    # Indices: constant tensor [1, 3]
    indices = numpy_helper.from_array(
        np.array([1, 3], dtype=np.int64), name='indices'
    )

    # Gather along axis=0: selects rows 1 and 3
    gather_node = helper.make_node(
        'Gather',
        inputs=['input', 'indices'],
        outputs=['output'],
        axis=0
    )

    graph = helper.make_graph(
        [gather_node],
        'gather_test',
        [X], [Y],
        initializer=[indices]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gather_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Gather model to {out_path}")
    print(f"  Data input: [4, 8]")
    print(f"  Indices:    [1, 3]")
    print(f"  Gather:     axis=0 -> [2, 8]")


if __name__ == '__main__':
    main()
