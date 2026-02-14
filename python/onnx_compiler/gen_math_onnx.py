#!/usr/bin/env python3
"""
Generate an ONNX model testing Exp, Log, and Sqrt math operations.
  [1, 16] -> Relu -> Exp -> Log -> Sqrt -> [1, 16]
Relu ensures non-negative input, Exp(>=0) >= 1, Log(>=1) >= 0, Sqrt(>=0) valid.
Saves to models/math_test.onnx
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

    # Input: [1, 16]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16])

    # Chain: input -> Relu -> Exp -> Log -> Sqrt -> output
    # Relu: max(x, 0) -> non-negative
    # Exp(non-negative) -> >= 1  (reasonable for int8)
    # Log(>= 1) -> >= 0
    # Sqrt(>= 0) -> >= 0

    relu_node = helper.make_node('Relu', ['input'], ['relu_out'])
    exp_node = helper.make_node('Exp', ['relu_out'], ['exp_out'])
    log_node = helper.make_node('Log', ['exp_out'], ['log_out'])
    sqrt_node = helper.make_node('Sqrt', ['log_out'], ['output'])

    graph = helper.make_graph(
        [relu_node, exp_node, log_node, sqrt_node],
        'math_test',
        [X], [Y],
        initializer=[]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'math_test.onnx')
    onnx.save(model, out_path)
    print(f"Saved Math model to {out_path}")
    print(f"  Input:  [1, 16]")
    print(f"  Chain:  Relu -> Exp -> Log -> Sqrt")
    print(f"  Output: [1, 16]")


if __name__ == '__main__':
    main()
