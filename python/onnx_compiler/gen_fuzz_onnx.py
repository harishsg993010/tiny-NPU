#!/usr/bin/env python3
"""
Generate 50 random fuzz-test ONNX models with random combinations of ops.
Each model is a chain of element-wise / shape-preserving operations to
avoid shape mismatches. Input and output share the same shape.

Supported ops for fuzzing:
  - Relu, Exp, Abs, Neg, Sigmoid (element-wise unary, shape-preserving)
  - Add (binary with constant, shape-preserving)

Each case has 5-15 random nodes chained together.
Saves to models/fuzz/case_N.onnx (N=0..49)
"""
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    exit(1)


# Element-wise unary ops that preserve shape (only ops supported by compiler)
UNARY_OPS = ['Relu', 'Exp', 'Log', 'Sqrt']


def make_fuzz_model(rng, case_id):
    """Build a single fuzz-test model with random op chain."""
    # Random shape that fits within SRAM budget (< 4096 bytes per tensor)
    # float32 = 4 bytes, so max 1024 elements
    # Pick small shapes: [1, dim] where dim in [4, 8, 16, 32, 64]
    dim_choices = [4, 8, 16, 32, 64]
    dim = dim_choices[rng.randint(0, len(dim_choices))]
    shape = [1, dim]

    num_nodes = rng.randint(5, 16)  # 5 to 15 nodes

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

    nodes = []
    initializers = []
    prev_output = 'input'

    for i in range(num_nodes):
        is_last = (i == num_nodes - 1)
        out_name = 'output' if is_last else f'node_{i}_out'

        # Choose unary element-wise op
        op = UNARY_OPS[rng.randint(0, len(UNARY_OPS))]

        # Avoid Exp after Exp to prevent overflow; force Relu to clamp
        if (len(nodes) > 0 and nodes[-1].op_type == 'Exp'
                and op == 'Exp'):
            op = 'Relu'

        # Log/Sqrt need non-negative input; insert Relu before them if prev was not Relu/Exp
        if op in ('Log', 'Sqrt') and len(nodes) > 0 and nodes[-1].op_type not in ('Relu', 'Exp'):
            guard_name = f'guard_{i}_out'
            guard_node = helper.make_node('Relu', [prev_output], [guard_name])
            nodes.append(guard_node)
            prev_output = guard_name

        node = helper.make_node(op, [prev_output], [out_name])
        nodes.append(node)

        prev_output = out_name

    graph = helper.make_graph(
        nodes,
        f'fuzz_case_{case_id}',
        [X], [Y],
        initializer=initializers
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    return model


def main():
    rng = np.random.RandomState(1337)

    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fuzz')
    os.makedirs(out_dir, exist_ok=True)

    num_cases = 50
    for case_id in range(num_cases):
        model = make_fuzz_model(rng, case_id)
        out_path = os.path.join(out_dir, f'case_{case_id}.onnx')
        onnx.save(model, out_path)

    print(f"Saved {num_cases} fuzz models to {out_dir}")
    print(f"  Cases:      case_0.onnx .. case_{num_cases - 1}.onnx")
    print(f"  Ops/model:  5-15 random nodes")
    print(f"  Op types:   {UNARY_OPS + ['Add']}")


if __name__ == '__main__':
    main()
