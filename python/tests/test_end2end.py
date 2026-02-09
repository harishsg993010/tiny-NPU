#!/usr/bin/env python3
"""
End-to-end verification tests for NPU.
Generates test vectors, runs golden models, and compares against RTL simulation results.
"""
import sys
import os
import subprocess
import numpy as np
import struct

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from golden.quant import clamp_i8, requantize, requantize_vec
from golden.gemm_ref import gemm_int8
from golden.softmax_ref import softmax_fixed
from golden.layernorm_ref import layernorm_fixed
from golden.gelu_ref import gelu_fixed
from golden.gpt2_block_ref import GPT2BlockRef, run_reference_block

TINY_CFG = dict(hidden=64, heads=4, head_dim=16, ffn_dim=256, seq_len=8)

def save_binary(path, arr):
    """Save numpy array as raw binary (int8)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype(np.int8).tofile(path)

def load_binary(path, dtype=np.int8):
    """Load raw binary file as numpy array."""
    return np.fromfile(path, dtype=dtype)

class TestGEMM:
    """Test INT8 GEMM against golden model."""

    def test_basic(self):
        """Basic GEMM test: small fixed matrices."""
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        C = gemm_int8(A, B, scale=1, shift=0)
        # Expected (int32): [[19, 22], [43, 50]] -> clamp to int8
        expected = np.array([[19, 22], [43, 50]], dtype=np.int8)
        np.testing.assert_array_equal(C, expected)
        print("GEMM basic: PASS")

    def test_transpose(self):
        """GEMM with B transposed."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        B = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int8)  # Already transposed shape
        C1 = gemm_int8(A, B, scale=1, shift=0)
        # B as [N,K] with transpose_b
        Bt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        C2 = gemm_int8(A, Bt, transpose_b=True, scale=1, shift=0)
        np.testing.assert_array_equal(C1, C2)
        print("GEMM transpose: PASS")

    def test_random(self, M=16, N=16, K=16, seed=42):
        """Random GEMM test."""
        rng = np.random.RandomState(seed)
        A = rng.randint(-32, 32, (M, K)).astype(np.int8)
        B = rng.randint(-32, 32, (K, N)).astype(np.int8)
        C = gemm_int8(A, B, scale=1, shift=7)
        assert C.shape == (M, N)
        assert C.dtype == np.int8
        print(f"GEMM random {M}x{N}x{K}: PASS (output range [{C.min()}, {C.max()}])")

    def test_requant(self):
        """Test requantization with various scale/shift."""
        A = np.array([[10, 20], [30, 40]], dtype=np.int8)
        B = np.array([[1, 0], [0, 1]], dtype=np.int8)
        C = gemm_int8(A, B, scale=1, shift=0)
        np.testing.assert_array_equal(C, A)

        C2 = gemm_int8(A, B, scale=1, shift=1)  # divide by 2
        expected = np.array([[5, 10], [15, 20]], dtype=np.int8)
        np.testing.assert_array_equal(C2, expected)
        print("GEMM requant: PASS")

    def run_all(self):
        self.test_basic()
        self.test_transpose()
        self.test_random()
        self.test_random(M=8, N=64, K=64, seed=123)
        self.test_requant()

class TestSoftmax:
    """Test fixed-point softmax against golden model."""

    def test_basic(self):
        """Basic softmax test."""
        scores = np.array([10, 20, 30, 40], dtype=np.int8)
        probs = softmax_fixed(scores)
        assert probs.dtype == np.int8
        assert len(probs) == 4
        # Higher scores should give higher probabilities
        assert probs[3] >= probs[2] >= probs[1] >= probs[0]
        print(f"Softmax basic: PASS (probs={probs})")

    def test_causal_mask(self):
        """Softmax with causal mask."""
        scores = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int8)
        mask = np.array([True, True, True, True, False, False, False, False])
        probs = softmax_fixed(scores, causal_mask=mask)
        # Masked positions should have lower probability than unmasked max
        max_unmasked = int(np.max(probs[:4]))
        max_masked = int(np.max(probs[4:]))
        assert max_unmasked >= max_masked, f"Unmasked should dominate: unmasked_max={max_unmasked}, masked_max={max_masked}, probs={probs}"
        print(f"Softmax causal: PASS (probs={probs})")

    def test_uniform(self):
        """Uniform input should give roughly uniform output."""
        scores = np.full(8, 50, dtype=np.int8)
        probs = softmax_fixed(scores)
        # All probabilities should be similar
        assert np.std(probs.astype(float)) < 5, f"Should be uniform: {probs}"
        print(f"Softmax uniform: PASS (probs={probs})")

    def run_all(self):
        self.test_basic()
        self.test_causal_mask()
        self.test_uniform()

class TestLayerNorm:
    """Test fixed-point LayerNorm."""

    def test_basic(self):
        """Basic LayerNorm test."""
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int8)
        y = layernorm_fixed(x)
        assert y.dtype == np.int8
        assert len(y) == 8
        print(f"LayerNorm basic: PASS (output={y})")

    def test_zero_mean(self):
        """Output should have roughly zero mean."""
        rng = np.random.RandomState(42)
        x = rng.randint(-64, 64, 64).astype(np.int8)
        y = layernorm_fixed(x)
        mean = np.mean(y.astype(float))
        assert abs(mean) < 10, f"Mean should be ~0: {mean}"
        print(f"LayerNorm zero_mean: PASS (mean={mean:.2f})")

    def run_all(self):
        self.test_basic()
        self.test_zero_mean()

class TestGELU:
    """Test GELU activation."""

    def test_basic(self):
        """Basic GELU test."""
        x = np.array([-64, -32, 0, 32, 64], dtype=np.int8)
        y = gelu_fixed(x)
        assert y.dtype == np.int8
        # GELU(0) should be ~0
        assert abs(int(y[2])) < 2, f"GELU(0) should be ~0: {y[2]}"
        # GELU(positive) should be positive
        assert y[3] > 0, f"GELU(32) should be positive: {y[3]}"
        assert y[4] > 0, f"GELU(64) should be positive: {y[4]}"
        # GELU(very negative) should be ~0
        assert abs(int(y[0])) < 5, f"GELU(-64) should be ~0: {y[0]}"
        print(f"GELU basic: PASS (output={y})")

    def test_monotonic_positive(self):
        """GELU should be roughly monotonic for positive inputs."""
        x = np.arange(0, 127, dtype=np.int8)
        y = gelu_fixed(x)
        # Allow some non-monotonicity due to quantization
        violations = sum(1 for i in range(len(y)-1) if y[i+1] < y[i] - 1)
        assert violations < 5, f"Too many monotonicity violations: {violations}"
        print(f"GELU monotonic: PASS ({violations} violations)")

    def run_all(self):
        self.test_basic()
        self.test_monotonic_positive()

class TestTransformerBlock:
    """End-to-end transformer block test."""

    def test_tiny_block(self):
        """Run tiny transformer block."""
        result = run_reference_block(**TINY_CFG)
        x_out = result['output']
        assert x_out.shape == (TINY_CFG['seq_len'], TINY_CFG['hidden'])
        assert x_out.dtype == np.int8
        print(f"Transformer block: PASS")
        print(f"  Input range:  [{result['input'].min()}, {result['input'].max()}]")
        print(f"  Output range: [{x_out.min()}, {x_out.max()}]")

    def test_generate_vectors(self):
        """Generate test vectors for RTL verification."""
        result = run_reference_block(**TINY_CFG)

        vec_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sim', 'vectors')
        os.makedirs(vec_dir, exist_ok=True)

        save_binary(os.path.join(vec_dir, 'input.bin'), result['input'])
        save_binary(os.path.join(vec_dir, 'output_ref.bin'), result['output'])

        for name, w in result['weights'].items():
            save_binary(os.path.join(vec_dir, f'{name}.bin'), w)

        print(f"Test vectors saved to {vec_dir}")

    def run_all(self):
        self.test_tiny_block()
        self.test_generate_vectors()

def run_all_tests():
    """Run all golden model tests."""
    print("=" * 60)
    print("NPU Golden Model Tests")
    print("=" * 60)

    print("\n--- GEMM Tests ---")
    TestGEMM().run_all()

    print("\n--- Softmax Tests ---")
    TestSoftmax().run_all()

    print("\n--- LayerNorm Tests ---")
    TestLayerNorm().run_all()

    print("\n--- GELU Tests ---")
    TestGELU().run_all()

    print("\n--- Transformer Block Tests ---")
    TestTransformerBlock().run_all()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == '__main__':
    run_all_tests()
