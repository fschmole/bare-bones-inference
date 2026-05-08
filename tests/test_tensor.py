"""Tests for the compute.Tensor class — compared against numpy as golden reference."""

import numpy as np
import pytest
from compute import Tensor, set_trace, set_trace_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(t: Tensor) -> np.ndarray:
    """Convert our Tensor to a numpy array."""
    return np.array(t.to_vec(), dtype=np.float32).reshape(t.shape)


def from_np(a: np.ndarray) -> Tensor:
    """Convert a numpy array to our Tensor."""
    return Tensor(a.astype(np.float32).flatten().tolist(), list(a.shape))


def assert_close(ours: Tensor, expected: np.ndarray, atol: float = 1e-5):
    """Assert our Tensor matches a numpy array within tolerance."""
    got = to_np(ours)
    np.testing.assert_allclose(got, expected, atol=atol, rtol=1e-5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_data(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        assert t.shape == [2, 3]
        assert t.numel() == 6
        assert t.to_vec() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_zeros(self):
        t = Tensor.zeros([3, 4])
        assert t.shape == [3, 4]
        assert all(x == 0.0 for x in t.to_vec())

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            Tensor([1.0, 2.0], [3])  # 2 elements but shape says 3

    def test_repr(self):
        t = Tensor([1.0, 2.0], [2])
        r = repr(t)
        assert "Tensor" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# Shape operations
# ---------------------------------------------------------------------------

class TestReshape:
    def test_basic(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = from_np(a)
        result = t.reshape([4, 3])
        assert result.shape == [4, 3]
        assert_close(result, a.reshape(4, 3))

    def test_flatten(self):
        t = Tensor(list(range(24)), [2, 3, 4])
        r = t.reshape([24])
        assert r.shape == [24]

    def test_mismatch_raises(self):
        t = Tensor.zeros([2, 3])
        with pytest.raises(ValueError):
            t.reshape([2, 4])


class TestTranspose:
    def test_2d(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = from_np(a)
        result = t.transpose(0, 1)
        assert_close(result, a.T)

    def test_3d_swap_01(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = from_np(a)
        result = t.transpose(0, 1)
        assert_close(result, a.transpose(1, 0, 2))

    def test_3d_swap_12(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = from_np(a)
        result = t.transpose(1, 2)
        assert_close(result, a.transpose(0, 2, 1))

    def test_same_dim_is_noop(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        result = t.transpose(0, 0)
        assert result.to_vec() == t.to_vec()


class TestSlice:
    def test_columns(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = from_np(a)
        result = t.slice(1, 1, 3)
        assert_close(result, a[:, 1:3])

    def test_rows(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = from_np(a)
        result = t.slice(0, 0, 2)
        assert_close(result, a[0:2, :])

    def test_3d_batch(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = from_np(a)
        result = t.slice(0, 1, 2)
        assert_close(result, a[1:2, :, :])

    def test_3d_last_dim(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = from_np(a)
        result = t.slice(2, 0, 2)
        assert_close(result, a[:, :, 0:2])


class TestSliceRow:
    def test_basic(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = from_np(a)
        assert_close(t.slice_row(0), a[0])
        assert_close(t.slice_row(1), a[1])
        assert_close(t.slice_row(2), a[2])

    def test_negative_index(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = from_np(a)
        assert_close(t.slice_row(-1), a[-1])

    def test_3d(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = from_np(a)
        result = t.slice_row(0)
        assert result.shape == [3, 4]
        assert_close(result, a[0])


# ---------------------------------------------------------------------------
# Math operations
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_2d(self):
        a = np.random.randn(4, 5).astype(np.float32)
        b = np.random.randn(5, 3).astype(np.float32)
        result = from_np(a).matmul(from_np(b))
        assert_close(result, a @ b, atol=1e-4)

    def test_2d_square(self):
        a = np.random.randn(8, 8).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        result = from_np(a).matmul(from_np(b))
        assert_close(result, a @ b, atol=1e-4)

    def test_3d_batched(self):
        a = np.random.randn(4, 5, 6).astype(np.float32)
        b = np.random.randn(4, 6, 3).astype(np.float32)
        result = from_np(a).matmul(from_np(b))
        assert_close(result, a @ b, atol=1e-4)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            from_np(np.zeros((2, 3), dtype=np.float32)).matmul(
                from_np(np.zeros((4, 5), dtype=np.float32))
            )

    def test_gpt2_sizes(self):
        """Test with dimensions matching GPT-2: hidden=768, ffn=3072."""
        a = np.random.randn(10, 768).astype(np.float32)
        b = np.random.randn(768, 3072).astype(np.float32)
        result = from_np(a).matmul(from_np(b))
        assert result.shape == [10, 3072]
        assert_close(result, a @ b, atol=1e-3)


class TestAdd:
    def test_same_shape(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        assert_close(from_np(a).add(from_np(b)), a + b)

    def test_broadcast_bias(self):
        """[M, N] + [N] — adding a bias vector to each row."""
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        assert_close(from_np(a).add(from_np(b)), a + b)

    def test_broadcast_3d_mask(self):
        """[B, M, N] + [M, N] — adding causal mask to batched attention scores."""
        a = np.random.randn(12, 5, 5).astype(np.float32)
        b = np.random.randn(5, 5).astype(np.float32)
        assert_close(from_np(a).add(from_np(b)), a + b)

    def test_broadcast_3d_bias(self):
        """[B, M, N] + [N] — adding bias to each row of each batch."""
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        assert_close(from_np(a).add(from_np(b)), a + b)


class TestMul:
    def test_elementwise(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        assert_close(from_np(a).mul(from_np(b)), a * b)


class TestMulScalar:
    def test_basic(self):
        a = np.random.randn(3, 4).astype(np.float32)
        assert_close(from_np(a).mul_scalar(0.5), a * 0.5)

    def test_scale_factor(self):
        """1/sqrt(d_k) scaling used in attention."""
        a = np.random.randn(12, 5, 5).astype(np.float32)
        scale = 1.0 / np.sqrt(64.0)
        assert_close(from_np(a).mul_scalar(scale), a * scale)


class TestGelu:
    def test_matches_numpy(self):
        """Compare against the exact GELU formula (not the approximation in some libs)."""
        a = np.random.randn(5, 10).astype(np.float32)
        # Same formula as in our Rust code
        c = np.sqrt(2.0 / np.pi)
        expected = 0.5 * a * (1.0 + np.tanh(c * (a + 0.044715 * a**3)))
        assert_close(from_np(a).gelu(), expected)

    def test_zero(self):
        t = Tensor([0.0], [1])
        assert abs(t.gelu().to_vec()[0]) < 1e-7

    def test_large_positive(self):
        t = Tensor([10.0], [1])
        result = t.gelu().to_vec()[0]
        assert abs(result - 10.0) < 0.01  # gelu(x) ≈ x for large positive x


class TestSoftmax:
    def test_1d(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = np.exp(a - a.max()) / np.exp(a - a.max()).sum()
        assert_close(from_np(a).softmax(0), expected)

    def test_2d_last_axis(self):
        a = np.random.randn(3, 4).astype(np.float32)
        # numpy softmax along last axis
        shifted = a - a.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        assert_close(from_np(a).softmax(-1), expected)

    def test_3d_last_axis(self):
        """Batched softmax for attention scores: [n_heads, seq, seq]."""
        a = np.random.randn(12, 5, 5).astype(np.float32)
        shifted = a - a.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        assert_close(from_np(a).softmax(-1), expected)

    def test_sums_to_one(self):
        a = np.random.randn(4, 8).astype(np.float32)
        result = to_np(from_np(a).softmax(-1))
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6)


class TestLayerNorm:
    def test_2d(self):
        x = np.random.randn(3, 4).astype(np.float32)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        eps = 1e-5

        # Manual numpy layer norm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * gamma + beta

        assert_close(
            from_np(x).layer_norm(from_np(gamma), from_np(beta), eps),
            expected,
        )

    def test_with_scale_and_shift(self):
        x = np.random.randn(3, 4).astype(np.float32)
        gamma = np.array([2.0, 0.5, 1.0, 3.0], dtype=np.float32)
        beta = np.array([0.1, -0.1, 0.0, 0.5], dtype=np.float32)
        eps = 1e-5

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * gamma + beta

        assert_close(
            from_np(x).layer_norm(from_np(gamma), from_np(beta), eps),
            expected,
        )

    def test_3d(self):
        """Layer norm on 3D input normalizes along the last dim."""
        x = np.random.randn(2, 3, 4).astype(np.float32)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        eps = 1e-5

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * gamma + beta

        assert_close(
            from_np(x).layer_norm(from_np(gamma), from_np(beta), eps),
            expected,
        )


class TestEmbeddingLookup:
    def test_basic(self):
        table = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float32,
        )
        indices = [2, 0, 1]
        result = from_np(table).embedding_lookup(indices)
        expected = table[indices]
        assert_close(result, expected)

    def test_repeated_indices(self):
        table = np.random.randn(100, 16).astype(np.float32)
        indices = [5, 5, 10, 0, 99]
        result = from_np(table).embedding_lookup(indices)
        assert result.shape == [5, 16]
        assert_close(result, table[indices])

    def test_out_of_range_raises(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        with pytest.raises(ValueError):
            t.embedding_lookup([5])


class TestTriMask:
    def test_basic(self):
        mask = Tensor.tri_mask(4)
        m = to_np(mask)
        # Diagonal and below should be 0
        for i in range(4):
            for j in range(4):
                if j <= i:
                    assert m[i, j] == 0.0
                else:
                    assert m[i, j] == -1e9

    def test_size_1(self):
        mask = Tensor.tri_mask(1)
        assert mask.to_vec() == [0.0]


class TestArgmax:
    def test_basic(self):
        t = Tensor([1.0, 5.0, 3.0, 2.0], [4])
        assert t.argmax() == 1

    def test_last_element(self):
        t = Tensor([1.0, 2.0, 3.0, 10.0], [4])
        assert t.argmax() == 3

    def test_negative_values(self):
        t = Tensor([-5.0, -1.0, -3.0], [3])
        assert t.argmax() == 1


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

class TestTracing:
    def test_trace_toggle(self, tmp_path):
        trace_log = str(tmp_path / "trace.log")
        set_trace_file(trace_log)

        set_trace(2)
        a = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = Tensor([1.0, 0.0, 0.0, 1.0], [2, 2])
        _ = a.matmul(b)

        content = open(trace_log).read()
        assert "[compute] matmul" in content

        # Clear file and disable trace
        open(trace_log, "w").close()
        set_trace(0)
        _ = a.matmul(b)

        content = open(trace_log).read()
        assert content == ""

        set_trace_file(None)
