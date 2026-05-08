// tensor.rs — Minimal tensor library for educational LLM inference.
//
// Storage is always contiguous row-major f32. No strides — transpose copies data.
// AVX2+FMA SIMD kernels with automatic runtime fallback to scalar loops.

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use std::io::Write;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Force-disable AVX2 (use naive scalar ops). When false, runtime detection decides.
static AVX2_DISABLED: AtomicBool = AtomicBool::new(false);

pub fn set_avx2_enabled(enabled: bool) {
    AVX2_DISABLED.store(!enabled, Ordering::Relaxed);
}

pub fn get_avx2_enabled() -> bool {
    !AVX2_DISABLED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Tracing
// ---------------------------------------------------------------------------

/// Trace verbosity: 0=off, 1=low (op name + total ms), 2=high (op + shapes + ms)
static TRACE_LEVEL: AtomicU8 = AtomicU8::new(0);

pub fn set_trace_level(level: u8) {
    TRACE_LEVEL.store(level.min(2), Ordering::Relaxed);
}

pub fn get_trace_level() -> u8 {
    TRACE_LEVEL.load(Ordering::Relaxed)
}

fn is_trace() -> bool {
    TRACE_LEVEL.load(Ordering::Relaxed) > 0
}

fn is_trace_high() -> bool {
    TRACE_LEVEL.load(Ordering::Relaxed) >= 2
}

/// Optional trace log file.
static TRACE_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

pub fn set_trace_file(path: Option<&str>) {
    let mut guard = TRACE_FILE.lock().unwrap();
    *guard = path.map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .expect("failed to open trace file")
    });
}

/// Write a trace message to the trace file only (not stderr).
fn trace_write(msg: &str) {
    if let Ok(mut guard) = TRACE_FILE.lock() {
        if let Some(ref mut f) = *guard {
            let _ = writeln!(f, "{}", msg);
        }
    }
}

fn fmt_shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Print a trace line if tracing is enabled.
/// Low verbosity: op name + time only.
/// High verbosity: op name + input shapes + output shape + time.
macro_rules! trace {
    ($name:expr, [$($input:expr),+], $output:expr, $start:expr) => {
        if is_trace() {
            let ms = $start.elapsed().as_secs_f64() * 1000.0;
            if is_trace_high() {
                let inputs = vec![$(fmt_shape($input)),+];
                trace_write(&format!(
                    "[compute] {} {} → {} ({:.2}ms)",
                    $name,
                    inputs.join(" × "),
                    fmt_shape($output),
                    ms,
                ));
            } else {
                trace_write(&format!("[compute] {} ({:.2}ms)", $name, ms));
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Row-major strides for a given shape.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Convert a flat index to a multi-dimensional index given strides.
fn flat_to_multi(mut flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(strides.len());
    for &s in strides {
        idx.push(flat / s);
        flat %= s;
    }
    idx
}

/// Convert a multi-dimensional index to a flat index given strides.
fn multi_to_flat(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides).map(|(i, s)| i * s).sum()
}

// ---------------------------------------------------------------------------
// SIMD Acceleration (AVX2 + FMA)
// ---------------------------------------------------------------------------
//
// Runtime feature detection dispatches to AVX2+FMA kernels when available,
// falling back to naive scalar loops otherwise. The #[target_feature] attribute
// tells the compiler to emit AVX2 instructions for these specific functions
// without affecting the rest of the binary.

#[cfg(target_arch = "x86_64")]
fn use_avx2() -> bool {
    if AVX2_DISABLED.load(Ordering::Relaxed) {
        return false;
    }
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn use_avx2() -> bool {
    false
}

// -- Horizontal reductions --------------------------------------------------

/// Horizontal sum of 8 floats in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// Horizontal max of 8 floats in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(max128);
    let maxs = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(maxs, maxs);
    let result = _mm_max_ss(maxs, shuf2);
    _mm_cvtss_f32(result)
}

// -- GEMM (general matrix multiply) ----------------------------------------

/// Naive scalar GEMM: C[m,n] = A[m,k] × B[k,n]
fn gemm_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// AVX2+FMA tiled GEMM. Broadcasts each A element across 8 B columns,
/// accumulating with FMA. Tiles over K for cache locality.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const K_TILE: usize = 128;
    let n8 = n / 8 * 8;

    c[..m * n].fill(0.0);

    for kt in (0..k).step_by(K_TILE) {
        let k_end = (kt + K_TILE).min(k);

        for i in 0..m {
            // AVX2: process 8 output columns at a time
            let mut j = 0;
            while j < n8 {
                let out_idx = i * n + j;
                let mut acc = _mm256_loadu_ps(c.as_ptr().add(out_idx));
                for p in kt..k_end {
                    let a_val = _mm256_set1_ps(*a.get_unchecked(i * k + p));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                    acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                }
                _mm256_storeu_ps(c.as_mut_ptr().add(out_idx), acc);
                j += 8;
            }
            // Scalar remainder (when n is not a multiple of 8)
            for j in n8..n {
                let mut sum = c[i * n + j];
                for p in kt..k_end {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

/// Dispatch: use AVX2 if available, otherwise naive.
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { gemm_avx2(a, b, c, m, k, n); }
        return;
    }
    gemm_naive(a, b, c, m, k, n);
}

// -- Elementwise addition ---------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    for i in n8..n {
        out[i] = a[i] + b[i];
    }
}

fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { vec_add_avx2(a, b, out); }
        return;
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

// -- Elementwise multiply ---------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(va, vb));
        i += 8;
    }
    for i in n8..n {
        out[i] = a[i] * b[i];
    }
}

fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { vec_mul_avx2(a, b, out); }
        return;
    }
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

// -- Scalar multiply --------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_mul_scalar_avx2(a: &[f32], s: f32, out: &mut [f32]) {
    let n = a.len();
    let n8 = n / 8 * 8;
    let vs = _mm256_set1_ps(s);
    let mut i = 0;
    while i < n8 {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(va, vs));
        i += 8;
    }
    for i in n8..n {
        out[i] = a[i] * s;
    }
}

fn vec_mul_scalar(a: &[f32], s: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { vec_mul_scalar_avx2(a, s, out); }
        return;
    }
    for i in 0..a.len() {
        out[i] = a[i] * s;
    }
}

// -- GELU activation -------------------------------------------------------

/// Padé approximation of tanh, accurate to ~1e-7 for |x| < 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(x: __m256) -> __m256 {
    // tanh(x) = x * P(x²) / Q(x²), clamped to [-1, 1]
    let x2 = _mm256_mul_ps(x, x);
    // P(x²) = 135135 + x² * (17325 + x² * (378 + x²))
    let p = _mm256_add_ps(x2, _mm256_set1_ps(378.0));
    let p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(17325.0));
    let p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(135135.0));
    let num = _mm256_mul_ps(x, p);
    // Q(x²) = 135135 + x² * (62370 + x² * (3150 + 28 * x²))
    let q = _mm256_fmadd_ps(_mm256_set1_ps(28.0), x2, _mm256_set1_ps(3150.0));
    let q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(62370.0));
    let q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(135135.0));
    let result = _mm256_div_ps(num, q);
    // Clamp to [-1, 1] for large inputs
    _mm256_min_ps(_mm256_max_ps(result, _mm256_set1_ps(-1.0)), _mm256_set1_ps(1.0))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_gelu_avx2(data: &[f32], out: &mut [f32]) {
    let n = data.len();
    let n8 = n / 8 * 8;
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let c = _mm256_set1_ps(0.7978845608028654); // sqrt(2/π)
    let coeff = _mm256_set1_ps(0.044715);
    let mut i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(data.as_ptr().add(i));
        let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        let inner = _mm256_mul_ps(c, _mm256_fmadd_ps(coeff, x3, x));
        let t = tanh_avx2(inner);
        let result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t)));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 8;
    }
    // Scalar remainder
    let c_s = 0.7978845608028654f32;
    for i in n8..n {
        let x = data[i];
        out[i] = 0.5 * x * (1.0 + (c_s * (x + 0.044715 * x * x * x)).tanh());
    }
}

fn vec_gelu(data: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { vec_gelu_avx2(data, out); }
        return;
    }
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    for i in 0..data.len() {
        let x = data[i];
        out[i] = 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
    }
}

// -- Fast exp (for softmax) -------------------------------------------------

/// Polynomial exp approximation with range reduction: exp(x) = 2^n * poly(r).
/// Uses a 6th-order Taylor series for the fractional part (error < 2e-7).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2(x: __m256) -> __m256 {
    // Clamp to avoid overflow in float→int conversion
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.3));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.7));

    let log2e = _mm256_set1_ps(1.4426950408889634);
    let ln2 = _mm256_set1_ps(0.6931471805599453);
    let one = _mm256_set1_ps(1.0);

    // Range reduction: n = round(x * log2(e)), r = x - n * ln(2)
    let ni = _mm256_cvtps_epi32(_mm256_mul_ps(x, log2e));
    let n = _mm256_cvtepi32_ps(ni);
    let r = _mm256_fnmadd_ps(n, ln2, x); // r = x - n*ln2

    // Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
    let c6 = _mm256_set1_ps(1.0 / 720.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c2 = _mm256_set1_ps(0.5);
    let p = _mm256_fmadd_ps(c6, r, c5);
    let p = _mm256_fmadd_ps(p, r, c4);
    let p = _mm256_fmadd_ps(p, r, c3);
    let p = _mm256_fmadd_ps(p, r, c2);
    let p = _mm256_fmadd_ps(p, r, one);
    let p = _mm256_fmadd_ps(p, r, one);

    // 2^n via exponent biasing: cast n to int, add 127, shift to exponent field
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2n)
}

// -- Softmax ----------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(data: &mut [f32]) {
    let n = data.len();
    let n8 = n / 8 * 8;

    // Max
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < n8 {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(data.as_ptr().add(i)));
        i += 8;
    }
    let mut max_val = hmax_avx2(max_v);
    for i in n8..n {
        if data[i] > max_val {
            max_val = data[i];
        }
    }

    // Exp and sum
    let max_bc = _mm256_set1_ps(max_val);
    let mut sum_v = _mm256_setzero_ps();
    i = 0;
    while i < n8 {
        let x = _mm256_sub_ps(_mm256_loadu_ps(data.as_ptr().add(i)), max_bc);
        let e = exp_avx2(x);
        _mm256_storeu_ps(data.as_mut_ptr().add(i), e);
        sum_v = _mm256_add_ps(sum_v, e);
        i += 8;
    }
    let mut sum = hsum_avx2(sum_v);
    for i in n8..n {
        data[i] = (data[i] - max_val).exp();
        sum += data[i];
    }

    // Normalize
    let inv_sum = _mm256_set1_ps(1.0 / sum);
    i = 0;
    while i < n8 {
        let x = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_mul_ps(x, inv_sum));
        i += 8;
    }
    let inv_sum_s = 1.0 / sum;
    for i in n8..n {
        data[i] *= inv_sum_s;
    }
}

fn softmax_row_naive(data: &mut [f32]) {
    let n = data.len();
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..n {
        if data[i] > max_val {
            max_val = data[i];
        }
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        data[i] = (data[i] - max_val).exp();
        sum += data[i];
    }
    for i in 0..n {
        data[i] /= sum;
    }
}

fn softmax_row(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { softmax_row_avx2(data); }
        return;
    }
    softmax_row_naive(data);
}

// -- Layer normalization ----------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_row_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    eps: f32,
) {
    let d = input.len();
    let d8 = d / 8 * 8;

    // Sum for mean
    let mut sum_acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < d8 {
        sum_acc = _mm256_add_ps(sum_acc, _mm256_loadu_ps(input.as_ptr().add(i)));
        i += 8;
    }
    let mut sum = hsum_avx2(sum_acc);
    for i in d8..d {
        sum += input[i];
    }
    let mean = sum / d as f32;

    // Variance
    let mean_v = _mm256_set1_ps(mean);
    let mut var_acc = _mm256_setzero_ps();
    i = 0;
    while i < d8 {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let diff = _mm256_sub_ps(x, mean_v);
        var_acc = _mm256_fmadd_ps(diff, diff, var_acc);
        i += 8;
    }
    let mut var = hsum_avx2(var_acc);
    for i in d8..d {
        let diff = input[i] - mean;
        var += diff * diff;
    }
    let inv_std = 1.0 / (var / d as f32 + eps).sqrt();

    // Normalize, scale, shift
    let inv_std_v = _mm256_set1_ps(inv_std);
    i = 0;
    while i < d8 {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let diff = _mm256_sub_ps(x, mean_v);
        let norm = _mm256_mul_ps(diff, inv_std_v);
        let g = _mm256_loadu_ps(gamma.as_ptr().add(i));
        let b = _mm256_loadu_ps(beta.as_ptr().add(i));
        let result = _mm256_fmadd_ps(norm, g, b);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        i += 8;
    }
    for i in d8..d {
        out[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

fn layer_norm_row_naive(
    input: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32], eps: f32,
) {
    let d = input.len();
    let mean: f32 = input.iter().sum::<f32>() / d as f32;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..d {
        out[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

fn layer_norm_row(
    input: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32], eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        unsafe { layer_norm_row_avx2(input, gamma, beta, out, eps); }
        return;
    }
    layer_norm_row_naive(input, gamma, beta, out, eps);
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[pymethods]
impl Tensor {
    // -- Constructors -------------------------------------------------------

    /// Create a tensor from flat data and a shape.
    /// Example (Python): Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            )));
        }
        Ok(Tensor { data, shape })
    }

    /// All-zeros tensor of the given shape.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Upper-triangular causal mask: 0 on/below diagonal, -1e9 above.
    /// Used to prevent attention to future tokens.
    #[staticmethod]
    fn tri_mask(size: usize) -> Self {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in (i + 1)..size {
                data[i * size + j] = -1e9;
            }
        }
        Tensor {
            data,
            shape: vec![size, size],
        }
    }

    // -- Accessors ----------------------------------------------------------

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn numel(&self) -> usize {
        self.data.len()
    }

    /// Return data as a flat Python list of floats.
    fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn __repr__(&self) -> String {
        let n = self.data.len();
        if n <= 6 {
            format!("Tensor(shape={:?}, data={:?})", self.shape, self.data)
        } else {
            format!(
                "Tensor(shape={:?}, data=[{:.4}, {:.4}, ... {:.4}, {:.4}])",
                self.shape,
                self.data[0],
                self.data[1],
                self.data[n - 2],
                self.data[n - 1],
            )
        }
    }

    // -- Shape operations ---------------------------------------------------

    /// Reshape (same data, new shape). Total element count must match.
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cannot reshape {:?} ({} elems) to {:?} ({} elems)",
                self.shape,
                self.data.len(),
                new_shape,
                expected,
            )));
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Transpose (swap two dimensions). Returns a new tensor with copied data.
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        let ndim = self.shape.len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transpose dims ({}, {}) out of range for {}D tensor",
                dim0, dim1, ndim
            )));
        }
        if dim0 == dim1 {
            return Ok(self.clone());
        }

        let t = Instant::now();

        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);

        let old_s = strides(&self.shape);
        let new_s = strides(&new_shape);
        let numel = self.data.len();
        let mut new_data = vec![0.0f32; numel];

        for i in 0..numel {
            let mut mi = flat_to_multi(i, &new_s); // index in output
            mi.swap(dim0, dim1); // corresponding index in input
            new_data[i] = self.data[multi_to_flat(&mi, &old_s)];
        }

        trace!("transpose", [&self.shape], &new_shape, t);
        Ok(Tensor {
            data: new_data,
            shape: new_shape,
        })
    }

    /// Slice along a dimension: tensor.slice(dim, start, end).
    /// Returns a new tensor containing elements [start..end) along `dim`.
    fn slice(&self, dim: usize, start: usize, end: usize) -> PyResult<Self> {
        let ndim = self.shape.len();
        if dim >= ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim {} out of range for {}D tensor",
                dim, ndim
            )));
        }
        if start >= end || end > self.shape[dim] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid slice [{}..{}) for dim {} of size {}",
                start, end, dim, self.shape[dim]
            )));
        }

        let t = Instant::now();

        let outer: usize = self.shape[..dim].iter().product();
        let inner: usize = self.shape[dim + 1..].iter().product();
        let old_dim = self.shape[dim];
        let new_dim = end - start;

        let mut new_data = Vec::with_capacity(outer * new_dim * inner);
        for o in 0..outer {
            for k in start..end {
                let src = (o * old_dim + k) * inner;
                new_data.extend_from_slice(&self.data[src..src + inner]);
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = new_dim;

        trace!("slice", [&self.shape], &new_shape, t);
        Ok(Tensor {
            data: new_data,
            shape: new_shape,
        })
    }

    /// Extract a single "row" along the first dimension.
    /// For a 2D tensor [M, N], slice_row(i) returns shape [N].
    /// Supports negative indexing: slice_row(-1) is the last row.
    fn slice_row(&self, idx: i32) -> PyResult<Self> {
        if self.shape.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "slice_row: tensor has no dimensions",
            ));
        }
        let rows = self.shape[0];
        let idx = if idx < 0 {
            (rows as i32 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= rows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row index {} out of range for {} rows",
                idx, rows
            )));
        }

        let row_size: usize = self.shape[1..].iter().product();
        let start = idx * row_size;
        Ok(Tensor {
            data: self.data[start..start + row_size].to_vec(),
            shape: self.shape[1..].to_vec(),
        })
    }

    // -- Math operations ----------------------------------------------------

    /// Matrix multiply. Supports 2D×2D and batched 3D×3D.
    ///   2D: [M, K] × [K, N] → [M, N]
    ///   3D: [B, M, K] × [B, K, N] → [B, M, N]
    fn matmul(&self, other: &Tensor) -> PyResult<Tensor> {
        let t = Instant::now();

        let result = match (self.shape.len(), other.shape.len()) {
            (2, 2) => {
                let (m, k) = (self.shape[0], self.shape[1]);
                let n = other.shape[1];
                if other.shape[0] != k {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "matmul: {:?} × {:?} — inner dims don't match",
                        self.shape, other.shape
                    )));
                }
                let mut out = vec![0.0f32; m * n];
                gemm(&self.data, &other.data, &mut out, m, k, n);
                Tensor {
                    data: out,
                    shape: vec![m, n],
                }
            }
            (3, 3) => {
                let (b, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
                let n = other.shape[2];
                if other.shape[0] != b || other.shape[1] != k {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "matmul: {:?} × {:?} — batch or inner dims don't match",
                        self.shape, other.shape
                    )));
                }
                let mut out = vec![0.0f32; b * m * n];
                for bi in 0..b {
                    let a_off = bi * m * k;
                    let b_off = bi * k * n;
                    let o_off = bi * m * n;
                    gemm(
                        &self.data[a_off..a_off + m * k],
                        &other.data[b_off..b_off + k * n],
                        &mut out[o_off..o_off + m * n],
                        m, k, n,
                    );
                }
                Tensor {
                    data: out,
                    shape: vec![b, m, n],
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "matmul: unsupported shapes {:?} × {:?} (need 2D×2D or 3D×3D)",
                    self.shape, other.shape
                )));
            }
        };

        trace!("matmul", [&self.shape, &other.shape], &result.shape, t);
        Ok(result)
    }

    /// Elementwise addition with broadcasting.
    /// Supports: same shape, or other's shape matches the tail of self's shape.
    ///   [M, N] + [N]       — add bias to each row
    ///   [B, M, N] + [M, N] — add mask to each batch element
    ///   [B, M, N] + [N]    — add bias to each row of each batch
    fn add(&self, other: &Tensor) -> PyResult<Tensor> {
        let t = Instant::now();

        let result = if self.shape == other.shape {
            // Same shape: elementwise add
            let mut data = vec![0.0f32; self.data.len()];
            vec_add(&self.data, &other.data, &mut data);
            Tensor {
                data,
                shape: self.shape.clone(),
            }
        } else if self.shape.len() > other.shape.len() {
            // Check that other's shape matches the tail of self's shape
            let offset = self.shape.len() - other.shape.len();
            if self.shape[offset..] != other.shape[..] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "add: cannot broadcast {:?} and {:?}",
                    self.shape, other.shape
                )));
            }
            let other_size = other.data.len();
            let mut data = vec![0.0f32; self.data.len()];
            for chunk in 0..(self.data.len() / other_size) {
                let off = chunk * other_size;
                vec_add(
                    &self.data[off..off + other_size],
                    &other.data,
                    &mut data[off..off + other_size],
                );
            }
            Tensor {
                data,
                shape: self.shape.clone(),
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "add: cannot broadcast {:?} and {:?}",
                self.shape, other.shape
            )));
        };

        trace!("add", [&self.shape, &other.shape], &result.shape, t);
        Ok(result)
    }

    /// Elementwise multiply (same shape required).
    fn mul(&self, other: &Tensor) -> PyResult<Tensor> {
        if self.shape != other.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mul: shape mismatch {:?} vs {:?}",
                self.shape, other.shape
            )));
        }
        let t = Instant::now();
        let mut data = vec![0.0f32; self.data.len()];
        vec_mul(&self.data, &other.data, &mut data);
        let result = Tensor {
            data,
            shape: self.shape.clone(),
        };
        trace!("mul", [&self.shape, &other.shape], &result.shape, t);
        Ok(result)
    }

    /// Multiply every element by a scalar.
    fn mul_scalar(&self, s: f32) -> Tensor {
        let t = Instant::now();
        let mut data = vec![0.0f32; self.data.len()];
        vec_mul_scalar(&self.data, s, &mut data);
        let result = Tensor {
            data,
            shape: self.shape.clone(),
        };
        trace!("mul_scalar", [&self.shape], &result.shape, t);
        result
    }

    /// GELU activation (used in GPT-2 FFN).
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    fn gelu(&self) -> Tensor {
        let t = Instant::now();
        let mut data = vec![0.0f32; self.data.len()];
        vec_gelu(&self.data, &mut data);
        let result = Tensor {
            data,
            shape: self.shape.clone(),
        };
        trace!("gelu", [&self.shape], &result.shape, t);
        result
    }

    /// Softmax along the given axis. Supports negative axis (-1 = last).
    /// Uses the numerically stable version: exp(x - max) / sum(exp(x - max)).
    fn softmax(&self, axis: i32) -> PyResult<Tensor> {
        let ndim = self.shape.len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "softmax: axis {} out of range for {}D tensor",
                axis, ndim
            )));
        }

        let t = Instant::now();

        let mut data = self.data.clone();
        let axis_size = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();

        if inner == 1 {
            // Fast path: contiguous rows (last-axis softmax, the common case)
            for o in 0..outer {
                let off = o * axis_size;
                softmax_row(&mut data[off..off + axis_size]);
            }
        } else {
            // General path: strided access
            for o in 0..outer {
                for i in 0..inner {
                    let base = o * axis_size * inner + i;

                    let mut max_val = f32::NEG_INFINITY;
                    for k in 0..axis_size {
                        let v = data[base + k * inner];
                        if v > max_val {
                            max_val = v;
                        }
                    }

                    let mut sum = 0.0f32;
                    for k in 0..axis_size {
                        let idx = base + k * inner;
                        data[idx] = (data[idx] - max_val).exp();
                        sum += data[idx];
                    }

                    for k in 0..axis_size {
                        data[base + k * inner] /= sum;
                    }
                }
            }
        }

        let result = Tensor {
            data,
            shape: self.shape.clone(),
        };
        trace!("softmax", [&self.shape], &result.shape, t);
        Ok(result)
    }

    /// Layer normalization along the last dimension.
    /// norm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
    /// gamma and beta must have shape [last_dim].
    fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> PyResult<Tensor> {
        let ndim = self.shape.len();
        let last_dim = self.shape[ndim - 1];

        if gamma.shape != [last_dim] || beta.shape != [last_dim] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "layer_norm: gamma/beta shape must be [{}], got {:?} / {:?}",
                last_dim, gamma.shape, beta.shape
            )));
        }

        let t = Instant::now();

        let num_rows = self.data.len() / last_dim;
        let mut data = vec![0.0f32; self.data.len()];

        for row in 0..num_rows {
            let off = row * last_dim;
            layer_norm_row(
                &self.data[off..off + last_dim],
                &gamma.data,
                &beta.data,
                &mut data[off..off + last_dim],
                eps,
            );
        }

        let result = Tensor {
            data,
            shape: self.shape.clone(),
        };
        trace!("layer_norm", [&self.shape], &result.shape, t);
        Ok(result)
    }

    /// Look up rows from an embedding table by index.
    /// self must be 2D [vocab_size, embed_dim].
    /// Returns a tensor of shape [len(indices), embed_dim].
    fn embedding_lookup(&self, indices: Vec<usize>) -> PyResult<Tensor> {
        if self.shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "embedding_lookup: table must be 2D [vocab_size, embed_dim]",
            ));
        }

        let t = Instant::now();

        let embed_dim = self.shape[1];
        let seq_len = indices.len();
        let mut data = vec![0.0f32; seq_len * embed_dim];

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[0] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "embedding_lookup: index {} out of range for vocab size {}",
                    idx, self.shape[0]
                )));
            }
            let src = idx * embed_dim;
            let dst = i * embed_dim;
            data[dst..dst + embed_dim].copy_from_slice(&self.data[src..src + embed_dim]);
        }

        let new_shape = vec![seq_len, embed_dim];
        trace!("embedding_lookup", [&self.shape], &new_shape, t);
        Ok(Tensor {
            data,
            shape: new_shape,
        })
    }

    /// Index of the maximum element (flat). Used for greedy decoding.
    fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
