// lib.rs — PyO3 module entry point.
//
// Exposes the Tensor class and configuration functions to Python.
// Usage:
//   from compute import Tensor, set_trace, set_avx2, set_trace_file
//   set_trace(2)         # 0=off, 1=low, 2=high
//   set_avx2(True)       # True=use AVX2 if available, False=force naive
//   set_trace_file("trace.log")  # also log to file

mod tensor;

use pyo3::prelude::*;
use tensor::Tensor;

/// Set trace verbosity level: 0=off, 1=low (op name + time), 2=high (shapes + time).
#[pyfunction]
fn set_trace(level: u8) {
    tensor::set_trace_level(level);
}

/// Get current trace verbosity level.
#[pyfunction]
fn get_trace() -> u8 {
    tensor::get_trace_level()
}

/// Enable or disable AVX2 SIMD acceleration.
/// When disabled, all ops use naive scalar loops (useful for benchmarking/debugging).
#[pyfunction]
fn set_avx2(enabled: bool) {
    tensor::set_avx2_enabled(enabled);
}

/// Check if AVX2 is currently enabled.
#[pyfunction]
fn get_avx2() -> bool {
    tensor::get_avx2_enabled()
}

/// Set a file path to log trace output to (in addition to stderr).
/// Pass None to disable file logging.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn set_trace_file(path: Option<&str>) {
    tensor::set_trace_file(path);
}

#[pymodule]
fn compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_function(wrap_pyfunction!(set_trace, m)?)?;
    m.add_function(wrap_pyfunction!(get_trace, m)?)?;
    m.add_function(wrap_pyfunction!(set_avx2, m)?)?;
    m.add_function(wrap_pyfunction!(get_avx2, m)?)?;
    m.add_function(wrap_pyfunction!(set_trace_file, m)?)?;
    Ok(())
}
