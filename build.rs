//! Compile the GPU shaders to SPIR-V with `glslc`, from shaderc.

use std::path::PathBuf;
use std::process::Command;

/// Each SPIR-V module: its name, its source and the macros it is compiled
/// with: the matrix product's tile shapes (rows and columns) and how its
/// shared memory is read.
const SHADERS: [(&str, &str, &[&str]); 15] = [
    ("dwconv", "dwconv", &[]),
    ("dw7", "dw7", &[]),
    ("eltwise", "eltwise", &[]),
    ("pool", "pool", &[]),
    ("mean", "mean", &[]),
    ("convt", "convt", &[]),
    ("layernorm", "layernorm", &[]),
    ("attention", "attention", &[]),
    ("ctc", "ctc", &[]),
    ("gemm64", "gemm", &["BM=64", "BN=64"]),
    ("gemm32", "gemm", &["BM=128", "BN=32"]),
    ("gemm16", "gemm", &["BM=256", "BN=16"]),
    ("gemm64_vec4", "gemm", &["BM=64", "BN=64", "VEC4"]),
    ("gemm32_vec4", "gemm", &["BM=128", "BN=32", "VEC4"]),
    ("gemm16_vec4", "gemm", &["BM=256", "BN=16", "VEC4"]),
];

fn main() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    println!("cargo::rerun-if-changed=shaders/act.glsl");
    for (name, source, defines) in SHADERS {
        let source = format!("shaders/{source}.comp");
        println!("cargo::rerun-if-changed={source}");
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O", &source])
            .args(defines.iter().map(|d| format!("-D{d}")))
            .arg("-o")
            .arg(out.join(format!("{name}.spv")))
            .status()
            .unwrap_or_else(|e| panic!("running glslc (from shaderc): {e}"));
        assert!(status.success(), "glslc failed on {source}: {status}");
    }
}
