// The activations folded into an operation, numbered as `Fused` in
// src/engine/plan.rs: 0 none, 1 ReLU, 2 sigmoid, 3 GELU, 4 swish, 5 hard
// sigmoid, 6 the neck's hard sigmoid. They follow the CPU's definitions
// (src/engine/layers.rs, src/engine/kernels/activation.rs).

// The flat index of this invocation in a dispatch of up to 65535 by N
// workgroups.
int invocation() {
    return int((gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x
        + gl_LocalInvocationID.x);
}

// exp(z) for z <= 0 as XNNPACK's f32-vsigmoid "p5-div" kernels compute it,
// 0 below the denormal cutoff. `precise` keeps the compiler from folding the
// magic bias away, which rounds n.
float exp_neg(float z) {
    precise float n = z * uintBitsToFloat(0x3fb8aa3bu) + uintBitsToFloat(0x4b40007fu);
    float s = uintBitsToFloat(floatBitsToUint(n) << 23);
    n -= uintBitsToFloat(0x4b40007fu);
    precise float t = n * uintBitsToFloat(0xb5bfbe8eu) + (n * uintBitsToFloat(0xbf317200u) + z);
    float p = uintBitsToFloat(0x3c07cfceu);
    p = p * t + uintBitsToFloat(0x3d2b9d0du);
    p = p * t + uintBitsToFloat(0x3e2aad40u);
    p = p * t + uintBitsToFloat(0x3efffee3u);
    p = p * t + uintBitsToFloat(0x3f7ffffbu);
    return z < uintBitsToFloat(0xc2aeac4fu) ? 0.0 : (t * s) * p + s;
}

float sigmoid(float x) {
    float e = exp_neg(-abs(x));
    float f = e / (e + 1.0);
    return x < 0.0 ? f : 1.0 - f;
}

// x Phi(x), Phi a rational fit on [-5.5, 5.5].
float gelu(float x) {
    float c = clamp(x, -5.5, 5.5);
    float s = c * c;
    float p = uintBitsToFloat(0x32c36a3cu);
    p = p * s + uintBitsToFloat(0x36d52462u);
    p = p * s + uintBitsToFloat(0x39310738u);
    p = p * s + uintBitsToFloat(0x3b9943c9u);
    p = p * s + uintBitsToFloat(0x3d0aa93cu);
    p = p * s + uintBitsToFloat(0x3ecc4228u);
    float q = uintBitsToFloat(0x35a0b9e3u);
    q = q * s + uintBitsToFloat(0x389a737au);
    q = q * s + uintBitsToFloat(0x3af6ffc6u);
    q = q * s + uintBitsToFloat(0x3ceaaaeeu);
    q = q * s + uintBitsToFloat(0x3e80c788u);
    q = q * s + 1.0;
    return x * clamp(c * (p / q) + 0.5, 0.0, 1.0);
}

float act(float v, int a) {
    switch (a) {
    case 1: return max(v, 0.0);
    case 2: return sigmoid(v);
    case 3: return gelu(v);
    case 4: return v * sigmoid(v);
    case 5: return clamp(v * uintBitsToFloat(0x3e2aaaadu) + 0.5, 0.0, 1.0);
    case 6: return clamp(v * 0.2 + 0.5, 0.0, 1.0);
    default: return v;
    }
}
