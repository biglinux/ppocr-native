# Development

## The GPU kernels

Measured on 10 pages at 300 dpi, on a Radeon Vega 7 beside a Ryzen 5 5600H
and a GeForce GTX 1660 Ti beside a Core i5-13400:

- An integrated GPU reads 40% of a page's line width: on the Vega 7, every
  line on the cores took 14.1 s; a GPU share of 0.4 took 11.3 s and 0.5 took
  12.5 to 13.5 s.
- The matrix kernel takes eight values of the reduction a step; 16 and 32 are
  slower on both GPUs, and 4 is slower on the Vega.
- Shared tiles read as `vec4` are faster on the GTX 1660 Ti and slower on the
  Vega (detector 268 to 274 ms against 311 to 315 ms), so AMD reads them one
  value at a time.
- Slower on the Vega 7, and left out: half-float tensors (15% faster, but 49
  of 440 boxes moved), weights read through the scalar cache (detector twice
  as long), two 1x1 convolutions passing their product through shared memory
  (267 to 270 ms against 308 to 312 ms), register tiles of 8x4 or 8x8, a max
  pool computed inside the convolution that reads it, and keys and values of
  the attention in shared memory (the page 10% slower; 1% faster on the GTX
  1660 Ti, within noise).
