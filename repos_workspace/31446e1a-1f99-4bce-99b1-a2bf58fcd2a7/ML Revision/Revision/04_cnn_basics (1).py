"""
CNN BASICS
===========
Interview must-knows:
- Motivation over plain fully-connected nets for images: FC layers on raw
  pixels (a) explode in parameter count (every pixel connects to every hidden
  unit) and (b) ignore spatial structure (a shifted cat is a totally different
  input vector to an FC layer). CNNs fix both via:
    1. PARAMETER SHARING: the same small filter (kernel) slides across the
       whole image, so one filter's weights are reused at every spatial
       location -> far fewer parameters, and a pattern learned in one corner
       is automatically detected anywhere else too.
    2. LOCAL CONNECTIVITY: each output unit only looks at a small local patch
       (the receptive field), matching the intuition that nearby pixels are
       more related than far-apart ones.
    3. TRANSLATION EQUIVARIANCE: shifting the input shifts the feature map by
       the same amount -- a direct consequence of weight sharing.
- Convolution operation: slide a KxK filter over the input, at each position
  compute an element-wise multiply + sum (technically cross-correlation in
  most DL frameworks, not "true" flipped convolution, but everyone calls it
  convolution anyway).
- Output spatial size: out = floor((in + 2*padding - kernel) / stride) + 1
    padding: "same" (pad so output size == input size) vs "valid" (no padding,
             output shrinks).
    stride: step size the filter moves each time; stride>1 downsamples.
- Multiple filters per layer -> multiple output "feature maps" stacked into a
  channel dimension; each filter learns to detect a different pattern (edges,
  textures early on -> more abstract/semantic parts in deeper layers).
- Pooling (e.g. Max Pooling): downsamples each feature map (e.g. take the max
  over each 2x2 block) -> reduces spatial size/computation, adds a small
  amount of translation invariance, no learnable parameters.
- Typical architecture pattern: [Conv -> Activation(ReLU) -> Pool] repeated,
  then Flatten -> Fully Connected -> Softmax for classification.
- Parameter count for one conv layer: (kernel_h * kernel_w * in_channels + 1
  bias) * out_channels -- MUCH smaller than an FC layer over the same input.
"""

import numpy as np

# -----------------------------------------------------------------
# 1. 2D CONVOLUTION FROM SCRATCH (single channel, single filter)
# -----------------------------------------------------------------
def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            row, col = i * stride, j * stride
            patch = image[row:row + kh, col:col + kw]
            output[i, j] = np.sum(patch * kernel)          # elementwise multiply + sum
    return output

# A simple 6x6 "image" with a vertical edge (left half dark, right half bright)
image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
], dtype=float)

vertical_edge_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=float)                                    # classic Sobel-like edge detector

feature_map = conv2d(image, vertical_edge_filter, stride=1, padding=0)
print("Input image (6x6):\n", image)
print("\nVertical-edge filter (3x3):\n", vertical_edge_filter)
print("\nOutput feature map (4x4) -- large response exactly at the edge column:\n", feature_map)

# -----------------------------------------------------------------
# 2. Effect of stride and padding on output size (the formula in practice)
# -----------------------------------------------------------------
def output_size(in_size, kernel, stride, padding):
    return (in_size + 2 * padding - kernel) // stride + 1

print("\nOutput size formula check, in_size=6, kernel=3:")
for stride, padding in [(1, 0), (1, 1), (2, 0), (2, 1)]:
    computed = output_size(6, 3, stride, padding)
    actual = conv2d(image, vertical_edge_filter, stride=stride, padding=padding).shape[0]
    print(f"  stride={stride} padding={padding} -> formula={computed}, actual={actual}")

# -----------------------------------------------------------------
# 3. MAX POOLING FROM SCRATCH
# -----------------------------------------------------------------
def max_pool2d(feature_map, size=2, stride=2):
    h, w = feature_map.shape
    out_h, out_w = (h - size) // stride + 1, (w - size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            row, col = i * stride, j * stride
            output[i, j] = feature_map[row:row + size, col:col + size].max()
    return output

pooled = max_pool2d(feature_map, size=2, stride=2)
print("\nFeature map after 2x2 max pooling (halves spatial size, keeps strongest signal):\n", pooled)

# -----------------------------------------------------------------
# 4. Parameter count: Conv layer vs equivalent Fully-Connected layer
# -----------------------------------------------------------------
in_h, in_w, in_c = 32, 32, 3           # e.g. a small RGB image
kernel_size, out_channels = 3, 16

conv_params = (kernel_size * kernel_size * in_c + 1) * out_channels
fc_equivalent_params = (in_h * in_w * in_c) * (in_h * in_w * out_channels)  # naive dense mapping

print(f"\nParameter comparison for a {in_h}x{in_w}x{in_c} input, {out_channels} output maps:")
print(f"  Conv layer ({kernel_size}x{kernel_size} kernel): {conv_params:,} parameters")
print(f"  Naive fully-connected equivalent:      {fc_equivalent_params:,} parameters")
print(f"  Ratio: FC has {fc_equivalent_params // conv_params:,}x more parameters "
      "-- this is the parameter-sharing payoff.")

# -----------------------------------------------------------------
# 5. Multiple filters -> multiple feature maps (stacked "channels")
# -----------------------------------------------------------------
horizontal_edge_filter = vertical_edge_filter.T
maps = np.stack([
    conv2d(image, vertical_edge_filter),
    conv2d(image, horizontal_edge_filter),
])
print(f"\nStacking 2 filters' outputs gives shape {maps.shape} "
      "(2 output channels, each a different learned pattern detector).")

print("\nKey talking points: weight sharing + local connectivity -> fewer "
      "params + translation equivariance, conv output-size formula, "
      "stride/padding trade-offs, pooling for downsampling/invariance with no "
      "learnable params, stacking Conv->ReLU->Pool blocks, why CNNs beat FC "
      "nets on images.")
