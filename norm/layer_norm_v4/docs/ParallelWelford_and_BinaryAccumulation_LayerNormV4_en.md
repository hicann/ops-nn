# Parallel Welford and the Extra-Long-Axis Binary Accumulation Algorithm: Precision Improvement for Norm Operators — Taking LayerNormV4 as an Example

## 0. Prerequisite Knowledge

### 0.1 Purpose of This Document and Target Readers

This document targets **users in the CANN ecosystem and practitioners in the software technology field**. It deeply analyzes, at the source code level, two core precision optimization technologies of LayerNormV4 on the Ascend 950 platform: **Parallel Welford** and **extra-long-axis binary accumulation**.

Before reading this document, familiarize yourself with the mathematical definition of LayerNorm (input/output shapes and the concept of normalization axes) and the basic properties of floating-point arithmetic. If you already read CANN operator source code proficiently, start directly from [Chapter 1](#1-background).

---

### 0.2 Background of the Ascend 950 and arch35 Platforms

| Concept | Description |
|------|------|
| **Ascend 950** | A model of Huawei Ascend AI processors. The precision optimization features of LayerNormV4 described in this document are designed for the hardware architecture of this chip, and may differ from the Ascend 910B and 910C series chips in parameters such as vector unit width and UB capacity. |
| **arch35** | The internal architecture code name corresponding to Ascend 950. In CANN source code, kernel code bound to a specific hardware architecture is placed under the `op_kernel/arch35/` path, indicating that this implementation applies only to the arch35 (that is, Ascend 950) architecture. Different arch values represent different hardware generations, so do not mistake this for an independent term. |
| **V4 suffix** | Indicates the fourth-generation implementation of the LayerNorm operator. In CANN, the same operator may have multiple versions (V1/V2/V3/V4), and each major version iteration usually introduces new algorithms or optimization strategies. The core upgrade of V4 is the introduction of the Parallel Welford and binary accumulation algorithms to improve precision. |

---

### 0.3 Basic Concepts of CANN Operator Development

#### 0.3.1 Division of Responsibilities Between op_kernel and op_host

CANN operators adopt the **Host-Device separation** programming model:

| Directory | Execution Location | Responsibility |
|------|---------|------|
| `op_kernel/` | **AI Core (device side/NPU)** | Executes the core computation logic of the operator (data loading → computation → write back). Kernel code runs on the AI Core of the Ascend chip and is the performance-critical path. |
| `op_host/` | **Host CPU** | Handles preparation before operator launch: tiling parameter calculation, task division and scheduling, operator attribute validation, and so on. Host-side code runs on the server CPU and does not directly participate in tensor computation. |

**Why are tiling parameters computed on the Host side?** Tiling parameters (such as the size of each tile and the number of loop iterations) depend on the shape of the input tensor, and the shape is known only at runtime. The Host CPU reads the shape, computes these parameters, and then sends them to the AI Core for execution. This "Host computes, Device runs" separation lets the AI Core focus only on pure computation and avoid handling control logic.

CANN operator developers need to maintain both Host-side and Kernel-side code.

#### 0.3.2 UB (Unified Buffer)

| Property | Description |
|------|------|
| **Essence** | An **on-chip memory** on the AI Core, similar to the Shared Memory of a GPU. |
| **Capacity** | Varies across architectures, usually several hundred KB. In this document, 64 alignment, tile size, and other parameters all treat this capacity as a hard constraint. |
| **Ownership** | **Exclusive to each AI Core**; different cores cannot directly access each other's UB. |
| **Usage** | Stores input data of the current tile, intermediate results (such as mean/rstd/M2 accumulators), gamma/beta weights, temporary buffers, and so on. |
| **Relationship with L1/L2** | UB sits at the L1 level and is 1-2 orders of magnitude faster than Global Memory (DDR/HBM). The core of operator performance optimization is to move data into UB, compute, and then move it out, minimizing accesses to Global Memory. |

**UB is the core constraint of tile division in this document.** When the total of one tile of data + gamma/beta + double buffer + temporary buffers exceeds the UB capacity, the tile must be split into smaller pieces.

#### 0.3.3 TILING_KEY — Algorithm Selection Key

**TILING_KEY** is an integer constant used in the CANN operator development framework to select an algorithm branch at **compile time** (essentially an enumeration value). It serves as the "algorithm selector" — different TILING_KEY values correspond to different algorithm implementations of this operator.

**Generation flow of TILING_KEY:**

```mermaid
Input tensor shape + data type
        ↓
  Host-side Tiling template matching (by priority)
        ↓
  Determine the best Tiling template
        ↓
  Generate the corresponding TILING_KEY (compile-time constant)
        ↓
  Kernel side dispatches to the specific implementation through the TILING_KEY_IS() macro
```

**Naming rule analysis (using `LNV4_REGBASE_WELFORD_FLOAT_FLOAT` as an example):**

| Part | Meaning |
|------|------|
| `LNV4` | LayerNorm Version 4 |
| `REGBASE` | Uses the register base (regbase) addressing mode |
| `WELFORD` | Uses the Parallel Welford algorithm |
| `FLOAT` (first) | Input data type (`src_dtype`), float32 |
| `FLOAT` (second) | Intermediate accumulation type (`accum_dtype`), float32 |

Similarly, `LNV4_REGBASE_TWO_PASS_FLOAT_FLOAT` indicates the Two-Pass (binary accumulation) algorithm in regbase mode, and `LNV4_REGBASE_TWO_PASS_PERF_FLOAT_FLOAT` follows the same pattern.

**Rules for the value range of TILING_KEY:**

| TILING_KEY range | Algorithm | Meaning of the base |
|----------------|------|---------|
| 800-822 | SplitR | Base 800 = starting number of the SplitR algorithm family |
| 400-422 | Parallel Welford | Base 400 = starting number of the Welford algorithm family |
| 300-322 | Two-Pass + Binary accumulation | Base 300 = starting number of the TwoPass algorithm family |
| 500-522 | Two-Pass Perf (optimized version) | Base 500 = starting number of the TwoPassPerf algorithm family |

The value range within each algorithm family (such as 400-422) covers different **data type combinations** (such as `FLOAT_FLOAT`, `HALF_FLOAT`, and `BF16_FLOAT`); that is, each concrete `(src_dtype, accum_dtype)` combination occupies one TILING_KEY value. Therefore, `400-422` means the Welford family covers about 23 data type combinations. These range values are naming conventions of the CANN framework and have no direct mathematical relationship with the underlying implementation.

**The compile-time macro `TILING_KEY_IS()`** is a conditional judgment macro provided by the CANN framework. It expands into a constant comparison at compile time. For example:

```cpp
if (TILING_KEY_IS(LNV4_REGBASE_WELFORD_FLOAT_FLOAT)) { ... }
// Equivalent to: if (TILING_KEY == some compile-time constant) { ... }
```

This lets the compiler eliminate mismatched branches at compile time and remove runtime branch overhead.

**Relationship between TILING_KEY and Tiling template priority:**

When a Tiling template is registered, it carries a priority value (for example, SplitR is 140, TwoPassPerf is 150, TwoPass is 200, and Welford is 4000). The framework tries matching in ascending priority order: it first checks whether SplitR (140) satisfies `IsCapable()`; if not, it tries TwoPassPerf (150), then TwoPass (200), and finally Welford (4000). This priority system and the TILING_KEY range values (400/300/500/800) are two independent numbering systems — the former controls the template matching order, and the latter is the algorithm identifier.

#### 0.3.4 regbase (Register Base Addressing Mode)

**regbase** (Register Base) is a **data addressing and layout mode** of the AI Core. When `isRegBase == true`, the kernel uses register-based relative addressing to access data. Compared with traditional immediate addressing, regbase mode:

- **Higher instruction density**: addresses are passed through registers, making instructions more compact
- **Supports vector fusion**: some CANN vector fusion instructions (such as `FusedMulDstAdd`) can be used only in regbase mode
- **Fewer UB access conflicts**: relaying through registers reduces the number of UB reads

`isRegBase` is a Boolean value computed on the Host side based on the input shape and data type, and it is passed to the Kernel side through tiling parameters. If the current shape does not meet the hardware requirements of regbase (such as alignment constraints), `isRegBase` is set to `false`; then none of the three algorithm branches is selected, and the operator falls back to the generic (non-regbase) path.

**Why do both Welford and TwoPass require isRegBase == true?** Because the parallel Welford and binary accumulation algorithms need frequent access to the M2 accumulator, temporary buffers, and other resources, and regbase mode ensures that these accesses happen in the most efficient way. Currently, no Welford/TwoPass downgrade path without regbase exists — if the regbase conditions are not met, the operator falls back to other LayerNorm implementations (such as V3 or the generic path).

#### 0.3.5 The `__aicore__` Modifier

`__aicore__` is a function attribute annotation provided by the CANN compiler to declare that the function runs on the **AI Core (the vector computation core of the Ascend NPU)**. After seeing this modifier, the compiler:

- Generates AI Core-specific instruction set code for the function (instead of ARM/x86 code for the Host CPU)
- Allows the function to use AI Core-specific intrinsics (vector instructions, matrix instructions, and so on)
- Compiles the function into the binary object of the operator kernel

#### 0.3.6 VL_F32 — Vector Processing Unit Width

| Property | Description |
|------|------|
| **Definition** | Vector Length for float32, that is, **the number of float32 elements that the vector processing unit (VPU) of the AI Core can process in a single pass**. |
| **Value** | On Ascend 950 (arch35), the value is **64**. That is, one SIMD instruction of the VPU processes 64 float32 values in parallel. |
| **Essence** | This is a **hardware constant** determined by the vector register width of the chip (64 × 32 bits = 2048 bits). VL_F32 may differ across architectures. |
| **Significance in this document** | VL_F32 = 64 is the base constant for all subsequent analysis, including tile size alignment (must be a multiple of 64), derivation of binary accumulation parameters, and IsCapable threshold calculation. |

`VL_F32² = 64² = 4096`, that is, the square of the number of float32 elements that the vector unit processes at full load in one pass. This value appears in the IsCapable condition of TwoPassPerf (R ≤ 2 × VL_F32² = 8192), because the size of the intermediate buffer of the binary merge tree of TwoPassPerf relates to VL_F32².

#### 0.3.7 double buffer (Double Buffering)

**double buffer** is a classic **pipeline optimization technique**: two buffers of the same size (Buffer A and Buffer B) are allocated in UB so that **data transfer** and **computation** can overlap:

```mermaid
Timeline →   [Load A] [Compute A + Load B] [Compute B + Load A] [Compute A +...] ...
              ↑        ↑              ↑
          A ready: transfer and computation in parallel, transfer and computation in parallel
```

In the context of LayerNormV4, double buffering is mainly used for **double buffering of input data tiles**: while the current tile is being computed by the VPU, the DMA engine moves the data of the next tile from Global Memory into the other buffer in UB in the background. This "hides" the DMA transfer latency so that the VPU almost never idles waiting for data.

---

### 0.4 Floating-Point Precision Basics

#### 0.4.1 Floating-Point Addition Is Not Associative

Due to the finite precision of floating-point numbers (float32 has about 7 significant decimal digits, FP16 about 3, and BF16 about 2), addition does not satisfy the mathematical associative law. That is, `(a + b) + c ≠ a + (b + c)` can hold under floating-point arithmetic.

**Concrete example:**

```Cpp
(1.0e10 + 1.0) - 1.0e10   →   0.0   (1.0 is "swallowed")
1.0e10 + (1.0 - 1.0e10)   →   0.0   (also lost)
1.0 + (1.0e10 - 1.0e10)   →   1.0   (correct)
```

When a large number and a small number are added, the low-order significant digits of the small number are drowned by rounding errors. In deep learning, this error is especially common in scenarios such as gradient accumulation and normalization statistics computation, because the magnitudes of element values within a batch often differ greatly.

#### 0.4.2 catastrophic cancellation (disastrous cancellation / cancellation error)

When two large numbers of similar magnitude are subtracted, the number of significant digits of the result drops sharply, and the relative error is extremely amplified. For example:

```Cpp
√(1.0000001 × 10^10) - √(1.0000000 × 10^10)
     ↑ true value is about 0.0005              ↑
However, because floating-point representation precision is limited, these two large numbers may be completely identical in floating-point registers, the subtraction result is directly 0, and the relative error is 100%.
```

Classic case: the naive variance formula `Var = E[X²] - (E[X])²`. When all x_i are very large (for example, all at the 10⁶ magnitude) and the variance is small, `E[X²]` and `(E[X])²` are two very close large numbers, and subtracting them directly encounters catastrophic cancellation, producing completely unreliable results. The Welford algorithm avoids this problem at the root through centering (always subtracting the current mean rather than the final mean).

---

## 1. Background

Layer Normalization is a fundamental operator in deep learning. Its core computation is:

```Cpp
mean = E[x]                                     // Mean
rstd = 1 / sqrt(Var[x] + eps)                  // Reciprocal standard deviation
y = gamma * ((x - mean) * rstd) + beta         // Normalization + affine transformation
```

Here **rstd** is the abbreviation of "reciprocal standard deviation", that is, the reciprocal of the standard deviation. Using the reciprocal of the standard deviation instead of the standard deviation itself converts one division in the subsequent normalization step into a multiplication (`(x - mean) * rstd` instead of `(x - mean) / std`), because multiplication has higher throughput than division on the AI Core.

On the Ascend 950 platform, the LayerNormV4 operator uses two core algorithms, **Parallel Welford** and **extra-long-axis binary accumulation**, to greatly improve computation precision while keeping high performance. This document analyzes the implementation principles of these two technologies at the source code level.

**Input tensor convention:** This document assumes that the input of LayerNorm has shape `[A, R]` (already flattened or with a specified normalization axis). Here R is the normalization axis length (mean/var are computed along this dimension), and A is the total number of elements outside the normalization axis (such as batch × seq_len).

---

## 2. Architecture

LayerNormV4 has four algorithm branches on the 950 (arch35) platform, dynamically selected at compile time through **TILING_KEY**:

| TILING_KEY | Algorithm | Template Priority | Applicable Scenario |
|-----------|------|--------|---------|
| 800-822 | SplitR cross-core segmented reduction | Highest (140, tried first) | Small A, large R: R≥65536 and A≤AIV cores/4; each core scans a contiguous segment, then exchanges via workspace, one whole-core synchronization, and a fixed-order merge |
| 400-422 | Parallel Welford | Lowest (4000, tried last) | General scenarios, single-pass scan |
| 300-322 | Two-Pass + Binary accumulation | Medium (200) | Extra-long axis scenarios (R is large and Welford does not satisfy the UB constraint) |
| 500-522 | Two-Pass Perf | Next (150) | Medium-length axes, extreme performance (additional requirements: rowAlign does not exceed the hard limit, and colSize/rowAlign does not trigger the joint threshold) |

**Dispatch logic** (entry code in `op_kernel/layer_norm_v4_apt.h:104-184`):

```cpp
__aicore__ inline void layer_norm_impl(...) {
    if (TILING_KEY_IS(LNV4_REGBASE_SPLIT_R_FLOAT_FLOAT)) {
        RegbaseSplitRImpl<float, float>(...);
    } else if (TILING_KEY_IS(LNV4_REGBASE_TWO_PASS_FLOAT_FLOAT)) {
        RegbaseTwoPassImpl<float, float, DTYPE_MEAN>(...);
    } else if (TILING_KEY_IS(LNV4_REGBASE_WELFORD_FLOAT_FLOAT)) {
        RegbaseWelfordImpl<float, float, DTYPE_MEAN>(...);
    } else if (TILING_KEY_IS(LNV4_REGBASE_TWO_PASS_PERF_FLOAT_FLOAT)) {
        RegbaseTwoPassPerfImpl<float, float, DTYPE_MEAN>(...);
    }
}
```

The `TILING_KEY_IS()` macro expands into a constant comparison at compile time. Only the matching one of the four branches is retained, and the others are removed during compile-time optimization, with no runtime overhead. For the naming rules of macro parameters, see [Prerequisite Knowledge 0.3.3](#033-tiling_key--algorithm-selection-key).

Welford and SplitR share buffer initialization, local statistics, normalization, and cache management in `LayerNormV4WelfordBase`. The whole-row Welford class controls row iteration. SplitR distributes one row across multiple cores and merges segment statistics. Unmatched SplitR shapes continue through Two-Pass Perf, Two-Pass, and Welford template selection. The selected key also encodes the supported input and parameter data types.

## 3. Parallel Welford Algorithm

### 3.1 Principle

The Welford algorithm [1] is an **online algorithm** that computes the mean and variance numerically stably in a **single-pass scan**, avoiding at the root the **catastrophic cancellation** caused by subtracting a small number from a large number in the naive two-pass algorithm (see [Prerequisite Knowledge 0.4.2](#042-catastrophic-cancellation-disastrous-cancellation--cancellation-error)).

#### 3.1.1 Derivation of the Welford Recurrence Formulas

The naive variance formula is:

```Cpp
σ² = Sum[(x_i - μ)²] / n      for i = 1 to n
```

But this formula requires two passes: the first pass computes μ, and the second pass computes σ². Moreover, if all x_i are at the 10⁶ magnitude while the variance is only 10⁻³, direct computation encounters catastrophic cancellation.

Welford's insight is to use recurrence-style updates. Let the mean and the second central moment of the first n-1 elements be μ_{n-1} and M_{2,n-1} respectively. When the n-th element x_n arrives:

**Step 1 Incrementally update the mean:**

```Cpp
μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
```

Define `delta = x_n - μ_{n-1}`; then `μ_n = μ_{n-1} + delta / n`.

**Step 2 Incrementally update the second central moment (M2):**

The second central moment is defined as `M_{2,n} = Sum[(x_i - μ_n)²]` (i=1 to n). Its recurrence is:

```Cpp
M_{2,n} = M_{2,n-1} + (x_n - μ_{n-1}) * (x_n - μ_n)
```

Substituting delta gives:

```Cpp
M_{2,n} = M_{2,n-1} + delta * (x_n - μ_n)
```

**Step 3 Normalize the variance:**

```Cpp
σ_n² = M_{2,n} / n
```

**Why does this recurrence avoid catastrophic cancellation?** The key is that every step **centers** the data: each update is based on the latest mean, subtracting the current mean (`x - μ_new`), ensuring that the two numbers participating in the multiplication use the latest center as the baseline and that their magnitudes stay controllable. In the traditional formula `E[X²] - (E[X])²`, the two large numbers are computed independently and then subtracted; once they are close, significant digits cancel out. Welford always centers locally, eliminating the need to subtract large numbers.

**Final update formulas (the form used in the code):**

```Cpp
delta = x - mean               // mean is the old mean μ_{n-1}
mean += delta / n              // Update to the new mean μ_n
M2 += delta * (x - mean_new)   // mean_new is the updated μ_n
variance = M2 / n              // Current variance
```

#### 3.1.2 Meaning of M2

**M2** is the abbreviation of "**Second Central Moment**". It is the **unnormalized variance**:

```Cpp
M2 = Sum[(x_i - μ_n)²] = n * σ²       for i = 1 to n
```

Dividing M2 by n gives the variance; taking the square root and then the reciprocal gives rstd. The core operation of Welford is to continuously accumulate and update M2 during streaming processing and finally normalize once to obtain the statistics.

---

### 3.2 Parallelization Design

The `LayerNormV4RegbaseWelford` class in `op_kernel/arch35/layer_norm_v4_welford.h` implements the complete parallel Welford algorithm.

**Core flow** (`Process()` method, lines 109-173):

```Cpp
For each row i (0 ~ currentBlockFactor):
    WelfordInitialize(mean, variance)     // Clear the mean accumulator and the M2 accumulator
    for each tile:
        ProcessWelfordUpdate(offset)       // Load the data tile and call the WelfordUpdate vector intrinsic to update element by element
    if tail > 0:
        ProcessWelfordUpdate(tail)          // Process the tail data shorter than one tile
    WelfordFinalize(mean, rstd)             // M2 / N → variance → square root and reciprocal → rstd
    for each tile:
        ProcessNormalize(offset)             // Normalize the raw data with mean/rstd + gamma/beta affine
```

#### 3.2.1 WelfordUpdate API Description

`WelfordUpdate` is a **vector intrinsic** provided by the CANN framework on the AI Core. It completes the Welford recurrence update for 64 float32 elements in one SIMD instruction. It is not a user-defined function but a compiler wrapper of a hardware instruction. Its logic is equivalent to:

```Cpp
for i in range(VL_F32):                    // VL_F32 = 64, SIMD parallel execution
    delta = data[i] - mean_acc
    mean_acc += delta / current_n
    M2_acc += delta * (data[i] - mean_acc)
```

The inputs are the data vector pointer plus the current mean/M2/n accumulators, and the output is the updated accumulators. This instruction uses the vector computing capability of the VPU and completes the parallel Welford update of 64 elements in one instruction.

**Overall algorithm complexity:** For each row, the data is loaded only once (single-pass scan), and the mean and M2 accumulation updates complete during loading; the normalization stage loads the data again to apply the (x - mean) * rstd + gamma/beta transformation. The total number of data accesses is 2 (1 for statistics + 1 for normalization).

**Tile division strategy** (`op_host/layer_norm_v4_welford_tiling.cpp`):

```Cpp
tileLength = the largest 64-aligned value that satisfies the UB space constraint
welfordUpdateTimes = N / tileLength
welfordUpdateTail = N % tileLength
```

The tile size is validated by the `IsValidTileLength()` function, ensuring that the total of **double buffer + gamma/beta weights + Welford temporary space** does not exceed the UB capacity (see [Prerequisite Knowledge 0.3.7](#037-double-buffer-double-buffering)).

**Space cost of Welford:** Compared with the naive algorithm, Welford needs to additionally store the M2 accumulator in UB (one float32 per row). When both R and A are large, the M2 accumulator array itself occupies considerable UB space. This is the main reason why Welford is unavailable under some extreme shapes.

---

### 3.3 Precision Advantage Analysis

Traditional single-pass computation uses the naive formula `Var = E[X²] - (E[X])²` and suffers precision loss from subtracting two large numbers. The Welford algorithm updates the mean and the centered second moment M2 element by element, always centering the data with the latest mean before updating, and avoids subtracting large numbers. Therefore:

- **FP16/BF16 input scenarios**: the precision of Welford is significantly better than the naive single-pass algorithm, because catastrophic cancellation has a more severe impact in low-precision formats (FP16 has only about 3 significant decimal digits, and BF16 about 2)
- **Large data volume scenarios**: the larger N is, the more obvious the advantage of Welford, because in the traditional method both E[X²] and (E[X])² grow with N, and the precision loss from the subtraction is amplified accordingly

---

## 4. Extra-Long-Axis Binary Accumulation Algorithm

### 4.1 Problem Definition

When the normalization axis (the R dimension, that is, the length of the normalization axis) exceeds the vector processing unit width (VL_F32 = 64 float32 values), relying solely on `ReduceSum` for linear accumulation over the long sequence introduces large **rounding errors**. The reasons are:

1. **Floating-point addition is not associative**, and the accumulation order directly affects result precision (see [Prerequisite Knowledge 0.4.1](#041-floating-point-addition-is-not-associative))
2. Under linear accumulation over a long sequence, the partial sum accumulated early is already a "large number", and subsequent elements are relatively "small numbers". Adding a large number and a small number causes the low-order significant digits of the small number to be rounded away (**the "large number swallows small number" phenomenon**)

**Concrete example:** Linearly accumulate 10,000 float32 elements. The sum of the first 5000 elements is about 10⁶, and the 5001-th element is 0.001. In float32 precision, 10⁶ + 0.001 = 10⁶ (the 7 significant digits of 0.001 cannot catch up with the magnitude of 10⁶ and are directly rounded to zero). The cumulative rounding error over 10,000 elements is not negligible.

**Why cannot Welford fully replace binary accumulation?** Although Welford handles this problem elegantly with high precision (because it updates element by element rather than accumulating first and subtracting later), Welford needs to maintain an M2 accumulator in UB for each row. When R is extremely long (for example, R = 10⁵), the number of M2 accumulators maintained by Welford equals the number of rows and exceeds the UB capacity. At this point, TwoPass + binary accumulation becomes the only choice — it needs two passes (slightly lower performance), but its intermediate state is more compact and fits in UB.

---

### 4.2 Principle of Binary Accumulation

The `CalculateMeanVarVF()` method in `op_kernel/arch35/layer_norm_v4_two_pass.h` implements **binary accumulation (Binary Accumulation Tree)** on the extra-long axis.

**Core idea:** Split the R-axis data into the **quotient (power-of-two part)** and the **remainder (remaining part)**, first compute the accumulation sum of each sub-block separately, and then merge them level by level in pairs through a **Binary Merge Tree**:

**Algorithm steps:**

```tex
Input: R elements
Step 1: Define two split parameters
    binaryAddQuotient: the largest power-of-two block size within R aligned to VL_F32
        (computed in IsCapable(), starting from vlFp32=64, repeatedly multiplying by 2 until ≥ R, then dividing by 2)
    powerOfTwoForR: the smallest power of two greater than or equal to R
        (computed separately in DoOpTiling(), starting from 1, repeatedly multiplying by 2 until ≥ R)

Step 2: Split at binaryAddQuotient
    left_part  = x[0 : binaryAddQuotient - 1]    // Power-of-two part
    right_part = x[binaryAddQuotient : R - 1]    // Remaining part (not a power of two)

Step 3: Obtain the sub-block accumulation sums of the left and right parts through ReduceSum
        and store them into the temporary buffer tmpBuf (in UB)

Step 4: Binary merge tree (lines 548-558 of the code)
    Let level 0 have binaryAddNum accumulation sums
    for (i = 0; i < binaryAddK; i++) {         // binaryAddK: number of merge levels
        curNum /= 2;
        for (j = 0; j < curNum; j++) {
            tmp[j] += tmp[j + curNum];          // Add adjacent pairs
        }
    }
    // After merging, tmp[0] is the final total sum
```

**Why is binary accumulation more precise?** At each level of the binary merge tree, the two numbers being added have similar magnitudes (because they come from sub-blocks of equal length), avoiding "large number swallows small number". Information propagates from leaves to the root over O(log n) levels, so the rounding error grows only with log n (instead of the O(n) growth of linear accumulation).

**Diagram (using 8 elements as an example):**

```mermaid
Level 0 (leaves):  a0  a1  a2  a3  a4  a5  a6  a7
                  \  /    \  /    \  /    \  /
Level 1:         s0+s1   s2+s3   s4+s5   s6+s7     ← similar magnitudes in pairs
                    \      /         \      /
Level 2:          (s0..s3)         (s4..s7)        ← still similar
                          \            /
Level 3:              final sum (s0..s7)
```

**Space and performance cost of TwoPass:** Two passes mean the data is moved from Global Memory into UB at least 3 times (the first pass computes the mean, the second pass computes the variance, and normalization takes 1 more time), about 50% more bandwidth overhead than the 2 times of Welford. Its additional bandwidth cost is one tradeoff considered by template admission and selection. But its advantage is an **extremely compact intermediate state** — it only needs to store the temporary buffer of the binary accumulation tree and does not depend on per-row M2 accumulators.

---

### 4.3 Parameter Calculation

binaryAddQuotient and powerOfTwoForR are two **independently computed** parameters, generated at different places on the Host side:

**binaryAddQuotient is computed in `IsCapable()` (about lines 63-67):**

Starting from `vlFp32` (VL_F32 = 64), multiply by 2 each time until the value is not less than R, then divide by 2, obtaining the largest VL_F32-aligned power-of-two block size within R:

```cpp
binaryAddQuotient = commonParams.vlFp32;   // Start from 64
while (binaryAddQuotient < r) {
    binaryAddQuotient *= 2;                // Double each time: 64→128→256→512→1024...
}
binaryAddQuotient /= 2;                    // Step back one level to ensure < r
// For example, R=1000: 64→128→256→512→1024 (stop when >= 1000) → 512 (=1024/2)
```

**powerOfTwoForR is computed separately in `DoOpTiling()` (about lines 131-135):**

Starting from 1, multiply by 2 each time until the value is not less than R, obtaining the smallest power of two greater than or equal to R:

```cpp
int64_t powerOfTwoForR = 1;
while (powerOfTwoForR < r) {
    powerOfTwoForR *= 2;                  // For example, R=1000 → powerOfTwoForR=1024
}
```

**Compute the number of binary merge levels binaryAddK (about lines 136-142):**

Use VL_F32 = 64 as the basic block size:

```cpp
vcaddNum = binaryAddQuotient / VL_F32;       // Number of blocks, for example 512/64 = 8
binaryAddNum = vcaddNum;                     // Initial number of merge blocks
curBinaryAddNum = 1;                         // Start from 1 and double level by level
while (curBinaryAddNum < binaryAddNum) {
    binaryAddK++;                              // Count the number of merge levels
    curBinaryAddNum *= 2;                      // Double at each level: 1→2→4→8...
}
// For example, binaryAddNum=8, then binaryAddK=3 (curBinaryAddNum goes 1→2→4→8, 3 doublings until >= 8)
```

`vcaddNum` is the number of vector accumulation blocks (each block of VL_F32=64 is one accumulation unit), and it is assigned to `binaryAddNum` as the number of leaf nodes of the merge tree. `curBinaryAddNum` starts from 1 and simulates the growth of the node count of the merge tree (1 → 2 → 4 → ... → binaryAddNum); each doubling corresponds to one merge level, until the value >= binaryAddNum.

---

### 4.4 Precision Advantage Analysis

Compared with naive linear accumulation, binary accumulation converts the O(n) chain of additions into multi-level tree-shaped addition of O(log n):

**Linear accumulation (poor precision):**

```Cpp
sum = ((...(a0 + a1) + a2) + ...) + a_n
```

The error grows linearly with N. Large number swallows small number may happen at every step.

**Binary accumulation (good precision):**

```Cpp
level 0: a0+a1, a2+a3, a4+a5, ...        // Accumulation within sub-blocks
level 1: (a0+a1)+(a2+a3), ...             // Merging between sub-blocks
...
level k: final sum
```

At each level of addition, the operands have similar magnitudes, avoiding large number swallows small number. The error grows only with log N.

(Binary accumulation reduces the error growth of linear accumulation from O(n) to O(log n). This conclusion comes from the standard analysis of rounding errors of floating-point addition; see Chapter 4 of the Higham textbook in Further Reading.)

---

### 4.5 Fusion Optimizations

`CalculateMeanVarRCommon<1/2>()` in `op_kernel/arch35/layer_norm_v4_two_pass_perf.h` stacks multiple extreme performance optimizations on top of binary accumulation:

**1. Save (x - mean) while computing the mean:**
During the first-pass scan that computes the mean, the result of subtracting the mean from each element is synchronously stored into the `xSubMeanUb` temporary buffer. In this way, the second pass that computes the variance does not need to reload the raw data and decode it, and directly uses the saved differences, **saving one complete Global Memory data transfer**, effectively reducing the number of data loads from 3 to 2.

**2. Reuse of variance computation:**
The second pass directly uses the (x - mean) differences saved in the first pass to compute the sum of squares (that is, the variance), avoiding the subtract-mean operation on the raw data again.

**3. Row pairing + FusedMulDstAdd fused instruction:**
`CalculateNormalizeVF` processes **two rows** each time (row pairing) and uses the `FusedMulDstAdd` instruction. `FusedMulDstAdd` is a CANN **vector fusion instruction** that completes the following in a single instruction cycle:

```Cpp
dst = src * gamma + beta
```

That is, it fuses the `* gamma + beta` two steps of `(x - mean) * rstd * gamma + beta` into one instruction, reducing the number of instruction issues and register usage.

**4. Multi-level reduction (lastLoopNums template parameter):**
The `LAST_LOOP_NUMS` template parameter controls the merge strategy of the **last level** of the binary accumulation tree:

- `LAST_LOOP_NUMS = 1`: the last level completes the reduction with a single `ReduceSum` instruction (simple and direct)
- `LAST_LOOP_NUMS = 2`: at the last level, `ShiftLefts` first shifts the elements left (aligning the exponent bits of the accumulators), and then `Add` completes the reduction. `ShiftLefts` is a vector shift instruction. In this context, it aligns the exponents of multiple accumulators before addition, reducing rounding errors caused by mismatched exponents. This method achieves higher accumulation precision when the R dimension is long enough, at the cost of a few extra instructions.

---

## 5. Algorithm Selection

### 5.1 Priority and SplitR admission

Templates are attempted in this order:

1. SplitR: priority 140.
2. Two-Pass Perf: priority 150.
3. Two-Pass: priority 200.
4. Welford: priority 4000.

SplitR uses the shared `isRegBase` platform check and excludes the V3 path. It does not add a deterministic-level restriction. Its shape conditions are:

```text
1 <= A <= floor(AIV_core_count / 4)
65536 <= R <= INT32_MAX
splitR = min(16, floor(AIV_core_count / A), floor(R / 16384))
splitR >= 2
```

For 64 AIV cores, A must be at most 16. Segment boundaries are aligned, and the last segment uses its actual element count.

Each core computes a segment mean and population variance and publishes them into separate workspace planes. Each GM record occupies one UB block. After `SyncAll`, `DataCopyPad<float, PaddingMode::Compact>` gathers the first float of each record into adjacent UB lanes. Each of the two planes is rounded up to a whole number of vectors; the factor two represents mean and variance, independently of double buffering.

At most sixteen segment statistics fit in one vector. The merge builds weighted contributions in vector-sized chunks and invokes the common `NormCommon::NormCommonRegbase::CalculateReduceSum` interface:

```text
weight_s = count_s / R
mean = mean_0 + sum((mean_s - mean_0) * weight_s)
variance = sum(variance_s * weight_s + (mean_s - mean)^2 * weight_s)
```

Centering the mean around the first segment retains the large-offset precision fix. Normalizing counts before multiplying variance avoids the intermediate `variance_s * count_s` overflow. The squared mean-difference term remains subject to FP32 range limits. No general multi-level BatchNorm reduction scratch is required.

Each core normalizes its own input segment. Only the first segment writes mean/rstd. Workspace includes the system synchronization allocation plus the two statistic planes. Host UB budgeting reserves both vector-aligned compact planes before selecting tile length.

### 5.2 Detailed IsCapable() Check Conditions

Each template has an `IsCapable()` method that determines at runtime on the Host side whether the template is available for the actual shape. The following table lists the Capable conditions of the original three algorithms; SplitR admission is described in 5.1:

| Algorithm | IsCapable Condition | Condition Explanation and Design Intent |
|------|---------------|------------------|
| **Welford** | `isRegBase == true` | regbase is a necessary condition for the Welford parallel algorithm to efficiently access the M2 accumulator (see [Prerequisite Knowledge 0.3.4](#034-regbase-register-base-addressing-mode)). If not satisfied, fall back |
| **TwoPass** | `isRegBase == true`<br>+ UB capacity constraint satisfied | Besides regbase, it must also ensure that the tile size + the temporary buffer of the binary accumulation tree fit in UB. If R is so large that the buffer overflows, it is unavailable |
| **TwoPassPerf** | `isRegBase == true`<br>+ rowAlign does not exceed the hard limit<br>+ colSize/rowAlign does not trigger the joint threshold<br>+ UB capacity constraint satisfied | See the item-by-item explanation below |

**Explanation of the additional conditions of TwoPassPerf:**

- **rowAlign does not exceed the hard limit**: the first geometric judgment of `IsCapable()` (`op_host/layer_norm_v4_regbase_two_pass_perf_tiling.cpp:70-73`) returns false when `rowAlign > LN_NUM_TWO * vlFp32 * vlFp32 * LN_NUM_TWO`, that is, the magnitude of about `4 × VL_F32²` (`LN_NUM_TWO = 2`, substituting gives `2 × 64 × 64 × 2 = 16384`). This means rowAlign has a hard upper limit related to `VL_F32²`; beyond it, the template is unavailable.

- **colSize/rowAlign does not trigger the joint threshold**: the second judgment (lines 76-78) returns false when `colSize >= LN_ROW_THRESHOLD && rowAlign >= LN_COL_THRESHOLD`. The two thresholds are static constants (= 4096 and 8192), corresponding to the upper limits of colSize and rowAlign respectively. Both conditions must be satisfied simultaneously to trigger rejection — this avoids misjudging unavailability when only one dimension is too large while the other is small.

- **UB capacity constraint satisfied**: `CanFitInBuffer(1)` validates that the total of the tile size + the (x-mean) difference buffer + other temporary space does not exceed the UB capacity.

**Logical relationship between the conditions (from strict to relaxed):**
```Cpp
Welford IsCapable condition:     isRegBase == true                                    ← most relaxed
TwoPass IsCapable condition:     isRegBase + UB constraint                            ← medium
TwoPassPerf IsCapable condition: isRegBase + rowAlign hard limit + joint threshold + UB constraint ← strictest
```

The strictness order above does not represent the matching order. The framework checks in the order SplitR → TwoPassPerf → TwoPass → Welford, and the first successful template generates the tiling.

### 5.3 Three-Way Trade-Off Analysis: Precision, Performance, and Applicability

No algorithm can be optimal in all three dimensions at the same time. Understanding what each algorithm "**sacrifices**" is as important as understanding what it "**gains**":

| Algorithm | What It Pursues | What It Sacrifices | Consequence of Mismatch |
|------|---------|-----------|-------------|
| **Welford** | Precision + single-pass performance | More UB space (M2 accumulator array) | UB overflow when R is too large; unavailable |
| **TwoPass** | Adaptability to extra-long axes | One extra global data transfer (3 times vs 2 times) | Obvious performance loss when R is small; no precision advantage |
| **TwoPassPerf** | Extreme performance + excellent precision | rowAlign and colSize/rowAlign are limited by the joint threshold | Unavailable when the shape exceeds the bounds; falls back to a slower branch |

**Design philosophy:** The three original algorithms do not replace each other; instead, they form a covering partition of the "**shape space**". The framework configures common cases optimally through priorities (140→150→200→4000), and IsCapable conditions guarantee a fallback in extreme cases. SplitR increases core utilization for small A and large R, adding workspace traffic and global synchronization. Shapes outside its threshold use the original templates.

For readers from the software field, this "multiple algorithms covering the shape space" pattern is similar to:

- **Multi-level optimization passes of compilers** — different optimization levels suit different code characteristics
- **Multi-plan candidates of database query optimizers** — the cost model selects the best execution plan
- **Kernel selection in BLAS libraries** — MKL/OpenBLAS dynamically switch kernels based on matrix size

## 6. Source Map

| File | Responsibility |
| --- | --- |
| `op_host/layer_norm_v4_split_r_tiling.cpp` | SplitR admission, UB budget, workspace and scheduling |
| `op_kernel/arch35/layer_norm_v4_split_r.h` | Segmentation, publication, compact gather, merge and normalization |
| `op_kernel/arch35/layer_norm_v4_welford_base.h` | Shared Welford buffers and computation |
| `op_kernel/arch35/layer_norm_v4_welford.h` | Whole-row Welford execution |
| `op_host/layer_norm_v4_welford_tiling.cpp` | Welford tile selection |
| `op_kernel/arch35/layer_norm_v4_two_pass.h` | Two-Pass computation |
| `op_kernel/arch35/layer_norm_v4_two_pass_perf.h` | Optimized Two-Pass computation |
| `op_host/layer_norm_v4_regbase_two_pass_tiling.cpp` | Two-Pass tiling |
| `op_host/layer_norm_v4_regbase_two_pass_perf_tiling.cpp` | Two-Pass Perf tiling |
| `op_kernel/layer_norm_v4_apt.h` | Kernel dispatch |
| `op_host/layer_norm_v4_tiling.h` | Tiling data definitions |

## 7. Summary

Centered statistics, tree accumulation, and shape-based template selection address different numerical and performance constraints. SplitR extends this selection to small-A/large-R inputs while reusing the local Welford and normalization paths. Changes to reduction order and memory layout require precision and bounds regression on the resulting binary.

---

### References

[1] B. P. Welford, "Note on a Method for Calculating Corrected Sums of Squares and Products," *Technometrics*, vol. 4, no. 3, pp. 419–420, 1962.

### Further Reading

- **IEEE 754 floating-point arithmetic standard**: the authoritative definition of floating-point representation formats, rounding modes, and special values (NaN/Inf). It is the basis for understanding all precision discussions in this document.
- **CANN Operator Development Guide** (official documentation of the Ascend community): covers the operator development process, the Tiling mechanism, UB management, debugging methods, and so on.
- **Higham, N. J. "Accuracy and Stability of Numerical Algorithms" (2nd ed.)**: a classic textbook on error analysis of floating-point algorithms. Chapters 1 and 4 give detailed analyses of precision problems in summation and variance computation.
- **Chan, T. F. et al. "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances" (1979)**: a comparative study of the Welford recurrence formula and the pairwise variance computation algorithm; it is the theoretical predecessor of the TwoPass binary accumulation idea in this document.
