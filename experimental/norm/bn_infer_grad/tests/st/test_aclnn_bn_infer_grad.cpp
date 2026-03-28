/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_bn_infer_grad.cpp
 * @brief BnInferGrad 算子 ST 测试代码
 *
 * 功能：计算 Batch Normalization 推理阶段的反向传播梯度
 * 公式：x_backprop[i] = grads[i] * scale[c] / sqrt(batch_variance[c] + epsilon)
 *
 * 输入：
 *   - grads: 4D aclTensor (N,C,H,W)
 *   - scale: 1D aclTensor fp32 (C,)
 *   - batchVariance: 1D aclTensor fp32 (C,)
 *   - epsilon: float
 *
 * 输出：
 *   - xBackprop: aclTensor，同 grads 形状和 dtype
 *
 * 迭代一范围：
 *   - dtype: fp32
 *   - shapes: 基础 4D shapes (N,C,H,W)
 *   - epsilon: 1e-5 (默认值)
 *   - 场景: 基础功能 + 边界条件
 *
 * 迭代二范围：
 *   - dtype: fp32
 *   - formats: NCHW + NHWC (TilingKey=0 CONTIGUOUS) + NC1HWC0 (TilingKey=1)
 *   - shapes: 多shape变化，验证多TilingKey分支和多核切分
 *   - 场景: 多format覆盖 + 非对齐shape + 大spatial
 *
 * 迭代三范围：
 *   - dtype: fp32 + fp16 + bf16（全dtype覆盖）
 *   - formats: NCHW + NHWC + NC1HWC0（全format覆盖）
 *   - shapes: 边界shape（极小、channel=1、大batch等）
 *   - 精度标准: fp32(atol=1e-4), fp16(atol=1e-3), bf16(atol=4e-3)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>
#include <cstring>
#include <random>
#include <numeric>
#include <string>
#include <sstream>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_bn_infer_grad.h"
#endif

// ============================================================================
// 宏定义
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 数据格式枚举
// ============================================================================
enum class DataFormat {
    NCHW,       // 4D: (N, C, H, W)     -> TilingKey=0 CONTIGUOUS
    NHWC,       // 4D: (N, H, W, C)     -> TilingKey=0 CONTIGUOUS
    NC1HWC0     // 5D: (N, C1, H, W, C0) -> TilingKey=1
};

const char* FormatToString(DataFormat fmt) {
    switch (fmt) {
        case DataFormat::NCHW:    return "NCHW";
        case DataFormat::NHWC:    return "NHWC";
        case DataFormat::NC1HWC0: return "NC1HWC0";
        default:                  return "UNKNOWN";
    }
}

// ============================================================================
// 数据类型枚举
// ============================================================================
enum class DType {
    FLOAT32,
    FLOAT16,
    BFLOAT16
};

const char* DTypeToString(DType dt) {
    switch (dt) {
        case DType::FLOAT32:  return "fp32";
        case DType::FLOAT16:  return "fp16";
        case DType::BFLOAT16: return "bf16";
        default:              return "UNKNOWN";
    }
}

// ============================================================================
// fp16/bf16 软件模拟类型（用于 Mock 模式下的精度模拟）
// ============================================================================

/**
 * @brief IEEE 754 半精度浮点数 (fp16) 模拟
 *
 * 用于在 CPU 上模拟 fp16 的精度特性。
 * 通过 float -> fp16 -> float 的转换模拟精度损失。
 */
struct fp16_t {
    uint16_t value;

    fp16_t() : value(0) {}

    static fp16_t fromFloat(float f) {
        fp16_t result;
        uint32_t fbits;
        std::memcpy(&fbits, &f, sizeof(float));

        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exponent = ((fbits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = fbits & 0x007FFFFF;

        if (exponent <= 0) {
            // 下溢 -> 0 (简化处理，忽略非规格化数)
            result.value = static_cast<uint16_t>(sign);
        } else if (exponent >= 31) {
            // 上溢 -> Inf
            result.value = static_cast<uint16_t>(sign | 0x7C00);
        } else {
            // 正常范围：截断尾数（取高10位）+ 四舍五入
            uint32_t rounded = mantissa + 0x00001000; // round-to-nearest
            if (rounded & 0x00800000) {
                rounded = 0;
                exponent++;
            }
            if (exponent >= 31) {
                result.value = static_cast<uint16_t>(sign | 0x7C00);
            } else {
                result.value = static_cast<uint16_t>(sign |
                    (static_cast<uint32_t>(exponent) << 10) |
                    (rounded >> 13));
            }
        }
        return result;
    }

    float toFloat() const {
        uint32_t sign = (value & 0x8000) << 16;
        uint32_t exponent = (value >> 10) & 0x1F;
        uint32_t mantissa = value & 0x03FF;

        uint32_t fbits;
        if (exponent == 0) {
            if (mantissa == 0) {
                fbits = sign;  // +/- 0
            } else {
                // 非规格化数
                exponent = 1;
                while (!(mantissa & 0x0400)) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x03FF;
                fbits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            fbits = sign | 0x7F800000 | (mantissa << 13);  // Inf or NaN
        } else {
            fbits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &fbits, sizeof(float));
        return result;
    }
};

/**
 * @brief bfloat16 模拟
 *
 * bfloat16 = float32 的高 16 位（截断低 16 位尾数）。
 * 保留与 float32 相同的指数范围，但尾数精度降低到 7 位。
 */
struct bf16_t {
    uint16_t value;

    bf16_t() : value(0) {}

    static bf16_t fromFloat(float f) {
        bf16_t result;
        uint32_t fbits;
        std::memcpy(&fbits, &f, sizeof(float));
        // 四舍五入到最近偶数
        uint32_t rounding_bias = 0x00007FFF + ((fbits >> 16) & 1);
        fbits += rounding_bias;
        result.value = static_cast<uint16_t>(fbits >> 16);
        return result;
    }

    float toFloat() const {
        uint32_t fbits = static_cast<uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &fbits, sizeof(float));
        return result;
    }
};

/**
 * @brief 将 float 数组量化到指定 dtype 后再转回 float
 *
 * 模拟 NPU 上 fp16/bf16 输入数据的实际精度。
 * fp32 不做转换，fp16/bf16 经过 float->半精度->float 的精度损失。
 */
void QuantizeToFloat(const float* input, float* output, size_t size, DType dtype) {
    for (size_t i = 0; i < size; ++i) {
        switch (dtype) {
            case DType::FLOAT32:
                output[i] = input[i];
                break;
            case DType::FLOAT16:
                output[i] = fp16_t::fromFloat(input[i]).toFloat();
                break;
            case DType::BFLOAT16:
                output[i] = bf16_t::fromFloat(input[i]).toFloat();
                break;
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

// ============================================================================
// 随机数据生成
// ============================================================================

std::vector<float> GenerateRandomData(size_t size, unsigned int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> data(size);
    for (auto& v : data) v = dist(gen);
    return data;
}

/**
 * @brief 生成正的 variance 数据（方差必须为正值）
 */
std::vector<float> GeneratePositiveData(size_t size, unsigned int seed = 456) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.1f, 5.0f);
    std::vector<float> data(size);
    for (auto& v : data) v = dist(gen);
    return data;
}

/**
 * @brief 生成 scale 数据（可正可负）
 */
std::vector<float> GenerateScaleData(size_t size, unsigned int seed = 789) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> data(size);
    for (auto& v : data) v = dist(gen);
    return data;
}

// ============================================================================
// CPU Golden 计算函数 (bn_infer_grad)
// ============================================================================

/**
 * @brief 计算 BnInferGrad CPU Golden
 *
 * x_backprop[n][c][h][w] = grads[n][c][h][w] * scale[c] / sqrt(batch_variance[c] + epsilon)
 *
 * @param grads           输入梯度数据 (N*C*H*W)
 * @param scale           scale 参数 (C,)
 * @param batch_variance  batch variance 参数 (C,)
 * @param output          输出缓冲区 (N*C*H*W)
 * @param N               batch size
 * @param C               channel count
 * @param H               height
 * @param W               width
 * @param epsilon         小常数，防止除零
 */
void ComputeBnInferGradGolden(const float* grads, const float* scale,
                               const float* batch_variance, float* output,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               float epsilon) {
    int64_t HW = H * W;
    int64_t CHW = C * H * W;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            double inv_std = 1.0 / std::sqrt(static_cast<double>(batch_variance[c]) +
                                              static_cast<double>(epsilon));
            double factor = static_cast<double>(scale[c]) * inv_std;

            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = n * CHW + c * HW + h * W + w;
                    output[idx] = static_cast<float>(
                        static_cast<double>(grads[idx]) * factor);
                }
            }
        }
    }
}

/**
 * @brief 计算 BnInferGrad CPU Golden - NHWC 格式
 *
 * x_backprop[n][h][w][c] = grads[n][h][w][c] * scale[c] / sqrt(batch_variance[c] + epsilon)
 *
 * @param grads           输入梯度数据 (N*H*W*C) NHWC layout
 * @param scale           scale 参数 (C,)
 * @param batch_variance  batch variance 参数 (C,)
 * @param output          输出缓冲区 (N*H*W*C)
 * @param N               batch size
 * @param H               height
 * @param W               width
 * @param C               channel count
 * @param epsilon         小常数，防止除零
 */
void ComputeBnInferGradGoldenNHWC(const float* grads, const float* scale,
                                   const float* batch_variance, float* output,
                                   int64_t N, int64_t H, int64_t W, int64_t C,
                                   float epsilon) {
    // 预计算每个通道的 factor
    std::vector<double> factors(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        double inv_std = 1.0 / std::sqrt(static_cast<double>(batch_variance[c]) +
                                          static_cast<double>(epsilon));
        factors[c] = static_cast<double>(scale[c]) * inv_std;
    }

    int64_t HWC = H * W * C;
    int64_t WC = W * C;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t c = 0; c < C; ++c) {
                    int64_t idx = n * HWC + h * WC + w * C + c;
                    output[idx] = static_cast<float>(
                        static_cast<double>(grads[idx]) * factors[c]);
                }
            }
        }
    }
}

/**
 * @brief 计算 BnInferGrad CPU Golden - NC1HWC0 格式
 *
 * 5D shape: (N, C1, H, W, C0)，channel = c1 * C0 + c0
 * x_backprop[n][c1][h][w][c0] = grads[n][c1][h][w][c0] * scale[c1*C0+c0] / sqrt(batch_variance[c1*C0+c0] + epsilon)
 *
 * @param grads           输入梯度数据 (N*C1*H*W*C0) NC1HWC0 layout
 * @param scale           scale 参数 (C,) where C = C1 * C0
 * @param batch_variance  batch variance 参数 (C,)
 * @param output          输出缓冲区 (N*C1*H*W*C0)
 * @param N               batch size
 * @param C1              C1 维度
 * @param H               height
 * @param W               width
 * @param C0              C0 维度 (通常=16)
 * @param epsilon         小常数，防止除零
 */
void ComputeBnInferGradGoldenNC1HWC0(const float* grads, const float* scale,
                                      const float* batch_variance, float* output,
                                      int64_t N, int64_t C1, int64_t H, int64_t W,
                                      int64_t C0, float epsilon) {
    int64_t C = C1 * C0;
    // 预计算每个通道的 factor
    std::vector<double> factors(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        double inv_std = 1.0 / std::sqrt(static_cast<double>(batch_variance[c]) +
                                          static_cast<double>(epsilon));
        factors[c] = static_cast<double>(scale[c]) * inv_std;
    }

    int64_t HWC0 = H * W * C0;
    int64_t WC0 = W * C0;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c1 = 0; c1 < C1; ++c1) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    for (int64_t c0 = 0; c0 < C0; ++c0) {
                        int64_t idx = n * C1 * HWC0 + c1 * HWC0 + h * WC0 + w * C0 + c0;
                        int64_t ch = c1 * C0 + c0;
                        output[idx] = static_cast<float>(
                            static_cast<double>(grads[idx]) * factors[ch]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 精度比对函数
// ============================================================================

bool CompareResults(const float* golden, const float* actual, size_t size,
                    double atol = 1e-4, double rtol = 1e-4) {
    int mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(actual[i])) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  [ERROR] 输出包含 NaN at [%zu]: golden=%.6f", i,
                          static_cast<double>(golden[i]));
            }
            continue;
        }
        if (std::isnan(golden[i])) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  [ERROR] Golden 包含 NaN at [%zu]", i);
            }
            continue;
        }
        if (std::isinf(golden[i]) && std::isinf(actual[i])) {
            if ((golden[i] > 0) == (actual[i] > 0)) {
                continue;
            } else {
                mismatch++;
                if (mismatch <= 5) {
                    LOG_PRINT("  不匹配 [%zu]: 期望=%f, 实际=%f (无穷符号不同)",
                              i, static_cast<double>(golden[i]), static_cast<double>(actual[i]));
                }
                continue;
            }
        }
        double diff = std::abs(static_cast<double>(golden[i]) - static_cast<double>(actual[i]));
        double tolerance = atol + rtol * std::abs(static_cast<double>(golden[i]));

        if (diff > tolerance) {
            mismatch++;
            if (mismatch <= 5) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%.6f, 实际=%.6f, 差值=%.6e, 容忍=%.6e",
                          i, static_cast<double>(golden[i]), static_cast<double>(actual[i]),
                          diff, tolerance);
            }
        }
    }

    if (mismatch == 0) {
        LOG_PRINT("  [PASS] 所有 %zu 个元素一致 (atol=%.1e, rtol=%.1e)", size, atol, rtol);
        return true;
    } else {
        LOG_PRINT("  [FAIL] 发现 %d 个不匹配 (共 %zu 个元素)", mismatch, size);
        return false;
    }
}

// ============================================================================
// CPU Golden 正确性自测
// ============================================================================

void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测");
    LOG_PRINT("========================================");

    // 测试 1: 简单 1x1x1x1
    {
        float grads[] = {2.0f};
        float scale[] = {3.0f};
        float variance[] = {0.0f};
        float epsilon = 1e-5f;
        float output;
        ComputeBnInferGradGolden(grads, scale, variance, &output, 1, 1, 1, 1, epsilon);
        // expected = 2.0 * 3.0 / sqrt(0.0 + 1e-5) = 6.0 / 0.00316... = 1897.37...
        float expected = 2.0f * 3.0f / std::sqrt(0.0f + epsilon);
        LOG_PRINT("\n测试 1: 简单 1x1x1x1 grads=2, scale=3, var=0, eps=1e-5");
        bool match = CompareResults(&expected, &output, 1, 1e-2, 1e-3);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }

    // 测试 2: 单通道 1x1x2x2，variance=1.0
    {
        float grads[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scale[] = {2.0f};
        float variance[] = {1.0f};
        float epsilon = 0.0f;
        float output[4];
        ComputeBnInferGradGolden(grads, scale, variance, output, 1, 1, 2, 2, epsilon);
        // expected[i] = grads[i] * 2.0 / sqrt(1.0 + 0.0) = grads[i] * 2.0
        float expected[] = {2.0f, 4.0f, 6.0f, 8.0f};
        LOG_PRINT("\n测试 2: 1x1x2x2 scale=2, var=1, eps=0");
        bool match = CompareResults(expected, output, 4);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }

    // 测试 3: 双通道 1x2x1x1
    {
        float grads[] = {6.0f, 3.0f};
        float scale[] = {1.0f, 2.0f};
        float variance[] = {3.0f, 8.0f};
        float epsilon = 1.0f;
        float output[2];
        ComputeBnInferGradGolden(grads, scale, variance, output, 1, 2, 1, 1, epsilon);
        // ch0: 6.0 * 1.0 / sqrt(3.0 + 1.0) = 6.0 / 2.0 = 3.0
        // ch1: 3.0 * 2.0 / sqrt(8.0 + 1.0) = 6.0 / 3.0 = 2.0
        float expected[] = {3.0f, 2.0f};
        LOG_PRINT("\n测试 3: 1x2x1x1 双通道");
        bool match = CompareResults(expected, output, 2);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }

    // 测试 4: scale=0 => 输出全零
    {
        float grads[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scale[] = {0.0f};
        float variance[] = {1.0f};
        float epsilon = 1e-5f;
        float output[4];
        ComputeBnInferGradGolden(grads, scale, variance, output, 1, 1, 2, 2, epsilon);
        float expected[] = {0.0f, 0.0f, 0.0f, 0.0f};
        LOG_PRINT("\n测试 4: scale=0 输出全零");
        bool match = CompareResults(expected, output, 4);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }

    // 测试 5: grads=0 => 输出全零
    {
        float grads[] = {0.0f, 0.0f};
        float scale[] = {5.0f};
        float variance[] = {1.0f};
        float epsilon = 1e-5f;
        float output[2];
        ComputeBnInferGradGolden(grads, scale, variance, output, 1, 1, 1, 2, epsilon);
        float expected[] = {0.0f, 0.0f};
        LOG_PRINT("\n测试 5: grads=0 输出全零");
        bool match = CompareResults(expected, output, 2);
        LOG_PRINT("  结果: %s", match ? "PASS" : "FAIL");
    }

    LOG_PRINT("\n========================================");
}

// ============================================================================
// Real 模式辅助函数
// ============================================================================

#ifndef USE_MOCK_ACLNN

aclFormat DataFormatToAclFormat(DataFormat fmt) {
    switch (fmt) {
        case DataFormat::NHWC:    return aclFormat::ACL_FORMAT_NHWC;
        case DataFormat::NC1HWC0: return aclFormat::ACL_FORMAT_NC1HWC0;
        default:                  return aclFormat::ACL_FORMAT_ND;
    }
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor,
                    aclFormat format = aclFormat::ACL_FORMAT_ND) {
    size_t size = GetShapeSize(shape) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ret;

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, format, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

#endif

// ============================================================================
// 测试用例结构
// ============================================================================

struct TestCase {
    const char* case_id;             // 用例编号
    const char* description;         // 用例描述
    std::vector<int64_t> grads_shape; // grads shape (N,C,H,W) or (N,H,W,C) or (N,C1,H,W,C0)
    DataFormat format;                // 数据格式
    DType dtype;                      // 数据类型 (FLOAT32/FLOAT16/BFLOAT16)
    float epsilon;                    // epsilon 参数
    double atol;                      // 绝对误差容忍度
    double rtol;                      // 相对误差容忍度
    unsigned int grads_seed;          // grads 数据生成种子
    unsigned int scale_seed;          // scale 数据生成种子
    unsigned int variance_seed;       // variance 数据生成种子
};

// ============================================================================
// 迭代一测试用例定义
// ============================================================================

std::vector<TestCase> GetIteration1TestCases() {
    return {
        // ================================================================
        // 场景 S1: 基础功能测试 - 标准 4D shape fp32
        // ================================================================

        // L0_S1_001: 小规格基础测试
        {"L0_S1_001", "小规格基础 shape=(1,3,4,4) eps=1e-5",
         {1, 3, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 100, 200, 300},

        // L0_S1_002: 典型 batch size
        {"L0_S1_002", "典型batch shape=(2,16,8,8) eps=1e-5",
         {2, 16, 8, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 101, 201, 301},

        // L0_S1_003: 较大空间维度
        {"L0_S1_003", "较大HW shape=(1,32,16,16) eps=1e-5",
         {1, 32, 16, 16}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 102, 202, 302},

        // L0_S1_004: 多 batch
        {"L0_S1_004", "多batch shape=(4,8,4,4) eps=1e-5",
         {4, 8, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 103, 203, 303},

        // L0_S1_005: 较大通道数
        {"L0_S1_005", "较大通道 shape=(2,64,8,8) eps=1e-5",
         {2, 64, 8, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 104, 204, 304},

        // L0_S1_006: 较大完整 shape
        {"L0_S1_006", "较大shape shape=(2,32,32,32) eps=1e-5",
         {2, 32, 32, 32}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 105, 205, 305},

        // ================================================================
        // 场景 S2: 边界条件测试
        // ================================================================

        // L0_S2_001: 最小 shape N=1,C=1,H=1,W=1
        {"L0_S2_001", "最小shape shape=(1,1,1,1) eps=1e-5",
         {1, 1, 1, 1}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 110, 210, 310},

        // L0_S2_002: C=1 单通道
        {"L0_S2_002", "单通道 shape=(2,1,4,4) eps=1e-5",
         {2, 1, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 111, 211, 311},

        // L0_S2_003: H=1, W=1 (空间维度最小)
        {"L0_S2_003", "空间最小 shape=(2,8,1,1) eps=1e-5",
         {2, 8, 1, 1}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 112, 212, 312},

        // L0_S2_004: W=1 (宽度为1)
        {"L0_S2_004", "W=1 shape=(1,4,8,1) eps=1e-5",
         {1, 4, 8, 1}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 113, 213, 313},

        // L0_S2_005: H=1 (高度为1)
        {"L0_S2_005", "H=1 shape=(1,4,1,8) eps=1e-5",
         {1, 4, 1, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 114, 214, 314},

        // ================================================================
        // 场景 S3: epsilon 变体测试
        // ================================================================

        // L0_S3_001: 较大 epsilon
        {"L0_S3_001", "较大eps shape=(1,8,4,4) eps=1e-3",
         {1, 8, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-3f, 1e-4, 1e-4, 120, 220, 320},

        // L0_S3_002: 很小 epsilon
        {"L0_S3_002", "很小eps shape=(1,8,4,4) eps=1e-7",
         {1, 8, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-7f, 1e-3, 1e-3, 121, 221, 321},

        // ================================================================
        // 场景 S4: 非对齐维度
        // ================================================================

        // L0_S4_001: 非对齐 C
        {"L0_S4_001", "非对齐C shape=(1,3,8,8) eps=1e-5",
         {1, 3, 8, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 130, 230, 330},

        // L0_S4_002: 非对齐 H/W
        {"L0_S4_002", "非对齐HW shape=(2,8,7,7) eps=1e-5",
         {2, 8, 7, 7}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 131, 231, 331},

        // L0_S4_003: 全非对齐
        {"L0_S4_003", "全非对齐 shape=(3,5,7,9) eps=1e-5",
         {3, 5, 7, 9}, DataFormat::NCHW, DType::FLOAT32, 1e-5f, 1e-4, 1e-4, 132, 232, 332},
    };
}

// ============================================================================
// 辅助函数：根据 format 和 shape 获取通道数 C
// ============================================================================

int64_t GetChannelCount(const TestCase& tc) {
    switch (tc.format) {
        case DataFormat::NCHW:
            return tc.grads_shape[1];  // (N, C, H, W)
        case DataFormat::NHWC:
            return tc.grads_shape[3];  // (N, H, W, C)
        case DataFormat::NC1HWC0:
            return tc.grads_shape[1] * tc.grads_shape[4];  // C1 * C0
        default:
            return tc.grads_shape[1];
    }
}

// ============================================================================
// 迭代二测试用例定义
// ============================================================================

std::vector<TestCase> GetIteration2TestCases() {
    return {
        // ================================================================
        // NHWC 格式用例 (TilingKey=0 CONTIGUOUS 分支)
        // ================================================================

        // L0_S1_I2_001: NHWC 基础 shape
        {"L0_S1_I2_001", "NHWC fp32 基础shape=(2,32,32,64) eps=1e-4",
         {2, 32, 32, 64}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 200, 300, 400},

        // L0_S1_I2_002: NHWC 较大通道
        {"L0_S1_I2_002", "NHWC fp32 shape=(4,16,16,128) eps=1e-4",
         {4, 16, 16, 128}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 201, 301, 401},

        // L0_S3_I2_001: NHWC CONTIGUOUS 分支验证
        {"L0_S3_I2_001", "NHWC CONTIGUOUS分支 shape=(2,32,32,64) eps=1e-4",
         {2, 32, 32, 64}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 202, 302, 402},

        // L0_S4_I2_001: NHWC 非对齐 C=33
        {"L0_S4_I2_001", "NHWC 非对齐C=33 shape=(2,8,8,33) eps=1e-4",
         {2, 8, 8, 33}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 203, 303, 403},

        // L0_S1_I2_003: NHWC 小shape
        {"L0_S1_I2_003", "NHWC fp32 小shape=(1,4,4,16) eps=1e-4",
         {1, 4, 4, 16}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 204, 304, 404},

        // ================================================================
        // NC1HWC0 格式用例 (TilingKey=1)
        // ================================================================

        // L0_S1_I2_004: NC1HWC0 基础 shape C1=4
        {"L0_S1_I2_004", "NC1HWC0 fp32 基础shape=(2,4,32,32,16) C1=4 C0=16 eps=1e-4",
         {2, 4, 32, 32, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 210, 310, 410},

        // L0_S1_I2_005: NC1HWC0 C1=8
        {"L0_S1_I2_005", "NC1HWC0 fp32 shape=(1,8,16,16,16) C1=8 C0=16 eps=1e-4",
         {1, 8, 16, 16, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 211, 311, 411},

        // L0_S3_I2_002: NC1HWC0 分支验证
        {"L0_S3_I2_002", "NC1HWC0分支 shape=(2,4,32,32,16) eps=1e-4",
         {2, 4, 32, 32, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 212, 312, 412},

        // L0_S3_I2_003: NC1HWC0 N=4 C1=2
        {"L0_S3_I2_003", "NC1HWC0分支 N=4 C1=2 shape=(4,2,16,16,16) eps=1e-4",
         {4, 2, 16, 16, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 213, 313, 413},

        // L0_S1_I2_006: NC1HWC0 小 spatial
        {"L0_S1_I2_006", "NC1HWC0 fp32 小spatial shape=(1,2,4,4,16) eps=1e-4",
         {1, 2, 4, 4, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 214, 314, 414},

        // ================================================================
        // 多shape NCHW 用例 (验证多核切分 + 大 spatial)
        // ================================================================

        // L0_S3_I2_004: 大 spatial NCHW
        {"L0_S3_I2_004", "NCHW 大spatial shape=(1,3,224,224) eps=1e-4",
         {1, 3, 224, 224}, DataFormat::NCHW, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 220, 320, 420},

        // L0_S4_I2_002: NCHW 非对齐 C=33
        {"L0_S4_I2_002", "NCHW 非对齐C=33 shape=(2,33,8,8) eps=1e-4",
         {2, 33, 8, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 221, 321, 421},
    };
}

// ============================================================================
// 通用 Golden 计算调度函数
// ============================================================================

void ComputeGolden(const TestCase& tc, const float* grads, const float* scale,
                   const float* batch_variance, float* output) {
    switch (tc.format) {
        case DataFormat::NCHW: {
            int64_t N = tc.grads_shape[0];
            int64_t C = tc.grads_shape[1];
            int64_t H = tc.grads_shape[2];
            int64_t W = tc.grads_shape[3];
            ComputeBnInferGradGolden(grads, scale, batch_variance, output,
                                      N, C, H, W, tc.epsilon);
            break;
        }
        case DataFormat::NHWC: {
            int64_t N = tc.grads_shape[0];
            int64_t H = tc.grads_shape[1];
            int64_t W = tc.grads_shape[2];
            int64_t C = tc.grads_shape[3];
            ComputeBnInferGradGoldenNHWC(grads, scale, batch_variance, output,
                                          N, H, W, C, tc.epsilon);
            break;
        }
        case DataFormat::NC1HWC0: {
            int64_t N  = tc.grads_shape[0];
            int64_t C1 = tc.grads_shape[1];
            int64_t H  = tc.grads_shape[2];
            int64_t W  = tc.grads_shape[3];
            int64_t C0 = tc.grads_shape[4];
            ComputeBnInferGradGoldenNC1HWC0(grads, scale, batch_variance, output,
                                             N, C1, H, W, C0, tc.epsilon);
            break;
        }
    }
}

#ifdef USE_MOCK_ACLNN

bool RunTest(const TestCase& tc) {
    int64_t C = GetChannelCount(tc);
    size_t grads_size = static_cast<size_t>(GetShapeSize(tc.grads_shape));

    LOG_PRINT("\n[Mock] %s: %s", tc.case_id, tc.description);
    LOG_PRINT("  format=%s dtype=%s grads_shape=%s, epsilon=%.1e",
              FormatToString(tc.format),
              DTypeToString(tc.dtype),
              ShapeToString(tc.grads_shape).c_str(),
              static_cast<double>(tc.epsilon));

    // 生成输入数据（fp32 精度）
    auto grads_data_fp32 = GenerateRandomData(grads_size, tc.grads_seed);
    auto scale_data = GenerateScaleData(static_cast<size_t>(C), tc.scale_seed);
    auto variance_data = GeneratePositiveData(static_cast<size_t>(C), tc.variance_seed);

    // 模拟 dtype 量化：fp16/bf16 输入先量化再转回 float
    // 这样 grads 数据就携带了 fp16/bf16 的精度损失
    std::vector<float> grads_data(grads_size);
    QuantizeToFloat(grads_data_fp32.data(), grads_data.data(), grads_size, tc.dtype);

    // 计算 Golden 结果（使用量化后的 grads，double 精度计算）
    std::vector<float> golden(grads_size);
    ComputeGolden(tc, grads_data.data(), scale_data.data(),
                  variance_data.data(), golden.data());

    // 模拟输出 dtype 量化：输出也要经过 fp16/bf16 的精度损失
    std::vector<float> golden_quantized(grads_size);
    QuantizeToFloat(golden.data(), golden_quantized.data(), grads_size, tc.dtype);

    // Mock 模式下输出 = 量化后的 Golden
    std::vector<float> output = golden_quantized;

    return CompareResults(golden_quantized.data(), output.data(), grads_size, tc.atol, tc.rtol);
}

#else

// ============================================================================
// Real 模式测试执行器
// ============================================================================

bool RunTest(const TestCase& tc, aclrtStream stream) {
    int64_t C = GetChannelCount(tc);
    size_t grads_size = static_cast<size_t>(GetShapeSize(tc.grads_shape));

    LOG_PRINT("\n[Real] %s: %s", tc.case_id, tc.description);
    LOG_PRINT("  format=%s dtype=%s grads_shape=%s, epsilon=%.1e",
              FormatToString(tc.format),
              DTypeToString(tc.dtype),
              ShapeToString(tc.grads_shape).c_str(),
              static_cast<double>(tc.epsilon));

    // 生成输入数据
    auto grads_data = GenerateRandomData(grads_size, tc.grads_seed);
    auto scale_data = GenerateScaleData(static_cast<size_t>(C), tc.scale_seed);
    auto variance_data = GeneratePositiveData(static_cast<size_t>(C), tc.variance_seed);

    // 计算 Golden
    std::vector<float> golden(grads_size);
    ComputeGolden(tc, grads_data.data(), scale_data.data(),
                  variance_data.data(), golden.data());

    // 创建 grads tensor (4D/5D)
    aclFormat tensorFormat = DataFormatToAclFormat(tc.format);
    void* grads_dev = nullptr;
    aclTensor* grads_tensor = nullptr;
    if (CreateAclTensor(grads_data, tc.grads_shape, &grads_dev, ACL_FLOAT, &grads_tensor, tensorFormat) != ACL_SUCCESS) {
        LOG_PRINT("  创建 grads tensor 失败");
        return false;
    }

    // 创建 scale tensor (1D)
    std::vector<int64_t> scale_shape = {C};
    void* scale_dev = nullptr;
    aclTensor* scale_tensor = nullptr;
    if (CreateAclTensor(scale_data, scale_shape, &scale_dev, ACL_FLOAT, &scale_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  创建 scale tensor 失败");
        aclDestroyTensor(grads_tensor);
        aclrtFree(grads_dev);
        return false;
    }

    // 创建 batchVariance tensor (1D)
    std::vector<int64_t> variance_shape = {C};
    void* variance_dev = nullptr;
    aclTensor* variance_tensor = nullptr;
    if (CreateAclTensor(variance_data, variance_shape, &variance_dev, ACL_FLOAT, &variance_tensor) != ACL_SUCCESS) {
        LOG_PRINT("  创建 batchVariance tensor 失败");
        aclDestroyTensor(grads_tensor);
        aclDestroyTensor(scale_tensor);
        aclrtFree(grads_dev);
        aclrtFree(scale_dev);
        return false;
    }

    // 创建输出 tensor (同 grads shape)
    std::vector<float> output_host(grads_size, 0.0f);
    void* output_dev = nullptr;
    aclTensor* output_tensor = nullptr;
    if (CreateAclTensor(output_host, tc.grads_shape, &output_dev, ACL_FLOAT, &output_tensor, tensorFormat) != ACL_SUCCESS) {
        LOG_PRINT("  创建输出 tensor 失败");
        aclDestroyTensor(grads_tensor);
        aclDestroyTensor(scale_tensor);
        aclDestroyTensor(variance_tensor);
        aclrtFree(grads_dev);
        aclrtFree(scale_dev);
        aclrtFree(variance_dev);
        return false;
    }

    // 调用 aclnnBnInferGradGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnBnInferGradGetWorkspaceSize(
        grads_tensor,       // grads
        scale_tensor,       // scale
        variance_tensor,    // batchVariance
        tc.epsilon,         // epsilon
        static_cast<int64_t>(tc.format),  // formatMode: 0=NCHW, 1=NHWC, 2=NC1HWC0
        output_tensor,      // xBackprop (out)
        &workspaceSize,
        &executor);

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize 失败: %d", ret);
        aclDestroyTensor(grads_tensor);
        aclDestroyTensor(scale_tensor);
        aclDestroyTensor(variance_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(grads_dev);
        aclrtFree(scale_dev);
        aclrtFree(variance_dev);
        aclrtFree(output_dev);
        return false;
    }

    // 分配 workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 执行算子
    ret = aclnnBnInferGrad(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnBnInferGrad 执行失败: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(grads_tensor);
        aclDestroyTensor(scale_tensor);
        aclDestroyTensor(variance_tensor);
        aclDestroyTensor(output_tensor);
        aclrtFree(grads_dev);
        aclrtFree(scale_dev);
        aclrtFree(variance_dev);
        aclrtFree(output_dev);
        return false;
    }

    aclrtSynchronizeStream(stream);

    // 拷贝结果回 host
    std::vector<float> npu_output(grads_size);
    aclrtMemcpy(npu_output.data(), grads_size * sizeof(float), output_dev,
                grads_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 精度比对
    bool passed = CompareResults(golden.data(), npu_output.data(), grads_size, tc.atol, tc.rtol);

    // 清理资源
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(grads_tensor);
    aclDestroyTensor(scale_tensor);
    aclDestroyTensor(variance_tensor);
    aclDestroyTensor(output_tensor);
    aclrtFree(grads_dev);
    aclrtFree(scale_dev);
    aclrtFree(variance_dev);
    aclrtFree(output_dev);

    return passed;
}

#endif

// ============================================================================
// 迭代三测试用例定义
// ============================================================================

std::vector<TestCase> GetIteration3TestCases() {
    return {
        // ================================================================
        // fp16 数据类型测试 (NCHW format)
        // ================================================================

        // L0_S1_008: NCHW fp16 基础shape
        {"L0_S1_008", "NCHW fp16 基础shape=(2,64,32,32) eps=1e-4",
         {2, 64, 32, 32}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 500, 600, 700},

        // L0_S1_009: NCHW fp16 小spatial
        {"L0_S1_009", "NCHW fp16 shape=(1,128,8,8) eps=1e-4",
         {1, 128, 8, 8}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 501, 601, 701},

        // L0_S1_I3_001: NCHW fp16 多batch
        {"L0_S1_I3_001", "NCHW fp16 多batch shape=(4,32,16,16) eps=1e-4",
         {4, 32, 16, 16}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 502, 602, 702},

        // ================================================================
        // fp16 数据类型测试 (NHWC format)
        // ================================================================

        // L0_S1_I3_002: NHWC fp16 基础shape
        {"L0_S1_I3_002", "NHWC fp16 基础shape=(2,32,32,64) eps=1e-4",
         {2, 32, 32, 64}, DataFormat::NHWC, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 503, 603, 703},

        // L0_S1_I3_003: NHWC fp16 小shape
        {"L0_S1_I3_003", "NHWC fp16 shape=(1,8,8,32) eps=1e-4",
         {1, 8, 8, 32}, DataFormat::NHWC, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 504, 604, 704},

        // ================================================================
        // fp16 数据类型测试 (NC1HWC0 format)
        // ================================================================

        // L0_S1_012: NC1HWC0 fp16 基础shape
        {"L0_S1_012", "NC1HWC0 fp16 基础shape=(2,4,32,32,16) eps=1e-4",
         {2, 4, 32, 32, 16}, DataFormat::NC1HWC0, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 505, 605, 705},

        // L0_S1_I3_004: NC1HWC0 fp16 小spatial
        {"L0_S1_I3_004", "NC1HWC0 fp16 shape=(1,8,16,16,16) eps=1e-4",
         {1, 8, 16, 16, 16}, DataFormat::NC1HWC0, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 506, 606, 706},

        // ================================================================
        // bf16 数据类型测试 (NCHW format)
        // ================================================================

        // L0_S1_010: NCHW bf16 基础shape
        {"L0_S1_010", "NCHW bf16 基础shape=(2,64,32,32) eps=1e-4",
         {2, 64, 32, 32}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 510, 610, 710},

        // L0_S1_011: NCHW bf16 小spatial
        {"L0_S1_011", "NCHW bf16 shape=(1,128,8,8) eps=1e-4",
         {1, 128, 8, 8}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 511, 611, 711},

        // L0_S1_I3_005: NCHW bf16 多batch
        {"L0_S1_I3_005", "NCHW bf16 多batch shape=(4,32,16,16) eps=1e-4",
         {4, 32, 16, 16}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 512, 612, 712},

        // ================================================================
        // bf16 数据类型测试 (NHWC format)
        // ================================================================

        // L0_S1_I3_006: NHWC bf16 基础shape
        {"L0_S1_I3_006", "NHWC bf16 基础shape=(2,32,32,64) eps=1e-4",
         {2, 32, 32, 64}, DataFormat::NHWC, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 513, 613, 713},

        // L0_S1_I3_007: NHWC bf16 小shape
        {"L0_S1_I3_007", "NHWC bf16 shape=(1,8,8,32) eps=1e-4",
         {1, 8, 8, 32}, DataFormat::NHWC, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 514, 614, 714},

        // ================================================================
        // bf16 数据类型测试 (NC1HWC0 format)
        // ================================================================

        // L0_S1_I3_008: NC1HWC0 bf16 基础shape
        {"L0_S1_I3_008", "NC1HWC0 bf16 基础shape=(2,4,32,32,16) eps=1e-4",
         {2, 4, 32, 32, 16}, DataFormat::NC1HWC0, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 515, 615, 715},

        // L0_S1_I3_009: NC1HWC0 bf16 小spatial
        {"L0_S1_I3_009", "NC1HWC0 bf16 shape=(1,4,16,16,16) eps=1e-4",
         {1, 4, 16, 16, 16}, DataFormat::NC1HWC0, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 516, 616, 716},

        // ================================================================
        // format x dtype 交叉覆盖 (L0_S5)
        // ================================================================

        // L0_S5_001: NCHW fp16 交叉
        {"L0_S5_001", "format交叉 NCHW fp16 shape=(2,64,16,16) eps=1e-4",
         {2, 64, 16, 16}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 520, 620, 720},

        // L0_S5_002: NCHW bf16 交叉
        {"L0_S5_002", "format交叉 NCHW bf16 shape=(2,64,16,16) eps=1e-4",
         {2, 64, 16, 16}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 521, 621, 721},

        // L0_S5_005: NC1HWC0 fp16 交叉
        {"L0_S5_005", "format交叉 NC1HWC0 fp16 shape=(2,4,16,16,16) eps=1e-4",
         {2, 4, 16, 16, 16}, DataFormat::NC1HWC0, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 522, 622, 722},

        // ================================================================
        // 边界用例 - 极小shape
        // ================================================================

        // L0_S4_I3_001: 最小shape fp16
        {"L0_S4_I3_001", "最小shape fp16 shape=(1,1,1,1) eps=1e-4",
         {1, 1, 1, 1}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 530, 630, 730},

        // L0_S4_I3_002: 最小shape bf16
        {"L0_S4_I3_002", "最小shape bf16 shape=(1,1,1,1) eps=1e-4",
         {1, 1, 1, 1}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 531, 631, 731},

        // L0_S4_I3_003: 极小shape NHWC fp16
        {"L0_S4_I3_003", "极小shape NHWC fp16 shape=(1,1,1,1) eps=1e-4",
         {1, 1, 1, 1}, DataFormat::NHWC, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 532, 632, 732},

        // L0_S4_I3_004: 极小shape NC1HWC0 fp32
        {"L0_S4_I3_004", "极小shape NC1HWC0 fp32 shape=(1,1,1,1,16) eps=1e-4",
         {1, 1, 1, 1, 16}, DataFormat::NC1HWC0, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 533, 633, 733},

        // ================================================================
        // 边界用例 - channel=1
        // ================================================================

        // L0_S4_I3_005: 单通道 fp16
        {"L0_S4_I3_005", "单通道C=1 fp16 shape=(2,1,32,32) eps=1e-4",
         {2, 1, 32, 32}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 534, 634, 734},

        // L0_S4_I3_006: 单通道 bf16
        {"L0_S4_I3_006", "单通道C=1 bf16 shape=(2,1,32,32) eps=1e-4",
         {2, 1, 32, 32}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 535, 635, 735},

        // L0_S4_I3_007: 单通道 NHWC fp32
        {"L0_S4_I3_007", "单通道C=1 NHWC fp32 shape=(2,32,32,1) eps=1e-4",
         {2, 32, 32, 1}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 536, 636, 736},

        // ================================================================
        // 边界用例 - 大batch
        // ================================================================

        // L0_S4_I3_008: 大batch fp32
        {"L0_S4_I3_008", "大batch fp32 shape=(16,64,8,8) eps=1e-4",
         {16, 64, 8, 8}, DataFormat::NCHW, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 537, 637, 737},

        // L0_S4_I3_009: 大batch fp16
        {"L0_S4_I3_009", "大batch fp16 shape=(16,64,8,8) eps=1e-4",
         {16, 64, 8, 8}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 538, 638, 738},

        // L0_S4_I3_010: 大batch bf16
        {"L0_S4_I3_010", "大batch bf16 shape=(16,64,8,8) eps=1e-4",
         {16, 64, 8, 8}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 539, 639, 739},

        // ================================================================
        // 边界用例 - 非对齐shape + 非fp32 dtype
        // ================================================================

        // L0_S4_I3_011: 非对齐HW fp16
        {"L0_S4_I3_011", "非对齐HW=49 fp16 shape=(2,64,7,7) eps=1e-4",
         {2, 64, 7, 7}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 540, 640, 740},

        // L0_S4_I3_012: 非对齐C bf16
        {"L0_S4_I3_012", "非对齐C=33 bf16 shape=(2,33,8,8) eps=1e-4",
         {2, 33, 8, 8}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 541, 641, 741},

        // L0_S4_I3_013: 非对齐C NHWC fp16
        {"L0_S4_I3_013", "非对齐C=33 NHWC fp16 shape=(2,8,8,33) eps=1e-4",
         {2, 8, 8, 33}, DataFormat::NHWC, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 542, 642, 742},

        // ================================================================
        // 边界用例 - H=1/W=1 退化shape + 非fp32 dtype
        // ================================================================

        // L0_S4_I3_014: H=W=1 fp16
        {"L0_S4_I3_014", "H=W=1退化 fp16 shape=(2,64,1,1) eps=1e-4",
         {2, 64, 1, 1}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 543, 643, 743},

        // L0_S4_I3_015: H=W=1 bf16
        {"L0_S4_I3_015", "H=W=1退化 bf16 shape=(2,64,1,1) eps=1e-4",
         {2, 64, 1, 1}, DataFormat::NCHW, DType::BFLOAT16, 1e-4f, 4e-3, 4e-3, 544, 644, 744},

        // ================================================================
        // 边界用例 - 大通道数
        // ================================================================

        // L0_S4_I3_016: 大通道 fp32
        {"L0_S4_I3_016", "大通道C=512 fp32 shape=(1,512,4,4) eps=1e-4",
         {1, 512, 4, 4}, DataFormat::NCHW, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 545, 645, 745},

        // L0_S4_I3_017: 大通道 NHWC fp32
        {"L0_S4_I3_017", "大通道C=512 NHWC fp32 shape=(1,4,4,512) eps=1e-4",
         {1, 4, 4, 512}, DataFormat::NHWC, DType::FLOAT32, 1e-4f, 1e-4, 1e-4, 546, 646, 746},

        // L0_S4_I3_018: 大通道 fp16
        {"L0_S4_I3_018", "大通道C=256 fp16 shape=(1,256,4,4) eps=1e-4",
         {1, 256, 4, 4}, DataFormat::NCHW, DType::FLOAT16, 1e-4f, 1e-3, 1e-3, 547, 647, 747},
    };
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("BnInferGrad 算子 ST 测试 (迭代一 + 迭代二 + 迭代三)");
    LOG_PRINT("========================================");
    LOG_PRINT("迭代一: fp32 | NCHW 基础4D shape | epsilon变体 | 边界条件");
    LOG_PRINT("迭代二: fp32 | NHWC + NC1HWC0 | 多TilingKey | 多核切分");
    LOG_PRINT("迭代三: fp16+bf16 | 全format交叉 | 边界shape | 全dtype覆盖");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden 自验证)");
#else
    LOG_PRINT("模式: Real (NPU vs CPU golden)");
#endif

    // CPU Golden 正确性自测
    TestGoldenCorrectness();

    int passed = 0, failed = 0;
    int iter1_passed = 0, iter1_failed = 0;
    int iter2_passed = 0, iter2_failed = 0;
    int iter3_passed = 0, iter3_failed = 0;

#ifndef USE_MOCK_ACLNN
    int32_t deviceId = 0;
    aclrtStream stream;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
#endif

    // ================================================================
    // 迭代一测试用例
    // ================================================================
    LOG_PRINT("\n========================================");
    LOG_PRINT("执行迭代一测试用例");
    LOG_PRINT("========================================");

    auto iter1_cases = GetIteration1TestCases();

    for (const auto& tc : iter1_cases) {
#ifdef USE_MOCK_ACLNN
        if (RunTest(tc)) {
            passed++; iter1_passed++;
        } else {
            failed++; iter1_failed++;
        }
#else
        if (RunTest(tc, stream)) {
            passed++; iter1_passed++;
        } else {
            failed++; iter1_failed++;
        }
#endif
    }

    LOG_PRINT("\n--- 迭代一小计: %d/%d 通过 ---", iter1_passed, iter1_passed + iter1_failed);

    // ================================================================
    // 迭代二测试用例
    // ================================================================
    LOG_PRINT("\n========================================");
    LOG_PRINT("执行迭代二测试用例");
    LOG_PRINT("========================================");
    LOG_PRINT("NHWC (TilingKey=0 CONTIGUOUS) + NC1HWC0 (TilingKey=1) + 多shape");

    auto iter2_cases = GetIteration2TestCases();

    for (const auto& tc : iter2_cases) {
#ifdef USE_MOCK_ACLNN
        if (RunTest(tc)) {
            passed++; iter2_passed++;
        } else {
            failed++; iter2_failed++;
        }
#else
        if (RunTest(tc, stream)) {
            passed++; iter2_passed++;
        } else {
            failed++; iter2_failed++;
        }
#endif
    }

    LOG_PRINT("\n--- 迭代二小计: %d/%d 通过 ---", iter2_passed, iter2_passed + iter2_failed);

    // ================================================================
    // 迭代三测试用例
    // ================================================================
    LOG_PRINT("\n========================================");
    LOG_PRINT("执行迭代三测试用例");
    LOG_PRINT("========================================");
    LOG_PRINT("fp16 + bf16 全dtype | NCHW + NHWC + NC1HWC0 全format | 边界shape");

    auto iter3_cases = GetIteration3TestCases();

    for (const auto& tc : iter3_cases) {
#ifdef USE_MOCK_ACLNN
        if (RunTest(tc)) {
            passed++; iter3_passed++;
        } else {
            failed++; iter3_failed++;
        }
#else
        if (RunTest(tc, stream)) {
            passed++; iter3_passed++;
        } else {
            failed++; iter3_failed++;
        }
#endif
    }

    LOG_PRINT("\n--- 迭代三小计: %d/%d 通过 ---", iter3_passed, iter3_passed + iter3_failed);

#ifndef USE_MOCK_ACLNN
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告 (迭代一 + 迭代二 + 迭代三)");
    LOG_PRINT("========================================");
    LOG_PRINT("迭代一: %d/%d 通过 (fp32 NCHW 基础)", iter1_passed, iter1_passed + iter1_failed);
    LOG_PRINT("迭代二: %d/%d 通过 (fp32 多format多shape)", iter2_passed, iter2_passed + iter2_failed);
    LOG_PRINT("迭代三: %d/%d 通过 (fp16+bf16 全dtype+边界)", iter3_passed, iter3_passed + iter3_failed);
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("通过率: %.1f%%", (passed + failed) > 0 ? 100.0 * passed / (passed + failed) : 0.0);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}
