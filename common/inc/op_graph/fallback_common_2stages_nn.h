/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_FALLBACK_COMMON_TWOSTAGES_NN_H_
#define OPS_NN_FALLBACK_COMMON_TWOSTAGES_NN_H_

#include <cstring>
#include <unordered_map>
#include <vector>

#include "op_common/op_graph/op_fallback.h"
#include "op_common/op_graph/op_fallback_internal.h"

namespace fallback {

using aclCubeMathType = enum : int8_t {
    KEEP_DTYPE = 0,
    ALLOW_FP32_DOWN_PRECISION = 1,
    USE_FP16 = 2,
    USE_HF32 = 3,
};

using OpImplMode = enum : int64_t {
    DEFAULT_MODE = 0x1,
    HIGH_PERFORMANCE_MODE = 0x2,
    ENABLE_FORCE_GRP_ACC_FOR_FP32 = 0x4,
    SUPER_PERFORMANCE_MODE = 0x8,
    SUPPORT_OUT_OF_BOUND_INDEX = 0x10,
    ENABLE_FLOAT_32_EXECUTION = 0x20,
    ENABLE_HF32 = 0x40,
};

// precision_mode 值: 1 表示允许 FP32 降精度到 FP16（高性能模式）
constexpr int32_t ALLOW_FP32_DOWN_PRECISION_MODE = 1;

// MatMulV3 proto 中 opImplMode 属性的索引（第 4 个属性，0-based）
constexpr size_t MATMUL_V3_OP_IMPL_MODE_ATTR_IDX = 3;

// GE allow_hf32 选项值: 格式为 "MC"，M=matmul 位, C=conv 位; 1=启用, 0=禁用
constexpr const char* GE_ALLOW_HF32_MATMUL_ONLY = "01";     // 仅 matmul 启用 HF32
constexpr const char* GE_ALLOW_HF32_MATMUL_AND_CONV = "11"; // matmul + conv 均启用 HF32

// ConvertMmType 中矩阵乘最后两维的索引偏移
constexpr int64_t MATMUL_DIM_M_OFFSET_FROM_END = 2; // 倒数第 2 维（M/行维度）
constexpr int64_t MATMUL_DIM_N_OFFSET_FROM_END = 1; // 倒数第 1 维（N/列维度）

// ACL_CUBE_MATH_TYPE_MAP 的 key 由 (allowHf32 << 1) | allowFp32ToFp16 组成
static const std::unordered_map<uint8_t, aclCubeMathType> ACL_CUBE_MATH_TYPE_MAP = {
    {0b00, KEEP_DTYPE}, {0b01, USE_FP16}, {0b10, USE_HF32}, {0b11, ALLOW_FP32_DOWN_PRECISION}};

static inline bool GetGlobalPrecisionMode(gert::OpExecutePrepareContext* host_api_ctx)
{
    int32_t precision_mode = host_api_ctx->GetPrecisionMode();
    return (precision_mode == ALLOW_FP32_DOWN_PRECISION_MODE);
}

static inline bool IsMatMulV3Hf32Enabled(gert::OpExecutePrepareContext* host_api_ctx)
{
    int64_t opImplMode = DEFAULT_MODE;
    const char* nodeType = host_api_ctx->GetNodeType();
    if (nodeType != nullptr && strcmp(nodeType, "MatMulV3") == 0) {
        const auto attrs = host_api_ctx->GetAttrs();
        if (attrs == nullptr) {
            OP_LOGE("aclnnfallback", "Get attrs of MatMulV3 failed");
            return false;
        }
        const int64_t* opImplModePtr = attrs->GetAttrPointer<int64_t>(MATMUL_V3_OP_IMPL_MODE_ATTR_IDX);
        if (opImplModePtr == nullptr) {
            OP_LOGE("aclnnfallback", "Get opImplMode of MatMulV3 failed");
            return false;
        }
        opImplMode = *opImplModePtr;
        OP_LOGD("aclnnfallback", "MatMulV3 opImplMode: %ld", opImplMode);
        return (opImplMode & ENABLE_HF32) != 0;
    }
    return false;
}

static inline bool GetMatmulPrecisionMode(gert::OpExecutePrepareContext* host_api_ctx)
{
    const bool enableHf32ByOpImplMode = IsMatMulV3Hf32Enabled(host_api_ctx);
    const char* ge_allow_hf32 = host_api_ctx->GetAllowHf32();
    if (ge_allow_hf32 == nullptr) {
        OP_LOGE("aclnnfallback", "Get allow_hf32 failed");
        return enableHf32ByOpImplMode;
    }
    const bool enableHf32ByGlobalOption = (strcmp(ge_allow_hf32, GE_ALLOW_HF32_MATMUL_ONLY) == 0 ||
                                           strcmp(ge_allow_hf32, GE_ALLOW_HF32_MATMUL_AND_CONV) == 0);
    return enableHf32ByOpImplMode || enableHf32ByGlobalOption;
}

static inline int8_t GetMathType(gert::OpExecutePrepareContext* host_api_ctx)
{
    bool allowHf32 = GetMatmulPrecisionMode(host_api_ctx);
    bool allowFp32ToFp16 = GetGlobalPrecisionMode(host_api_ctx);

    uint8_t cubeMathTypeCode = (static_cast<uint8_t>(allowHf32) << 1) + static_cast<uint8_t>(allowFp32ToFp16);
    auto iter = ACL_CUBE_MATH_TYPE_MAP.find(cubeMathTypeCode);
    if (iter == ACL_CUBE_MATH_TYPE_MAP.end()) {
        return ALLOW_FP32_DOWN_PRECISION;
    }
    OP_LOGI("aclnnfallback", "GetMathType: %d", iter->second);
    return iter->second;
}

static inline aclTensor* ConvertMmType(const gert::Tensor* ge_tensor, bool transpose, bool enable_NZ = false)
{
    if (ge_tensor == nullptr) {
        return nullptr;
    }
    auto gert_shape = ge_tensor->GetStorageShape();
    // 1D 及以下维度无需 transpose，直接走通用转换
    if (gert_shape.GetDimNum() <= 1) {
        return ConvertType(ge_tensor);
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

    void* device_addr = (void*)ge_tensor->GetAddr();
    auto dataType = GetConvertType(ge_tensor);

    std::vector<int64_t> shape;
    for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
        shape.push_back(gert_shape.GetDim(i));
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - MATMUL_DIM_M_OFFSET_FROM_END; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto viewShape = shape;
    if (transpose) {
        auto dimM = shape.size() - MATMUL_DIM_M_OFFSET_FROM_END;
        auto dimN = shape.size() - MATMUL_DIM_N_OFFSET_FROM_END;
        auto swap = strides[dimN];
        strides[dimN] = strides[dimM];
        strides[dimM] = swap;
        viewShape[dimN] = shape[dimM];
        viewShape[dimM] = shape[dimN];
    }

    auto acl_format = aclFormat::ACL_FORMAT_ND;
    if (enable_NZ && GetPrimaryFormat(ge_tensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
        acl_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    aclTensor* out = aclCreateTensor(viewShape.data(), shape.size(), dataType, strides.data(), 0, acl_format,
                                     shape.data(), shape.size(), device_addr);
    OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "out nullptr"), return nullptr);
    return out;
}

static inline ge::graphStatus ExecuteOpLaunch(gert::OpExecuteLaunchContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("aclnnfallback", "launch_ctx is null"), return ge::GRAPH_FAILED);
    auto* params = static_cast<OpApiParams*>(context->GetOpApiParams());
    OP_CHECK_IF(params == nullptr, OP_LOGE("aclnnfallback", "params is null"), return ge::GRAPH_FAILED);

    void* workspace_addr = nullptr;
    auto* workspace_addrs = context->GetWorkspaceAddrs();
    if (workspace_addrs != nullptr && workspace_addrs->GetSize() > 0) {
        workspace_addr = workspace_addrs->GetData()[0]->GetAddr();
    }
    uint64_t workspace_size = 0;
    auto* workspace_sizes = context->GetWorkspaceSizes();
    if (workspace_sizes != nullptr && workspace_sizes->GetSize() > 0) {
        workspace_size = workspace_sizes->GetData()[0];
    }

    auto acl_stream = context->GetStream();
    auto op_api_ret = params->op_api_func(workspace_addr, workspace_size, params->executor, acl_stream);
    for (auto& val : params->converted_params) {
        if (val.pointer != nullptr && val.deleter != nullptr) {
            val.deleter(val.pointer);
        }
    }
    params->converted_params.clear();
    OP_CHECK_IF(op_api_ret != 0,
                OP_LOGE("aclnnfallback", "call %s launch failed op_api_ret: %d", context->GetNodeName(), op_api_ret),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace fallback

#endif // OPS_NN_FALLBACK_COMMON_TWOSTAGES_NN_H_
