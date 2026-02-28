/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_tiling_custom_backward_compatible.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom_backward_compatible.h"
#include "op_host/tiling_key.h"
#include "../../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

namespace {
constexpr uint64_t SUPPORT_C0_SIZE = 32;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr int32_t CUSTOM_DEPRECATED_PRIORITY = 11;
} // namespace

namespace optiling {
bool WeightQuantBatchMatmulV2TilingCustomBackwardCompatible::IsCapable()
{
    OP_LOGI(opName_, "Begin check custom backward compatible template");
    OP_TILING_CHECK(
        matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ ||
            (matmulInfoPtr_->aDtype != ge::DT_FLOAT16 && matmulInfoPtr_->aDtype != ge::DT_BF16) ||
            matmulInfoPtr_->bDtype != ge::DT_INT8 || matmulInfoPtr_->c0Size != SUPPORT_C0_SIZE,
        OP_LOGI(opName_, "the custom backward compatible template only support a16w8 nz format and C0 only support 32"),
        return false);

    OP_TILING_CHECK(matmulInfoPtr_->transA, OP_LOGI(opName_, "A16W8 Nz cannot support x transpose"), return false);
    OP_TILING_CHECK(
        matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
        OP_LOGI(opName_, "A16W8 Nz only support n < 65536 and k < 65536"), return false);
    OP_TILING_CHECK(
        matmulInfoPtr_->antiQuantType != QuantType::PER_CHANNEL &&
            matmulInfoPtr_->antiQuantType != QuantType::PER_GROUP,
        OP_LOGI(opName_, "A16W8 Nz only support perchannel and per-group quant mode"), return false);

    if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
        OP_TILING_CHECK(
            matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128,
            OP_LOGI(
                opName_, "A16W8 Nz only support group_size = 64 or 128 for per-group scene, but is [%lu]",
                matmulInfoPtr_->groupSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0,
            OP_LOGI(
                opName_,
                "A16W8 Nz only support kSize align to group_size for per-group scene, "
                "but kSize is [%lu], group_size is [%lu]",
                matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->kSize % 64 != 0 || matmulInfoPtr_->nSize % 64 != 0,
            OP_LOGI(
                opName_,
                "A16W8 Nz only support kSize and nSize align to 64 for per-group scene, "
                "but kSize is [%lu], nSize is [%lu]",
                matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
            return false);
        OP_TILING_CHECK(
            matmulInfoPtr_->transB, OP_LOGI(opName_, "A16W8 Nz cannot support weight transpose for per-group scene"),
            return false);
    }

    OP_LOGI(opName_, "Check custom backward compatible template success");
    return true;
}

uint64_t WeightQuantBatchMatmulV2TilingCustomBackwardCompatible::GetTilingKey() const
{
    uint64_t socVersionType = WQBMMV2_SOC_SUPPORT_L1_TO_BT_BF16;
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    uint64_t algorithm = WQBMMV2_ALGO_VECTOR_ANTIQUANT;
    uint64_t subAlgorithm = WQBMMV2_SUB_ALGO_CUSTOM_BACKWARD_COMPATIBLE;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    OP_LOGD(
        opName_,
        "tiling key params: socVersionType[%lu], subSocVersionType[%lu], antiquantScenario[%lu], algorithm[%lu],"
        "subAlgorithm[%lu], templateCustom[%lu], apiConstexpr[%lu], transA[%s], transB[%s], antiquantType[%lu],"
        "quantType[%lu], hasAntiquantOffset[%s], hasBias[%s], isBiasFp32[%s], isWeightNz[%s]",
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, templateCustom, apiConstexpr,
        transA ? "true" : "false", transB ? "true" : "false", antiquantType, quantType,
        hasAntiquantOffset ? "true" : "false", hasBias ? "true" : "false", isBiasFp32 ? "true" : "false",
        isWeightNz ? "true" : "false");
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, templateCustom, apiConstexpr,
        transA, transB, antiquantType, quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    WeightQuantBatchMatmulV2, WeightQuantBatchMatmulV2TilingCustomBackwardCompatible,
    static_cast<int32_t>(NpuArch::DAV_3510), CUSTOM_DEPRECATED_PRIORITY);

} // namespace optiling