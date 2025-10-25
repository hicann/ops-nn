/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_tiling_msd_group.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (30) USING BLANK LINES.
 * 
 * 
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "tiling_base/tiling_key.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2MsdGroupTilingData)
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, vec1SingleCoreM);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, reserve1);
TILING_DATA_FIELD_DEF(uint16_t, reserve2);
TILING_DATA_FIELD_DEF(uint32_t, reserve3);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365333139882753, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365058261975809, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367514982809857, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367240104902913, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365315960602881, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365041082695937, WeightQuantBatchMatmulV2MsdGroupTilingData)

REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367514983858433, WeightQuantBatchMatmulV2MsdGroupTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367240105951489, WeightQuantBatchMatmulV2MsdGroupTilingData)

class WeightQuantBatchMatmulV2TilingMsdGroup : public WeightQuantBatchMatmulV2Tiling
{
public:
    explicit WeightQuantBatchMatmulV2TilingMsdGroup(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    };
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2TilingMsdGroup() override = default;

protected:
    std::unique_ptr<WeightQuantBatchMatmulV2MsdGroupTilingData> tilingData_;

    void Reset();

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    bool CheckL1Size() const;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        if (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP &&
            (matmulInfoPtr_->innerPrecise != 0 || matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)) {
            // 在A16W4 pergroup 高性能/高精度 tilingkey
            TilingKeyConfigure tilingKeyConfigure;
            SetCommonTilingKeyElement(tilingKeyConfigure);

            uint64_t socVersionType = tilingKeyConfigure.socVersionType / 10UL;
            uint64_t subSocVersionType = 0UL;
            uint64_t antiquantScenario = tilingKeyConfigure.quantizationScenario;
            uint64_t algorithm = static_cast<uint64_t>(OptimizationAlgorithmCategory::MULTI_SCALE_DEQUANT);
            uint64_t subAlgorithm = static_cast<uint64_t>(OptimizationAlgorithmSubCategory::VDEFAULT);
            uint64_t subAlgorithmCustom = 0UL;
            uint64_t innerPrecise = matmulInfoPtr_->innerPrecise;
            uint64_t templateCustom = 0UL;
            uint64_t apiConstexpr = 0UL;
            bool transA = ((tilingKeyConfigure.transposeSituation >> 1) & 1) != 0;
            bool transB = (tilingKeyConfigure.transposeSituation & 1) != 0;
            uint64_t antiquantType = tilingKeyConfigure.antiquantType;
            uint64_t quantType = tilingKeyConfigure.quantType;
            bool hasAntiquantOffset = ((tilingKeyConfigure.optionInputSituation >> 1) & 1) != 0;
            bool hasBias = false;
            bool isBiasFp32 = false;
            bool isWeightNz = (tilingKeyConfigure.weightFormat == 1UL) ? true : false;
            uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            uint64_t fullLoadMode = 5UL; // 5 means FULL_LOAD_MODE_NOT_USED
            uint64_t batch = 0UL;
            uint64_t tilingKey_ = GET_TPL_TILING_KEY(
                socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
                innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        } else {
            uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
            uint64_t subSocVersionType = 0UL;
            uint64_t antiquantScenario = 0UL;
            uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
            uint64_t subAlgorithm = 0UL;
            uint64_t subAlgorithmCustom = static_cast<uint64_t>(KernelTemplateType::MSD_GROUP);
            uint64_t innerPrecise = 0UL;
            uint64_t templateCustom = 0UL;
            uint64_t apiConstexpr = 0UL;
            bool transA = matmulInfoPtr_->transA;
            bool transB = matmulInfoPtr_->transB;
            uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
            uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
            bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
            bool hasBias = false;
            bool isBiasFp32 = false;
            bool isWeightNz = false;
            uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            uint64_t fullLoadMode = 5UL; // 5 means FULL_LOAD_MODE_NOT_USED
            uint64_t batch = 0UL;
            uint64_t tilingKey_ = GET_TPL_TILING_KEY(
                socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
                innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        }
    }

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = 64 * 1024 * 1024; // workspace 固定使用 64 * 1024 * 1024
        return ge::GRAPH_SUCCESS;
    }

    bool GetMatMulTiling();
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H
