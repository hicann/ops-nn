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
 * \file quant_batch_matmul_v4_weight_quant_mx_cube_tiling.h
 * \brief
 */
#pragma once

#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {
class AdaptiveSlidingWindowWeightQuantMxCubeTiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowWeightQuantMxCubeTiling(gert::TilingContext* context)
        : AdaptiveSlidingWindowTiling(context)
    {
        tilingDataSize_ = sizeof(withoutBatchTilingData_);
    }
    ~AdaptiveSlidingWindowWeightQuantMxCubeTiling() override = default;

    uint64_t GetTilingKey() const override;
    ge::graphStatus DoLibApiTiling() override;

protected:
    uint32_t GetX1Idx() const override { return X1_INDEX_V4; }
    uint32_t GetX2Idx() const override { return X2_INDEX_V4; }
    uint32_t GetScaleIdx() const override { return X2_SCALE_INDEX_V4; }
    uint32_t GetPertokenIdx() const override { return X1_SCALE_INDEX_V4; }
    uint32_t GetBiasIdx() const override { return BIAS_INDEX_V4; }
    uint32_t GetOffsetIdx() const override { return X2_OFFSET_INDEX_V4; }
    const char* GetDefaultOpName() const override { return "QuantBatchMatmulV4"; }

    ge::graphStatus GetShapeAttrsInfo() override;
    bool SetPlatformInfoForTiling() override;
    bool AnalyzeAttrs() override;
    bool CheckDtype() const override;
    bool AnalyzeInputs() override;
    bool IsCapable() override;

    void ResetTilingData() override;
    const void* GetTilingData() const override;
    bool CalcBasicBlock() override;
    void AnalyseFullLoadInfo() override;
    void CalcTailRoundBasicBlockSplit() override;
    bool CalL1Tiling() override;
    void SetTilingData() override;

private:
    struct MxL1EstimateParams {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t baseM;
        uint64_t baseN;
        bool isAFullLoad;
    };

    bool AnalyzeGroupSize();
    bool AnalyzeTranspose();
    bool CheckOperandDtypes() const;
    bool CheckOutputAndBiasDtypes() const;
    bool CheckDisabledInputs() const;
    bool CheckMatrixShape(const char* name, const gert::Shape& shape) const;
    bool CheckMxShapes() const;
    bool CheckTensorLayout(const char* name, const gert::CompileTimeTensorDesc* desc, const gert::StorageShape* shape,
                           ge::Format format, const gert::Shape& originShape, const gert::Shape& storageShape) const;
    void UpdateAFullLoadStatus();
    uint64_t DeriveScaleKL1(uint32_t scaleFactorA, uint64_t stepKa, uint32_t scaleFactorB, uint64_t stepKb,
                            uint64_t baseK) const;
    void CalculateNBufferNum();
    void PrintTilingData() const;
    uint64_t CalcUsedL1Size(const MxL1EstimateParams& params, uint32_t l1BufferNum) const;
    uint64_t CalcMxFullKLoadSize(uint64_t outerSize, ge::DataType dataDtype, ge::DataType scaleDtype) const;

    DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData withoutBatchTilingData_{};
};
} // namespace optiling
