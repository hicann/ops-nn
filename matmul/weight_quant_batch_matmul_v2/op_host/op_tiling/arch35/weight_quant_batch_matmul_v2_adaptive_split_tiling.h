/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_split_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SPLIT_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ADAPTIVE_SPLIT_TILING_H

#include "../weight_quant_batch_matmul_v2_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "error_util.h"

namespace optiling {
namespace weight_quant_batch_matmul_v2 {
class WeightQuantBatchMatmulV2TilingAS : public WeightQuantBatchMatmulV2Tiling
{
public:
    explicit WeightQuantBatchMatmulV2TilingAS(gert::TilingContext* context) : WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    };
    ~WeightQuantBatchMatmulV2TilingAS() override = default;

protected:
    std::unique_ptr<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams> tilingData_;
    size_t tilingDataSize_ = sizeof(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams);

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

private:
    OptimizationAlgorithmSubCategory algorithmSubCategory_ = OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT;
    Mte2Configuration mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2;
    uint64_t l1NMaxSize_ = 0;
    bool weightMxFp4Flag_ = false;
    bool weightInt4Flag_ = false;
    bool nzSceneFlag_ = false;

    bool CheckHighPerfScene() const;
    bool CheckWeightMicroscalingFp4Scene() const;
    void ComputeCubeTiling(bool highPerfFlag);
    void ComputeCubeSplit(bool highPerfFlag);
    void SetAttrs();
    void ComputeHighPerfSceneCubeSplit();
    void ComputeBasicTiling();
    void ComputeTailResplitTiling();
    void OptimizeMatmulTiling();
    void OptimizeMatmulTilingInA16W4Nz();
    bool CheckL1SizeAfterExtending(uint64_t extendedBaseN) const;
    bool CheckUbSizeAfterExtending(uint64_t extendedWeightMte2N) const;
    bool CheckL0CSizeAfterExtending(uint64_t extendedBaseN) const;
    void RecalculateBaseBlockSize();
    void ResplitTail();
    void SetInnerSize();
    void SetDefaultMatmulTiling();
    void SetPreLoad();
    void EnlargeBaseK(uint64_t l0aMaxBaseK);
    void EnlargeBaseKInFullloadA(
        uint64_t maxBaseK, uint64_t minKL1, uint64_t maxL1, const std::vector<uint64_t>& l0BaseKList,
        const std::vector<uint64_t>& l1KList);
    void EnlargeBaseKNotFullloadA(
        uint64_t maxBaseK, uint64_t maxKL1, const std::vector<uint64_t>& l0BaseKList,
        const std::vector<uint64_t>& l1KList);
    bool IsWeight4Nz() const;
    void ReSetTilingAfterExtendedBaseN(
        uint64_t extendedBaseN, uint64_t firstTailBlockL1Size, uint64_t firstTailBlockCount,
        uint64_t secondTailBlockL1Size, uint64_t secondTailBlockCount) const;
};
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling

#endif