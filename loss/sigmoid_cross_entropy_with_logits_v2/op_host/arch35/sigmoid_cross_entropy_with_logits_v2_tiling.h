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
 * \file sigmoid_cross_entropy_with_logits_v2_tiling.h
 * \brief
 */
#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_H

#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "loss/sigmoid_cross_entropy_with_logits_v2/op_kernel/arch35/sigmoid_cross_entropy_with_logits_v2_dag.h"
#include "loss/sigmoid_cross_entropy_with_logits_v2/op_kernel/arch35/sigmoid_cross_entropy_with_logits_v2_tiling_key.h"
#include "error_util.h"
namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace Ops::Base;

struct SigmoidCEWithLogitsV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SigmoidCEWithLogitsV2TilingClass : public TilingBaseClass
{
public:
    explicit SigmoidCEWithLogitsV2TilingClass(gert::TilingContext* context) : TilingBaseClass(context){}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();

    template <typename T, typename R>
    ge::graphStatus RunBroadcastTiling();

    ge::DataType outputDtype;
    ge::DataType predictDtype;
    uint32_t reduction = 0;
    uint32_t hasWeight = 0;
    uint32_t hasPosWeight = 0;
    uint64_t tilingKey_ = 0;
};

template <typename T, typename R>
ge::graphStatus SigmoidCEWithLogitsV2TilingClass::RunBroadcastTiling() {
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (this->hasWeight && this->hasPosWeight) {
        auto brcBaseTiling = BroadcastBaseTiling<
            typename SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2HasTwoWeight<T, R>::OpDag>(context_);
        
        ret = brcBaseTiling.DoTiling();
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->reduction, 
                                            this->hasWeight, this->hasPosWeight);
    } else if (this->hasWeight) {
        auto brcBaseTiling = BroadcastBaseTiling<
            typename SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2WeightOnly<T, R>::OpDag>(context_);
        ret = brcBaseTiling.DoTiling();
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->reduction, 
                                            this->hasWeight, this->hasPosWeight);
    } else if (this->hasPosWeight) {
        auto brcBaseTiling = BroadcastBaseTiling<
            typename SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2PosWeightOnly<T, R>::OpDag>(context_);
        ret = brcBaseTiling.DoTiling();
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->reduction, 
                                            this->hasWeight, this->hasPosWeight);
    } else {
        auto brcBaseTiling = BroadcastBaseTiling<
            typename SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2<T, R>::OpDag>(context_);
        ret = brcBaseTiling.DoTiling();
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->reduction,
                                            this->hasWeight, this->hasPosWeight);
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                    "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
    return ret;
}

} // namespace optiling
#endif  // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_H