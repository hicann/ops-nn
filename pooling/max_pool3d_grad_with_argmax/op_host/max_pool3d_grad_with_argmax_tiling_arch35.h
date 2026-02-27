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
 * \file max_pool3d_grad_with_argmax_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_ARCH35_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "max_pool3d_grad_with_argmax_tiling.h"
#include "../op_kernel/arch35/max_pool3d_grad_with_argmax_struct.h"
#include "../../pool_3d_common/op_host/arch32/max_pool3d_grad_tiling_constants.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"


namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace MaxPool3DGradWithArgmaxOp;
struct MaxPool3DGradWithArgmaxInputInfo {
    int64_t dPad{0};
    int64_t hPad{0};
    int64_t wPad{0};
    int64_t dStride{1};
    int64_t hStride{1};
    int64_t wStride{1};
    int64_t dKernel{1};
    int64_t hKernel{1};
    int64_t wKernel{1};
    int64_t dDilation{1};
    int64_t hDilation{1};
    int64_t wDilation{1};
    int64_t nX{1};
    int64_t cX{1};
    int64_t dX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t nGrad{1};
    int64_t cGrad{1};
    int64_t dGrad{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    bool ceilMode{false};
    int64_t gradShapeSize{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::DataType indexDtype{ge::DataType::DT_INT32};
    ge::Format inputFormat{ge::Format::FORMAT_NCDHW};
    int64_t isInt32Meet{1};
};

struct MaxPool3DGradWithArgmaxNCDHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t dProBatchSize{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct MaxPool3DGradWithArgmaxNCDHWSplitInfo {
    // DoUBTiling
    uint32_t isCheckRange{0};

    int64_t highAxisInner{0};
    int64_t highAxisTail{0};
    int64_t highAxisOuter{0};

    int64_t dOutputInner{0};
    int64_t dOutputTail{0};
    int64_t dOutputOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
};

class MaxPool3DGradWithArgmaxTilingBaseV35 : public TilingBaseClass {
public:
    explicit MaxPool3DGradWithArgmaxTilingBaseV35(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~MaxPool3DGradWithArgmaxTilingBaseV35() override
    {}

    const std::string nodeName = "MaxPool3DGradWithArgmax";
    MaxPool3DGradWithArgmaxTilingDataV35* tilingData_ = context_->GetTilingData<MaxPool3DGradWithArgmaxTilingDataV35>();
    MaxPool3DGradWithArgmaxInputInfo inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetCntTailTilingParams();
    void SetOtherInputParams();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class MaxPool3DGradWithArgmaxTilingSimt : public MaxPool3DGradWithArgmaxTilingBaseV35 {
public:
    explicit MaxPool3DGradWithArgmaxTilingSimt(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBaseV35(context)
    {}
    ~MaxPool3DGradWithArgmaxTilingSimt() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class MaxPool3DGradWithArgmaxNCDHWTiling : public MaxPool3DGradWithArgmaxTilingBaseV35 {
public:
    explicit MaxPool3DGradWithArgmaxNCDHWTiling(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBaseV35(context)
    {}

    ~MaxPool3DGradWithArgmaxNCDHWTiling() override
    {}
    MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData* tilingData =
        context_->GetTilingData<MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData>();

private:
    void DoUBTiling();
    void InitializationVars();
    bool TrySplitNC();
    bool TrySplitAlignD();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignDHW();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentDWH();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPool3DGradWithArgmaxNCDHWBaseInfo baseData;
    MaxPool3DGradWithArgmaxNCDHWSplitInfo splitData;
};


}  // namespace optiling

#endif
