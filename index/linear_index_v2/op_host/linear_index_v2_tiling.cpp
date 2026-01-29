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
 * \file linear_index_v2_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "linear_index_v2_tiling.h"

constexpr int64_t SCALE_SPACE = 20480;
constexpr int64_t UB_PART = 4;
constexpr int64_t UB_PART_FOR_DISCONTINOUS = 2;
constexpr int64_t MIN_UB_SIZE = 1024;
constexpr int64_t MASK_INDEX = 2;
constexpr int64_t INT_SIZE = 4;
constexpr uint64_t VALID_INDICES = 1;
constexpr uint64_t INVALID_INDICES = 0;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t REGBASE_CCEC_CACHE_SIZE = 8 * 1024;

namespace optiling {
struct computeParam {
    uint64_t formerTime = 0;
    uint64_t tailTime = 0;
    uint64_t formerDataNum = 0;
    uint64_t tailDataNum = 0;
};

class LinearIndexV2Tiling
{
public:
    explicit LinearIndexV2Tiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    void SetTilingKeyMode(bool isInt32Dtype);
    void TilingCompute(const uint64_t idxNum, const uint64_t coreNum);
    void ComputeParam(computeParam& param, uint64_t dataNum) const;
    gert::TilingContext* tilingContext_ = nullptr;
    LinearIndexV2TilingData tilingData_;
    uint64_t workspaceSize_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t dataMaxInUb_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t tensorId_ = 0;
    uint64_t formerCoreNum_ = 0;
    uint64_t formerCoreDataNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t tailCoreDataNum_ = 0;
    uint64_t indicesMask_[MAX_DIM_NUM] = {0};

    computeParam formerCoreParam_;
    computeParam tailCoreParam_;
};

void LinearIndexV2Tiling::ComputeParam(computeParam& param, uint64_t dataNum) const
{
    uint64_t copyTime = (dataNum + dataMaxInUb_ - 1) / dataMaxInUb_;
    param.formerTime = dataNum % copyTime;
    param.tailTime = copyTime - param.formerTime;
    param.formerDataNum = (dataNum + copyTime - 1) / copyTime;
    param.tailDataNum = dataNum / copyTime;
}

void LinearIndexV2Tiling::TilingCompute(const uint64_t idxNum, const uint64_t coreNum)
{
    OP_LOGD(tilingContext_, "common tiling begin");
    dataMaxInUb_ = (ubSize_ / UB_PART) / sizeof(int);
    formerCoreNum_ = idxNum % coreNum;
    tailCoreNum_ = coreNum - formerCoreNum_;
    formerCoreDataNum_ = (idxNum + coreNum - 1) / coreNum;
    tailCoreDataNum_ = idxNum / coreNum;

    ComputeParam(formerCoreParam_, formerCoreDataNum_);
    ComputeParam(tailCoreParam_, tailCoreDataNum_);
    OP_LOGD(tilingContext_, "common tiling end");
}

void LinearIndexV2Tiling::SetTilingKeyMode(bool isInt32Dtype)
{
    tilingKey_ = isInt32Dtype ? 1 : 0;
}

ge::graphStatus LinearIndexV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling start.");
    if (tilingContext_ == nullptr) {
        OP_LOGE(tilingContext_, "tilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = static_cast<const LinearIndexV2CompileInfo*>(tilingContext_->GetCompileInfo());
    auto computeNodeInfoPtr = tilingContext_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, computeNodeInfoPtr);
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxInstanceInfoPtr);
    tensorId_ = idxInstanceInfoPtr->GetInstanceNum();
    OP_CHECK_IF(
        tensorId_ == 0, OP_LOGE(tilingContext_, "indices can not be a empty tensor list"), return ge::GRAPH_FAILED);
    // for ascend950, indices can be [0], which means support non-continuous indices
    uint64_t validIdx = 0;
    if (compileInfo->isAscend950) {
        for (uint64_t i = 0; i < tensorId_; i++) {
            auto idxTensorShapePtr = tilingContext_->GetDynamicInputShape(0, i);
            auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
            auto idxShape = 1;
            for (uint32_t j = 0; j < idxTensorShape.GetDimNum(); j++) {
                idxShape *= idxTensorShape.GetDim(j);
            }
            indicesMask_[i] = idxShape == 0 ? INVALID_INDICES : VALID_INDICES;
            validIdx = idxShape != 0 ? i : validIdx;
        }
    }
    // 1. 取出tensorList中第一个tensor的shape, aclnn中保证了第一个tensor是非空的(For ascend950, need find a valid
    // shape)
    auto idxTensorShapePtr = compileInfo->isAscend950 ? tilingContext_->GetDynamicInputShape(0, validIdx) :
                                                           tilingContext_->GetDynamicInputShape(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxTensorShapePtr);
    auto idxTensorDtypePtr = compileInfo->isAscend950 ? tilingContext_->GetDynamicInputDesc(0, validIdx) :
                                                           tilingContext_->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, idxTensorDtypePtr);
    auto idxDtype = idxTensorDtypePtr->GetDataType();
    if (idxDtype != ge::DT_INT32 && idxDtype != ge::DT_INT64) {
        OP_LOGE(tilingContext_, "indice only support in32 or int64");
        return ge::GRAPH_FAILED;
    }
    SetTilingKeyMode(idxDtype == ge::DT_INT32);
    auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
    uint64_t idxNum = 1;
    for (uint32_t i = 0; i < idxTensorShape.GetDimNum(); i++) {
        idxNum *= idxTensorShape.GetDim(i);
    }

    usedCoreNum_ = std::min(static_cast<uint64_t>(compileInfo->totalCoreNum), idxNum);
    OP_CHECK_IF(
        (usedCoreNum_ == 0), OP_LOGE(tilingContext_, "coreNum in common must greater than 0."),
        return ge::GRAPH_FAILED);

    workspaceSize_ = compileInfo->workspaceSize;
    // 2. 计算ub内一次最多能存放的数据量
    ubSize_ = compileInfo->ubSizePlatForm - SCALE_SPACE;
    OP_CHECK_IF(
        (ubSize_ < MIN_UB_SIZE), OP_LOGE(tilingContext_, "ub size %lu is less than 1024", ubSize_),
        return ge::GRAPH_FAILED);
    // 计算连续场景下的参数
    TilingCompute(idxNum, usedCoreNum_);
    OP_LOGD(tilingContext_, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LinearIndexV2Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext_, "set tiling data start.");
    tilingData_.params.set_usedCoreNum(usedCoreNum_);
    tilingData_.params.set_tensorId(tensorId_);
    tilingData_.params.set_formerCoreNum(formerCoreNum_);
    tilingData_.params.set_formerCoreDataNum(formerCoreDataNum_);
    tilingData_.params.set_tailCoreNum(tailCoreNum_);
    tilingData_.params.set_tailCoreDataNum(tailCoreDataNum_);
    tilingData_.params.set_indicesMask(indicesMask_);

    tilingData_.params.set_formerCoreFormerDataNum(formerCoreParam_.formerDataNum);
    tilingData_.params.set_formerCoreTailDataNum(formerCoreParam_.tailDataNum);
    tilingData_.params.set_formerCoreFormerTime(formerCoreParam_.formerTime);
    tilingData_.params.set_formerCoreTailTime(formerCoreParam_.tailTime);
    tilingData_.params.set_tailCoreFormerDataNum(tailCoreParam_.formerDataNum);
    tilingData_.params.set_tailCoreTailDataNum(tailCoreParam_.tailDataNum);
    tilingData_.params.set_tailCoreFormerTime(tailCoreParam_.formerTime);
    tilingData_.params.set_tailCoreTailTime(tailCoreParam_.tailTime);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    tilingContext_->SetTilingKey(tilingKey_);
    tilingContext_->SetBlockDim(usedCoreNum_);
    auto compileInfo = static_cast<const LinearIndexV2CompileInfo*>(tilingContext_->GetCompileInfo());
    if (compileInfo->isAscend950) {
        tilingContext_->SetLocalMemorySize(compileInfo->ubSizePlatForm - DCACHE_SIZE - REGBASE_CCEC_CACHE_SIZE);
    }
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    TilingDataPrint();
    OP_LOGD(tilingContext_, "set tiling data end.");
    return ge::GRAPH_SUCCESS;
}

void LinearIndexV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "tilingKey:              %lu", tilingKey_);
    OP_LOGD(tilingContext_, "usedCoreNum:            %lu", usedCoreNum_);
    OP_LOGD(tilingContext_, "tensorId:               %lu", tensorId_);
    OP_LOGD(tilingContext_, "formerCoreNum:          %lu", formerCoreNum_);
    OP_LOGD(tilingContext_, "formerCoreDataNum:      %lu", formerCoreDataNum_);
    OP_LOGD(tilingContext_, "formerCoreFormerDataNum:%lu", formerCoreParam_.formerDataNum);
    OP_LOGD(tilingContext_, "formerCoreTailDataNum:  %lu", formerCoreParam_.tailDataNum);
    OP_LOGD(tilingContext_, "formerCoreFormerTime:   %lu", formerCoreParam_.formerTime);
    OP_LOGD(tilingContext_, "formerCoreTailTime:     %lu", formerCoreParam_.tailTime);
    OP_LOGD(tilingContext_, "tailCoreNum:            %lu", tailCoreNum_);
    OP_LOGD(tilingContext_, "tailCoreDataNum:        %lu", tailCoreDataNum_);
    OP_LOGD(tilingContext_, "tailCoreFormerDataNum:  %lu", tailCoreParam_.formerDataNum);
    OP_LOGD(tilingContext_, "tailCoreTailDataNum:    %lu", tailCoreParam_.tailDataNum);
    OP_LOGD(tilingContext_, "tailCoreFormerTime:     %lu", tailCoreParam_.formerTime);
    OP_LOGD(tilingContext_, "tailCoreTailTime:       %lu", tailCoreParam_.tailTime);
    for (size_t i = 0; i < MAX_DIM_NUM; i++) {
        OP_LOGD(tilingContext_, "indicesMask_%lu:     %lu", i, indicesMask_[i]);
    }
}

ge::graphStatus TilingLinearIndexV2(gert::TilingContext* context)
{
    LinearIndexV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForLinearIndexV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForLinearIndexV2 start");
    auto compileInfo = context->GetCompiledInfo<LinearIndexV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo->isAscend950 =
        ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 ? true : false;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForLinearIndexV2 end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LinearIndexV2)
    .InputsDataDependency({MASK_INDEX})
    .Tiling(TilingLinearIndexV2)
    .TilingParse<LinearIndexV2CompileInfo>(TilingPrepareForLinearIndexV2);
} // namespace optiling