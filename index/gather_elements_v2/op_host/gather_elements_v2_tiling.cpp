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
 * \file gather_elements_v2_tiling.cpp
 * \brief
 */

#include "gather_elements_v2_last_dim_tiling.h"
#include "platform/platform_infos_def.h"

namespace optiling {
constexpr uint64_t RESERVED_UB_SIZE = 2048;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t TRANSPOSE_WS_LEN = 128;
constexpr uint64_t X_IDX = 0;
constexpr uint64_t INDEX_IDX = 1;
constexpr uint64_t NUM_TWO = 2;

class GatherElementsV2Tiling
{
public:
    explicit GatherElementsV2Tiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline void SetTilingKeyMode();
    inline void Tiling4GatherElementsV2();
    inline void Tiling4Scalar();
    inline bool CalcMaxBufferSize();
    inline void CalcMaxIdxDataAlign();

    GatherElementsV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;

private:
    uint64_t usedCoreNum_ = 0;
    uint64_t tilingKey_ = 0;
    ge::DataType xDtype_;
    ge::DataType idxDtype_;
    uint64_t xDtypeSize_;

    uint64_t xPreDim_ = 1;        // x的前置shape
    uint64_t xGatherDim_ = 1;     // x在Gather轴的shape
    uint64_t xPostDim_ = 1;       // x的后置shape
    uint64_t idxPreDim_ = 1;      // idx的前置shape
    uint64_t idxGatherDim_ = 1;   // idx在Gather轴的shape
    uint64_t idxPostDim_ = 1;     // idx的后置shape
    uint64_t coreGroupNum_ = 0;   // 核的组数
    uint64_t formerGroupNum_ = 0; // 大组数
    uint64_t tailGroupNum_ = 0;   // 小组
    uint64_t formerGroupPreDim_ = 0;
    uint64_t tailGroupPreDim_ = 0;
    uint64_t formerGroupCoreNum_ = 0; // 大组中的大核数
    uint64_t tailGroupCoreNum_ = 0;   // 大组中的小核数

    uint64_t formerGroupFormerNum_ = 0;
    uint64_t formerGroupTailNum_ = 0;
    uint64_t formerGroupFormerPostDim_ = 0;
    uint64_t formerGroupTailPostDim_ = 0;

    uint64_t tailGroupFormerNum_ = 0;
    uint64_t tailGroupTailNum_ = 0;
    uint64_t tailGroupFormerPostDim_ = 0;
    uint64_t tailGroupTailPostDim_ = 0;

    // transpose
    uint64_t carryNumAlign_ = 0;
    uint64_t xCarryNumAlign_ = 0;
    uint64_t idxCarryNumAlign_ = 0;
    uint64_t inBufferSize_ = 0;
    uint64_t outBufferSize_ = 0;
    uint64_t transGatherDimSlice_ = TRANSPOSE_WS_LEN;
    uint64_t idxGatherDimSlice_ = 0;
    uint64_t workspacePerBlock_ = 0;
    bool needTrans_ = false;

    // scalar
    uint64_t formerGroupFormerData_ = 0;
    uint64_t formerGroupTailData_ = 0;
    uint64_t tailGroupFormerData_ = 0;
    uint64_t tailGroupTailData_ = 0;
    uint64_t maxIdxDataAlign_ = 0;
};

inline void GatherElementsV2Tiling::SetTilingKeyMode()
{
    tilingKey_ = needTrans_;
    tilingContext_->SetTilingKey(tilingKey_);
}

inline bool GatherElementsV2Tiling::CalcMaxBufferSize()
{
    carryNumAlign_ = CACHELINE / xDtypeSize_;
    xCarryNumAlign_ = CACHELINE / xDtypeSize_;
    idxCarryNumAlign_ = CACHELINE / sizeof(int32_t);
    uint64_t xAlign = BLOCK_SIZE / xDtypeSize_;
    uint64_t idxAlign = BLOCK_SIZE / sizeof(int32_t);
    uint64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    uint64_t minIdxGatherDimSlice = CACHELINE;
    uint64_t gatherInBufferSize = Ops::Base::CeilAlign(xGatherDim_, xAlign) * xDtypeSize_ +
                                  Ops::Base::CeilAlign(minIdxGatherDimSlice, idxAlign) * sizeof(int32_t) * NUM_TWO;
    uint64_t gatherOutBufferSize = Ops::Base::CeilAlign(minIdxGatherDimSlice, xAlign) * xDtypeSize_;
    uint64_t transInBufferSize = TRANSPOSE_WS_LEN * CACHELINE;  // 计算transpose所需输入UB
    uint64_t transOutBufferSize = TRANSPOSE_WS_LEN * CACHELINE; // 计算transpose所需输出UB
    inBufferSize_ = std::max(gatherInBufferSize, transInBufferSize);
    outBufferSize_ = std::max(gatherOutBufferSize, transOutBufferSize);
    if (availableUb < (inBufferSize_ + outBufferSize_)) {
        return false;
    }
    uint64_t ubLeft =
        availableUb -
        (std::max(transInBufferSize, Ops::Base::CeilAlign(xGatherDim_, xAlign) * xDtypeSize_) + transOutBufferSize);
    uint64_t maxIdxGatherDimSlice = ubLeft / (BLOCK_SIZE * NUM_TWO) * idxAlign;
    uint64_t idxGatherDimAlign = Ops::Base::CeilAlign(idxGatherDim_, idxAlign);
    idxGatherDimSlice_ = std::min(maxIdxGatherDimSlice, idxGatherDimAlign);
    gatherInBufferSize = Ops::Base::CeilAlign(xGatherDim_, xAlign) * xDtypeSize_ +
                         Ops::Base::CeilAlign(idxGatherDimSlice_, idxAlign) * sizeof(int32_t);
    gatherOutBufferSize = Ops::Base::CeilAlign(idxGatherDimSlice_, xAlign) * xDtypeSize_;
    inBufferSize_ = std::max(gatherInBufferSize, transInBufferSize);
    outBufferSize_ = std::max(gatherOutBufferSize, transOutBufferSize);
    return true;
}

inline void GatherElementsV2Tiling::CalcMaxIdxDataAlign()
{
    uint64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    maxIdxDataAlign_ = availableUb / BLOCK_SIZE * BLOCK_SIZE / (sizeof(int32_t) + xDtypeSize_);
}

inline void GatherElementsV2Tiling::Tiling4Scalar()
{
    OP_LOGD(tilingContext_, "Tiling4ScalarMode start");
    uint64_t idxDataPerPre = idxGatherDim_ * idxPostDim_;
    uint64_t idxAllData = idxPreDim_ * idxDataPerPre;
    usedCoreNum_ = std::min(idxAllData, coreNum_);
    if (idxPreDim_ >= usedCoreNum_) {
        formerGroupFormerData_ = idxDataPerPre;
        formerGroupTailData_ = idxDataPerPre;
        tailGroupFormerData_ = idxDataPerPre;
        tailGroupTailData_ = idxDataPerPre;
    } else {
        coreGroupNum_ = idxPreDim_;
        tailGroupNum_ = (coreGroupNum_ - usedCoreNum_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;

        formerGroupPreDim_ = 1;
        tailGroupPreDim_ = 1;
        formerGroupCoreNum_ = (usedCoreNum_ + coreGroupNum_ - 1) / coreGroupNum_;
        tailGroupCoreNum_ = usedCoreNum_ / coreGroupNum_;

        formerGroupTailNum_ = (formerGroupCoreNum_ - idxDataPerPre % formerGroupCoreNum_) % formerGroupCoreNum_;
        formerGroupFormerNum_ = formerGroupCoreNum_ - formerGroupTailNum_;
        formerGroupFormerData_ = (idxDataPerPre + formerGroupCoreNum_ - 1) / formerGroupCoreNum_;
        formerGroupTailData_ = idxDataPerPre / formerGroupCoreNum_;

        tailGroupTailNum_ = (tailGroupCoreNum_ - idxDataPerPre % tailGroupCoreNum_) % tailGroupCoreNum_;
        tailGroupFormerNum_ = tailGroupCoreNum_ - tailGroupTailNum_;
        tailGroupFormerData_ = (idxDataPerPre + tailGroupCoreNum_ - 1) / tailGroupCoreNum_;
        tailGroupTailData_ = idxDataPerPre / tailGroupCoreNum_;
    }
    CalcMaxIdxDataAlign();
    OP_LOGD(tilingContext_, "Tiling4ScalarMode finished");
}

inline void GatherElementsV2Tiling::Tiling4GatherElementsV2()
{
    OP_LOGD(tilingContext_, "Tiling4GatherElementsV2 start");

    usedCoreNum_ = std::min(idxPreDim_ * idxPostDim_, coreNum_);
    // 行分核
    if (idxPreDim_ > usedCoreNum_) {
        coreGroupNum_ = usedCoreNum_;
        tailGroupNum_ = (coreGroupNum_ - idxPreDim_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;
        if (usedCoreNum_ == 0UL) {
            usedCoreNum_ = 1UL;
        }
        formerGroupPreDim_ = (idxPreDim_ + usedCoreNum_ - 1) / usedCoreNum_;
        tailGroupPreDim_ = idxPreDim_ / usedCoreNum_;
        formerGroupCoreNum_ = 1;
        tailGroupCoreNum_ = 1;

        formerGroupTailNum_ = 0;
        formerGroupFormerNum_ = 1;
        tailGroupTailNum_ = 0;
        tailGroupFormerNum_ = 1;

        formerGroupFormerPostDim_ = idxPostDim_;
        formerGroupTailPostDim_ = idxPostDim_;
        tailGroupFormerPostDim_ = idxPostDim_;
        tailGroupTailPostDim_ = idxPostDim_;
    } else { // 行和列分核
        coreGroupNum_ = idxPreDim_;
        tailGroupNum_ = (coreGroupNum_ - usedCoreNum_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;

        formerGroupPreDim_ = 1;
        tailGroupPreDim_ = 1;
        formerGroupCoreNum_ = (usedCoreNum_ + coreGroupNum_ - 1) / coreGroupNum_;
        tailGroupCoreNum_ = usedCoreNum_ / coreGroupNum_;

        formerGroupTailNum_ = (formerGroupCoreNum_ - idxPostDim_ % formerGroupCoreNum_) % formerGroupCoreNum_;
        formerGroupFormerNum_ = formerGroupCoreNum_ - formerGroupTailNum_;
        formerGroupFormerPostDim_ = (idxPostDim_ + formerGroupCoreNum_ - 1) / formerGroupCoreNum_;
        formerGroupTailPostDim_ = idxPostDim_ / formerGroupCoreNum_;

        tailGroupTailNum_ = (tailGroupCoreNum_ - idxPostDim_ % tailGroupCoreNum_) % tailGroupCoreNum_;
        tailGroupFormerNum_ = tailGroupCoreNum_ - tailGroupTailNum_;
        tailGroupFormerPostDim_ = (idxPostDim_ + tailGroupCoreNum_ - 1) / tailGroupCoreNum_;
        tailGroupTailPostDim_ = idxPostDim_ / tailGroupCoreNum_;
    }

    OP_LOGD(tilingContext_, "Tiling4GatherElementsV2 finished");
}

ge::graphStatus GatherElementsV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "GatherElementsV2Tiling initing");
    auto compileInfo = static_cast<const GatherElementsV2CompileInfo*>(tilingContext_->GetCompileInfo());
    auto attrs = tilingContext_->GetAttrs();
    auto xShape = tilingContext_->GetInputShape(X_IDX)->GetStorageShape();
    auto idxShape = tilingContext_->GetInputShape(INDEX_IDX)->GetStorageShape();
    auto dim = *(attrs->GetAttrPointer<uint64_t>)(0);
    auto dimNum = xShape.GetDimNum();
    idxGatherDim_ = idxShape.GetDim(dim);
    xGatherDim_ = xShape.GetDim(dim);
    for (size_t i = 0; i < dim; i++) {
        idxPreDim_ *= idxShape.GetDim(i);
        xPreDim_ *= xShape.GetDim(i);
    }
    for (size_t i = dim + 1; i < dimNum; i++) {
        idxPostDim_ *= idxShape.GetDim(i);
        xPostDim_ *= xShape.GetDim(i);
    }
    xDtype_ = tilingContext_->GetInputDesc(X_IDX)->GetDataType();
    idxDtype_ = tilingContext_->GetInputDesc(INDEX_IDX)->GetDataType();
    xDtypeSize_ = GetSizeByDataType(xDtype_);

    coreNum_ = compileInfo->totalCoreNum;
    ubSize_ = compileInfo->ubSizePlatForm;
    Tiling4GatherElementsV2();
    size_t transWorkspaceSize = 0;
    if (CalcMaxBufferSize()) {
        needTrans_ = true;
        if (xDtypeSize_ == 0UL) {
            OP_LOGE("GatherElementsV2", "xDtypeSize is 0.");
            return ge::GRAPH_FAILED;
        }
        uint64_t xGatherDimAlign = Ops::Base::CeilAlign(xGatherDim_ * xDtypeSize_, sizeof(int32_t)) / xDtypeSize_;
        uint64_t usedWorkspaceLen =
            std::min(carryNumAlign_, std::max(formerGroupFormerPostDim_, tailGroupFormerPostDim_));
        workspacePerBlock_ = usedWorkspaceLen * (xGatherDimAlign * xDtypeSize_ + idxGatherDim_ * sizeof(int32_t));
        transWorkspaceSize = usedCoreNum_ * workspacePerBlock_;
    } else {
        needTrans_ = false;
        Tiling4Scalar();
    }
    size_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    sysWorkspaceSize = sysWorkspaceSize + transWorkspaceSize;
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    SetTilingKeyMode();

    OP_LOGD(tilingContext_, "GatherElementsV2Tiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingData_.params.set_xPreDim(xPreDim_);
    tilingData_.params.set_xGatherDim(xGatherDim_);
    tilingData_.params.set_xPostDim(xPostDim_);
    tilingData_.params.set_idxPreDim(idxPreDim_);
    tilingData_.params.set_idxGatherDim(idxGatherDim_);
    tilingData_.params.set_idxPostDim(idxPostDim_);

    tilingData_.params.set_coreGroupNum(coreGroupNum_);
    tilingData_.params.set_formerGroupNum(formerGroupNum_);
    tilingData_.params.set_tailGroupNum(tailGroupNum_);

    tilingData_.params.set_formerGroupPreDim(formerGroupPreDim_);
    tilingData_.params.set_tailGroupPreDim(tailGroupPreDim_);
    tilingData_.params.set_formerGroupCoreNum(formerGroupCoreNum_);
    tilingData_.params.set_tailGroupCoreNum(tailGroupCoreNum_);

    tilingData_.params.set_formerGroupFormerNum(formerGroupFormerNum_);
    tilingData_.params.set_formerGroupTailNum(formerGroupTailNum_);
    tilingData_.params.set_formerGroupFormerPostDim(formerGroupFormerPostDim_);
    tilingData_.params.set_formerGroupTailPostDim(formerGroupTailPostDim_);

    tilingData_.params.set_tailGroupFormerNum(tailGroupFormerNum_);
    tilingData_.params.set_tailGroupTailNum(tailGroupTailNum_);
    tilingData_.params.set_tailGroupFormerPostDim(tailGroupFormerPostDim_);
    tilingData_.params.set_tailGroupTailPostDim(tailGroupTailPostDim_);

    tilingData_.transTiling.set_carryNumAlign(carryNumAlign_);
    tilingData_.transTiling.set_xCarryNumAlign(xCarryNumAlign_);
    tilingData_.transTiling.set_idxCarryNumAlign(idxCarryNumAlign_);
    tilingData_.transTiling.set_inBufferSize(inBufferSize_);
    tilingData_.transTiling.set_outBufferSize(outBufferSize_);
    tilingData_.transTiling.set_transGatherDimSlice(transGatherDimSlice_);
    tilingData_.transTiling.set_idxGatherDimSlice(idxGatherDimSlice_);
    tilingData_.transTiling.set_workspacePerBlock(workspacePerBlock_);

    tilingData_.scalarTiling.set_formerGroupFormerData(formerGroupFormerData_);
    tilingData_.scalarTiling.set_formerGroupTailData(formerGroupTailData_);
    tilingData_.scalarTiling.set_tailGroupFormerData(tailGroupFormerData_);
    tilingData_.scalarTiling.set_tailGroupTailData(tailGroupTailData_);
    tilingData_.scalarTiling.set_maxIdxDataAlign(maxIdxDataAlign_);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void GatherElementsV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "tilingKey:                  %lu", tilingKey_);
    OP_LOGD(tilingContext_, "usedCoreNum:                %lu", usedCoreNum_);
    OP_LOGD(tilingContext_, "xPreDim:                    %lu", xPreDim_);
    OP_LOGD(tilingContext_, "xGatherDim:                 %lu", xGatherDim_);
    OP_LOGD(tilingContext_, "xPostDim:                   %lu", xPostDim_);
    OP_LOGD(tilingContext_, "idxPreDim:                  %lu", idxPreDim_);
    OP_LOGD(tilingContext_, "idxGatherDim:               %lu", idxGatherDim_);
    OP_LOGD(tilingContext_, "idxPostDim_:                %lu", idxPostDim_);

    OP_LOGD(tilingContext_, "coreGroupNum_:              %lu", coreGroupNum_);
    OP_LOGD(tilingContext_, "formerGroupNum:             %lu", formerGroupNum_);
    OP_LOGD(tilingContext_, "tailGroupNum:               %lu", tailGroupNum_);

    OP_LOGD(tilingContext_, "formerGroupPreDim:          %lu", formerGroupPreDim_);
    OP_LOGD(tilingContext_, "tailGroupPreDim:            %lu", tailGroupPreDim_);
    OP_LOGD(tilingContext_, "formerGroupCoreNum:         %lu", formerGroupCoreNum_);
    OP_LOGD(tilingContext_, "tailGroupCoreNum:           %lu", tailGroupCoreNum_);

    OP_LOGD(tilingContext_, "formerGroupFormerNum:       %lu", formerGroupFormerNum_);
    OP_LOGD(tilingContext_, "formerGroupTailNum:         %lu", formerGroupTailNum_);
    OP_LOGD(tilingContext_, "formerGroupFormerPostDim:   %lu", formerGroupFormerPostDim_);
    OP_LOGD(tilingContext_, "formerGroupTailPostDim:     %lu", formerGroupTailPostDim_);

    OP_LOGD(tilingContext_, "tailGroupFormerNum:         %lu", tailGroupFormerNum_);
    OP_LOGD(tilingContext_, "tailGroupTailNum:           %lu", tailGroupTailNum_);
    OP_LOGD(tilingContext_, "tailGroupFormerPostDim:     %lu", tailGroupFormerPostDim_);
    OP_LOGD(tilingContext_, "tailGroupTailPostDim:       %lu", tailGroupTailPostDim_);

    OP_LOGD(tilingContext_, "carryNumAlign:              %lu", carryNumAlign_);
    OP_LOGD(tilingContext_, "xCarryNumAlign:             %lu", xCarryNumAlign_);
    OP_LOGD(tilingContext_, "idxCarryNumAlign:           %lu", idxCarryNumAlign_);

    OP_LOGD(tilingContext_, "inBufferSize:               %lu", inBufferSize_);
    OP_LOGD(tilingContext_, "outBufferSize:              %lu", outBufferSize_);
    OP_LOGD(tilingContext_, "transGatherDimSlice:        %lu", transGatherDimSlice_);
    OP_LOGD(tilingContext_, "idxGatherDimSlice:          %lu", idxGatherDimSlice_);
    OP_LOGD(tilingContext_, "workspacePerBlock:          %lu", workspacePerBlock_);

    OP_LOGD(tilingContext_, "formerGroupFormerData:      %lu", formerGroupFormerData_);
    OP_LOGD(tilingContext_, "formerGroupTailData:        %lu", formerGroupTailData_);
    OP_LOGD(tilingContext_, "tailGroupFormerData:        %lu", tailGroupFormerData_);
    OP_LOGD(tilingContext_, "tailGroupTailData:          %lu", tailGroupTailData_);
    OP_LOGD(tilingContext_, "maxIdxDataAlign:            %lu", maxIdxDataAlign_);
}

ge::graphStatus TilingGatherElementsV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingGatherElementsV2 enter.");
    if (context == nullptr) {
        OP_LOGE("GatherElementsV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr"), return ge::GRAPH_FAILED);
    auto dim = *(attr->GetAttrPointer<uint64_t>)(0);
    auto dimNum = context->GetInputShape(X_IDX)->GetStorageShape().GetDimNum();
    if (dimNum == 0UL) {
        OP_LOGE("GatherElementsV2", "x's dim num is 0.");
        return ge::GRAPH_FAILED;
    }
    if ((dim + dimNum) % dimNum != dimNum - 1) {
        GatherElementsV2Tiling tilingObject(context);
        OP_CHECK_IF(
            tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "tiling init fail"),
            return ge::GRAPH_FAILED);
        return tilingObject.SetKernelTiling();
    } else {
        OP_LOGD(context->GetNodeName(), "TilingGatherElementsV2 last dim enter.");
        GatherElementsV2LastDimTiling tilingObject(context);
        OP_CHECK_IF(
            tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "tiling init fail"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForGatherElementsV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For GatherElementsV2 start");
    auto compileInfo = context->GetCompiledInfo<GatherElementsV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "Tiling prepare for gather elements v2 end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherElementsV2)
    .Tiling(TilingGatherElementsV2)
    .TilingParse<GatherElementsV2CompileInfo>(TilingPrepareForGatherElementsV2);
} // namespace optiling