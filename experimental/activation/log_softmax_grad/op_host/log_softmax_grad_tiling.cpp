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
 * \file log_softmax_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h" // Ops::Base::GetUbBlockSize
#include "../op_kernel/log_softmax_grad_tiling_data.h"
#include "../op_kernel/log_softmax_grad_tiling_key.h"

namespace optiling {
using namespace ge;
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

struct LogSoftmaxGradCompileInfo {};

constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint64_t BUFFER_NUM = 2; // 是否开启Double buffer
constexpr uint64_t NO_REDUCE_BUF_NUM = 3 * BUFFER_NUM;
constexpr uint64_t REDUCE_BUF_NUM = 3 * BUFFER_NUM + 1;
constexpr uint64_t REDUCE_MID_NODB_BUF_NUM = 3;
constexpr uint64_t REDUCE_TAIL_NODB_BUF_NUM = 4;
constexpr uint64_t TRANS_MIN_MULT = 16;
constexpr uint64_t BPE_4 = 4;
constexpr uint64_t DY_IDX = 0;
constexpr uint64_t X_IDX = 1;
constexpr uint64_t Z_IDX = 0;

static graphStatus TilingParseForLogSoftmaxGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingParseForLogSoftmaxGrad start\n");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return GRAPH_NULL_PTR);
    OP_LOGD(context, "TilingParseForLogSoftmaxGrad end\n");
    return GRAPH_SUCCESS;
}

class LogSoftmaxGradTiling {
public:
    LogSoftmaxGradTiling() = delete;

    explicit LogSoftmaxGradTiling(gert::TilingContext* context) : context_(context) {}

    graphStatus Init()
    {
        tilingData_ = context_->GetTilingData<LogSoftmaxGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
        graphStatus ret = GetUbSizeAndCoreNum();
        OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context_, "GetUbSizeAndCoreNum failed"), return ret);
        blockSize_ = Ops::Base::GetUbBlockSize(context_);
        fp32ElemsPerBlock_ = blockSize_ / sizeof(float);
        transMinSize_ = TRANS_MIN_MULT * fp32ElemsPerBlock_;
        OP_LOGD(context_, "ubSize: %lu, coreNum: %lu, blockSize: %lu", ubSize_, coreNum_, blockSize_);

        // 上面的CheckParams函数会校验参数的合理性，这里无需重复校验
        const gert::Shape& inShape = context_->GetInputShape(DY_IDX)->GetStorageShape();
        const DataType inDtype = context_->GetInputDesc(DY_IDX)->GetDataType();
        inBpe_ = GetSizeByDataType(inDtype);
        dimNum_ = static_cast<int64_t>(inShape.GetDimNum());
        ret = GetReduceAxes();
        OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context_, "GetReduceAxes failed"), return ret);

        // 合并轴
        int64_t axisStart = *(reduceAxes_.begin()), axisEnd = *(reduceAxes_.rbegin()) + 1;
        for (int64_t i = 0; i < axisStart; i++) {
            mergedDim0_ *= inShape.GetDim(i);
        }
        for (int64_t i = axisStart; i < axisEnd; i++) {
            mergedDim1_ *= inShape.GetDim(i);
        }
        for (int64_t i = axisEnd; i < dimNum_; i++) {
            mergedDim2_ *= inShape.GetDim(i);
        }

        return GRAPH_SUCCESS;
    }

    graphStatus CheckParams()
    {
        const auto dyShapePtr = context_->GetInputShape(DY_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dyShapePtr);
        const auto xShapePtr = context_->GetInputShape(X_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
        const auto zShapePtr = context_->GetOutputShape(Z_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, zShapePtr);
        const auto dyDescPtr = context_->GetInputDesc(DY_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dyDescPtr);
        const auto xDescPtr = context_->GetInputDesc(X_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xDescPtr);
        const auto zDescPtr = context_->GetOutputDesc(Z_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, zDescPtr);

        const auto dyShape = dyShapePtr->GetStorageShape();
        const auto xShape = xShapePtr->GetStorageShape();
        const auto zShape = zShapePtr->GetStorageShape();
        OP_CHECK_IF(dyShape != xShape, OP_LOGE(context_, "dy and x shape must be the same"), return GRAPH_FAILED);
        OP_CHECK_IF(dyShape != zShape, OP_LOGE(context_, "dy and z shape must be the same"), return GRAPH_FAILED);

        const auto dyDtype = dyDescPtr->GetDataType();
        const auto xDtype = xDescPtr->GetDataType();
        const auto zDtype = zDescPtr->GetDataType();
        OP_CHECK_IF(dyDtype != xDtype, OP_LOGE(context_, "dy and x dtype must be the same"), return GRAPH_FAILED);
        OP_CHECK_IF(dyDtype != zDtype, OP_LOGE(context_, "dy and z dtype must be the same"), return GRAPH_FAILED);
        OP_CHECK_IF(dyDtype != DataType::DT_FLOAT && dyDtype != DataType::DT_BF16 && dyDtype != DataType::DT_FLOAT16,
                    OP_LOGE(context_, "Dtype only support float32, float16 or bfloat16"), return GRAPH_FAILED);

        return GRAPH_SUCCESS;
    }

    graphStatus DoTiling()
    {
        if (mergedDim1_ == 1) {
            schMode_ = NO_NEED_REDUCE;
            ProcNoNeedReduce();
        } else {
            if (mergedDim2_ == 1) {
                schMode_ = REDUCE_TAIL;
                ProcReduceTail();
            } else {
                schMode_ = REDUCE_MID;
                ProcReduceMid();
            }
        }

        tilingData_->singleBufElems = singleBufElems_;
        tilingData_->mergedDim0 = mergedDim0_;
        tilingData_->mergedDim1 = mergedDim1_;
        tilingData_->mergedDim2 = mergedDim2_;
        tilingData_->dim0Tile = dim0Tile_;
        tilingData_->dim1Tile = dim1Tile_;
        tilingData_->dim2Tile = dim2Tile_;
        tilingData_->totalElems = totalElems_;
        tilingData_->dim0LoopTime = dim0LoopTime_;
        tilingData_->dim0Remained = dim0Remained_;
        tilingData_->dim1LoopTime = dim1LoopTime_;
        tilingData_->dim1Remained = dim1Remained_;
        tilingData_->dim2LoopTime = dim2LoopTime_;
        tilingData_->dim2Remained = dim2Remained_;

        context_->SetTilingKey(GET_TPL_TILING_KEY(schMode_, isSmallShape_, isContiguous_));
        context_->SetBlockDim(usedCoreNum_);

        std::string schModeStr = schMode_ == NO_NEED_REDUCE ? "NO_NEED_REDUCE" :
                                                              (schMode_ == REDUCE_TAIL ? "REDUCE_TAIL" : "REDUCE_MID");

        OP_LOGD(context_,
                "TilingData info, bpe: %d, schMode: %s, usedCoreNum: %lu, singleBufElems: %lu, isSmallShape: %s, "
                "isContiguous: %s, mergedDim0: %lu, mergedDim1: %lu, mergedDim2: %lu, dim0Tile: %lu, dim1Tile: %lu, "
                "dim2Tile: %lu, totalElems: %lu, dim0LoopTime: %lu, dim0Remained: %lu, dim1LoopTime: %lu, "
                "dim1Remained: %lu, dim2LoopTime: %lu, dim2Remained: %lu",
                inBpe_, schModeStr.c_str(), usedCoreNum_, tilingData_->singleBufElems, isSmallShape_ ? "true" : "false",
                isContiguous_ ? "true" : "false", tilingData_->mergedDim0, tilingData_->mergedDim1,
                tilingData_->mergedDim2, tilingData_->dim0Tile, tilingData_->dim1Tile, tilingData_->dim2Tile,
                tilingData_->totalElems, tilingData_->dim0LoopTime, tilingData_->dim0Remained,
                tilingData_->dim1LoopTime, tilingData_->dim1Remained, tilingData_->dim2LoopTime,
                tilingData_->dim2Remained);

        graphStatus ret = SetWorkspaceSize();
        OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context_, "SetWorkspaceSize failed"), return ret);
        return GRAPH_SUCCESS;
    }

private:
    graphStatus SetWorkspaceSize(const size_t usrSize = 0)
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        return GRAPH_SUCCESS;
    }

    graphStatus GetUbSizeAndCoreNum()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        coreNum_ = ascendcPlatform.GetCoreNum();
        OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "Invalid coreNum_, must > 0"), return GRAPH_FAILED);
        OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "Invalid ubSize, must > 0"), return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }

    graphStatus GetReduceAxes()
    {
        const gert::RuntimeAttrs* attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        const auto axesPtr = attrs->GetListInt(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, axesPtr);
        const size_t axesSize = axesPtr->GetSize();
        OP_CHECK_IF(axesSize == 0, OP_LOGE(context_, "Invalid attr 'axis', axis size is 0"), return GRAPH_FAILED);
        const int64_t* axesData = axesPtr->GetData();
        OP_CHECK_NULL_WITH_CONTEXT(context_, axesData);

        for (size_t i = 0; i < axesSize; i++) {
            const auto dimVal = axesData[i] >= 0 ? axesData[i] : dimNum_ + axesData[i];
            OP_CHECK_IF(dimVal < 0 || dimVal >= dimNum_,
                        OP_LOGE(context_, "Invalid reduce dim, axesData[%zu]: %ld", i, axesData[i]),
                        return GRAPH_FAILED);
            reduceAxes_.insert(dimVal);
        }

        // 如果存在多个 reduce 轴，为了符合算子语义，这些轴必须是连续的，比如 reduceAxes_ = [3, 4, 5]
        auto preIter = reduceAxes_.begin();
        auto curIter = reduceAxes_.begin();
        for (curIter++; curIter != reduceAxes_.end(); ++curIter, ++preIter) {
            OP_CHECK_IF(*curIter != *preIter + 1,
                        OP_LOGE(context_, "Reduce dims should be continuous, %ld != %ld + 1", *curIter, *preIter),
                        return GRAPH_FAILED);
        }

        return GRAPH_SUCCESS;
    }

    uint64_t GetSingleBufElems(uint64_t bufNum)
    {
        // 因为中间计算都是float32，所以计算单个buf最大容量时使用bpe 4
        auto bufElems = FloorAlign(ubSize_ / bufNum, blockSize_) / BPE_4;
        if (inBpe_ != BPE_4) {
            // 当bpe不等于4时，因为要将数据搬运到buffer的后半部分，再做cast到整个buffer，需要使用singleBufElems做偏移，因此需要对齐
            bufElems = FloorAlign(bufElems, blockSize_ / inBpe_);
        }
        return bufElems;
    }

    void ProcNoNeedReduce()
    {
        singleBufElems_ = GetSingleBufElems(NO_REDUCE_BUF_NUM);
        totalElems_ = mergedDim0_ * mergedDim2_;
        usedCoreNum_ = std::min(CeilDiv(totalElems_, singleBufElems_), coreNum_);
    }

    void ProcReduceTail()
    {
        const uint64_t elemsPerBlock = blockSize_ / inBpe_;
        mergedDim2_ = mergedDim1_;
        mergedDim1_ = mergedDim0_;
        mergedDim0_ = 1;
        singleBufElems_ = GetSingleBufElems(REDUCE_TAIL_NODB_BUF_NUM);
        uint64_t mergedDim2Align = CeilAlign(mergedDim2_, elemsPerBlock);
        if (mergedDim2Align <= singleBufElems_) {
            isSmallShape_ = true;
            dim2Tile_ = mergedDim2_;
            dim2LoopTime_ = 1;
            dim2Remained_ = 0;
            dim1Tile_ = CeilDiv(mergedDim1_, coreNum_);
            uint64_t dim1Max = singleBufElems_ / mergedDim2_;
            if (mergedDim2_ % fp32ElemsPerBlock_ == 0) {
                isContiguous_ = true;
                dim1Tile_ = std::min(dim1Tile_, dim1Max);
            } else {
                if (dim1Max >= transMinSize_ && mergedDim2_ <= MAX_REPEAT_TIME) {
                    isContiguous_ = true;
                    dim1Tile_ = std::min(dim1Tile_, FloorAlign(dim1Max, transMinSize_));
                } else {
                    dim1Tile_ = std::min(dim1Tile_, singleBufElems_ / mergedDim2Align);
                }
            }
            dim1Tile_ = std::min(dim1Tile_, mergedDim1_);
            dim1LoopTime_ = mergedDim1_ / dim1Tile_;
            dim1Remained_ = mergedDim1_ - dim1LoopTime_ * dim1Tile_;
        } else {
            isContiguous_ = true;
            singleBufElems_ = GetSingleBufElems(REDUCE_BUF_NUM);
            dim2Tile_ = singleBufElems_;
            dim2LoopTime_ = mergedDim2_ / dim2Tile_;
            dim2Remained_ = mergedDim2_ - dim2LoopTime_ * dim2Tile_;
            dim1Tile_ = 1;
            dim1LoopTime_ = mergedDim1_;
            dim1Remained_ = 0;
        }
        uint64_t needCoreNum = dim1LoopTime_ + (dim1Remained_ ? 1 : 0);
        usedCoreNum_ = std::min(needCoreNum, coreNum_);
    }

    void ProcReduceMid()
    {
        const uint64_t elemsPerBlock = blockSize_ / inBpe_;
        uint64_t mergedDim2Align = CeilAlign(mergedDim2_, elemsPerBlock);
        singleBufElems_ = GetSingleBufElems(REDUCE_MID_NODB_BUF_NUM);
        dim2Tile_ = singleBufElems_ / mergedDim1_;
        if (dim2Tile_ >= elemsPerBlock) {
            isSmallShape_ = true;
            dim1Tile_ = mergedDim1_;
            dim1LoopTime_ = 1;
            dim1Remained_ = 0;
            dim2Tile_ = FloorAlign(dim2Tile_, elemsPerBlock);
            uint64_t dim2SliceCount = CeilDiv(coreNum_, mergedDim0_);
            uint64_t dim2TempTile = CeilAlign(CeilDiv(mergedDim2Align, dim2SliceCount), elemsPerBlock);
            dim2Tile_ = std::min(dim2Tile_, dim2TempTile);
            dim2Tile_ = std::min(dim2Tile_, mergedDim2_);
            dim2LoopTime_ = mergedDim2_ / dim2Tile_;
            dim2Remained_ = mergedDim2_ - dim2LoopTime_ * dim2Tile_;
        } else {
            singleBufElems_ = GetSingleBufElems(REDUCE_BUF_NUM);
            dim2Tile_ = std::min(elemsPerBlock, mergedDim2_);
            dim2LoopTime_ = mergedDim2_ / dim2Tile_;
            dim2Remained_ = mergedDim2_ - dim2LoopTime_ * dim2Tile_;
            dim1Tile_ = singleBufElems_ / elemsPerBlock;
            dim1Tile_ = std::min(dim1Tile_, mergedDim1_);
            dim1LoopTime_ = mergedDim1_ / dim1Tile_;
            dim1Remained_ = mergedDim1_ - dim1LoopTime_ * dim1Tile_;
        }
        if (dim2Tile_ == mergedDim2_ && mergedDim2_ % fp32ElemsPerBlock_ == 0) {
            isContiguous_ = true;
        }
        uint64_t needCoreNum = mergedDim0_ * (dim2LoopTime_ + (dim2Remained_ ? 1 : 0));
        if (dim1Tile_ == mergedDim1_ && dim2Tile_ == mergedDim2_ && (mergedDim1_ * mergedDim2Align) < singleBufElems_) {
            dim0Tile_ = std::min(CeilDiv(mergedDim0_, coreNum_), singleBufElems_ / (mergedDim1_ * mergedDim2Align));
            dim0Tile_ = std::min(dim0Tile_, mergedDim0_);
            dim0LoopTime_ = mergedDim0_ / dim0Tile_;
            dim0Remained_ = mergedDim0_ - dim0LoopTime_ * dim0Tile_;
            needCoreNum = dim0LoopTime_ + (dim0Remained_ ? 1 : 0);
        }
        usedCoreNum_ = std::min(needCoreNum, coreNum_);
    }

private:
    gert::TilingContext* context_ = nullptr;
    LogSoftmaxGradTilingData* tilingData_ = nullptr;
    std::set<int64_t> reduceAxes_;
    uint64_t ubSize_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t blockSize_ = 0;         // UB block 字节数，运行时通过 GetUbBlockSize 获取
    uint64_t fp32ElemsPerBlock_ = 0; // 单个 block 可容纳的 float32 元素数
    uint64_t transMinSize_ = 0;      // transpose 最小分块阈值，等于 TRANS_MIN_MULT * fp32ElemsPerBlock_
    int64_t dimNum_ = 0;
    uint64_t usedCoreNum_ = 0;
    int32_t inBpe_ = 0;
    // TilingKey params
    uint64_t schMode_ = 0;
    bool isSmallShape_ = false;
    bool isContiguous_ = false;
    // TilingData params
    uint64_t singleBufElems_ = 0;
    uint64_t mergedDim0_ = 1;
    uint64_t mergedDim1_ = 1;
    uint64_t mergedDim2_ = 1;
    uint64_t dim0Tile_ = 0;
    uint64_t dim1Tile_ = 0;
    uint64_t dim2Tile_ = 0;
    uint64_t totalElems_ = 0;
    uint64_t dim0LoopTime_ = 0;
    uint64_t dim0Remained_ = 0;
    uint64_t dim1LoopTime_ = 0;
    uint64_t dim1Remained_ = 0;
    uint64_t dim2LoopTime_ = 0;
    uint64_t dim2Remained_ = 0;
};

static graphStatus LogSoftmaxGradTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "LogSoftmaxGradTilingFunc start\n");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return GRAPH_NULL_PTR);
    auto tilingObject = LogSoftmaxGradTiling(context);
    graphStatus ret = tilingObject.Init();
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context, "Tiling object init failed"), return ret);

    ret = tilingObject.CheckParams();
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context, "Tiling object check params failed"), return ret);

    ret = tilingObject.DoTiling();
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(context, "Tiling object do tiling failed"), return ret);
    OP_LOGD(context, "LogSoftmaxGradTilingFunc end\n");
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogSoftmaxGrad)
    .Tiling(LogSoftmaxGradTilingFunc)
    .TilingParse<LogSoftmaxGradCompileInfo>(TilingParseForLogSoftmaxGrad);
} // namespace optiling
