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
 * \file repeat_interleave_grad_david_tiling.cc
 * \brief
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/atvoss/reduce/reduce_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "repeat_interleave_grad_tiling_arch35.h"
#include "repeat_interleave_grad_int_repeat_tiling.h"
#include "repeat_interleave_grad.h"
#include "../op_kernel/v35/repeat_interleave_grad_dag.h"
#include "../op_kernel/v35/repeat_interleave_grad_tiling_key.h"
#include "../op_kernel/v35/repeat_interleave_grad_david_tiling_data.h"

namespace optiling {

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_REPEAT_INDEX = 1;
static constexpr size_t OUTPUT_Y_INDEX = 0;
static constexpr size_t ATTR_AXIS_INDEX = 0;
static constexpr int32_t DOUBLE_BUFFER = 2;
static constexpr size_t RIG_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
static constexpr int32_t RIG_CACHE_BUF_SIZE = 16 * 1024;
static constexpr int32_t RES_BUF_SIZE = 8 * 1024;
static constexpr int32_t CAST_MULT = 3;
static constexpr int32_t BASE_SPLIT_LENGTH = 128;
static constexpr int32_t nTwo = 2;
static constexpr int64_t MAX_YGRAD_DIM = 8;
constexpr size_t WORKSPACE_SIZE_EXT = static_cast<size_t>(64 * 8); // 额外的workspace，用于block切人Repeat时使用
constexpr int32_t INT32_SIZE = 4;

using namespace Ops::Base;

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape, const gert::Shape& oneShape)
{
    if (inShape.IsScalar()) {
        return oneShape;
    }
    return inShape;
}

class RIGDavidTilingImpl
{
public:
    explicit RIGDavidTilingImpl(gert::TilingContext* context) : context_(context)
    {}

    ge::graphStatus Init(const RepeatInterleaveGradCompileInfo* compileInfo);

    ge::graphStatus DoTiling(gert::TilingContext* context);

private:
    void TilingStrategy();

    void TilingStrategyBlockSplitR();

    void DoBlockSplit(int64_t splitLen, int32_t coreNum, RIGBlockPara& blockPara) const;

    void DoUbSplit(int64_t splitLen, int64_t splitFactor, UbParaUnit& ubPara) const;

    void FillTilingData(gert::TilingContext* context);

    void PrintTilingData();

    void PrintBlockPara();

    void PrintUbPara();

    void MergeAxis(
        const gert::Shape& xShape, const gert::Shape& yShape, const gert::Shape& repeatShape, int64_t xDimNum);

private:
    int32_t clSize_ = 0;
    int32_t blockSize_ = 0;
    int32_t vRegSize_ = 0;
    int32_t ubSize_ = 0;
    int32_t coreNum_ = 0;
    int64_t axis_ = 0;
    bool isAxisNone_ = false;
    int64_t lenM_ = 1;
    int64_t lenR_ = 1;
    int64_t lenN_ = 1;
    int64_t lenP_ = 1;
    int64_t lenRepeat_ = 1;
    bool isRepeatScalar_ = false;
    int64_t repeatScalarValue_ = 0;
    int32_t dtSize_ = 0;
    int32_t dtSizeCast_ = 0;
    int32_t rtSize_ = 0;
    int64_t alignN_ = 0; // 32B对齐，单位B
    int32_t realCoreNum_ = 0;
    int32_t coreNumPerM_ = 0;
    RIGBlockPara mBlockPara_;
    RIGBlockPara repeatBlockPara_;
    RIGBlockPara nBlockPara_;
    RIGUbPara mUbPara_;
    RIGUbPara repeatUbPara_;
    RIGUbPara repeatLoopPara_; // ub内每次只载入一部分repeat，因此ub内部repeat还需要切分
    RIGUbPara nUbPara_;
    RIGUbPara repeatSumUbPara; // Repeat计算各个核的sum时，产生的Ub切分策略
    int32_t xBufferSize_ = 0;
    int32_t yBufferSize_ = 0;
    int32_t repeatBufferSize_ = 0;
    int32_t repeatSumBufferSize_ = 0; // Repeat计算各个核的sum时，所需要的bufferSize
    int32_t inutDataBufSzie_ = 0;
    int32_t basicBlockSize_ = 0; // 单位：B
    int32_t rFactor_ = 1;        // 进行二分累加时，切分的RA块数据的R.i的大小
    int32_t nFactor_ = 1;        // 进行二分累加时，切分的RA块数据的n.i的大小
    bool dtCast_ = false;
    uint32_t templateNum_ = 0;
    gert::TilingContext* context_;
}; // class RIGDavidTilingImpl

ge::graphStatus RIGDavidTilingImpl::Init(const RepeatInterleaveGradCompileInfo* compileInfo)
{
    clSize_ = static_cast<int32_t>(compileInfo->clSize);
    blockSize_ = static_cast<int32_t>(compileInfo->blockSize);
    vRegSize_ = static_cast<int32_t>(compileInfo->vRegSize);
    ubSize_ = static_cast<int32_t>(compileInfo->ubSize);
    coreNum_ = static_cast<int32_t>(compileInfo->coreNum);
    // 获取y_grad
    auto xStorage = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    auto oneShapeX = gert::Shape{1};
    const gert::Shape& xShape = EnsureNotScalar(xStorage->GetStorageShape(), oneShapeX);

    auto repeatStorage = context_->GetInputShape(INPUT_REPEAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, repeatStorage);
    auto oneShapeRepeat = gert::Shape{1};
    const gert::Shape& repeatShape = EnsureNotScalar(repeatStorage->GetStorageShape(), oneShapeRepeat);

    auto ytStorage = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ytStorage);
    auto oneShapeY = gert::Shape{1};
    const gert::Shape& yShape = EnsureNotScalar(ytStorage->GetStorageShape(), oneShapeY);

    auto repeatShapeSize = repeatShape.GetShapeSize();

    OP_CHECK_IF(
        repeatShapeSize == 0L, OP_LOGE(context_->GetNodeName(), "The repeat shape is empty!"),
        return ge::GRAPH_FAILED);

    // 获取axis 如果axis为null，处理逻辑不同
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* axis = attrs->GetInt(ATTR_AXIS_INDEX);
    if (axis == nullptr) {
        isAxisNone_ = true;
    } else {
        axis_ = *axis;
    }

    int64_t xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(
        xDimNum > MAX_YGRAD_DIM,
        OP_LOGE(context_->GetNodeName(), "The yGrad Dim should not be greater than 8!"),
        return ge::GRAPH_FAILED);
    if (!isAxisNone_ && axis_ < 0) {
        axis_ = axis_ + xDimNum;
    }

    // 校验axis是none时，输入的长度是否为1
    OP_CHECK_IF(
        (isAxisNone_ && xDimNum != 1),
        OP_LOGE(context_->GetNodeName(), "Axis is none, yGrad shape length should be 1"),
        return ge::GRAPH_FAILED);
    // 合轴
    MergeAxis(xShape, yShape, repeatShape, xDimNum);

    lenRepeat_ = repeatShape.GetDim(0);

    // 获取数据类型
    auto inputXDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    ge::DataType dType = inputXDesc->GetDataType();
    dtSize_ = GetSizeByDataType(dType);
    dtCast_ = (dType == ge::DataType::DT_FLOAT16 || dType == ge::DataType::DT_BF16) ? true : false;
    alignN_ = CeilAlign(lenN_ * dtSize_, static_cast<int64_t>(blockSize_));

    auto inputRepeatDesc = context_->GetInputDesc(INPUT_REPEAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputRepeatDesc);
    ge::DataType rType = inputRepeatDesc->GetDataType();
    rtSize_ = GetSizeByDataType(rType);

    repeatBufferSize_ = clSize_;
    inutDataBufSzie_ = ubSize_ - repeatBufferSize_ * nTwo - RIG_CACHE_BUF_SIZE - RES_BUF_SIZE;

    return ge::GRAPH_SUCCESS;
}

void RIGDavidTilingImpl::MergeAxis(
    const gert::Shape& xShape, const gert::Shape& yShape, const gert::Shape& repeatShape, int64_t xDimNum)
{
    if (isAxisNone_) {
        lenM_ = 1;
        lenN_ = 1;
        lenP_ = xShape[xDimNum - 1];
        lenR_ = repeatShape.GetDim(0);
    } else {
        lenP_ = xShape.GetDim(axis_);
        lenR_ = yShape.GetDim(axis_);
        for (int64_t i = 0; i < axis_; i++) {
            lenM_ *= xShape[i];
        }
        for (int64_t i = axis_ + 1; i < static_cast<int64_t>(xDimNum); i++) {
            lenN_ *= xShape[i];
        }
    }
}

ge::graphStatus RIGDavidTilingImpl::DoTiling(gert::TilingContext* context)
{
    OP_LOGD(context_->GetNodeName(), "Enter RIGDavidTilingImpl DoTiling");

    TilingStrategy();

    FillTilingData(context);

    PrintTilingData();

    context_->SetBlockDim(realCoreNum_);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(0, 0, 0, templateNum_);

    context_->SetTilingKey(tilingKey);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = templateNum_ == RIG::BLOCK_SPLIT_R ? RIG_WORKSPACE_SIZE + WORKSPACE_SIZE_EXT : RIG_WORKSPACE_SIZE;

    OP_LOGD(context_->GetNodeName(), "Exit RIGDavidTilingImpl DoTiling");
    return ge::GRAPH_SUCCESS;
}

void RIGDavidTilingImpl::TilingStrategy()
{
    // 空tensor
    if (lenM_ == 0 || lenR_ == 0 || lenN_ == 0) {
        realCoreNum_ = 0;
        return;
    }

    // repeat为tensor
    if (dtCast_) {
        basicBlockSize_ = inutDataBufSzie_ / DOUBLE_BUFFER / CAST_MULT;
    } else {
        basicBlockSize_ = inutDataBufSzie_ / DOUBLE_BUFFER;
    }
    basicBlockSize_ = basicBlockSize_ / blockSize_ * blockSize_; // 32B对齐

    if (lenM_ > static_cast<int64_t>(coreNum_ / nTwo) || (alignN_ * lenRepeat_ <= ubSize_)) { // M 开核
        // N,  repeat不切核
        DoBlockSplit(lenN_, 1, nBlockPara_);
        DoBlockSplit(lenRepeat_, 1, repeatBlockPara_);

        // M切核
        DoBlockSplit(lenM_, coreNum_, mBlockPara_); // M.o M.i

        // repeat要按照128B切分
        DoUbSplit(
            repeatBlockPara_.blockTailFactor, static_cast<int64_t>(repeatBufferSize_ / rtSize_),
            repeatLoopPara_.tailCoreUbPara); // repeat.o repeat.i

        if (lenN_ != 1) { // RA pattern
            nFactor_ = BASE_SPLIT_LENGTH;
            rFactor_ = basicBlockSize_ / nFactor_ / dtSize_;
            DoUbSplit(nBlockPara_.blockTailFactor, nFactor_, nUbPara_.tailCoreUbPara); // N.o  N.i
            templateNum_ = RIG::BLOCK_SPLIT_M;
        } else { // AR pattern
            nFactor_ = 1;
            DoUbSplit(mBlockPara_.blockFactor, BASE_SPLIT_LENGTH, mUbPara_.mainCoreUbPara);     // main M.i.o  M.i.i
            DoUbSplit(mBlockPara_.blockTailFactor, BASE_SPLIT_LENGTH, mUbPara_.tailCoreUbPara); // M.i.o  M.i.i
            rFactor_ = basicBlockSize_ / mUbPara_.mainCoreUbPara.ubFactor / dtSize_; // 使用主核主块计算rfactor
            if (rFactor_ <= 1) {
                OP_LOGE(context_->GetNodeName(), "RAxis in ub is too small, cannot calculate");
                return;
            }
            templateNum_ = RIG::BLOCK_SPLIT_M_N_1;
        }
        realCoreNum_ = mBlockPara_.blockCount;
    } else {
        if (lenM_ * lenN_ >= coreNum_) {
            DoBlockSplit(lenRepeat_, 1, repeatBlockPara_);
            // block切N, M
            coreNumPerM_ = coreNum_ / static_cast<int32_t>(lenM_);
            DoBlockSplit(lenN_, coreNumPerM_, nBlockPara_); // N.o N.i
            coreNumPerM_ = nBlockPara_.blockCount;
            DoBlockSplit(lenM_, lenM_, mBlockPara_); // M.o M.i

            DoUbSplit(
                repeatBlockPara_.blockTailFactor, static_cast<int64_t>(repeatBufferSize_ / rtSize_),
                repeatLoopPara_.tailCoreUbPara); // repeat.o repeat.i

            nFactor_ = BASE_SPLIT_LENGTH;
            rFactor_ = basicBlockSize_ / nFactor_ / dtSize_;
            DoUbSplit(nBlockPara_.blockFactor, nFactor_, nUbPara_.mainCoreUbPara); // N.i.o  N.i.i
            DoUbSplit(nBlockPara_.blockTailFactor, nFactor_, nUbPara_.tailCoreUbPara);
            templateNum_ = RIG::BLOCK_SPLIT_MN;
            realCoreNum_ = coreNumPerM_ * static_cast<int32_t>(lenM_);
        } else if (lenM_ * lenRepeat_ >= coreNum_) {
            // block切repeat
            return TilingStrategyBlockSplitR();
        }
    }
}
/* M和R一起切block，R轴最多借64切核 */
void RIGDavidTilingImpl::TilingStrategyBlockSplitR()
{
    // 借R轴切核，借轴数量等于M剩余的
    int64_t borrowRCount = FloorDiv(static_cast<int64_t>(coreNum_), lenM_);

    // M轴切核，全部分核
    DoBlockSplit(lenM_, coreNum_, mBlockPara_);
    // R轴切核，借轴
    DoBlockSplit(lenR_, borrowRCount, repeatBlockPara_);
    // N轴不分核核
    DoBlockSplit(lenN_, 1, nBlockPara_);

    // M轴切UB，每次处理1个M
    DoUbSplit(mBlockPara_.blockFactor, 1, mUbPara_.mainCoreUbPara);
    DoUbSplit(mBlockPara_.blockTailFactor, 1, mUbPara_.mainCoreUbPara);
    // R轴切UB（这里切UB是为了每个核上sum而做的切分），UB尽量多的载入repeat，保险起见预留一点buff(8K)
    int32_t maxForR = (rtSize_ == INT32_SIZE) ? FloorDiv(ubSize_ - RES_BUF_SIZE, rtSize_ * CAST_MULT) :
                                                FloorDiv(ubSize_ - RES_BUF_SIZE, rtSize_);
    DoUbSplit(repeatBlockPara_.blockFactor, maxForR, repeatSumUbPara.mainCoreUbPara);
    DoUbSplit(repeatBlockPara_.blockTailFactor, maxForR, repeatSumUbPara.tailCoreUbPara);
    // R轴切UB（这里切UB才是真正做算子处理时的切分）,一次处理128B
    DoUbSplit(
        repeatBlockPara_.blockFactor, static_cast<int64_t>(repeatBufferSize_ / rtSize_),
        repeatLoopPara_.mainCoreUbPara);
    DoUbSplit(
        repeatBlockPara_.blockTailFactor, static_cast<int64_t>(repeatBufferSize_ / rtSize_),
        repeatLoopPara_.tailCoreUbPara);
    // N轴切UB，全载
    DoUbSplit(nBlockPara_.blockTailFactor, nBlockPara_.blockTailFactor, nUbPara_.tailCoreUbPara);

    basicBlockSize_ -= static_cast<int32_t>(WORKSPACE_SIZE_EXT); // 还需要搬入workspace，所以要减掉workspace
    rFactor_ = basicBlockSize_ / static_cast<int32_t>(lenN_) / dtSize_;
    int32_t repeatSumMainCoreMaxFactor = repeatSumUbPara.mainCoreUbPara.ubCount > 1 ?
                                             repeatSumUbPara.mainCoreUbPara.ubFactor :
                                             repeatSumUbPara.mainCoreUbPara.ubTailFactor;
    repeatSumBufferSize_ = CeilAlign(repeatSumMainCoreMaxFactor * rtSize_, blockSize_);
    templateNum_ = RIG::BLOCK_SPLIT_R;
    realCoreNum_ = mBlockPara_.blockCount * repeatBlockPara_.blockCount;
}

// RIGBlockPara -> RIGBlockPara
void RIGDavidTilingImpl::DoBlockSplit(int64_t splitLen, int32_t coreNum, RIGBlockPara& blockPara) const
{
    if (splitLen <= 0) {
        return;
    }
    blockPara.blockFactor = CeilDiv(splitLen, static_cast<int64_t>(coreNum));
    blockPara.blockCount = CeilDiv(splitLen, blockPara.blockFactor);
    blockPara.blockTailFactor =
        (splitLen % blockPara.blockFactor == 0) ? blockPara.blockFactor : splitLen % blockPara.blockFactor;
}

// UbParaUnit -> UbParaUnit
void RIGDavidTilingImpl::DoUbSplit(int64_t splitLen, int64_t splitFactor, UbParaUnit& ubPara) const
{
    if (splitLen <= 0) {
        return;
    }
    ubPara.ubFactor = splitLen < splitFactor ? splitLen : splitFactor;
    ubPara.ubCount = CeilDiv(splitLen, static_cast<int64_t>(ubPara.ubFactor));
    ubPara.ubTailFactor = (splitLen % ubPara.ubFactor == 0) ? ubPara.ubFactor : splitLen % ubPara.ubFactor;
}

void RIGDavidTilingImpl::FillTilingData(gert::TilingContext* context)
{
    RepeatInterleaveGradDavidTilingData* tiling = context->GetTilingData<RepeatInterleaveGradDavidTilingData>();
    tiling->lenM = lenM_;
    tiling->lenR = lenR_;
    tiling->lenN = lenN_;
    tiling->lenP = lenP_;
    tiling->rFactor = rFactor_;
    tiling->lenRepeat = lenRepeat_;
    tiling->mBlockPara = mBlockPara_;
    tiling->repeatBlockPara = repeatBlockPara_;
    tiling->nBlockPara = nBlockPara_;

    tiling->mUbPara.mainCoreUbPara = mUbPara_.mainCoreUbPara;
    tiling->repeatUbPara.mainCoreUbPara = repeatUbPara_.mainCoreUbPara;
    tiling->nUbPara.mainCoreUbPara = nUbPara_.mainCoreUbPara;
    tiling->repeatSumUbPara.mainCoreUbPara = repeatSumUbPara.mainCoreUbPara;
    tiling->mUbPara.tailCoreUbPara = mUbPara_.tailCoreUbPara;
    tiling->repeatUbPara.tailCoreUbPara = repeatUbPara_.tailCoreUbPara;
    tiling->nUbPara.tailCoreUbPara = nUbPara_.tailCoreUbPara;
    tiling->repeatSumUbPara.tailCoreUbPara = repeatSumUbPara.tailCoreUbPara;

    tiling->repeatLoopPara = repeatLoopPara_;
    tiling->realCoreNum = realCoreNum_;
    tiling->repeatBufferSize = repeatBufferSize_;
    tiling->repeatSumBufferSize = repeatSumBufferSize_;
    tiling->basicBlockSize = basicBlockSize_;
    tiling->coreNumPerM = coreNumPerM_;
}

void RIGDavidTilingImpl::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is templateNum:%u, realCoreNum:%d, lenM:%ld, lenR:%ld, lenN:%ld, lenP:%ld, lenRepeat:%ld,\
        repeatBufferSize:%d, repeatSumBufferSize_:%d, rFactor:%d, basicBlockSize:%d, ubSize_:%d, coreNumPerM:%d",
        templateNum_, realCoreNum_, lenM_, lenR_, lenN_, lenP_, lenRepeat_, repeatBufferSize_, repeatSumBufferSize_,
        rFactor_, basicBlockSize_, ubSize_, coreNumPerM_);

    PrintBlockPara();
    PrintUbPara();
}

void RIGDavidTilingImpl::PrintBlockPara()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is mBlockPara:%d %ld %ld, repeatBlockPara:%d %ld %ld, nBlockPara:%d %ld %ld",
        mBlockPara_.blockCount, mBlockPara_.blockFactor, mBlockPara_.blockTailFactor, repeatBlockPara_.blockCount,
        repeatBlockPara_.blockFactor, repeatBlockPara_.blockTailFactor, nBlockPara_.blockCount, nBlockPara_.blockFactor,
        nBlockPara_.blockTailFactor);
}

void RIGDavidTilingImpl::PrintUbPara()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is mUbPara:%d %d %d %d %d %d, repeatUbPara:%d %d %d %d %d %d, nUbPara:%d %d %d %d %d %d, "
        "repeatLoopPara:%d %d %d %d %d %d, repeatSumUbPara:%d %d %d %d %d %d",
        mUbPara_.mainCoreUbPara.ubCount, mUbPara_.mainCoreUbPara.ubFactor, mUbPara_.mainCoreUbPara.ubTailFactor,
        mUbPara_.tailCoreUbPara.ubCount, mUbPara_.tailCoreUbPara.ubFactor, mUbPara_.tailCoreUbPara.ubTailFactor,
        repeatUbPara_.mainCoreUbPara.ubCount, repeatUbPara_.mainCoreUbPara.ubFactor,
        repeatUbPara_.mainCoreUbPara.ubTailFactor, repeatUbPara_.tailCoreUbPara.ubCount,
        repeatUbPara_.tailCoreUbPara.ubFactor, repeatUbPara_.tailCoreUbPara.ubTailFactor,
        nUbPara_.mainCoreUbPara.ubCount, nUbPara_.mainCoreUbPara.ubFactor, nUbPara_.mainCoreUbPara.ubTailFactor,
        nUbPara_.tailCoreUbPara.ubCount, nUbPara_.tailCoreUbPara.ubFactor, nUbPara_.tailCoreUbPara.ubTailFactor,
        repeatLoopPara_.mainCoreUbPara.ubCount, repeatLoopPara_.mainCoreUbPara.ubFactor,
        repeatLoopPara_.mainCoreUbPara.ubTailFactor, repeatLoopPara_.tailCoreUbPara.ubCount,
        repeatLoopPara_.tailCoreUbPara.ubFactor, repeatLoopPara_.tailCoreUbPara.ubTailFactor,
        repeatSumUbPara.mainCoreUbPara.ubCount, repeatSumUbPara.mainCoreUbPara.ubFactor,
        repeatSumUbPara.mainCoreUbPara.ubTailFactor, repeatSumUbPara.tailCoreUbPara.ubCount,
        repeatSumUbPara.tailCoreUbPara.ubFactor, repeatSumUbPara.tailCoreUbPara.ubTailFactor);
}

static ge::graphStatus TilingGetCompileInfo(gert::TilingContext* context, RepeatInterleaveGradCompileInfo* compileInfo)
{
    OP_LOGD(context, "Enter TilingPrepare4RepeatInterleaveGradDavid.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context, "core num is negative."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0L), OP_LOGE(context, "fail to get ub size."),
        return ge::GRAPH_FAILED);

    compileInfo->clSize = GetCacheLineSize(context);
    OP_CHECK_IF(
        (compileInfo->clSize <= 0U),
        OP_LOGE(context, "fail to get cache line size."),
        return ge::GRAPH_FAILED);

    compileInfo->blockSize = GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->blockSize <= 0),
        OP_LOGE(context, "fail to get block size."), return ge::GRAPH_FAILED);

    compileInfo->vRegSize = GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vRegSize <= 0U),
        OP_LOGE(context, "fail to get vReg size."), return ge::GRAPH_FAILED);

    OP_LOGD(context, "Exit TilingPrepare4CumsumAscendc.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RepeatInterleaveGradDavid(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter Tiling4RepeatInterleaveGradDavid.");
    RepeatInterleaveGradCompileInfo compileInfo;
    TilingGetCompileInfo(context, &compileInfo);

    if (RIGintRepeat::IsIntRepeat(context)) {
        return RIGintRepeat::Tiling4RIGIntRepeat(context, &compileInfo);
    }

    RIGDavidTilingImpl tilingImpl = RIGDavidTilingImpl(context);
    if (tilingImpl.Init(&compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4RepeatInterleaveGradDavid init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4RepeatInterleaveGradDavid DoTiling failed.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Exit Tiling4RepeatInterleaveGradDavid.");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling