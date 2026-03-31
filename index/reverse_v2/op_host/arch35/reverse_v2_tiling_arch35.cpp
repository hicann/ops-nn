/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file reverse_v2_tiling.cc
 * \brief
 */
#include "reverse_v2_tiling_arch35.h"
#include <algorithm>
#include "tiling/tiling_api.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "reverse_v2_tiling.h"
#include "platform/platform_info.h"

namespace optiling {
constexpr int64_t INPUT_X = 0;
constexpr int64_t INPUT_AXIS = 1;
constexpr int64_t OUTPUT_Y = 2;

constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t TWO_DIMS = 2;
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIM3 = 3;
constexpr int64_t DIM4 = 4;
constexpr int64_t DIM5 = 5;
constexpr int64_t DIM6 = 6;
constexpr int64_t DIM7 = 7;
constexpr size_t DIM8 = 8;
static constexpr uint64_t REVERSE_V2_SIMD_TILING_KEY = 10001;

constexpr uint64_t TILING_REVERSE_DIM = 100;
constexpr uint64_t TILING_NOT_REVERSE_DIM = 10;

constexpr int64_t MAX_INT32_NUM = 2147483647;
constexpr int64_t RESERVED_UB_SIZE = static_cast<int64_t>(9) * 1024;
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

constexpr int64_t SIMD_THRESHOLD = 128;
constexpr int64_t UB_THRESHOLD = 1024;

constexpr int64_t SPLIT_NEG_ONE_DIM = 0;
constexpr int64_t SPLIT_NEG_TWO_DIM = 1;
constexpr int64_t SPLIT_NEG_THREE_DIM = 2;
constexpr int64_t NO_SPLIT_OUT_DIM = 3;
constexpr int64_t BUFFER_NUM = 2;

constexpr int64_t UB_FACTOR_MIN_BETY = 2048;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t ONE_BLK_BYTE = 32;
constexpr int64_t WORKSPACE_SIZE = 32;
constexpr int64_t DIGIT_THOUSAND = 1000;

int64_t GetRemainder(int64_t u_value, int64_t d_value) {
  int64_t res_value = 0;
  if (d_value == 0) {
    return u_value;
  }
  res_value = u_value % d_value;

  return res_value;
}

bool ReverseV2Tiling::IsCapable()
{
    return true;
}

ge::graphStatus ReverseV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseV2Tiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const ReverseV2CompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->totalCoreNum;
    ubSize_ = compileInfo->ubSize;
    ubSize_ -= RESERVED_UB_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseV2Tiling::GetInputShape()
{
    auto inputX = context_->GetInputTensor(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = inputX->GetStorageShape();
    size_t inputDimNum = inputShape.GetDimNum();
    if (inputDimNum > DIM8) {
        OP_LOGE(context_->GetNodeName(), "input dim num must be less than or equal to 8");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputDimNum; i++) {
        int64_t curDim = inputShape.GetDim(i);
        inputShape_.push_back(curDim);
        inputSize_ *= curDim;
    }
    auto inputDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dtype = inputDesc->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(dtype);
    OP_CHECK_IF(
        dtypeSize_ <= 0,
        OP_LOGE(context_, "dtypeSize must be greater than 0, dtypeSize: %ld", dtypeSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReverseV2Tiling::GetReversedDims()
{
    auto axis = context_->GetInputTensor(INPUT_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    int64_t axisShapeSize = axis->GetShapeSize();
    const T* dimsPtr = axis->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, dimsPtr);
    for (int64_t i = 0; i < axisShapeSize; i++) {
        reversedDims_.push_back(static_cast<int64_t>(dimsPtr[i]));
    }
    // convert negative axis to positive
    int64_t size = inputShape_.size() > 0 ? inputShape_.size() : DIM1;
    for (int64_t &num : reversedDims_) {
        if (num < 0) {
            num = size + num;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseV2Tiling::GetShapeAttrsInfo()
{
    // has been executed before
    if (flag_) {
        OP_LOGD("Enter ReverseV2 tiling.");
        return ge::GRAPH_SUCCESS;
    }
    flag_ = true;
    // get inputShape_
    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // get reversedDims_
    auto axisXDesc = context_->GetInputDesc(INPUT_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axisXDesc);
    ge::DataType axisDataType = axisXDesc->GetDataType();
    if (axisDataType == ge::DT_INT64) {
        if (GetReversedDims<int64_t>() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (axisDataType == ge::DT_INT32) {
        if (GetReversedDims<int32_t>() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context_->GetNodeName(), "axis must be int32 or int64");
        return ge::GRAPH_FAILED;
    }
    // check axis if valid
    // 1. reversedDims_ cannot be duplicated
    std::set<int64_t> dimsSet(reversedDims_.begin(), reversedDims_.end());
    if (dimsSet.size() != reversedDims_.size()) {
        OP_LOGE(context_->GetNodeName(), "values within axis cannot be duplicated");
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> dimsVec(dimsSet.begin(), dimsSet.end());
    reversedDims_ = dimsVec;
    // 2. inputShape_ dim num must be greater than or equal to reversedDims_ size
    if (reversedDims_.size() > inputShape_.size() && !inputShape_.empty()) {
        OP_LOGE(context_->GetNodeName(), "axis size must be less than or equal to input shape dim num");
        return ge::GRAPH_FAILED;
    }
    auto axis = context_->GetInputTensor(INPUT_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    auto axisShape = axis->GetStorageShape();
    size_t axisDimNum = axisShape.GetDimNum();
    // 3. axis must be 1D
    if (axisDimNum != 1U) {
        OP_LOGE(context_->GetNodeName(), "axis dim num must be 1D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> ReverseV2Tiling::GetNonReversedDims(const std::vector<int64_t> &shapeVec,
                                                         const std::vector<int64_t> &dims,
                                                         bool isReversedDim)
{
    if (isReversedDim && shapeVec.empty()) {
        return reversedDims_;
    }
    std::vector<int64_t> nonReversedDims;
    for (int64_t i = 0; i < static_cast<int64_t>(shapeVec.size()); i++) {
        if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
            nonReversedDims.push_back(i);
        }
    }
    return nonReversedDims;
}

std::vector<int64_t> MergeInputShape(const std::vector<int64_t> &shapeVec, const std::vector<int64_t> &dims)
{
    std::vector<int64_t> mergedInputShape;
    size_t startDim = 0;
    for (size_t dim = 0; dim < shapeVec.size(); dim++) {
        auto dimCur = std::find(dims.begin(), dims.end(), dim);
        auto dimNext = std::find(dims.begin(), dims.end(), dim + 1);
        if (!(dim + 1 < shapeVec.size() && dimCur != dims.end() && dimNext != dims.end() && *dimNext - *dimCur == 1)) {
            int64_t ans = 1;
            for (size_t idx = startDim; idx <= dim; idx++) {
                ans *= shapeVec[idx];
            }
            mergedInputShape.push_back(ans);
            startDim = dim + 1;
        }
    }
    return mergedInputShape;
}

int64_t GetParam(const std::vector<int64_t> &shapeVec, int64_t dim)
{
    int64_t param = 1;
    for (size_t i = dim + 1; i < shapeVec.size(); i++) {
        param *= shapeVec[i];
    }
    return param;
}

std::vector<int64_t> MergeDims(const std::vector<int64_t> &dims)
{
    std::vector<int64_t> mergedDims;
    size_t cnt = 0;
    if (dims.size() > 0) {
        mergedDims.push_back(dims[0]);
    }
    for (size_t i = 1; i < dims.size(); i++) {
        if (dims[i] - mergedDims.back() - cnt == 1) {
            cnt++;
            continue;
        }
        mergedDims.push_back(dims[i] - cnt);
    }
    return mergedDims;
}

ge::graphStatus ReverseV2Tiling::DoOpTiling()
{
    // calc tilingData
    int64_t threadNum = 1024;
    usedCoreNum_ = std::min((inputSize_ + threadNum - 1) / threadNum, totalCoreNum_);
    blockFactor_ = (inputSize_ + usedCoreNum_ - 1) / usedCoreNum_;

    // get nonReversedDims
    std::vector<int64_t> nonReversedDims = GetNonReversedDims(inputShape_, reversedDims_, false);
    // merge input shape according to nonReversedDims
    inputShape_ = MergeInputShape(inputShape_, nonReversedDims);
    // merge nonReversedDims
    nonReversedDims = MergeDims(nonReversedDims);

    // get reversedDims according to nonReversedDims
    reversedDims_ = GetNonReversedDims(inputShape_, nonReversedDims, true);
    // merge inputShape
    inputShape_ = MergeInputShape(inputShape_, reversedDims_);
    reversedDims_ = MergeDims(reversedDims_);
    int64_t inputShapeSize = static_cast<int64_t>(inputShape_.size());
    dim0_ = inputShapeSize > DIM0 ? inputShape_[DIM0] : 0;
    dim0_ = dim0_ == 0 ? inputSize_ : dim0_;
    dim1_ = inputShapeSize > DIM1 ? inputShape_[DIM1] : 0;
    dim2_ = inputShapeSize > DIM2 ? inputShape_[DIM2] : 0;
    dim3_ = inputShapeSize > DIM3 ? inputShape_[DIM3] : 0;
    dim4_ = inputShapeSize > DIM4 ? inputShape_[DIM4] : 0;
    dim5_ = inputShapeSize > DIM5 ? inputShape_[DIM5] : 0;
    dim6_ = inputShapeSize > DIM6 ? inputShape_[DIM6] : 0;
    dim7_ = inputShapeSize > DIM7 ? inputShape_[DIM7] : 0;
    param0_ = GetParam(inputShape_, DIM0);
    param1_ = GetParam(inputShape_, DIM1);
    param2_ = GetParam(inputShape_, DIM2);
    param3_ = GetParam(inputShape_, DIM3);
    param4_ = GetParam(inputShape_, DIM4);
    param5_ = GetParam(inputShape_, DIM5);
    param6_ = GetParam(inputShape_, DIM6);
    DoSimdTiling();
    return ge::GRAPH_SUCCESS;
}

void ReverseV2Tiling::DoSimdTiling()
{
    dimNum_ = inputShape_.size();
    int64_t reverseDimNum = reversedDims_.size();
    //判断输入轴是否有效
    if (reverseDimNum < 1  || dimNum_ < TWO_DIMS) {
        isSimd_ = false;
        return;
    }
    if (reversedDims_[reverseDimNum - 1] == dimNum_ - 1 ||
        inputShape_[dimNum_ - 1] * dtypeSize_ < SIMD_THRESHOLD) {
        isSimd_ = false;
        return;
    }
    isSimd_ = true;
    if (reversedDims_[0] == 0) {
        dim0Reversed_ = 1;
    }
    availableUb_ = ubSize_ / DIGIT_TWO / dtypeSize_ / BUFFER_NUM;
    do {
        SingleUbTiling();
        if (totalLoop_ >= totalCoreNum_) {
            break;
        }
        availableUb_ = availableUb_ / DIGIT_TWO;
    } while (availableUb_ > UB_THRESHOLD);
    if (dim0Reversed_ && splitDim_ % DIGIT_TWO == 0) {
        splitDimReversed_ = 1;
    }
    if (!dim0Reversed_ && splitDim_ % DIGIT_TWO != 0) {
        splitDimReversed_ = 1;
    }
    for (int64_t i = splitDim_ - 1; i >= 0; i--) {
        loopStride_[i] = CalcLoopStride(i);
    }
    blockFactor_ = totalLoop_ / totalCoreNum_;
    blockTail_ = totalLoop_ - blockFactor_ * totalCoreNum_;
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : totalCoreNum_;
}

void ReverseV2Tiling::SingleUbTiling()
{
    int64_t inDimNum = inputShape_.size();
    int64_t negOneDim = inDimNum - 1;
    int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / dtypeSize_;
    int64_t inNegOneSize = inputShape_[negOneDim];
    inNegOneSize = Ops::Base::CeilAlign(inNegOneSize, oneBlockNum);
    SplitProcess(inNegOneSize);
    return;
}

void ReverseV2Tiling::SplitProcess(int64_t inNegOneSize)
{
    totalLoop_ = 1;
    int64_t tmpAll = 1;
    int64_t splitSize = 1;
    //at least two dims will in
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        if (dimNum_ - 1 == i ) {
            tmpAll *= inNegOneSize;
        } else {
            tmpAll *= inputShape_[i];
        }
        // vf 循环使用uint16_t
        if ((availableUb_ / tmpAll) < 1 || inputShape_[i] >= MAX_INPUT_ELEMENTS || tmpAll >= MAX_INPUT_ELEMENTS) {
            splitDim_ = i;
            break;
        }
        splitSize = tmpAll;
    }

    for (int64_t i = 0; i < splitDim_; i++) {
        totalLoop_ *= inputShape_[i];
    }
    splitDimInNum_ = availableUb_ / splitSize;
    splitDimLoop_ = Ops::Base::CeilDiv(inputShape_[splitDim_], splitDimInNum_);
    splitDimTailInNum_ = inputShape_[splitDim_] - (splitDimLoop_ - 1) * splitDimInNum_;
    inUbSize_ = splitDimInNum_ * splitSize;
    totalLoop_ *= splitDimLoop_;
}


int64_t ReverseV2Tiling::CalcLoopStride(int64_t index) {
    if (index == splitDim_ - 1) {
        return splitDimLoop_;
    }
    return loopStride_[index + 1] * inputShape_[index + 1];
}

uint64_t ReverseV2Tiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ReverseV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(usedCoreNum_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_inputSize(inputSize_);
    tilingData_.set_param0(param0_);
    tilingData_.set_param1(param1_);
    tilingData_.set_param2(param2_);
    tilingData_.set_param3(param3_);
    tilingData_.set_param4(param4_);
    tilingData_.set_param5(param5_);
    tilingData_.set_param6(param6_);
    tilingData_.set_dim0(dim0_);
    tilingData_.set_dim1(dim1_);
    tilingData_.set_dim2(dim2_);
    tilingData_.set_dim3(dim3_);
    tilingData_.set_dim4(dim4_);
    tilingData_.set_dim5(dim5_);
    tilingData_.set_dim6(dim6_);
    tilingData_.set_dim7(dim7_);
    tilingData_.set_blockTail(blockTail_);
    tilingData_.set_dtypeSize(dtypeSize_);
    tilingData_.set_inUbSize(inUbSize_);
    tilingData_.set_splitDim(splitDim_);
    tilingData_.set_splitDimInNum(splitDimInNum_);
    tilingData_.set_splitDimTailInNum(splitDimTailInNum_);
    tilingData_.set_splitDimLoop(splitDimLoop_);
    tilingData_.set_dimNum(dimNum_);
    tilingData_.set_dim0Reversed(dim0Reversed_);
    tilingData_.set_splitDimReversed(splitDimReversed_);
    tilingData_.set_loopStride(loopStride_);
    tilingData_.set_usedCoreNum(usedCoreNum_);

    uint64_t dimNum = inputShape_.size() == 0 ?
                      static_cast<uint64_t>(inputSize_) : static_cast<uint64_t>(inputShape_.size());
    uint64_t THRESHOLD = 1000;
    uint64_t mergedInputDimNum = dimNum * THRESHOLD;
    uint64_t reversedDim = reversedDims_[0];
    if (reversedDim == static_cast<uint64_t>(0)) {
        reversedDim = TILING_REVERSE_DIM;
    } else {
        reversedDim = TILING_NOT_REVERSE_DIM;
    }
    // if inputSize_ in the range of uint32
    uint64_t isInputSizeInUint32 = 1;
    if (inputSize_ > MAX_INT32_NUM) {
        isInputSizeInUint32 = static_cast<uint64_t>(0);
    }
    if (isSimd_) {
        tilingKey_ = REVERSE_V2_SIMD_TILING_KEY;
    } else {
        tilingKey_ = mergedInputDimNum + reversedDim + isInputSizeInUint32;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ReverseV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalCoreNum: " << totalCoreNum_ << ", ";
    info << "usedCoreNum: " << usedCoreNum_ << ", ";
    info << "blockFactor: " << blockFactor_ << ", ";
    info << "param0: " << param0_ << ", ";
    info << "param1: " << param1_ << ", ";
    info << "param2: " << param2_ << ", ";
    info << "param3: " << param3_ << ", ";
    info << "param4: " << param4_ << ", ";
    info << "param5: " << param5_ << ", ";
    info << "param6: " << param6_ << ", ";
    info << "dim0: " << dim0_ << ", ";
    info << "dim1: " << dim1_ << ", ";
    info << "dim2: " << dim2_ << ", ";
    info << "dim3: " << dim3_ << ", ";
    info << "dim4: " << dim4_ << ", ";
    info << "dim5: " << dim5_ << ", ";
    info << "dim6: " << dim6_ << ", ";
    info << "dim7: " << dim7_ << ", ";
    info << "inputSize: " << inputSize_ << ", ";
    info << "tilingKey: " << tilingKey_ << ", ";
    info << "isSimd: " << isSimd_ << ", ";
    info << "dtypeSize: " << dtypeSize_ << ", ";
    info << "blockTail: " << blockTail_ << ", ";
    info << "inUbSize: " << inUbSize_ << ", ";
    info << "splitDim: " << splitDim_ << ", ";
    info << "splitDimInNum: " << splitDimInNum_ << ", ";
    info << "splitDimTailInNum: " << splitDimTailInNum_ << ", ";
    info << "splitDimLoop: " << splitDimLoop_ << ", ";
    info << "dimNum: " << dimNum_ << ", ";
    info << "dim0Reversed: " << dim0Reversed_ << ", ";
    info << "splitDimReversed: " << splitDimReversed_ << ", ";
    info << "loopStride0: " << loopStride_[DIM0] << ", ";
    info << "loopStride1: " << loopStride_[DIM1] << ", ";
    info << "loopStride2: " << loopStride_[DIM2] << ", ";
    info << "loopStride3: " << loopStride_[DIM3] << ", ";
    info << "loopStride4: " << loopStride_[DIM4] << ", ";
    info << "loopStride5: " << loopStride_[DIM5] << ", ";
    info << "loopStride6: " << loopStride_[DIM6] << ", ";
    info << "loopStride7: " << loopStride_[DIM7];

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

bool ReverseV2Tiling::IsTensorMove()
{
    // check InputTensor == OutputTensor
    if (inputShape_.size() < 1) {
        return true;
    }
    for (auto revdim : reversedDims_) {
        if (inputShape_[revdim] != 1) {
           return false;
        }
    }
    return true;
}

void CalcBlockFactor4ReverseV2(ReverseV2TilingParam &tilingParam, int64_t numel)
{
    tilingParam.uo = Ops::Base::CeilDiv(numel, tilingParam.ubFactor);
    tilingParam.tailBlockTailUbFactor = GetRemainder(numel, tilingParam.ubFactor);

    int64_t coreData = Ops::Base::CeilDiv(tilingParam.uo, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = Ops::Base::CeilDiv(tilingParam.uo, coreData);
    tilingParam.blockFactor = Ops::Base::CeilDiv(tilingParam.uo, tilingParam.usedCoreNum);
    tilingParam.tailBlockFactor = tilingParam.uo - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;
    if (tilingParam.tailBlockTailUbFactor == 0) {
        tilingParam.tailBlockTailUbFactor = tilingParam.ubFactor;
    }
}

static ge::graphStatus DoTiling4ReverseV2(const gert::TilingContext *context, ReverseV2TilingParam &tilingParam)
{
    auto xShapePtr = context->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    int64_t numel = xShape.GetShapeSize();
    // 获取输入数据类型所占的字节数
    auto inputXPtr = context->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto dtype = inputXPtr->GetDataType();
    tilingParam.bytesForOneData = ge::GetSizeByDataType(dtype);

    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / tilingParam.bytesForOneData;
    // 计算ubFactor
    if (numel >= maxUbAvailable) {
        tilingParam.ubFactor = maxUbAvailable;
    } else {
        tilingParam.ubFactor = numel;
    }
    CalcBlockFactor4ReverseV2(tilingParam, numel);
    if (tilingParam.usedCoreNum == tilingParam.totalCoreNum || tilingParam.blockFactor > 1) {
        return ge::GRAPH_SUCCESS;
    }
    // 当使用核数不满并且每个核只有一次循环时可以进行调整，增大使用核数，来提高性能
    if (GetRemainder(numel, tilingParam.totalCoreNum) == 0) {
        tilingParam.ubFactor = Ops::Base::FloorDiv(numel, tilingParam.totalCoreNum);
    } else {
        tilingParam.ubFactor = Ops::Base::FloorDiv(numel, tilingParam.totalCoreNum - 1);
    }
    tilingParam.ubFactor = Ops::Base::CeilAlign(tilingParam.ubFactor, ONE_BLK_BYTE / tilingParam.bytesForOneData);
    int64_t ubFactorMin = UB_FACTOR_MIN_BETY / tilingParam.bytesForOneData;
    tilingParam.ubFactor = tilingParam.ubFactor < ubFactorMin ? ubFactorMin : tilingParam.ubFactor;
    CalcBlockFactor4ReverseV2(tilingParam, numel);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData4ReverseV2(TensorMoveTilingData& tilingData, const ReverseV2TilingParam& tilingParam)
{
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_tailBlockTailUbFactor(tilingParam.tailBlockTailUbFactor);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_tilingKey(tilingParam.tilingKey);
}

static void PrintTilingData4ReverseV2(const gert::TilingContext* context, TensorMoveTilingData& tilingData)
{
    OP_LOGI(context->GetNodeName(),
        "ReverseV2 tilingdata: totalCoreNum:%ld, usedCoreNum:%ld,  ubFactor:%ld, tailBlockTailUbFactor:%ld, "
        "blockFactor:%ld, tailBlockFactor:%ld, tilingKey:%ld ",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_ubFactor(),
        tilingData.get_tailBlockTailUbFactor(), tilingData.get_blockFactor(), tilingData.get_tailBlockFactor(),
        tilingData.get_tilingKey());
}

static ge::graphStatus ReverseV2SetTilingData(gert::TilingContext* context, TensorMoveTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReverseV2ToTensorMoveTilingForAscendC(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "ReverseV2ToTensorMoveTilingForAscendC running begin.");

    auto compileInfo = reinterpret_cast<const ReverseV2CompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReverseV2TilingParam tilingParam;
    tilingParam.totalCoreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSize;

    OP_CHECK_IF(DoTiling4ReverseV2(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Dotiling failed."), return ge::GRAPH_FAILED);

    // tilingkey由数据类型所占字节(1/2/4/8)*1000 表示
    tilingParam.tilingKey = tilingParam.bytesForOneData * DIGIT_THOUSAND;

    TensorMoveTilingData tilingData;
    SetTilingData4ReverseV2(tilingData, tilingParam);
    OP_CHECK_IF(ReverseV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ReverseV2SetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData4ReverseV2(context, tilingData);
    OP_LOGD(context->GetNodeName(), "ReverseV2ToTensorMoveTilingForAscendC running END ops-nn cang.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseV2TilingForAscendC(gert::TilingContext *context)
{
    ReverseV2Tiling tiling(context);
    if (tiling.GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tiling.IsTensorMove()) {
        return ReverseV2ToTensorMoveTilingForAscendC(context);
    }
    return tiling.DoTiling();
}
} // namespace optiling