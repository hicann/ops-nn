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
 * \file in_training_update_grad_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "in_training_update_grad_tiling_arch35.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr int64_t NDC1HWC0_DIM_NUM = 6;
// C0 不做 ==16 的校验: 平台常量 C0_SIZE 虽为 16(tbe platform_info.py:39 / cce_params.py:144),
// 但本算子的全部尺寸都从 dy 的 C0 推导, 六个张量 C0 一致时(由下面的形状校验保证)结果对该布局即正确,
// C0 取值不影响我方输出正确性; 校验它只会拒掉本能正常工作的输入。A2 亦无此约束。
constexpr int64_t DIM_N = 0;
constexpr int64_t DIM_D = 1;
constexpr int64_t DIM_C1 = 2;
constexpr int64_t DIM_H = 3;
constexpr int64_t DIM_W = 4;
constexpr int64_t DIM_C0 = 5;

constexpr int64_t INPUT_DY_IDX = 0;
constexpr int64_t INPUT_X_IDX = 1;
constexpr int64_t INPUT_VARIANCE_IDX = 2;
constexpr int64_t INPUT_MEAN_IDX = 3;
constexpr int64_t OUTPUT_RES_GAMMA_IDX = 0;
constexpr int64_t OUTPUT_RES_BETA_IDX = 1;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MAX_BLOCK_COUNT = 65535; // DataCopyExtParams.blockCount is uint16
constexpr int64_t SINGLE_AIV_THRESHOLD_BYTES = 32L * 1024L;
// The kernel lays dy/x UB rows out with a 32B-aligned padded stride (register loads need 32B-aligned
// row starts), so capacity is accounted with the padded row size, not the raw C0 footprint.
constexpr int64_t UB_ALIGN_BYTES = 32;
// The arch35 register implementation computes at most 64 C0 lanes per row (one fp32 vector
// register); C0 beyond that silently truncates lanes >= 64 or raises a device aivec error, so the
// tiling gate rejects it up front instead of producing wrong results.
constexpr int64_t MAX_C0_LANES = 64;
// TilingData members and DataCopyExtParams fields are uint32: every derived product must fit
// before the narrowing cast, or the kernel would compute wrapped offsets/strides.
constexpr int64_t UINT32_LIMIT = 0xFFFFFFFFLL;

int64_t CeilAlign32(int64_t bytes) { return (bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES * UB_ALIGN_BYTES; }

// Tiling keys come from the shared tiling_data.h (single source with the kernel dispatch).
namespace Keys = InTrainingUpdateGradKeys;

constexpr float DEFAULT_EPSILON = 1e-6f;
} // namespace

namespace optiling {

ge::graphStatus InTrainingUpdateGradTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const InTrainingUpdateGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    vectorLength_ = compileInfoPtr->vectorLength;
    vlfp32_ = compileInfoPtr->vectorLength / sizeof(float);
    ubBlockSize_ = compileInfoPtr->ubBlockSize;

    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InTrainingUpdateGradTilingBase::CheckDtype()
{
    auto dyDesc = context_->GetInputDesc(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    dyDataType_ = dyDesc->GetDataType();
    OP_CHECK_IF(
        (dyDataType_ != ge::DT_FLOAT16 && dyDataType_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dy", ToString(dyDataType_).c_str(), "FLOAT or FLOAT16"),
        return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF((xDesc->GetDataType() != dyDataType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "dy and x",
                                                       ToString(xDesc->GetDataType()).c_str(),
                                                       "The dtypes of input dy and input x must be the same"),
                return ge::GRAPH_FAILED);

    const std::vector<std::pair<int64_t, std::string>> fp32Params = {{INPUT_VARIANCE_IDX, "variance"},
                                                                     {INPUT_MEAN_IDX, "mean"}};
    for (const auto& [idx, name] : fp32Params) {
        auto desc = context_->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_IF((desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name.c_str(),
                                              ToString(desc->GetDataType()).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    const std::vector<std::pair<int64_t, std::string>> fp32Outputs = {{OUTPUT_RES_GAMMA_IDX, "res_gamma"},
                                                                      {OUTPUT_RES_BETA_IDX, "res_beta"}};
    for (const auto& [idx, name] : fp32Outputs) {
        auto desc = context_->GetOutputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        OP_CHECK_IF((desc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name.c_str(),
                                              ToString(desc->GetDataType()).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InTrainingUpdateGradTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("INTrainingUpdateGrad", "TilingContext is nullptr."),
                return ge::GRAPH_FAILED);

    auto dyDesc = context_->GetInputDesc(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    // storage format 高位可能携带 C0 编码(GE 在部分图优化路径上标注),需取 primary 再比较
    format_ = static_cast<ge::Format>(ge::GetPrimaryFormat(dyDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(format_ != ge::FORMAT_NDC1HWC0,
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "dy", ToString(format_).c_str(), "NDC1HWC0"),
                return ge::GRAPH_FAILED);

    auto dyShape = context_->GetInputShape(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShape);
    auto storageShape = dyShape->GetStorageShape();
    int64_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF(dimNum != NDC1HWC0_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "dy", std::to_string(dimNum).c_str(),
                                                         "The shape dim of input dy must be 6 (NDC1HWC0)"),
                return ge::GRAPH_FAILED);

    numN_ = storageShape.GetDim(DIM_N);
    numD_ = storageShape.GetDim(DIM_D);
    numC1_ = storageShape.GetDim(DIM_C1);
    numH_ = storageShape.GetDim(DIM_H);
    numW_ = storageShape.GetDim(DIM_W);
    numC0_ = storageShape.GetDim(DIM_C0);

    OP_CHECK_IF((numN_ <= 0 || numC1_ <= 0 || numC0_ <= 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(storageShape).c_str(),
                                                      "N, C1 and C0 dims of input dy must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numD_ < 0 || numH_ < 0 || numW_ < 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", ToString(storageShape).c_str(),
                                                      "spatial dims (D, H, W) of input dy must not be negative"),
                return ge::GRAPH_FAILED);

    // 跨输入/输出 shape 一致性。tiling 的全部维度只从 dy 取(见上), kernel 也按这套几何读写其余
    // 张量: variance/mean 按每组 C0 个元素连续读(full_load.h:121 DataCopyPad varianceGm_[scalarOffset]),
    // 两个输出同样是 groupNum*C0。形状不符时不会报错, 而是按 dy 的几何去读写别人的内存 —— 越界或
    // 静默读错数据, 必须在入口拦住。
    //
    // A2(canndev in_training_update_grad.py)对此无显式校验, 但其 DSL 隐式要求等价:
    //   vadd(x, broadcast(mean, shape_dy))  -> x 必须与 dy 同形;
    //   broadcast(variance, shape_dy)       -> variance/mean 必须可广播到 dy;
    //   tuple_sum(..., [1,3,4], keepdims)   -> 输出形状由 dy 的归约轴决定, 与 variance 无关。
    // 故这里按**dy 推出的期望形状**判定, 不拿 variance 当基准(那样会把 A2 的语义搞反)。
    auto xShapePtr = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const auto& xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape != storageShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xShape).c_str(),
                                                      "The shape of x must be the same as dy"),
                return ge::GRAPH_FAILED);

    // 统计量与输出的期望形状: (N, 1, C1, 1, 1, C0) —— D/H/W 三个归约轴折叠为 1。
    gert::Shape statShape;
    statShape.SetDimNum(NDC1HWC0_DIM_NUM);
    statShape.SetDim(DIM_N, numN_);
    statShape.SetDim(DIM_D, 1);
    statShape.SetDim(DIM_C1, numC1_);
    statShape.SetDim(DIM_H, 1);
    statShape.SetDim(DIM_W, 1);
    statShape.SetDim(DIM_C0, numC0_);
    const std::vector<std::pair<const gert::Shape*, std::string>> statTensors = {
        {context_->GetInputShape(INPUT_VARIANCE_IDX) == nullptr ?
             nullptr :
             &context_->GetInputShape(INPUT_VARIANCE_IDX)->GetStorageShape(),
         "variance"},
        {context_->GetInputShape(INPUT_MEAN_IDX) == nullptr ?
             nullptr :
             &context_->GetInputShape(INPUT_MEAN_IDX)->GetStorageShape(),
         "mean"},
        {context_->GetOutputShape(OUTPUT_RES_GAMMA_IDX) == nullptr ?
             nullptr :
             &context_->GetOutputShape(OUTPUT_RES_GAMMA_IDX)->GetStorageShape(),
         "res_gamma"},
        {context_->GetOutputShape(OUTPUT_RES_BETA_IDX) == nullptr ?
             nullptr :
             &context_->GetOutputShape(OUTPUT_RES_BETA_IDX)->GetStorageShape(),
         "res_beta"}};
    for (const auto& [shapePtr, name] : statTensors) {
        OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "shape of %s is null", name.c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            *shapePtr != statShape,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name.c_str(), ToString(*shapePtr).c_str(),
                                                  "The shape must be (N, 1, C1, 1, 1, C0) derived from dy"),
            return ge::GRAPH_FAILED);
    }

    epsilon_ = DEFAULT_EPSILON;

    // Shared kernel-envelope gate (all three tiling templates): C0 lane bound and uint32 fits.
    // Every product is validated by pairwise quotient BEFORE it is computed (dims are
    // non-negative here), so extreme dim values cannot overflow the int64 check itself.
    OP_CHECK_IF((numC0_ > MAX_C0_LANES),
                OP_LOGE(context_->GetNodeName(),
                        "C0 dim of input dy must not exceed %lld (one fp32 vector "
                        "register); got %lld",
                        static_cast<long long>(MAX_C0_LANES), static_cast<long long>(numC0_)),
                return ge::GRAPH_FAILED);
    const auto pairFits = [](int64_t a, int64_t b) {
        return a <= UINT32_LIMIT && b <= (a > 0 ? UINT32_LIMIT / a : UINT32_LIMIT);
    };
    const auto rejectProduct = [this](const std::string& name) {
        OP_LOGE(context_->GetNodeName(), "Dim product %s of input dy exceeds the uint32 kernel envelope", name.c_str());
        return ge::GRAPH_FAILED;
    };
    OP_CHECK_IF(!pairFits(numH_, numW_), rejectProduct("H*W"), return ge::GRAPH_FAILED);
    numHW_ = numH_ * numW_;
    OP_CHECK_IF(!pairFits(numD_, numHW_), rejectProduct("D*H*W"), return ge::GRAPH_FAILED);
    reduceR_ = numD_ * numHW_;
    OP_CHECK_IF(!pairFits(numN_, numC1_), rejectProduct("N*C1"), return ge::GRAPH_FAILED);
    groupNum_ = numN_ * numC1_;
    OP_CHECK_IF(!pairFits(numHW_, numC0_), rejectProduct("H*W*C0"), return ge::GRAPH_FAILED);
    blockLenElem_ = numHW_ * numC0_;
    OP_CHECK_IF(!pairFits(groupNum_, numC0_), rejectProduct("N*C1*C0"), return ge::GRAPH_FAILED);

    return CheckDtype();
}

ge::graphStatus InTrainingUpdateGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus InTrainingUpdateGradTilingBase::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// ReduceEmpty (TilingKey 50000): R == 0 -> write 0.0 to both outputs (N*C1*C0 elements each).
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradReduceEmptyTiling::IsCapable() { return reduceR_ == 0; }

ge::graphStatus InTrainingUpdateGradReduceEmptyTiling::DoOpTiling()
{
    int64_t totalLength = groupNum_ * numC0_;
    int64_t aivNum = static_cast<int64_t>(aicoreParams_.blockDim);
    int64_t elemSize = FP32_BYTE;
    int64_t perLoopMax = static_cast<int64_t>(aicoreParams_.ubSize) / elemSize;
    int64_t alignUnit = ubBlockSize_ / elemSize; // 8 fp32 elements per 32B block

    blockNum_ = Ops::Base::CeilDiv(totalLength * elemSize, SINGLE_AIV_THRESHOLD_BYTES);
    if (blockNum_ > aivNum) {
        blockNum_ = aivNum;
    }
    if (blockNum_ < 1) {
        blockNum_ = 1;
    }

    int64_t perCoreElements = Ops::Base::CeilDiv(totalLength, blockNum_);
    int64_t lastCoreElements = totalLength - (blockNum_ - 1) * perCoreElements;
    // Non-negative invariant before the uint32 narrowing casts: a negative tail count wraps
    // to a huge uint32 in the kernel's copy loop and overruns the output buffers. This holds
    // by construction (perCoreElements = ceil(total / blockNum) implies (blockNum-1)*perCore <
    // total), but is checked so that a future formula change fails loudly here instead of on
    // device.
    OP_CHECK_IF((perCoreElements < 0 || lastCoreElements < 0),
                OP_LOGE(context_->GetNodeName(),
                        "ReduceEmpty element split produced a negative count: "
                        "perCoreElements=%lld, lastCoreElements=%lld",
                        static_cast<long long>(perCoreElements), static_cast<long long>(lastCoreElements)),
                return ge::GRAPH_FAILED);
    td_.perCoreElements = static_cast<uint32_t>(perCoreElements);
    td_.lastCoreElements = static_cast<uint32_t>(lastCoreElements);

    int64_t perCorePerLoop = Ops::Base::FloorAlign(std::min(perLoopMax, perCoreElements), alignUnit);
    if (perCorePerLoop < 1) {
        perCorePerLoop = perCoreElements;
    }
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perCorePerLoop);
    int64_t perCoreLastLoop = perCoreElements - (perCoreLoops - 1) * perCorePerLoop;
    OP_CHECK_IF((perCoreLastLoop < 0),
                OP_LOGE(context_->GetNodeName(), "ReduceEmpty per-core loop split produced a negative count: %lld",
                        static_cast<long long>(perCoreLastLoop)),
                return ge::GRAPH_FAILED);
    td_.perCoreLoops = static_cast<uint32_t>(perCoreLoops);
    td_.perCorePerLoopElements = static_cast<uint32_t>(perCorePerLoop);
    td_.perCoreLastLoopElements = static_cast<uint32_t>(perCoreLastLoop);

    int64_t lastCorePerLoop = Ops::Base::FloorAlign(std::min(perLoopMax, lastCoreElements), alignUnit);
    if (lastCorePerLoop < 1) {
        lastCorePerLoop = lastCoreElements;
    }
    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, lastCorePerLoop);
    int64_t lastCoreLastLoop = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoop;
    OP_CHECK_IF((lastCoreLastLoop < 0),
                OP_LOGE(context_->GetNodeName(), "ReduceEmpty last-core loop split produced a negative count: %lld",
                        static_cast<long long>(lastCoreLastLoop)),
                return ge::GRAPH_FAILED);
    td_.lastCoreLoops = static_cast<uint32_t>(lastCoreLoops);
    td_.lastCorePerLoopElements = static_cast<uint32_t>(lastCorePerLoop);
    td_.lastCoreLastLoopElements = static_cast<uint32_t>(lastCoreLastLoop);
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradReduceEmptyTiling::GetTilingKey() const { return Keys::REDUCE_EMPTY; }

ge::graphStatus InTrainingUpdateGradReduceEmptyTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// FullLoad (TilingKey 100000): one (n,c1) group's R*C0 block fits UB (double buffered).
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradFullLoadTiling::IsCapable()
{
    if (reduceR_ <= 0) {
        return false;
    }
    if (numD_ > MAX_BLOCK_COUNT) { // strided aggregate needs blockCount = D <= uint16 max
        return false;
    }
    int64_t elemSize = (dyDataType_ == ge::DT_FLOAT16) ? FP16_BYTE : FP32_BYTE;
    // The aggregate copy's srcStride is a uint32 byte gap: (numC1-1)*blockLenElem*elemSize.
    // Validate by quotient (blockBytes = blockLenElem*elemSize cannot overflow int64 because
    // blockLenElem already fits uint32) instead of multiplying the three factors first.
    const int64_t blockBytes = blockLenElem_ * elemSize;
    if (numC1_ > 1 && (blockBytes > UINT32_LIMIT || (numC1_ - 1) > UINT32_LIMIT / blockBytes)) {
        return false;
    }
    int64_t regPad = vectorLength_;
    // Padded row footprint: rows are stored at a 32B-aligned stride even when C0*elemSize is not.
    int64_t spatialBytes = reduceR_ * CeilAlign32(numC0_ * elemSize);
    // TPipe::InitBuffer rounds every buffer length up to a 32B multiple before allocating, so each
    // term must be accounted at its aligned length. dy/x stay aligned as is (rows are 32B-strided
    // and regPad is a 32B multiple); scalar buffers need explicit rounding.
    int64_t scalarBuf = CeilAlign32(numC0_ * FP32_BYTE + regPad);        // variance/mean/rstd/cGamma/cBeta
    int64_t need = DOUBLE_BUFFER * (spatialBytes + regPad) * 2 +         // dy + x
                   2 * scalarBuf +                                       // variance + mean
                   DOUBLE_BUFFER * CeilAlign32(numC0_ * FP32_BYTE) * 2 + // res_gamma + res_beta (masked stores)
                   scalarBuf +                                           // rstd (full-register reload, padded)
                   2 * scalarBuf;                                        // Kahan cGamma + cBeta (reloaded, padded)
    return need <= static_cast<int64_t>(aicoreParams_.ubSize);
}

ge::graphStatus InTrainingUpdateGradFullLoadTiling::DoOpTiling()
{
    int64_t perCoreGroups = Ops::Base::CeilDiv(groupNum_, static_cast<int64_t>(aicoreParams_.blockDim));
    if (perCoreGroups < 1) {
        perCoreGroups = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(groupNum_, perCoreGroups);

    td_.numC1 = static_cast<uint32_t>(numC1_);
    td_.numD = static_cast<uint32_t>(numD_);
    td_.numHW = static_cast<uint32_t>(numHW_);
    td_.numC0 = static_cast<uint32_t>(numC0_);
    td_.reduceR = static_cast<uint32_t>(reduceR_);
    td_.groupNum = static_cast<uint32_t>(groupNum_);
    td_.usedCoreNum = static_cast<uint32_t>(blockNum_);
    td_.perCoreGroups = static_cast<uint32_t>(perCoreGroups);
    td_.blockLenElem = static_cast<uint32_t>(blockLenElem_);
    td_.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradFullLoadTiling::GetTilingKey() const { return Keys::FULL_LOAD; }

ge::graphStatus InTrainingUpdateGradFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Stream (TilingKey 200000): fallback for large R*C0 (or large D); stream D-slices in row chunks.
// ---------------------------------------------------------------------------------------------------
bool InTrainingUpdateGradStreamTiling::IsCapable() { return reduceR_ > 0; }

ge::graphStatus InTrainingUpdateGradStreamTiling::DoOpTiling()
{
    int64_t aivNum = static_cast<int64_t>(aicoreParams_.blockDim);
    if (aivNum < 1) {
        aivNum = 1;
    }

    int64_t elemSize = (dyDataType_ == ge::DT_FLOAT16) ? FP16_BYTE : FP32_BYTE;
    int64_t regPad = vectorLength_;
    // TPipe::InitBuffer rounds every buffer length up to a 32B multiple before allocating, so the
    // scalar buffers must be accounted at their aligned length. dy/x chunks stay aligned (rows are
    // 32B-strided, regPad is a 32B multiple) and keep the register-pad reservation below.
    int64_t scalarBuf = CeilAlign32(numC0_ * FP32_BYTE + regPad);
    int64_t wspBuf = CeilAlign32(4 * CeilAlign32(numC0_ * FP32_BYTE) + regPad); // D-split partial 读入缓冲
    int64_t overhead = 2 * scalarBuf +                                          // variance + mean
                       DOUBLE_BUFFER * scalarBuf * 2 +                          // res_gamma + res_beta (reloaded)
                       scalarBuf +                                              // rstd (full-register reload, padded)
                       DOUBLE_BUFFER * scalarBuf * 2 + // Kahan cGamma + cBeta (VECOUT 双缓冲队列)
                       wspBuf +                        // partial 读入队列(始终分配)
                       DOUBLE_BUFFER * regPad * 2;     // register pad reserved for dy + x
    int64_t avail = static_cast<int64_t>(aicoreParams_.ubSize) - overhead;
    // Padded row footprint: rows are stored at a 32B-aligned stride even when C0*elemSize is not.
    int64_t perRowBytes = CeilAlign32(numC0_ * elemSize);
    int64_t rawTileRows = avail / (DOUBLE_BUFFER * perRowBytes * 2);
    if (rawTileRows < 1) {
        rawTileRows = 1;
    }
    // streamTileRows 直接下发 rawTileRows(不按 numHW 夹取): kernel 用它同时给 dy/x chunk 缓冲定尺寸,
    // 而聚合路径一次要放 dBatch 个 D 切片 = dBatch*numHW 行, 可能远大于 numHW。行块路径里
    // loops = ceil(numHW/streamTileRows), rawTileRows >= numHW 时 loops == 1, rows == numHW,
    // 行为与"夹到 numHW"一致。
    int64_t streamTileRows = rawTileRows;

    // D 方向聚合 DMA: 一个 UB chunk 能放下整数个 D 切片时, 用一条 strided 描述符批量搬
    // dBatch 个切片(full_load 的整组聚合写法), 把"逐 d 两次 32B DMA"的指令条数从 numD 降到
    // numD/dBatch。C0*elemSize 非 32B 对齐时(c0Stride != C0)退回逐 d 行块路径。
    int64_t c0Stride = CeilAlign32(numC0_ * elemSize) / elemSize;
    int64_t dBatch = 0;
    if (c0Stride == numC0_) {
        dBatch = rawTileRows / numHW_; // numHW_ > 0 (reduceR_ > 0)
        if (dBatch > static_cast<int64_t>(MAX_BLOCK_COUNT)) {
            dBatch = MAX_BLOCK_COUNT; // DataCopyExtParams.blockCount is uint16
        }
        if (dBatch > numD_) {
            dBatch = numD_;
        }
        if (dBatch < 1) {
            dBatch = 0; // 单个切片都放不进 UB -> 行块路径
        }
        // 聚合 DMA 的 srcStride 是 uint32 字节数: (numC1-1)*blockLenElem*elemSize 必须放得下,
        // 否则回绕成小值会让 DMA 按错误的片间隔取数(静默读错). 与 FullLoad IsCapable 同一契约:
        // 溢出时退回逐 d 行块路径(该路径不使用 srcStride)。
        if (dBatch > 0 && numC1_ > 1) {
            const int64_t blockBytes = blockLenElem_ * elemSize;
            if (blockBytes <= 0 || blockBytes > UINT32_LIMIT || (numC1_ - 1) > UINT32_LIMIT / blockBytes) {
                dBatch = 0;
            }
        }
    }

    // 并行: group 数不足核数时把 D 切成多份, 每份一个任务; dSplits == 1 时沿用 group 并行
    // (此时 splitNum = groupNum, 每核 perCoreTasks 个 group, 与旧实现一致)。
    int64_t dSplits = 1;
    if (groupNum_ < aivNum) {
        dSplits = aivNum / groupNum_;
        if (dSplits > numD_) {
            dSplits = numD_;
        }
        if (dSplits < 1) {
            dSplits = 1;
        }
    }
    int64_t splitNum = groupNum_ * dSplits;
    int64_t perCoreTasks = Ops::Base::CeilDiv(splitNum, aivNum);
    if (perCoreTasks < 1) {
        perCoreTasks = 1;
    }
    blockNum_ = Ops::Base::CeilDiv(splitNum, perCoreTasks);

    int64_t dPerSplit = Ops::Base::CeilDiv(numD_, dSplits);
    // partial: 每任务 4 段(sum/comp x gamma/beta) x C0 floats
    wsBytes_ = (dSplits > 1) ? static_cast<uint64_t>(splitNum) * 4 * numC0_ * FP32_BYTE : 0;
    (void)dPerSplit; // kernel 侧按 ceil(numD/dSplits) 现算, 不下发(结构体容量受限)

    td_.numC1 = static_cast<uint32_t>(numC1_);
    td_.numD = static_cast<uint32_t>(numD_);
    td_.numHW = static_cast<uint32_t>(numHW_);
    td_.numC0 = static_cast<uint32_t>(numC0_);
    td_.groupNum = static_cast<uint32_t>(groupNum_);
    td_.usedCoreNum = static_cast<uint32_t>(blockNum_);
    td_.blockLenElem = static_cast<uint32_t>(blockLenElem_);
    td_.streamTileRows = static_cast<uint32_t>(streamTileRows);
    td_.dSplits = static_cast<uint32_t>(dSplits);
    td_.dBatch = static_cast<uint32_t>(dBatch);
    td_.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

uint64_t InTrainingUpdateGradStreamTiling::GetTilingKey() const { return Keys::STREAM; }

ge::graphStatus InTrainingUpdateGradStreamTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_ + static_cast<size_t>(wsBytes_);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(td_), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------------------
static ge::graphStatus Tiling4INTrainingUpdateGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("INTrainingUpdateGrad", "Tiling context is nullptr"),
                return ge::GRAPH_FAILED);
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4INTrainingUpdateGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("INTrainingUpdateGrad", "TilingParse context is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<InTrainingUpdateGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context, "Get core num failed, core num: %lu", compileInfo->coreNum), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Get ub size failed, ub size: %lu", compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((compileInfo->ubBlockSize <= 0),
                OP_LOGE(context, "Get block size failed, block size: %lu", compileInfo->ubBlockSize),
                return ge::GRAPH_FAILED);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((compileInfo->vectorLength <= 0),
                OP_LOGE(context, "Get vector length failed, vector length: %u", compileInfo->vectorLength),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INTrainingUpdateGrad)
    .Tiling(Tiling4INTrainingUpdateGrad)
    .TilingParse<InTrainingUpdateGradCompileInfo>(TilingPrepare4INTrainingUpdateGrad);

REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradReduceEmptyTiling, IN_UG_REDUCE_EMPTY_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradFullLoadTiling, IN_UG_FULL_LOAD_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(INTrainingUpdateGrad, InTrainingUpdateGradStreamTiling, IN_UG_STREAM_PRIORITY);

} // namespace optiling
