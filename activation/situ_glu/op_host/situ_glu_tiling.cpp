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
 * \file situ_glu_tiling.cpp
 * \brief
 */
#include "situ_glu_tiling.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace ge;
namespace optiling {
constexpr int64_t X_INDEX = 0;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_INDEX = 0;
constexpr int64_t BETA_INDEX = 1;
constexpr int64_t LINEAR_BETA_INDEX = 2;
constexpr int64_t ACTIVATE_LEFT_INDEX = 3;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float); // 8 floats per 32B block
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t FLOATS_PER_PAIR = 7; // xQueue(DB2*2=4) + yQueue(1) + tmpBuf1(1) + tmpBuf2(1)

constexpr float BETA_DEFAULT = 1.0f;
constexpr float LINEAR_BETA_DEFAULT = 0.0f;
constexpr bool ACTIVATE_LEFT_DEFAULT = true;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
// 获取INPUT/OUTPUT/ATTR信息
ge::graphStatus SituGluTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }
// 获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus SituGluTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<SituGluCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNumAll_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNumAll_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

bool SituGluTiling::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93 &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
        return false;
    }
    return true;
}
// 计算数据切分TilingData
ge::graphStatus SituGluTiling::DoOpTiling()
{
    // 校验并获取输入参数
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 计算UB可以计算的最多pair数量（1个gate元素+1个up元素为1pair）
    CountMaxPair();
    // 设置tilingKey
    tilingKey_ = 1;
    // 设置tiling结构体参数
    tilingData_.set_coreNumAll(coreNumAll_);
    tilingData_.set_dimBatchSize(dimBatchSize_);
    tilingData_.set_dim2H(dim2H_);
    tilingData_.set_isLongH(isLongH_);
    tilingData_.set_ubMaxPair(ubMaxPair_);
    tilingData_.set_beta(beta_);
    tilingData_.set_linearBeta(linearBeta_);
    tilingData_.set_activateLeft(activateLeft_);
    return ge::GRAPH_SUCCESS;
}
// 计算高阶API的TilingData
ge::graphStatus SituGluTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }
// 计算Workspace 大小
ge::graphStatus SituGluTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
// 保存Tiling数据
ge::graphStatus SituGluTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(coreNumAll_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t SituGluTiling::GetTilingKey() const { return tilingKey_; }

// Dump Tiling数据
void SituGluTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey_: " << tilingKey_;
    info << ", coreNumAll: " << tilingData_.get_coreNumAll();
    info << ", ubSize_: " << ubSize_;
    info << ", xDims_: " << xDims_;
    info << ", cutDim_: " << cutDim_;
    info << ", dimBatchSize: " << tilingData_.get_dimBatchSize();
    info << ", dim2H: " << tilingData_.get_dim2H();
    info << ", isLongH: " << tilingData_.get_isLongH();
    info << ", beta: " << tilingData_.get_beta();
    info << ", linearBeta: " << tilingData_.get_linearBeta();
    info << ", activateLeft: " << tilingData_.get_activateLeft();
    info << ", ubMaxPair: " << tilingData_.get_ubMaxPair();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus SituGluTiling::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(CheckAndGetXAndAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "check x and attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckY() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "check y param failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluTiling::CheckAndGetXAndAttrs()
{
    // 获取attr参数
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* attrDim = attrs->GetAttrPointer<int>(DIM_INDEX);
    cutDim_ = attrDim == nullptr ? -1 : *attrDim;
    auto* attrBeta = attrs->GetAttrPointer<float>(BETA_INDEX);
    beta_ = attrBeta == nullptr ? BETA_DEFAULT : *attrBeta;
    auto* attrLinearBeta = attrs->GetAttrPointer<float>(LINEAR_BETA_INDEX);
    linearBeta_ = attrLinearBeta == nullptr ? LINEAR_BETA_DEFAULT : *attrLinearBeta;
    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ACTIVATE_LEFT_INDEX);
    bool activateLeft = attrActivateLeft == nullptr ? ACTIVATE_LEFT_DEFAULT : *attrActivateLeft;
    activateLeft_ = activateLeft ? 1 : 0;
    // 获取x shape
    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    xDims_ = inputShapeX.GetDimNum();
    OP_CHECK_IF(
        (cutDim_ > (xDims_ - 1) || cutDim_ < -1 * xDims_),
        OP_LOGE(context_->GetNodeName(), "dim should in [-%ld, %ld], but get %ld,", xDims_, xDims_ - 1, cutDim_),
        return ge::GRAPH_FAILED);
    cutDim_ = cutDim_ < 0 ? (cutDim_ + xDims_) : cutDim_; // cutDim统一为正数
    cutDimSize_ = inputShapeX.GetDim(cutDim_);
    // x合轴为2维：dimBatchSize为cutDim之前的各维乘积，dim2H为cutDim及之后的各维乘积
    if (xDims_ == 1) {
        dimBatchSize_ = 1;
        dim2H_ = inputShapeX.GetDim(0);
    } else {
        for (int64_t i = 0; i < cutDim_; i++) {
            dimBatchSize_ *= inputShapeX.GetDim(i);
        }
        for (int64_t j = cutDim_; j < xDims_; j++) {
            dim2H_ *= inputShapeX.GetDim(j);
        }
    }
    OP_CHECK_IF((cutDimSize_ % 2 == 1),
                OP_LOGE(context_->GetNodeName(), "x[dim] should be divisible by 2, but get %ld", cutDimSize_),
                return ge::GRAPH_FAILED);
    auto descX = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descX);
    xDtype_ = descX->GetDataType();
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype_) == SUPPORT_DTYPE.end()),
                OP_LOGE(context_->GetNodeName(), "x dtype only support float32/float16/bfloat16, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluTiling::CheckY()
{
    auto shapeY = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
    const gert::Shape& inputShapeY = shapeY->GetStorageShape();
    int64_t yDims = inputShapeY.GetDimNum();
    auto descY = context_->GetInputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, descY);
    auto yDtype = descY->GetDataType();
    OP_CHECK_IF((yDims != xDims_),
                OP_LOGE(context_->GetNodeName(),
                        "the number of dimensions of y should be equal to dimensions of x, but get %ld.", yDims),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (inputShapeY.GetDim(cutDim_) != (cutDimSize_ / SWI_FACTOR)),
        OP_LOGE(context_->GetNodeName(), "y[dim] should be equal to x[dim] divided by 2, but get %ld, expected %ld.",
                inputShapeY.GetDim(cutDim_), cutDimSize_ / SWI_FACTOR),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDtype != xDtype_),
                OP_LOGE(context_->GetNodeName(), "y dtype should be the same as x, please cheack."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluTiling::CountMaxPair()
{
    ubMaxPair_ = 1;
    int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE;
    // fp32: xQueue(DB2*2 halves=4) + yQueue(1) + tmpBuf1(1) + tmpBuf2(1) = 7 floats/pair
    // fp16/bf16: xQueue(4 T) + yQueue(1 T) + 5 float bufs(tmp1/tmp2/gateF/upF/yF) = 5*4 + 5*T bytes/pair
    int64_t bytesPerPair = FLOATS_PER_PAIR * static_cast<int64_t>(sizeof(float));
    int64_t alignElem = BLOCK_ELEM; // 32B / sizeof(float) = 8
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        constexpr int64_t MIXED_TBUFS = 5;                                             // tmp1 + tmp2 + gateF + upF + yF
        bytesPerPair = SWI_FACTOR * SWI_FACTOR * static_cast<int64_t>(sizeof(int16_t)) // xQueue: 2 halves * DB2 * T
                       + static_cast<int64_t>(sizeof(int16_t))                         // yQueue: 1 * T
                       + MIXED_TBUFS * static_cast<int64_t>(sizeof(float));
        alignElem = BLOCK_SIZE / static_cast<int64_t>(sizeof(int16_t)); // 16, 同时满足 T(16) 与 float(8) 的 32B 对齐
    }
    ubMaxPair_ = numerator / bytesPerPair;
    ubMaxPair_ = ubMaxPair_ / alignElem * alignElem; // 32字节对齐
    OP_CHECK_IF((numerator <= 0 || ubMaxPair_ <= 0),
                OP_LOGE(context_->GetNodeName(), "Input not supported, ub size is too small."),
                return ge::GRAPH_FAILED);
    int64_t dimH = dim2H_ / SWI_FACTOR;
    int64_t dsize = (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) ? static_cast<int64_t>(sizeof(int16_t)) :
                                                                            static_cast<int64_t>(sizeof(float));
    // short-H 路径按行 stride 拷贝(blockCount>1)，要求每个半行(dimH*dsize) 32B 对齐；
    // 否则回退到 long-H 连续单块拷贝路径，由 DataCopyPad 处理 sub-block 对齐。
    bool halfRowAligned = (dimH * dsize) % BLOCK_SIZE == 0;
    isLongH_ = (ubMaxPair_ < dimH || !halfRowAligned) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SituGlu", SituGluTiling, 20000);

ge::graphStatus TilingForSituGlu(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSituGlu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSituGlu enter.");
    auto compileInfo = context->GetCompiledInfo<SituGluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %u",
                        static_cast<uint32_t>(compileInfo->coreNum)),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepareForSituGlu exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SituGlu).Tiling(TilingForSituGlu).TilingParse<SituGluCompileInfo>(TilingPrepareForSituGlu);
} // namespace optiling
