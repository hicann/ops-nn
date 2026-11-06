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
 * \file situ_glu_grad_tiling.cpp
 * \brief
 */
#include "situ_glu_grad_tiling.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace ge;
namespace optiling {
constexpr int64_t GRADY_INDEX = 0;
constexpr int64_t X_INDEX = 1;
constexpr int64_t GRADX_INDEX = 0;
constexpr int64_t DIM_INDEX = 0;
constexpr int64_t BETA_INDEX = 1;
constexpr int64_t LINEAR_BETA_INDEX = 2;
constexpr int64_t ACTIVATE_LEFT_INDEX = 3;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr int64_t UB_RESERVE = 1024;

// fp32: gradYQueue(DB2*1=2) + xQueue(DB2*2=4) + gradXQueue(1*2=2) + 5 tmpBufs = 13 floats/pair
constexpr int64_t FLOATS_PER_PAIR_FP32 = 13;
// fp16/bf16: gradYQueue(2T) + xQueue(4T) + gradXQueue(2T) = 8T + gateF+upF+gradYF+5tmp = 8 float bufs
constexpr int64_t MIXED_TBUFS = 8; // gateF + upF + gradYF + tmp1..tmp5

constexpr float BETA_DEFAULT = 1.0f;
constexpr float LINEAR_BETA_DEFAULT = 0.0f;
constexpr bool ACTIVATE_LEFT_DEFAULT = true;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

ge::graphStatus SituGluGradTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SituGluGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<SituGluGradCompileInfo>();
        if (compileInfoPtr == nullptr) {
            const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context_));
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "compile info is null");
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), "compile info is null"}));
            return ge::GRAPH_FAILED;
        };
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

bool SituGluGradTiling::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93 &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND950 && !Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    return true;
}

ge::graphStatus SituGluGradTiling::DoOpTiling()
{
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CountMaxPair() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingKey_ = 1;
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

ge::graphStatus SituGluGradTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SituGluGradTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGradTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(coreNumAll_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t SituGluGradTiling::GetTilingKey() const { return tilingKey_; }

void SituGluGradTiling::DumpTilingInfo()
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

ge::graphStatus SituGluGradTiling::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(CheckAndGetXAndAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "check x and attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGradY() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "check grad_y param failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGradTiling::CheckAndGetXAndAttrs()
{
    auto* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context_) == nullptr || (context_)->GetNodeName() == nullptr ?
                                                       "situ_glu_grad" :
                                                       (context_)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto* attrDim = attrs->GetAttrPointer<int>(DIM_INDEX);
    cutDim_ = attrDim == nullptr ? -1 : *attrDim;
    auto* attrBeta = attrs->GetAttrPointer<float>(BETA_INDEX);
    beta_ = attrBeta == nullptr ? BETA_DEFAULT : *attrBeta;
    auto* attrLinearBeta = attrs->GetAttrPointer<float>(LINEAR_BETA_INDEX);
    linearBeta_ = attrLinearBeta == nullptr ? LINEAR_BETA_DEFAULT : *attrLinearBeta;
    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ACTIVATE_LEFT_INDEX);
    bool activateLeft = attrActivateLeft == nullptr ? ACTIVATE_LEFT_DEFAULT : *attrActivateLeft;
    activateLeft_ = activateLeft ? 1 : 0;

    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(shapeX == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "situ_glu_grad" :
                                                                                     (context_)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    xDims_ = inputShapeX.GetDimNum();
    OP_CHECK_IF((cutDim_ > (xDims_ - 1) || cutDim_ < -1 * xDims_),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "dim", std::to_string(cutDim_),
                    ("dim should in [-" + std::to_string(xDims_) + ", " + std::to_string(xDims_ - 1) + "], but get " +
                     std::to_string(cutDim_) + ",")),
                return ge::GRAPH_FAILED);
    cutDim_ = cutDim_ < 0 ? (cutDim_ + xDims_) : cutDim_;
    cutDimSize_ = inputShapeX.GetDim(cutDim_);
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
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(cutDimSize_),
                    ("x[dim] should be divisible by 2, but get " + std::to_string(cutDimSize_))),
                return ge::GRAPH_FAILED);
    auto descX = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(descX == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "situ_glu_grad" :
                                                                                     (context_)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    xDtype_ = descX->GetDataType();
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype_) == SUPPORT_DTYPE.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      std::to_string(static_cast<int>(descX->GetDataType())),
                                                      "x dtype only support float32/float16/bfloat16, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGradTiling::CheckGradY() const
{
    auto shapeGradY = context_->GetInputShape(GRADY_INDEX);
    OP_CHECK_IF(shapeGradY == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "situ_glu_grad" :
                                                                                     (context_)->GetNodeName()),
                    "grad_y"),
                return ge::GRAPH_FAILED);
    const gert::Shape& inputShapeGradY = shapeGradY->GetStorageShape();
    int64_t gradYDims = inputShapeGradY.GetDimNum();
    auto descGradY = context_->GetInputDesc(GRADY_INDEX);
    OP_CHECK_IF(descGradY == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "situ_glu_grad" :
                                                                                     (context_)->GetNodeName()),
                    "grad_y"),
                return ge::GRAPH_FAILED);
    auto gradYDtype = descGradY->GetDataType();
    OP_CHECK_IF((gradYDims != xDims_),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "x,grad_y", std::to_string(xDims_) + "," + std::to_string(gradYDims),
                    ("the number of dimensions of grad_y should be equal to dimensions of x, but get " +
                     std::to_string(gradYDims) + ".")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputShapeGradY.GetDim(cutDim_) != (cutDimSize_ / SWI_FACTOR)),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "grad_y", std::to_string(inputShapeGradY.GetDim(cutDim_)),
                    ("grad_y[dim] should be equal to x[dim] divided by 2, but get " +
                     std::to_string(inputShapeGradY.GetDim(cutDim_)) + ", expected " +
                     std::to_string(cutDimSize_ / SWI_FACTOR) + ".")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((gradYDtype != xDtype_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x,grad_y",
                                                       std::to_string(static_cast<int>(xDtype_)) + "," +
                                                           std::to_string(static_cast<int>(descGradY->GetDataType())),
                                                       "grad_y dtype should be the same as x, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGradTiling::CountMaxPair()
{
    ubMaxPair_ = 1;
    int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE;
    int64_t bytesPerPair = FLOATS_PER_PAIR_FP32 * static_cast<int64_t>(sizeof(float));
    int64_t alignElem = BLOCK_ELEM;
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        // T-typed queues: gradYQ(2T) + xQ(4T) + gradXQ(2T) = 8T
        // float bufs: gateF+upF+gradYF+5tmp = 8 float bufs
        bytesPerPair = SWI_FACTOR * SWI_FACTOR * static_cast<int64_t>(sizeof(int16_t)) // gradYQueue: DB2*1*T
                       + SWI_FACTOR * SWI_FACTOR * SWI_FACTOR * static_cast<int64_t>(sizeof(int16_t)) // xQueue: DB2*2*T
                       + SWI_FACTOR * static_cast<int64_t>(sizeof(int16_t)) // gradXQueue: 1*2*T
                       + MIXED_TBUFS * static_cast<int64_t>(sizeof(float));
        alignElem = BLOCK_SIZE / static_cast<int64_t>(sizeof(int16_t));
    }
    ubMaxPair_ = numerator / bytesPerPair;
    ubMaxPair_ = ubMaxPair_ / alignElem * alignElem;
    if ((numerator <= 0 || ubMaxPair_ <= 0)) {
        const std::string msgOpName = (context_ == nullptr || context_->GetNodeName() == nullptr ?
                                           "situ_glu_grad" :
                                           context_->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Validated inputs leave no usable UB tile: UBBytes=") +
                                       std::to_string(ubSize_) + ", remainingBytes=" + std::to_string(numerator) +
                                       ", bytesPerPair=" + std::to_string(bytesPerPair));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    int64_t dimH = dim2H_ / SWI_FACTOR;
    int64_t dsize = (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) ? static_cast<int64_t>(sizeof(int16_t)) :
                                                                            static_cast<int64_t>(sizeof(float));
    bool halfRowAligned = (dimH * dsize) % BLOCK_SIZE == 0;
    isLongH_ = (ubMaxPair_ < dimH || !halfRowAligned) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SituGluGrad", SituGluGradTiling, 20000);

ge::graphStatus TilingForSituGluGrad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSituGluGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSituGluGrad enter.");
    auto compileInfo = context->GetCompiledInfo<SituGluGradCompileInfo>();
    if (compileInfo == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object compileInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object compileInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object platformInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object platformInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if ((compileInfo->coreNum <= 0)) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "situ_glu_grad" :
                                           context->GetNodeName());
        const std::string msgReason = std::string(
            ("Get core num failed, core num: " + std::to_string(static_cast<uint32_t>(compileInfo->coreNum))));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    if ((compileInfo->ubSize <= 0)) {
        const std::string msgOpName = (context == nullptr || context->GetNodeName() == nullptr ?
                                           "situ_glu_grad" :
                                           context->GetNodeName());
        const std::string msgReason = std::string(
            ("Get ub size failed, ub size: " + std::to_string(static_cast<uint32_t>(compileInfo->ubSize))));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };

    OP_LOGD(context, "TilingPrepareForSituGluGrad exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SituGluGrad)
    .Tiling(TilingForSituGluGrad)
    .TilingParse<SituGluGradCompileInfo>(TilingPrepareForSituGluGrad);
} // namespace optiling
