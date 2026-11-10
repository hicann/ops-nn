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
 * \file clipped_swiglu_tiling.cc
 * \brief
 */
#include "clipped_swiglu_tiling.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using Ops::NN::Optiling::TilingRegistry;
// using namespace AscendC;
using namespace ge;
namespace optiling {
constexpr int64_t X_INDEX = 0;
constexpr int64_t GROUP_INDEX_INDEX = 1;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_INDEX = 0;
constexpr int64_t ALPHA_INDEX = 1;
constexpr int64_t LIMIT_INDEX = 2;
constexpr int64_t BIAS_INDEX = 3;
constexpr int64_t INTERLEAVED_INDEX = 4;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t DTYPE_FACTOR = 2;
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t SIZE_OF_BF16_FP16 = 2;
constexpr int64_t BATCH_MODE = 1;

constexpr float CLAMP_LIMIT_DEFAULT = 7.0;
constexpr float GLU_ALPHA_DEFAULT = 1.702;
constexpr float GLU_BIAS_DEFAULT = 1.0;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
// 获取INPUT/OUTPUT/ATTR信息
ge::graphStatus ClippedSwigluTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }
// 获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus ClippedSwigluTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<ClippedSwigluCompileInfo>();
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
    // maxPreCore_ = static_cast<int64_t>(coreNum_);
    return ge::GRAPH_SUCCESS;
}

bool ClippedSwigluTiling::IsCapable()
{
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93 &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        return false;
    }
    return true;
}
// 计算数据切分TilingData
ge::graphStatus ClippedSwigluTiling::DoOpTiling()
{
    // 校验并获取输入参数
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 计算UB可以计算的最多pair数量（1奇1偶为1pair 或者 1前1后为1pair）
    if (CountMaxPair() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 设置tilingKey
    tilingKey_ = 1;
    // 设置tiling结构体参数
    tilingData_.set_coreNumAll(coreNumAll_);
    tilingData_.set_dimBatchSize(dimBatchSize_);
    tilingData_.set_dim2H(dim2H_);
    tilingData_.set_isLongH(isLongH_);
    tilingData_.set_isGroup(isGroup_);
    tilingData_.set_isInterleaved(isInterleaved_);
    tilingData_.set_gluAlpha(gluAlpha_);
    tilingData_.set_gluLimit(gluLimit_);
    tilingData_.set_gluBias(gluBias_);
    tilingData_.set_ubMaxPair(ubMaxPair_);
    tilingData_.set_groupNum(groupNum_);
    return ge::GRAPH_SUCCESS;
}
// 计算高阶API的TilingData
ge::graphStatus ClippedSwigluTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }
// 计算Workspace 大小
ge::graphStatus ClippedSwigluTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
// 保存Tiling数据
ge::graphStatus ClippedSwigluTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(coreNumAll_);
    if (isGroup_ == 1) {
        // group 路径 kernel 使用 SyncAll 核间同步，需设置为 batch mode；
        // 非 group 路径核间无依赖，保持默认调度，避免多余的同步开销
        uint32_t batchMode = static_cast<uint32_t>(BATCH_MODE);
        auto scheduleRet = context_->SetScheduleMode(batchMode);
        OP_CHECK_IF(scheduleRet != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetScheduleMode error"),
                    return ge::GRAPH_FAILED);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t ClippedSwigluTiling::GetTilingKey() const { return tilingKey_; }

// Dump Tiling数据
void ClippedSwigluTiling::DumpTilingInfo()
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
    info << ", xCutDimNum_: " << xCutDimNum_;
    info << ", isGroup: " << tilingData_.get_isGroup();
    info << ", isInterleaved: " << tilingData_.get_isInterleaved();
    info << ", gluLimit: " << tilingData_.get_gluLimit();
    info << ", gluAlpha: " << tilingData_.get_gluAlpha();
    info << ", gluBias: " << tilingData_.get_gluBias();
    info << ", ubMaxPair: " << tilingData_.get_ubMaxPair();
    info << ", groupNum: " << tilingData_.get_groupNum();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus ClippedSwigluTiling::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(CheckAndGetXAndAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "check x and attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAndGetGroupIndex() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "check group_index param failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckY() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "check y param failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluTiling::CheckAndGetXAndAttrs()
{
    // 获取attr参数
    auto* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context_) == nullptr || (context_)->GetNodeName() == nullptr ?
                                                       "clipped_swiglu" :
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
    auto* attrAlpha = attrs->GetAttrPointer<float>(ALPHA_INDEX);
    gluAlpha_ = attrAlpha == nullptr ? GLU_ALPHA_DEFAULT : *attrAlpha;
    auto* attrLimit = attrs->GetAttrPointer<float>(LIMIT_INDEX);
    gluLimit_ = attrLimit == nullptr ? CLAMP_LIMIT_DEFAULT : *attrLimit;
    auto* attrBias = attrs->GetAttrPointer<float>(BIAS_INDEX);
    gluBias_ = attrBias == nullptr ? GLU_BIAS_DEFAULT : *attrBias;
    auto* attrInterleaved = attrs->GetAttrPointer<bool>(INTERLEAVED_INDEX);
    bool interleaved = attrInterleaved == nullptr ? true : *attrInterleaved;
    isInterleaved_ = interleaved ? 1 : 0;
    // 获取x shape
    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(shapeX == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "clipped_swiglu" :
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
    cutDim_ = cutDim_ < 0 ? (cutDim_ + xDims_) : cutDim_; // cutDim统一为正数
    xCutDimNum_ = inputShapeX.GetDim(cutDim_);
    // x合轴为2维
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
    OP_CHECK_IF((inputShapeX.GetDim(cutDim_) % 2 == 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(inputShapeX.GetDim(cutDim_)),
                    ("x[dim] should be divisible by 2, but get " + std::to_string(inputShapeX.GetDim(cutDim_)))),
                return ge::GRAPH_FAILED);
    auto descX = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(descX == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "clipped_swiglu" :
                                                                                     (context_)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    xDtype_ = descX->GetDataType();
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype_) == SUPPORT_DTYPE.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      std::to_string(static_cast<int>(descX->GetDataType())),
                                                      "x dtype only support float, half or bfloat16, please check."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluTiling::CheckAndGetGroupIndex()
{
    auto shapeGroupIndex = context_->GetInputShape(GROUP_INDEX_INDEX);
    if (shapeGroupIndex == nullptr) {
        isGroup_ = 0;
    } else {
        isGroup_ = 1;
        const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
        int64_t groupIndexDim = inputShapeGroupIndex.GetDimNum();
        auto descGroupIndex = context_->GetInputDesc(GROUP_INDEX_INDEX);
        if (descGroupIndex == nullptr) {
            const std::string msgOpName = std::string(
                (context_->GetNodeName() == nullptr ? "clipped_swiglu" : context_->GetNodeName()));
            OP_LOGE_WITHOUT_REPORT(
                msgOpName.c_str(), "%s",
                "group_index shape is already present; its missing descriptor violates the Tiling context contract.");
            REPORT_PREDEFINED_ERR_MSG(
                "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                (std::vector<const char*>{msgOpName.c_str(), "group_index shape is already present; its missing "
                                                             "descriptor violates the Tiling context contract."}));
            return ge::GRAPH_FAILED;
        };
        auto groupIndexDtype = descGroupIndex->GetDataType();
        OP_CHECK_IF((groupIndexDim != 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "group_index", std::to_string(groupIndexDim),
                        ("the number of dimensions of group_index should be 1, but get " +
                         std::to_string(groupIndexDim) + ".")),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (groupIndexDtype != ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                  std::to_string(static_cast<int>(descGroupIndex->GetDataType())),
                                                  "groupIndex dtype only support int64, please check."),
            return ge::GRAPH_FAILED);
        groupNum_ = inputShapeGroupIndex.GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluTiling::CheckY()
{
    auto shapeY = context_->GetOutputShape(Y_INDEX);
    if (shapeY == nullptr) {
        const std::string msgOpName = std::string(((context_) == nullptr || (context_)->GetNodeName() == nullptr ?
                                                       "clipped_swiglu" :
                                                       (context_)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object shapeY is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object shapeY is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    const gert::Shape& inputShapeY = shapeY->GetStorageShape();
    int64_t yDims = inputShapeY.GetDimNum();
    auto descY = context_->GetOutputDesc(Y_INDEX);
    if (descY == nullptr) {
        const std::string msgOpName = (context_->GetNodeName() == nullptr ? "clipped_swiglu" : context_->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Output descriptor y is unavailable; check the registered Tiling context.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "Output descriptor y is unavailable; check the registered Tiling context."}));
        return ge::GRAPH_FAILED;
    }
    auto yDtype = descY->GetDataType();
    OP_CHECK_IF((yDims != xDims_),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "x,y", std::to_string(xDims_) + "," + std::to_string(yDims),
                    ("the number of dimensions of y should be equal to dimensions of x, but get " +
                     std::to_string(yDims) + ".")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputShapeY.GetDim(cutDim_) != (xCutDimNum_ / 2)),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "y",
                                                          std::to_string(inputShapeY.GetDim(cutDim_)),
                                                          ("y[dim] should be equal to x[dim] divided by 2, but get " +
                                                           std::to_string(inputShapeY.GetDim(cutDim_)) + ".")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDtype != xDtype_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x,y",
                                                       std::to_string(static_cast<int>(xDtype_)) + "," +
                                                           std::to_string(static_cast<int>(descY->GetDataType())),
                                                       "y dtype should be the same as x, please cheack."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClippedSwigluTiling::CountMaxPair()
{
    ubMaxPair_ = 1;
    int64_t xBuffer = DB_BUFFER * SWI_FACTOR * DTYPE_FACTOR;
    int64_t yBuffer = DTYPE_FACTOR;
    int64_t groupindexBuffer = isGroup_ ?
                                   DB_BUFFER * ((groupNum_ * static_cast<int64_t>(sizeof(int64_t))) + BLOCK_SIZE - 1) /
                                       BLOCK_SIZE * BLOCK_SIZE :
                                   0;
    int64_t tmpBuffer1 = SWI_FACTOR * DTYPE_FACTOR;
    int64_t tmpBuffer2 = SWI_FACTOR * DTYPE_FACTOR;
    int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - groupindexBuffer;
    ubMaxPair_ = ((numerator / (xBuffer + yBuffer + tmpBuffer1 + tmpBuffer2) / BLOCK_SIZE * BLOCK_SIZE) -
                  (BLOCK_SIZE - 1)) /
                 SIZE_OF_BF16_FP16; // 32字节对齐
    OP_CHECK_IF(
        (numerator <= 0 || ubMaxPair_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "group_index", std::to_string(groupNum_),
            std::string("The group_index buffer leaves no usable UB tile: group_count=") + std::to_string(groupNum_) +
                ", availableUB=" + std::to_string(ubSize_) + ", remainingBytes=" + std::to_string(numerator)),
        return ge::GRAPH_FAILED);
    isLongH_ = ubMaxPair_ * SWI_FACTOR < dim2H_ ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("ClippedSwiglu", ClippedSwigluTiling, 20000);

ge::graphStatus TilingForClippedSwiglu(gert::TilingContext* context)
{
    bool regBase = Ops::NN::OpTiling::IsRegbaseSocVersion(context);
    if (regBase) {
        return Tiling4ClippedSwigluArch35(context);
    }
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForClippedSwiglu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForClippedSwiglu enter.");
    auto compileInfo = context->GetCompiledInfo<ClippedSwigluCompileInfo>();
    if (compileInfo == nullptr) {
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "clipped_swiglu" : (context)->GetNodeName()));
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
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "clipped_swiglu" : (context)->GetNodeName()));
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
                                           "clipped_swiglu" :
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
                                           "clipped_swiglu" :
                                           context->GetNodeName());
        const std::string msgReason = std::string(
            ("Get ub size failed, ub size: " + std::to_string(static_cast<uint32_t>(compileInfo->ubSize))));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };

    OP_LOGD(context, "TilingPrepareForClippedSwiglu exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClippedSwiglu)
    .Tiling(TilingForClippedSwiglu)
    .TilingParse<ClippedSwigluCompileInfo>(TilingPrepareForClippedSwiglu);
} // namespace optiling
