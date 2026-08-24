/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file bn_infer_tiling_infer_last_channel_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include "bn_infer_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL = 900000;
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL_SMALL_A = 902000;
constexpr int64_t TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A = 901000;
constexpr int64_t TILING_TEMPLATE_PRIORITY_LAST_CHANNEL = 91000;
constexpr int64_t MAX_SMALL_A = 8;
constexpr int64_t MIN_CONTINUOUS_A_LEN = 64;
constexpr int64_t MAX_CONTINUOUS_A_OUTER = 6;
constexpr int64_t MAX_CONTINUOUS_A_OUTER_FP16 = 3;
constexpr int64_t MIN_SMALL_A_B_LEN = 65536;

constexpr int64_t NHWC_DIM_NUM = 4;
constexpr int64_t NDHWC_DIM_NUM = 5;
constexpr int64_t WEIGHT_BIAS_NUM = 2;
constexpr int64_t INPUT_OUTPUT_NUM = 2;
constexpr int64_t MEAN_VAR_NUM = 2;

constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UINT32_BYTES = 4;
constexpr int64_t SMALL_LAST_CHANNEL_CACHE_BUFFER_NUM = 4;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

constexpr float DEFAULT_EPSILON = 1e-5;
static const int32_t INDEX_EPSILON = 0;

// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
namespace {
inline bool IsMulOverflow(int64_t lhs, int64_t rhs) { return lhs > 0 && rhs > 0 && lhs > INT64_MAX / rhs; }

inline bool TryMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || IsMulOverflow(lhs, rhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}
} // namespace

class BNInferLastChannelTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BNInferLastChannelTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~BNInferLastChannelTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        auto xDesc = context_->GetInputDesc(0);
        auto xShape = context_->GetInputShape(0);
        if (xDesc == nullptr || xShape == nullptr) {
            return false;
        }
        auto format = xDesc->GetFormat().GetStorageFormat();
        const gert::Shape& xStorageShape = xShape->GetStorageShape();
        if (format != FORMAT_NHWC && format != FORMAT_NDHWC) {
            bool isNchwPointwise = format == FORMAT_NCHW && xStorageShape.GetDimNum() == NHWC_DIM_NUM &&
                                   xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1;
            bool isNcdhwPointwise = format == FORMAT_NCDHW && xStorageShape.GetDimNum() == NDHWC_DIM_NUM &&
                                    xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1 &&
                                    xStorageShape.GetDim(DIM_4) == 1;
            if (!isNchwPointwise && !isNcdhwPointwise) {
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(format).c_str(),
                                           "NHWC, NDHWC, pointwise NCHW or pointwise NCDHW");
                return false;
            }
        }
        aTileBase = vlFp16;
        bytesPerElement = FLOAT16_BYTES;
        if (dataType == ge::DT_FLOAT) {
            aTileBase = vlFp32;
            bytesPerElement = FLOAT32_BYTES;
        }
        bytesPerWeightElement = weightDataType == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;

        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 5、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();

private:
    ge::graphStatus FillLastChannelTilingForBSplit(int64_t paramBytes, int64_t cacheBytes);
    ge::graphStatus ParseNhwcShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseNdhwcShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseNchwAsLastChannelShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseNcdhwAsLastChannelShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseSupportedShape(const gert::Shape& xStorageShape, ge::Format format);
    ge::graphStatus ValidateTilingParams() const;
    ge::graphStatus DoSmallLastChannelTiling();
    ge::graphStatus DoContinuousLastChannelTiling();
    ge::graphStatus DoGeneralLastChannelTiling();
    ge::graphStatus FillGeneralLastChannelTiling(int64_t bInner, int64_t bOuter, int64_t tileBlockBTail);
    bool IsLastChannelUnaligned() const;
    int64_t GetBaseTilingAOuter() const;
    ge::graphStatus FillGeneralTilingData(int64_t aOuter, int64_t bOuter, int64_t bInner, int64_t tileBlockALen,
                                          int64_t tileBlockATail, int64_t tileBlockAPaddingNum, int64_t tileBlockBTail);

    const char* opName = "BNInferLastChannel";

    int64_t usedCoreNums;

    uint64_t blockSize;
    uint64_t vlFp32;
    uint64_t vlFp16;
    int64_t bytesPerElement;
    int64_t bytesPerWeightElement;

    int64_t fusedALen;
    int64_t fusedBLen;
    int64_t aTileBase;
    bool isSmallLastChannel;
    bool isContinuousLastChannel;
    bool isUnalignedLastChannel;
    float epsilon;

    ge::DataType dataType = ge::DT_UNDEFINED;
    ge::DataType weightDataType = ge::DT_UNDEFINED;
    BNInferLastChannelTilingData tilingData;
};

void BNInferLastChannelTiling::Reset()
{
    usedCoreNums = 0;
    blockSize = 0;
    vlFp32 = 0;
    vlFp16 = 0;
    bytesPerElement = 0;
    bytesPerWeightElement = 0;

    fusedALen = 0;
    fusedBLen = 0;
    aTileBase = 0;
    isSmallLastChannel = false;
    isContinuousLastChannel = false;
    isUnalignedLastChannel = false;
    epsilon = 0;
}

ge::graphStatus BNInferLastChannelTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BNInferCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    opName = context_->GetNodeName();
    blockSize = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32 = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    vlFp16 = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT16_BYTES;

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    } else {
        aicoreParams_.ubSize = compileInfo->ubSize;
        aicoreParams_.numBlocks = compileInfo->coreNum;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ParseNhwcShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != NHWC_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "4D with NHWC format"),
                return ge::GRAPH_FAILED);
    fusedALen = xStorageShape.GetDim(DIM_3);
    int64_t outerLen = 0;
    OP_CHECK_IF(!TryMul(xStorageShape.GetDim(DIM_0), xStorageShape.GetDim(DIM_1), outerLen) ||
                    !TryMul(outerLen, xStorageShape.GetDim(DIM_2), fusedBLen),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x shape", "product overflows int64",
                                                       "x shape product must fit in int64"),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ParseNdhwcShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != NDHWC_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "5D with NDHWC format"),
                return ge::GRAPH_FAILED);
    fusedALen = xStorageShape.GetDim(DIM_4);
    int64_t outerLen = 0;
    OP_CHECK_IF(!TryMul(xStorageShape.GetDim(DIM_0), xStorageShape.GetDim(DIM_1), outerLen) ||
                    !TryMul(outerLen, xStorageShape.GetDim(DIM_2), outerLen) ||
                    !TryMul(outerLen, xStorageShape.GetDim(DIM_3), fusedBLen),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x shape", "product overflows int64",
                                                       "x shape product must fit in int64"),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ParseNchwAsLastChannelShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != NHWC_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "4D with NCHW format"),
                return ge::GRAPH_FAILED);
    bool hwIsOne = xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1;
    if (!hwIsOne) {
        return ge::GRAPH_PARAM_INVALID;
    }
    fusedALen = xStorageShape.GetDim(DIM_1);
    fusedBLen = xStorageShape.GetDim(DIM_0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ParseNcdhwAsLastChannelShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != NDHWC_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "5D with NCDHW format"),
                return ge::GRAPH_FAILED);
    bool dhwIsOne = xStorageShape.GetDim(DIM_2) == 1 && xStorageShape.GetDim(DIM_3) == 1 &&
                    xStorageShape.GetDim(DIM_4) == 1;
    if (!dhwIsOne) {
        return ge::GRAPH_PARAM_INVALID;
    }
    fusedALen = xStorageShape.GetDim(DIM_1);
    fusedBLen = xStorageShape.GetDim(DIM_0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ParseSupportedShape(const gert::Shape& xStorageShape, ge::Format format)
{
    if (format == FORMAT_NHWC) {
        auto ret = ParseNhwcShape(xStorageShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS && ret != ge::GRAPH_PARAM_INVALID,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x", "parse nhwc shape failed",
                                                              "NHWC shape parse failed"),
                    return ge::GRAPH_FAILED);
        if (ret == ge::GRAPH_PARAM_INVALID) {
            return ret;
        }
    } else if (format == FORMAT_NDHWC) {
        auto ret = ParseNdhwcShape(xStorageShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS && ret != ge::GRAPH_PARAM_INVALID,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x", "parse ndhwc shape failed",
                                                              "NDHWC shape parse failed"),
                    return ge::GRAPH_FAILED);
        if (ret == ge::GRAPH_PARAM_INVALID) {
            return ret;
        }
    } else if (format == FORMAT_NCHW) {
        auto ret = ParseNchwAsLastChannelShape(xStorageShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS && ret != ge::GRAPH_PARAM_INVALID,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x",
                                                              "parse nchw-as-last-channel shape failed",
                                                              "NCHW as last-channel shape parse failed"),
                    return ge::GRAPH_FAILED);
        if (ret == ge::GRAPH_PARAM_INVALID) {
            return ret;
        }
    } else if (format == FORMAT_NCDHW) {
        auto ret = ParseNcdhwAsLastChannelShape(xStorageShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS && ret != ge::GRAPH_PARAM_INVALID,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x",
                                                              "parse ncdhw-as-last-channel shape failed",
                                                              "NCDHW as last-channel shape parse failed"),
                    return ge::GRAPH_FAILED);
        if (ret == ge::GRAPH_PARAM_INVALID) {
            return ret;
        }
    } else {
        OP_LOGI(context_, "Only supported format NHWC or NDHWC.");
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("BNInferLastChannel", "context", "nullptr",
                                              "TilingContext must not be nullptr");
        return ge::GRAPH_FAILED;
    }

    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xDesc = context_->GetInputDesc(0);
    auto weightDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    dataType = xDesc->GetDataType();
    weightDataType = weightDesc->GetDataType();
    auto format = xDesc->GetFormat().GetStorageFormat();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(INDEX_EPSILON);
    epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    auto parseStatus = ParseSupportedShape(xShape->GetStorageShape(), format);
    OP_CHECK_IF(parseStatus != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "x", "parse last-channel shape failed", "last-channel shape parse failed"),
                return parseStatus);

    OP_CHECK_IF(
        fusedALen <= 0 || fusedBLen <= 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "fusedALen, fusedBLen",
                                               (std::to_string(fusedALen) + ", " + std::to_string(fusedBLen)).c_str(),
                                               "BNInfer does not support empty tensor on Ascend950"),
        return ge::GRAPH_FAILED);

    isSmallLastChannel = dataType != ge::DT_BF16 && fusedALen <= MAX_SMALL_A && fusedBLen > MIN_SMALL_A_B_LEN;
    return ge::GRAPH_SUCCESS;
}

bool BNInferLastChannelTiling::IsLastChannelUnaligned() const
{
    if (blockSize == 0 || bytesPerElement <= 0) {
        return false;
    }
    const int64_t elementsPerBlock = static_cast<int64_t>(blockSize) / bytesPerElement;
    return elementsPerBlock <= 0 || fusedALen % elementsPerBlock != 0;
}

ge::graphStatus BNInferLastChannelTiling::FillGeneralTilingData(int64_t aOuter, int64_t bOuter, int64_t bInner,
                                                                int64_t tileBlockALen, int64_t tileBlockATail,
                                                                int64_t tileBlockAPaddingNum, int64_t tileBlockBTail)
{
    int64_t totalTiles = 0;
    if (!TryMul(aOuter, bOuter, totalTiles)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "aOuter, bOuter", "product overflows int64",
                                               "total tile count must fit in int64");
        return ge::GRAPH_FAILED;
    }
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData.totalTiles = totalTiles;
    tilingData.tilesPerCore = tilesPerCore;
    tilingData.totalALen = fusedALen;
    tilingData.aOuter = aOuter;
    tilingData.bOuter = bOuter;
    tilingData.tileBlockALen = tileBlockALen;
    tilingData.tileBlockATail = tileBlockATail;
    tilingData.tileBlockAPaddingNum = tileBlockAPaddingNum;
    tilingData.tileBlockBLen = bInner;
    tilingData.tileBlockBTail = tileBlockBTail;
    tilingData.epsilon = epsilon;
    return ge::GRAPH_SUCCESS;
}

int64_t BNInferLastChannelTiling::GetBaseTilingAOuter() const
{
    if (aTileBase <= 0 || bytesPerElement <= 0 || aicoreParams_.ubSize <= 0) {
        return 0;
    }
    int64_t paramBytes = (MEAN_VAR_NUM * FLOAT32_BYTES + WEIGHT_BIAS_NUM * bytesPerWeightElement) * aTileBase;
    int64_t ubBufferSize = (static_cast<int64_t>(aicoreParams_.ubSize) / DOUBLE_BUFFER - paramBytes) / bytesPerElement /
                           INPUT_OUTPUT_NUM;
    int64_t bFactorMax = ubBufferSize / aTileBase;
    if (bFactorMax <= 0) {
        return 0;
    }
    int64_t bInner = fusedBLen <= bFactorMax ? fusedBLen : bFactorMax;
    int64_t elemBytes = bInner * INPUT_OUTPUT_NUM * bytesPerElement + WEIGHT_BIAS_NUM * bytesPerWeightElement +
                        MEAN_VAR_NUM * FLOAT32_BYTES;
    if (elemBytes <= 0) {
        return 0;
    }
    int64_t aFactorMax = static_cast<int64_t>(aicoreParams_.ubSize) / DOUBLE_BUFFER / aTileBase / elemBytes;
    int64_t aInnerMax = fusedALen / aTileBase;
    int64_t aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;
    int64_t tileBlockALen = aInner == 0 ? aTileBase : aInner * aTileBase;
    return Ops::Base::CeilDiv(fusedALen, tileBlockALen);
}

ge::graphStatus BNInferLastChannelTiling::FillLastChannelTilingForBSplit(int64_t paramBytes, int64_t cacheBytes)
{
    int64_t perElemBytes = INPUT_OUTPUT_NUM * DOUBLE_BUFFER * bytesPerElement;
    OP_CHECK_IF(perElemBytes <= 0 || fusedALen <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "perElemBytes, fusedALen",
                    (std::to_string(perElemBytes) + ", " + std::to_string(fusedALen)).c_str(),
                    "perElemBytes and fusedALen must be greater than 0"),
                return ge::GRAPH_FAILED);
    int64_t elemFactorMax = (static_cast<int64_t>(aicoreParams_.ubSize) - paramBytes - cacheBytes) / perElemBytes;
    int64_t bInner = elemFactorMax / fusedALen;
    bInner = bInner <= 0 ? 1 : bInner;
    bInner = fusedBLen <= bInner ? fusedBLen : bInner;
    while ((paramBytes + cacheBytes + bInner * fusedALen * INPUT_OUTPUT_NUM * DOUBLE_BUFFER * bytesPerElement >
            static_cast<int64_t>(aicoreParams_.ubSize)) &&
           bInner > 1) {
        bInner--;
    }
    int64_t bOuter = Ops::Base::CeilDiv(fusedBLen, bInner);
    int64_t bTail = fusedBLen % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;
    int64_t totalTiles = bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData.totalTiles = totalTiles;
    tilingData.tilesPerCore = tilesPerCore;
    tilingData.totalALen = fusedALen;
    tilingData.aOuter = 1;
    tilingData.bOuter = bOuter;
    tilingData.tileBlockALen = fusedALen;
    tilingData.tileBlockATail = fusedALen;
    tilingData.tileBlockAPaddingNum = 0;
    tilingData.tileBlockBLen = bInner;
    tilingData.tileBlockBTail = tileBlockBTail;
    tilingData.epsilon = epsilon;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::ValidateTilingParams() const
{
    OP_CHECK_IF(aTileBase <= 0 || aicoreParams_.ubSize <= 0 || aicoreParams_.numBlocks <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "aTileBase, ubSize, numBlocks",
                    (std::to_string(aTileBase) + ", " + std::to_string(aicoreParams_.ubSize) + ", " +
                     std::to_string(aicoreParams_.numBlocks))
                        .c_str(),
                    "tiling platform parameters must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::DoSmallLastChannelTiling()
{
    auto alignUp = [](int64_t value, int64_t base) { return (value + base - 1) / base * base; };
    int64_t paramBytes = DOUBLE_BUFFER * (MEAN_VAR_NUM * sizeof(float) + WEIGHT_BIAS_NUM * bytesPerWeightElement) *
                         fusedALen;
    int64_t paramCacheElemLen = (static_cast<int64_t>(vlFp32) / fusedALen) * fusedALen;
    int64_t offsetBytes = alignUp(paramCacheElemLen * UINT32_BYTES, static_cast<int64_t>(blockSize));
    int64_t cacheBytes = offsetBytes + SMALL_LAST_CHANNEL_CACHE_BUFFER_NUM *
                                           alignUp(paramCacheElemLen * FLOAT32_BYTES, static_cast<int64_t>(blockSize));
    return FillLastChannelTilingForBSplit(paramBytes, cacheBytes);
}

ge::graphStatus BNInferLastChannelTiling::DoContinuousLastChannelTiling()
{
    auto alignUp = [](int64_t value, int64_t base) { return (value + base - 1) / base * base; };
    int64_t paramAlignLen = alignUp(fusedALen, static_cast<int64_t>(vlFp32));
    int64_t paramBytes = DOUBLE_BUFFER * (MEAN_VAR_NUM * sizeof(float) + WEIGHT_BIAS_NUM * bytesPerWeightElement) *
                         paramAlignLen;
    int64_t paramCacheBytes = (MEAN_VAR_NUM + WEIGHT_BIAS_NUM) * FLOAT32_BYTES * paramAlignLen;
    return FillLastChannelTilingForBSplit(paramBytes, paramCacheBytes);
}

ge::graphStatus BNInferLastChannelTiling::FillGeneralLastChannelTiling(int64_t bInner, int64_t bOuter,
                                                                       int64_t tileBlockBTail)
{
    int64_t aFactorMax = aicoreParams_.ubSize / DOUBLE_BUFFER / aTileBase /
                         (bInner * INPUT_OUTPUT_NUM * bytesPerElement + WEIGHT_BIAS_NUM * bytesPerWeightElement +
                          MEAN_VAR_NUM * FLOAT32_BYTES);
    int64_t aInnerMax = fusedALen / aTileBase;
    int64_t aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;
    int64_t tileBlockALen = aInner == 0 ? aTileBase : aInner * aTileBase;
    OP_CHECK_IF(tileBlockALen <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "tileBlockALen",
                                                       std::to_string(tileBlockALen).c_str(),
                                                       "tileBlockALen must be greater than 0"),
                return ge::GRAPH_FAILED);
    int64_t aOuter = Ops::Base::CeilDiv(fusedALen, tileBlockALen);
    int64_t aTail = fusedALen % tileBlockALen;
    int64_t tileBlockATail = aTail == 0 ? tileBlockALen : aTail;
    int64_t tileBlockAPaddingNum = tileBlockALen - tileBlockATail;

    int64_t totalTiles = 0;
    OP_CHECK_IF(!TryMul(aOuter, bOuter, totalTiles),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "aOuter, bOuter",
                                                       (std::to_string(aOuter) + ", " + std::to_string(bOuter)).c_str(),
                                                       "totalTiles exceeds int64 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillGeneralTilingData(aOuter, bOuter, bInner, tileBlockALen, tileBlockATail, tileBlockAPaddingNum,
                                      tileBlockBTail) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "invalid",
                                                      "fill general tiling data failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::DoGeneralLastChannelTiling()
{
    int64_t baseAOuter = GetBaseTilingAOuter();
    int64_t maxContinuousAOuter = dataType == ge::DT_FLOAT ? MAX_CONTINUOUS_A_OUTER : MAX_CONTINUOUS_A_OUTER_FP16;
    bool canUseContinuousA = dataType == ge::DT_FLOAT ? fusedALen > MIN_CONTINUOUS_A_LEN :
                                                        baseAOuter > 1 && baseAOuter <= maxContinuousAOuter;
    bool meetsContinuousShape = dataType == ge::DT_FLOAT ?
                                    fusedALen > MIN_CONTINUOUS_A_LEN :
                                    fusedALen > MIN_CONTINUOUS_A_LEN && fusedBLen > MIN_SMALL_A_B_LEN;
    isContinuousLastChannel = isUnalignedLastChannel || (meetsContinuousShape && canUseContinuousA);
    if (isContinuousLastChannel) {
        return DoContinuousLastChannelTiling();
    }

    // 切分A、B基本块， （B,A） -- >(Bouter, Aouter, Binner*Ainner*ATileBase)
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (MEAN_VAR_NUM * FLOAT32_BYTES + WEIGHT_BIAS_NUM * bytesPerWeightElement) * aInner *
                                aTileBase) /
                           bytesPerElement / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    int64_t bFactorMax = ubBufferSize / aTileBase;
    int64_t bInner = fusedBLen <= bFactorMax ? fusedBLen : bFactorMax;
    OP_CHECK_IF(bInner <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "bInner",
                                                       std::to_string(bInner).c_str(), "bInner must be greater than 0"),
                return ge::GRAPH_FAILED);
    int64_t bOuter = Ops::Base::CeilDiv(fusedBLen, bInner);
    int64_t bTail = fusedBLen % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;
    return FillGeneralLastChannelTiling(bInner, bOuter, tileBlockBTail);
}

ge::graphStatus BNInferLastChannelTiling::DoOpTiling()
{
    OP_CHECK_IF(ValidateTilingParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "invalid",
                                                      "validate tiling params failed"),
                return ge::GRAPH_FAILED);
    // Refresh after platform information is available; GetShapeAttrsInfo may run earlier.
    isUnalignedLastChannel = IsLastChannelUnaligned();
    if (isSmallLastChannel) {
        return DoSmallLastChannelTiling();
    }
    return DoGeneralLastChannelTiling();
}

ge::graphStatus BNInferLastChannelTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t BNInferLastChannelTiling::GetTilingKey() const
{
    if (isSmallLastChannel) {
        return TILINGKEY_INFER_LAST_CHANNEL_SMALL_A;
    }
    if (isContinuousLastChannel) {
        return TILINGKEY_INFER_LAST_CHANNEL_CONTINUOUS_A;
    }
    return TILINGKEY_INFER_LAST_CHANNEL;
}

ge::graphStatus BNInferLastChannelTiling::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferLastChannelTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BNInferLastChannelTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = tilingData;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BNInfer", BNInferLastChannelTiling, TILING_TEMPLATE_PRIORITY_LAST_CHANNEL);
} // namespace optiling
