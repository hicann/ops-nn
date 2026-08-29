/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file bn_infer_tiling_infer_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include "bn_infer_tiling.h"
using namespace ge;

namespace {
constexpr int64_t TILINGKEY_INFER = 910000;
constexpr int64_t TILING_TEMPLATE_PRIORITY_INFER = 90000;

constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_NUM_5 = 5;
constexpr int64_t MIN_ND_DIM_NUM = 2;
constexpr int64_t WEIGHT_BIAS_NUM = 2;
constexpr int64_t MEAN_VAR_NUM = 2;
constexpr int64_t INPUT_OUTPUT_NUM = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

static const int32_t INDEX_EPSILON = 0;
constexpr float DEFAULT_EPSILON = 1e-5;

// 框架侧占位可以只预留32B（ttk正常），debugTool执行时需要预留16M
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
} // namespace

namespace optiling {
class BNInferTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BNInferTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) { Reset(); }
    ~BNInferTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        auto xDesc = context_->GetInputDesc(0);
        if (xDesc == nullptr) {
            return false;
        }
        auto format = xDesc->GetFormat().GetStorageFormat();
        if (format != FORMAT_ND && format != FORMAT_NCHW && format != FORMAT_NCDHW) {
            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", Ops::Base::ToString(format).c_str(),
                                       "ND, NCHW or NCDHW");
            return false;
        }
        aTileBase_ = vlFp16_;
        bytesPerElement_ = FLOAT16_BYTES;
        if (dataType_ == ge::DT_FLOAT) {
            aTileBase_ = vlFp32_;
            bytesPerElement_ = FLOAT32_BYTES;
        }
        bytesPerWeightElement_ = weightDataType_ == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;

        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 7、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    void Reset();

private:
    ge::graphStatus ParseNdShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseNchwShape(const gert::Shape& xStorageShape);
    ge::graphStatus ParseNcdhwShape(const gert::Shape& xStorageShape);
    ge::graphStatus ValidateTilingParams() const;
    ge::graphStatus CalcGeneralFactors(struct InferTilingFactors& factors);
    int64_t GetGeneralAInnerAlignedUbSize(const InferTilingFactors& factors, int64_t aInnerCandidate) const;
    ge::graphStatus FillTilingData(int64_t b0Outer, int64_t aOuter, int64_t b1Outer, int64_t b0Inner, int64_t aInner,
                                   int64_t b1Inner);

    const char* opName = "BNInfer";

    int64_t usedCoreNums_;
    uint64_t blockSize_;
    uint64_t vlFp32_;
    uint64_t vlFp16_;
    int64_t bytesPerElement_;
    int64_t bytesPerWeightElement_;
    int64_t fusedB0Len_;
    int64_t fusedALen_;
    int64_t fusedB1Len_;
    int64_t aTileBase_;
    float epsilon_;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType weightDataType_ = ge::DT_UNDEFINED;
    BNInferTilingData tilingData_;
};

void BNInferTiling::Reset()
{
    usedCoreNums_ = 0;
    blockSize_ = 0;
    vlFp32_ = 0;
    vlFp16_ = 0;
    bytesPerElement_ = 0;
    bytesPerWeightElement_ = 0;
    fusedB0Len_ = 0;
    fusedALen_ = 0;
    fusedB1Len_ = 0;
    aTileBase_ = 0;
    epsilon_ = 0;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    }

    return (value + factor - 1) / factor;
}

inline static int64_t AlignUp(int64_t value, int64_t align) { return CeilDiv(value, align) * align; }

inline static bool IsMulOverflow(int64_t lhs, int64_t rhs) { return lhs > 0 && rhs > 0 && lhs > INT64_MAX / rhs; }

inline static bool TryMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || IsMulOverflow(lhs, rhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline static bool TryAdd(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || lhs > INT64_MAX - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

inline static bool TryAlignUp(int64_t value, int64_t align, int64_t& result)
{
    if (value < 0 || align <= 0) {
        return false;
    }
    int64_t quotient = value / align;
    if (value % align != 0) {
        if (quotient == INT64_MAX || !TryAdd(quotient, 1, quotient)) {
            return false;
        }
    }
    return TryMul(quotient, align, result);
}

struct InferTilingFactors {
    int64_t b0Outer = 1;
    int64_t aOuter = 1;
    int64_t b1Outer = 1;
    int64_t b0Inner = 1;
    int64_t aInner = 1;
    int64_t b1Inner = 1;
};

ge::graphStatus BNInferTiling::ParseNdShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() < MIN_ND_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "at least 2D with ND format"),
                return ge::GRAPH_FAILED);
    fusedB0Len_ = xStorageShape.GetDim(DIM_0);
    fusedALen_ = xStorageShape.GetDim(DIM_1);
    fusedB1Len_ = 1;
    for (size_t i = static_cast<size_t>(DIM_2); i < xStorageShape.GetDimNum(); ++i) {
        int64_t nextB1Len = 0;
        OP_CHECK_IF(!TryMul(fusedB1Len_, xStorageShape.GetDim(i), nextB1Len),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x shape", "product overflows int64",
                                                           "x shape product must fit in int64"),
                    return ge::GRAPH_PARAM_INVALID);
        fusedB1Len_ = nextB1Len;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::ParseNchwShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != DIM_NUM_4,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "4D with NCHW format"),
                return ge::GRAPH_FAILED);
    fusedB0Len_ = xStorageShape.GetDim(DIM_0);
    fusedALen_ = xStorageShape.GetDim(DIM_1);
    OP_CHECK_IF(!TryMul(xStorageShape.GetDim(DIM_2), xStorageShape.GetDim(DIM_3), fusedB1Len_),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x shape", "product overflows int64",
                                                       "x shape product must fit in int64"),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::ParseNcdhwShape(const gert::Shape& xStorageShape)
{
    OP_CHECK_IF(xStorageShape.GetDimNum() != DIM_NUM_5,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x", std::to_string(xStorageShape.GetDimNum()).c_str(),
                                             "5D with NCDHW format"),
                return ge::GRAPH_FAILED);
    fusedB0Len_ = xStorageShape.GetDim(DIM_0);
    fusedALen_ = xStorageShape.GetDim(DIM_1);
    int64_t spatialLen = 0;
    OP_CHECK_IF(!TryMul(xStorageShape.GetDim(DIM_2), xStorageShape.GetDim(DIM_3), spatialLen) ||
                    !TryMul(spatialLen, xStorageShape.GetDim(DIM_4), fusedB1Len_),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x shape", "product overflows int64",
                                                       "x shape product must fit in int64"),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const BNInferCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    opName = context_->GetNodeName();
    blockSize_ = static_cast<uint64_t>(compileInfo->blockSize);
    vlFp32_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    vlFp16_ = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT16_BYTES;

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    } else {
        aicoreParams_.numBlocks = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("BNInfer", "context", "nullptr", "TilingContext must not be nullptr");
        return ge::GRAPH_FAILED;
    }

    // 获取输入shape
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    const gert::Shape& xStorageShape = xShape->GetStorageShape();
    auto xDesc = context_->GetInputDesc(0);
    auto weightDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    dataType_ = xDesc->GetDataType();
    weightDataType_ = weightDesc->GetDataType();
    auto format = xDesc->GetFormat().GetStorageFormat();
    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(INDEX_EPSILON);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    if (format == FORMAT_ND) {
        OP_CHECK_IF(ParseNdShape(xStorageShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x", "parse nd shape failed",
                                                              "ND shape parse failed"),
                    return ge::GRAPH_FAILED);
    } else if (format == FORMAT_NCHW) {
        OP_CHECK_IF(ParseNchwShape(xStorageShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x", "parse nchw shape failed",
                                                              "NCHW shape parse failed"),
                    return ge::GRAPH_FAILED);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(ParseNcdhwShape(xStorageShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "x", "parse ncdhw shape failed",
                                                              "NCDHW shape parse failed"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGI(context_->GetNodeName(), "Only supported infer ND, NCHW or NCDHW.");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(fusedB0Len_ <= 0 || fusedALen_ <= 0 || fusedB1Len_ <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "fusedB0Len, fusedALen, fusedB1Len",
                                                       (std::to_string(fusedB0Len_) + ", " +
                                                        std::to_string(fusedALen_) + ", " + std::to_string(fusedB1Len_))
                                                           .c_str(),
                                                       "BNInfer does not support empty tensor on Ascend950"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::FillTilingData(int64_t b0Outer, int64_t aOuter, int64_t b1Outer, int64_t b0Inner,
                                              int64_t aInner, int64_t b1Inner)
{
    int64_t b0aTiles = 0;
    int64_t totalTiles = 0;
    OP_CHECK_IF(!TryMul(b0Outer, aOuter, b0aTiles) || !TryMul(b0aTiles, b1Outer, totalTiles),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "b0Outer, aOuter, b1Outer",
                    (std::to_string(b0Outer) + ", " + std::to_string(aOuter) + ", " + std::to_string(b1Outer)).c_str(),
                    "totalTiles exceeds int64 range"),
                return ge::GRAPH_FAILED);
    int64_t tilesPerCore = CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t tileBlockB0Tail = fusedB0Len_ - b0Inner * (b0Outer - 1);
    int64_t tileBlockATail = fusedALen_ - aInner * (aOuter - 1);
    int64_t tileBlockB1Len = b1Inner * aTileBase_;
    int64_t tileBlockB1Tail = fusedB1Len_ - tileBlockB1Len * (b1Outer - 1);

    tilingData_.totalTiles = totalTiles;
    tilingData_.tilesPerCore = tilesPerCore;
    tilingData_.totalALen = fusedALen_;
    tilingData_.totalB1Len = fusedB1Len_;
    tilingData_.b0Outer = b0Outer;
    tilingData_.aOuter = aOuter;
    tilingData_.b1Outer = b1Outer;
    tilingData_.tileBlockB0Len = b0Inner;
    tilingData_.tileBlockB0Tail = tileBlockB0Tail;
    tilingData_.tileBlockALen = aInner;
    tilingData_.tileBlockATail = tileBlockATail;
    tilingData_.tileBlockB1Len = tileBlockB1Len;
    tilingData_.tileBlockB1Tail = tileBlockB1Tail;
    tilingData_.tileBlockAPaddingNum = 0;
    tilingData_.epsilon = epsilon_;
    usedCoreNums_ = CeilDiv(totalTiles, tilesPerCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::ValidateTilingParams() const
{
    OP_CHECK_IF(aTileBase_ <= 0 || aicoreParams_.ubSize <= 0 || aicoreParams_.numBlocks <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "aTileBase, ubSize, numBlocks",
                    (std::to_string(aTileBase_) + ", " + std::to_string(aicoreParams_.ubSize) + ", " +
                     std::to_string(aicoreParams_.numBlocks))
                        .c_str(),
                    "tiling platform parameters must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::CalcGeneralFactors(InferTilingFactors& factors)
{
    // 默认策略: 先按照B0, B1把UB切满，再切A。
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (MEAN_VAR_NUM * FLOAT32_BYTES + WEIGHT_BIAS_NUM * bytesPerWeightElement_) * factors.aInner *
                                aTileBase_) /
                           bytesPerElement_ / INPUT_OUTPUT_NUM;
    int64_t factorMax = ubBufferSize / aTileBase_;
    int64_t b1FactorMax = CeilDiv(fusedB1Len_, aTileBase_);
    factors.b1Inner = std::min(factorMax, b1FactorMax);
    OP_CHECK_IF(factors.b1Inner <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "b1Inner",
                                                       std::to_string(factors.b1Inner).c_str(),
                                                       "b1Inner must be greater than 0"),
                return ge::GRAPH_FAILED);
    factors.b1Outer = CeilDiv(fusedB1Len_, factors.b1Inner * aTileBase_);

    factorMax = factorMax / factors.b1Inner;
    factors.b0Inner = std::min(factorMax, fusedB0Len_);
    OP_CHECK_IF(factors.b0Inner <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "b0Inner",
                                                       std::to_string(factors.b0Inner).c_str(),
                                                       "b0Inner must be greater than 0"),
                return ge::GRAPH_FAILED);
    factors.b0Outer = CeilDiv(fusedB0Len_, factors.b0Inner);

    factorMax = factorMax / factors.b0Inner;
    factors.aInner = std::min(factorMax, fusedALen_);
    int64_t maxAInnerByUb = aicoreParams_.ubSize / DOUBLE_BUFFER /
                            (INPUT_OUTPUT_NUM * factors.b0Inner * factors.b1Inner * aTileBase_ * bytesPerElement_ +
                             WEIGHT_BIAS_NUM * bytesPerWeightElement_ + MEAN_VAR_NUM * FLOAT32_BYTES);
    factors.aInner = std::min(factors.aInner, maxAInnerByUb);
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
    while (factors.aInner > 0 && GetGeneralAInnerAlignedUbSize(factors, factors.aInner) > ubSize) {
        --factors.aInner;
    }
    OP_CHECK_IF(factors.aInner <= 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(),
                    "aInner, b0Inner, b1Inner, aTileBase, bytesPerElement, bytesPerWeightElement, ubSize",
                    (std::to_string(factors.aInner) + ", " + std::to_string(factors.b0Inner) + ", " +
                     std::to_string(factors.b1Inner) + ", " + std::to_string(aTileBase_) + ", " +
                     std::to_string(bytesPerElement_) + ", " + std::to_string(bytesPerWeightElement_) + ", " +
                     std::to_string(aicoreParams_.ubSize))
                        .c_str(),
                    "aInner must be greater than 0 after UB tiling"),
                return ge::GRAPH_FAILED);
    factors.aOuter = CeilDiv(fusedALen_, factors.aInner);
    return ge::GRAPH_SUCCESS;
}

int64_t BNInferTiling::GetGeneralAInnerAlignedUbSize(const InferTilingFactors& factors, int64_t aInnerCandidate) const
{
    int64_t tileBlockB1Len = 0;
    int64_t xyBufferSize = 0;
    int64_t weightBufferSize = 0;
    int64_t meanVarBufferSize = 0;
    int64_t alignedXyBufferSize = 0;
    int64_t alignedWeightBufferSize = 0;
    int64_t alignedMeanVarBufferSize = 0;
    int64_t totalBufferSize = 0;
    int64_t weightedBufferSize = 0;
    int64_t meanVarBufferSizeWithWeight = 0;
    if (!TryMul(factors.b1Inner, aTileBase_, tileBlockB1Len) ||
        !TryMul(factors.b0Inner, aInnerCandidate, xyBufferSize) ||
        !TryMul(xyBufferSize, tileBlockB1Len, xyBufferSize) || !TryMul(xyBufferSize, bytesPerElement_, xyBufferSize) ||
        !TryMul(aInnerCandidate, bytesPerWeightElement_, weightBufferSize) ||
        !TryMul(aInnerCandidate, FLOAT32_BYTES, meanVarBufferSize) ||
        !TryAlignUp(xyBufferSize, blockSize_, alignedXyBufferSize) ||
        !TryAlignUp(weightBufferSize, blockSize_, alignedWeightBufferSize) ||
        !TryAlignUp(meanVarBufferSize, blockSize_, alignedMeanVarBufferSize) ||
        !TryMul(INPUT_OUTPUT_NUM, alignedXyBufferSize, totalBufferSize) ||
        !TryMul(WEIGHT_BIAS_NUM, alignedWeightBufferSize, weightedBufferSize) ||
        !TryAdd(totalBufferSize, weightedBufferSize, totalBufferSize) ||
        !TryMul(MEAN_VAR_NUM, alignedMeanVarBufferSize, meanVarBufferSizeWithWeight) ||
        !TryAdd(totalBufferSize, meanVarBufferSizeWithWeight, totalBufferSize) ||
        !TryMul(DOUBLE_BUFFER, totalBufferSize, totalBufferSize)) {
        return INT64_MAX;
    }
    return totalBufferSize;
}

ge::graphStatus BNInferTiling::DoOpTiling()
{
    OP_CHECK_IF(ValidateTilingParams() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "invalid",
                                                      "validate tiling params failed"),
                return ge::GRAPH_FAILED);
    InferTilingFactors factors;
    ge::graphStatus status = CalcGeneralFactors(factors);
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "invalid",
                                                      "calc tiling factors failed"),
                return status);

    OP_CHECK_IF(
        FillTilingData(factors.b0Outer, factors.aOuter, factors.b1Outer, factors.b0Inner, factors.aInner,
                       factors.b1Inner) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "invalid", "fill tiling data failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t BNInferTiling::GetTilingKey() const { return TILINGKEY_INFER; }

ge::graphStatus BNInferTiling::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingData = context_->GetTilingData<BNInferTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    *tilingData = tilingData_;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BNInfer", BNInferTiling, TILING_TEMPLATE_PRIORITY_INFER);
} // namespace optiling
