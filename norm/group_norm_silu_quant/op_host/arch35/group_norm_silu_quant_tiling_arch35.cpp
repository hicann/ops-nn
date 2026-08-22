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
 * \file group_norm_silu_quant_tiling_arch35.cpp
 * \brief GroupNormSiluQuant arch35(Ascend950) regbase tiling 实现: dtype/shape 校验、核间(block)与 UB 切分、
 *        shapeQuantScale 推导与 quantScale UB 预留、TilingKey 选择(r_partial_load/r_full_load/empty)。
 */
#include "group_norm_silu_quant_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
namespace optiling {
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t X_SHAPE_MAX_LEN = 8;
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t INPUT_IDX_QUANT_SCALE = 3;
static const uint64_t INDEX_NUM_GROUPS = 0;
static const uint64_t INDEX_EPSILON = 1;
static const uint64_t INDEX_ACTIVATE_SILU = 2;
static const uint64_t OUTPUT_IDX_Y = 0;
static const uint64_t EMPTY_SHAPE_SIZE = 0;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t BUFFER_NUM = 2;
static const uint64_t DOUBLE_BUFFER = 2;
static const uint64_t MAX_CHANNEL_SIZE = 4096;
static const uint64_t MAX_NUM_PER_CORE = 2048;
static const uint64_t DICHOTOMY_ADD_COEFF = 2;
static const uint64_t ULONG_BIT_LEN = 64;
static const float DEFAULT_EPS = 1e-5;
static const bool DEFAULT_ACTIVATESILU = true;
static const uint64_t NDDMA_MAX_SIZE = 32; // 可根据实际测试结果更改此值

inline static int64_t CeilDiv(int64_t dividendVal, int64_t divisorVal)
{
    if (divisorVal == 0) {
        return dividendVal;
    } else if (dividendVal % divisorVal == 0) {
        return dividendVal / divisorVal;
    } else {
        return dividendVal / divisorVal + 1;
    }
}

inline static int64_t DownAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

inline static int64_t RoundUp(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

// quantScale 常驻 UB 字节数(per-tensor=1 或 per-channel=shapeC 个 fp32, 按 blockSize 对齐)。
inline static uint64_t CalcQuantScaleUbSize(uint64_t shapeQuantScale, uint64_t blockSize)
{
    return RoundUp(static_cast<uint64_t>(shapeQuantScale) * FLOAT32_BYTES, blockSize);
}

static ge::graphStatus CheckGammaParam(const gert::TilingContext* context, const ge::DataType& xDtype, uint64_t channel)
{
    auto gammaShapePtr = context->GetOptionalInputShape(INPUT_IDX_GAMMA);
    auto gammaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_GAMMA);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF(gammaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "gamma",
                                                         std::to_string(gammaShape.GetDimNum()).c_str(),
                                                         "The shape dim of gamma must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaSizes != channel,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape).c_str(),
                    ("The shape size of gamma must be the same as channel(dim[1] of input x), "
                     "got gamma shape size = " +
                     std::to_string(gammaSizes) + ", channel = " + std::to_string(channel))
                        .c_str()),
                return ge::GRAPH_FAILED);
    auto gammaDtype = gammaDtypePtr->GetDataType();
    OP_CHECK_IF((gammaDtype != xDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, gamma",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gammaDtype))
                                                           .c_str(),
                                                       "The dtype of gamma must be the same as x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBetaParam(const gert::TilingContext* context, const ge::DataType& xDtype, uint64_t channel)
{
    auto betaShapePtr = context->GetOptionalInputShape(INPUT_IDX_BETA);
    auto betaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_BETA);
    auto betaShape = betaShapePtr->GetStorageShape();
    uint64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF(
        (betaShape.GetDimNum() != 1 || betaSizes != channel),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "beta", Ops::Base::ToString(betaShape).c_str(),
                                              ("Beta dimension should be one, and the shape of beta must be "
                                               "the same as channel(dim[1] of input x), got beta size = " +
                                               std::to_string(betaSizes) + ", channel = " + std::to_string(channel))
                                                  .c_str()),
        return ge::GRAPH_FAILED);
    auto betaDtype = betaDtypePtr->GetDataType();
    OP_CHECK_IF((betaDtype != xDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, beta",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(betaDtype))
                                                           .c_str(),
                                                       "The dtype of beta must be the same as x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGammaAndBetaParams(const gert::TilingContext* context, const ge::DataType& xDtype,
                                               uint64_t channel)
{
    auto gammaShapePtr = context->GetOptionalInputShape(INPUT_IDX_GAMMA);
    auto gammaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_GAMMA);
    bool hasGamma = gammaShapePtr != nullptr;
    auto betaShapePtr = context->GetOptionalInputShape(INPUT_IDX_BETA);
    auto betaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_BETA);
    bool hasBeta = betaShapePtr != nullptr;
    if (!hasGamma && !hasBeta) {
        return ge::GRAPH_SUCCESS;
    }

    if (hasGamma && hasBeta) {
        auto gammaDtype = gammaDtypePtr->GetDataType();
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
        auto betaDtype = betaDtypePtr->GetDataType();
        uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDtype);
        OP_CHECK_IF((gammaDtypeSize != betaDtypeSize),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "gamma and beta",
                                                           (ge::TypeUtils::DataTypeToSerialString(gammaDtype) +
                                                            " and " + ge::TypeUtils::DataTypeToSerialString(betaDtype))
                                                               .c_str(),
                                                           "The datatype sizes of gamma and beta must be the same"),
                    return ge::GRAPH_FAILED);
    }

    if (hasGamma && CheckGammaParam(context, xDtype, channel) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (hasBeta && CheckBetaParam(context, xDtype, channel) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputXShape(const gert::TilingContext* context, const gert::Shape& xShape)
{
    uint64_t xDims = xShape.GetDimNum();
    OP_CHECK_IF((xDims < X_SHAPE_MIN_LEN),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDims).c_str(),
                                             "greater than or equal to 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDims > X_SHAPE_MAX_LEN),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDims).c_str(),
                                             "less than or equal to 8"),
                return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < xDims; i++) {
        int64_t curDim = xShape.GetDim(i);
        OP_CHECK_IF(
            (curDim < 0),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                ("The dim[" + std::to_string(i) + "] of x must be greater than 0, got " + std::to_string(curDim))
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // check x
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF((xDtypeSize <= 0),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                                      "The datatype size of x must be greater than 0"),
                return ge::GRAPH_FAILED);
    auto xShape = inputX->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    if (CheckInputXShape(context, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check gamma and beta
    if (CheckGammaAndBetaParams(context, xDtype, channel) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check quantScale: 长度必须为 1(per-tensor) 或 channel(per-channel);否则 kernel per-channel 逐通道读会越界。
    auto quantScaleShapePtr = context->GetOptionalInputShape(INPUT_IDX_QUANT_SCALE);
    if (quantScaleShapePtr != nullptr) {
        auto quantScaleShape = quantScaleShapePtr->GetStorageShape();
        uint64_t qsSize = static_cast<uint64_t>(quantScaleShape.GetShapeSize());
        OP_CHECK_IF((qsSize != 1 && qsSize != channel),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "quantScale", Ops::Base::ToString(quantScaleShape).c_str(),
                        ("The element count of quantScale must be 1 (per-tensor) or channel (per-channel), "
                         "got quantScale size = " +
                         std::to_string(qsSize) + ", channel = " + std::to_string(channel))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_IF((*numGroups <= 0),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "num_groups", std::to_string(*numGroups).c_str(),
                                          "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (channel % *numGroups != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_groups", std::to_string(*numGroups).c_str(),
                                              ("channel(dim[1] of input x) must be integer multiples of num_groups, "
                                               "got channel = " +
                                               std::to_string(channel) + ", num_groups = " + std::to_string(*numGroups))
                                                  .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void GetDichotomyAddParams(const gert::TilingContext* context, uint64_t r, uint64_t& power, uint64_t& dichotomyK,
                                  uint64_t& extraSize, uint64_t& lastNum)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t vl = compileInfo->vectorLength / FLOAT32_BYTES;
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t basePower = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r)));
    power = basePower == r ? basePower / DICHOTOMY_ADD_COEFF : basePower;
    uint64_t extraOriSize = power / vl;
    extraSize = RoundUp(extraOriSize * FLOAT32_BYTES, blockSize);
    dichotomyK = 0;
    if (extraOriSize < vl) {
        lastNum = extraOriSize;
        return;
    }
    uint64_t totalNum = extraOriSize / vl;
    uint64_t base = 1;
    lastNum = vl;
    while (base < totalNum) {
        dichotomyK++;
        base *= DICHOTOMY_ADD_COEFF;
    }
}

static uint64_t GetOptionalInputTensorSize(const gert::TilingContext* context,
                                           GroupNormSiluQuantRegbaseTilingData& tilingData, uint64_t index,
                                           uint64_t specifiedValue = 0, bool useNddma = false)
{
    auto tensorDesc = context->GetOptionalInputDesc(index);
    if (tensorDesc == nullptr) {
        return 0;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    auto dtypeSize = ge::GetSizeByDataType(tensorDesc->GetDataType());
    if (specifiedValue != 0) {
        return RoundUp(specifiedValue * dtypeSize, blockSize);
    }

    auto storageShape = context->GetOptionalInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    auto shape = storageShape->GetStorageShape();
    uint64_t num = 1;
    for (uint64_t i = 0; i < shape.GetDimNum(); i++) {
        num = num * shape.GetDim(i);
    }
    auto numUbSize = RoundUp(num * dtypeSize, blockSize);
    uint64_t hwNum = tilingData.get_hwNum();
    if (hwNum <= NDDMA_MAX_SIZE && useNddma) {
        int32_t xDtype = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
        if (xDtype == 0) {
            OP_LOGE(context, "Division by zero!");
            return 0;
        }
        int64_t eleNumAlign = RoundUp(tilingData.get_elemNum(), blockSize / xDtype);
        return RoundUp(eleNumAlign * dtypeSize, blockSize) * tilingData.get_numGroups();
    }
    return numUbSize;
}

// quantScale 常驻 UB 字节数, 区分 fold(hwNum<=NDDMA 且 channel 不大)+per-channel(按 numGroups*elemNumAlign 预留)
// 与其它(per-tensor 或非 fold per-channel, 按 shapeQuantScale 预留)。逐行原样保留。
static uint64_t CalcFoldQuantScaleUbSize(const gert::TilingContext* context,
                                         GroupNormSiluQuantRegbaseTilingData& tilingData, uint32_t blockSize)
{
    // fold(hwNum<=NDDMA)+per-channel 时 quantScale 按 fold 布局复制成 numGroups*elemNumAlign(同 gamma); 否则
    // per-channel 只需 shapeC。 大 channel(shapeC>MAX_CHANNEL_SIZE)走 R_PARTIAL_LOAD_GENERALIZED, kernel 按 shapeC
    // 逐通道读、不 fold; 故大 channel 不 fold。
    // 判据必须与 kernel r_full_load.h 的 `isFold = hwNum <= NDDMA_SIZE || (!hasGamma && !hasBeta)` 完全一致。
    // issue #21 的 L1_014: hwNum=28224(>NDDMA) 但没有 gamma/beta, kernel 照样 fold, 而这里旧判据只看 hwNum,
    // 于是 per-channel quantScale 只按 shapeC 个 fp32 预留, kernel 却按 numGroups*elemNumAlign 展开写
    // (4*28224 个 fp32 = 451KB, UB 仅 253952B) → MTE2 写越界 errcode 82。
    bool hasGammaOrBeta = context->GetOptionalInputDesc(INPUT_IDX_GAMMA) != nullptr ||
                          context->GetOptionalInputDesc(INPUT_IDX_BETA) != nullptr;
    bool couldFold = ((tilingData.get_hwNum() <= static_cast<int64_t>(NDDMA_MAX_SIZE)) || !hasGammaOrBeta) &&
                     (static_cast<uint64_t>(tilingData.get_shapeC()) <= MAX_CHANNEL_SIZE);
    // fold 布局的 quantScale 仅 per-channel 需要; per-tensor 只广播 1 个标量。
    bool perChannelScale = (tilingData.get_shapeQuantScale() != 1);
    if (couldFold && perChannelScale) {
        // 必须与 kernel 侧一致:CopyQuantScale2UBByNDDMA 以 elemNumAlign=RoundUp<x_dtype>(elemNum)(fp16/bf16 下为
        // 16)为每组 dst stride 复制 quantScale。此前误用 blockSize/FLOAT32_BYTES(=8), elemNum 非 16 倍数时少预留一半 →
        // VEC_ERROR。
        uint64_t xDtypeSizeForQs = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
        uint64_t qsBlockElem = (xDtypeSizeForQs == 0) ? (blockSize / FLOAT32_BYTES) : (blockSize / xDtypeSizeForQs);
        uint64_t eleNumAlign = RoundUp(tilingData.get_elemNum(), qsBlockElem);
        return RoundUp(eleNumAlign * FLOAT32_BYTES, blockSize) * tilingData.get_numGroups();
    }
    // per-tensor(shapeQuantScale==1, 只需 1 标量) 或 非 fold 的 per-channel(需 shapeC 个)。
    return CalcQuantScaleUbSize(tilingData.get_shapeQuantScale(), blockSize);
}

// 尝试走 R_FULL_LOAD(1110): 直接全载, 或用 move_align 搬 gamma/beta 后再试。设 key 成功则返回 true。逐行原样保留。
static bool TrySetFullLoad(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                           bool& isReduceFullLoad, uint64_t& maxReduceCount, uint64_t ubRemain, uint64_t& otherUbSize,
                           uint64_t gammaUbSize, uint64_t betaUbSize, uint64_t gammaNewUbSize, uint64_t betaNewUbSize,
                           uint64_t reduceCount, bool hasOptional, uint64_t xDtypeSize)
{
    if (xDtypeSize == 0) {
        return false;
    }
    // 大 channel(shapeC>MAX_CHANNEL) + 小 hwNum(<=NDDMA) 时不能走 R_FULL_LOAD(1110): 其 kernel isFold 会 fold 复制 →
    // 越界 VEC_ERROR。
    bool foldMismatch = (static_cast<uint64_t>(tilingData.get_shapeC()) > MAX_CHANNEL_SIZE) &&
                        (tilingData.get_hwNum() <= static_cast<int64_t>(NDDMA_MAX_SIZE));
    if (maxReduceCount > reduceCount && !foldMismatch) {
        isReduceFullLoad = true;
        int64_t tilingKey = static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_FULL_LOAD);
        tilingData.set_tilingKey(tilingKey);
        return true;
    }
    // 如果使用了nddma之后, 无法走全载模板，则尝试使用move_align搬入gamma或者beta，然后再次尝试是否能采用R轴全载性能模板
    bool isUseNddma = hasOptional && (tilingData.get_hwNum() <= static_cast<int64_t>(NDDMA_MAX_SIZE));
    if (isUseNddma) {
        otherUbSize = otherUbSize + (gammaUbSize - gammaNewUbSize) + (betaUbSize - betaNewUbSize);
        maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (maxReduceCount > reduceCount && !foldMismatch) {
            isReduceFullLoad = true;
            int64_t tilingKey = static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_FULL_LOAD);
            tilingData.set_tilingKey(tilingKey);
            return true;
        }
    }
    return false;
}

// R 轴过大无法全载的最终回落: 按"整段常驻装不装得下"在 R_PARTIAL_LOAD(1100)与
// R_PARTIAL_LOAD_GENERALIZED(1120)之间选。
// 1100 会在 welford 结束后把整段 gamma/beta/quantScale 一次性载入 UB(见 kernel r_partial_load.h 的
// LoadOptionalInputs), 因此只有整段装得下才能选它; 装不下必须走 1120(按通道范围流式载入)。
// issue #21: 旧判据是 isLargeChannel && hasOptional, 把"要不要整段常驻"与"有没有 gamma/beta"绑在了一起,
// 而 quantScale 是否整段常驻只取决于 C。大 C 且无 gamma/beta 时被错选进 1100, 随后
// ubSize - ... 无符号下溢成 2^61 级记账, 内核据此搬运即 UB 越界。
static void SetPartialFallback(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                               bool& isReduceFullLoad, uint64_t& ubRemain, uint64_t meanUbSize, uint64_t rstdUbSize,
                               uint64_t meanUbExtraSize, uint64_t rstdUbExtraSize, uint64_t ubSize, uint32_t blockSize,
                               bool isLargeChannel)
{
    isReduceFullLoad = false;
    uint64_t meanAndRstdSize = meanUbSize + rstdUbSize + meanUbExtraSize + rstdUbExtraSize;
    // 1100 的整段常驻账(均按非 fold 计: per-channel quantScale 只需 shapeC 个 fp32)
    uint64_t gammaFullUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, false);
    uint64_t betaFullUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, false);
    uint64_t quantScaleFullUbSize = CalcQuantScaleUbSize(tilingData.get_shapeQuantScale(), blockSize);
    uint64_t fullResidentSize = meanAndRstdSize + gammaFullUbSize + betaFullUbSize + quantScaleFullUbSize;
    if (isLargeChannel || fullResidentSize >= ubSize) {
        // 1120: gamma/beta/quantScale 都按通道范围流式载入, 这里只预留 mean/rstd;
        // quantScale 的实际用量(rowsForQuant 个 fp32)在 AdjustUbForQuantScale 里按解出的 count 扣除。
        ubRemain = ubSize > meanAndRstdSize ? ubSize - meanAndRstdSize : 0;
        int64_t tilingKey = static_cast<int64_t>(
            GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_PARTIAL_LOAD_GENERALIZED);
        tilingData.set_tilingKey(tilingKey);
    } else {
        ubRemain = ubSize - fullResidentSize; // 上面的分支条件已保证不下溢
        int64_t tilingKey = static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_PARTIAL_LOAD);
        tilingData.set_tilingKey(tilingKey);
    }
}

// 全载失败后的回落: 尝试大 channel 全载泛化(1130), 否则落
// R_PARTIAL_LOAD_GENERALIZED(1120)/R_PARTIAL_LOAD(1100)。逐行原样保留。
static void SetGeneralizedOrPartial(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                                    bool& isReduceFullLoad, uint64_t& maxReduceCount, uint64_t& ubRemain,
                                    uint64_t gammaUbSize, uint64_t betaUbSize, uint64_t meanUbSize, uint64_t rstdUbSize,
                                    uint64_t meanUbExtraSize, uint64_t rstdUbExtraSize, uint64_t outDtypeSize,
                                    uint64_t dichotomyAddExtraSize, uint64_t reduceCount, bool hasOptional,
                                    uint64_t xDtypeSize, uint64_t ubSize, uint32_t blockSize)
{
    if (xDtypeSize == 0) {
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    // 1110(唯一会 fold 的 key)的全载检查已在上面完成。后续 1130/1120/1100 均不 fold, per-channel quantScale 只需
    // shapeC。
    uint64_t quantScaleUbSize = CalcQuantScaleUbSize(tilingData.get_shapeQuantScale(), blockSize);
    uint64_t otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize + quantScaleUbSize;
    if (outDtypeSize != FLOAT32_BYTES) {
        otherUbSize = otherUbSize + meanUbExtraSize + rstdUbExtraSize;
    }
    otherUbSize += dichotomyAddExtraSize;
    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    bool isLargeChannel = static_cast<uint64_t>(tilingData.get_shapeC()) > MAX_CHANNEL_SIZE;
    uint64_t newUbRemain = ubRemain;
    // 对于Channel轴过大的场景，尝试将gamma大小限制为ShapeD，beta大小限制为shapeD，重新计算最大可全载的R轴
    if (isLargeChannel || reduceCount <= compileInfo->vectorLength / FLOAT32_BYTES) {
        uint64_t gammaSplitUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA,
                                                               tilingData.get_shapeD());
        uint64_t betaSplitUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA,
                                                              tilingData.get_shapeD());
        otherUbSize = otherUbSize - gammaUbSize - betaUbSize + gammaSplitUbSize + betaSplitUbSize;
        newUbRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
        uint64_t newMaxReduceCount = (newUbRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (newMaxReduceCount > reduceCount) {
            isReduceFullLoad = true;
            maxReduceCount = newMaxReduceCount;
            ubRemain = newUbRemain;
            int64_t tilingKey = static_cast<int64_t>(
                GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_FULL_LOAD_GENERALIZED);
            tilingData.set_tilingKey(tilingKey);
            return;
        }
    }
    // R轴过大，无法走全载模版，此时需要将二分累加所需要的ub空间释放，重新更新ubRemain和newMaxReduceCount
    SetPartialFallback(context, tilingData, isReduceFullLoad, ubRemain, meanUbSize, rstdUbSize, meanUbExtraSize,
                       rstdUbExtraSize, ubSize, blockSize, isLargeChannel);
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
}

// 基础 otherUbSize(gamma/beta/mean/rstd/quantScale) + 非 fp32 输出时的 mean/rstd 额外 UB。输出 extras 与
// outDtypeSize。逐行原样保留。
static uint64_t AccumulateBaseOtherUb(const gert::TilingContext* context,
                                      GroupNormSiluQuantRegbaseTilingData& tilingData, uint64_t gammaUbSize,
                                      uint64_t betaUbSize, uint64_t meanUbSize, uint64_t rstdUbSize,
                                      uint64_t quantScaleUbSize, uint64_t realNumPerCore, uint64_t xDtypeSize,
                                      uint32_t blockSize, uint64_t& meanUbExtraSize, uint64_t& rstdUbExtraSize,
                                      uint64_t& outDtypeSize)
{
    uint64_t otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize + quantScaleUbSize;
    meanUbExtraSize = 0;
    rstdUbExtraSize = 0;
    outDtypeSize = xDtypeSize;
    if (context->GetInputDesc(INPUT_IDX_GAMMA) != nullptr) {
        outDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType());
    } else if (context->GetInputDesc(INPUT_IDX_BETA) != nullptr) {
        outDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_BETA)->GetDataType());
    }
    if (outDtypeSize != FLOAT32_BYTES) {
        meanUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        rstdUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        otherUbSize = otherUbSize + meanUbExtraSize + rstdUbExtraSize;
    }
    return otherUbSize;
}

// 只取二分累加的额外 UB 字节(内部临时量丢弃)。逐行原样保留。
static uint64_t GetDichotomyExtraSize(const gert::TilingContext* context, uint64_t reduceCount)
{
    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    return dichotomyAddExtraSize;
}

/*
regbase tilingKey划分逻辑如下:
1. UB内默认存放2048根A轴，gamma和beta在UB内全载，在开启DB条件下，计算出能够全载的最大R轴
    1.1 R轴小于可全载的最大R轴,则走TILINGKEY_R_FULL_LOAD性能模板
    1.2 尝试切分gamma和beta，每次拷入gamma和beta时，拷贝一个完整的D大小的数据
计算出新的可全载的最大R轴，如果大于R轴实际大小，则走TILINGKEY_R_FULL_LOAD_GENERALIZED泛化模板
    1.3
当R轴不能全载时，如果gamma和beta小于4096，则走TILINGKEY_R_PARTIAL_LOAD模板，否则走TILINGKEY_R_PARTIAL_LOAD_GENERALIZED模板
二分累加所需要的UB大小，计算规则如下:
1. 优先按照R轴可以全载计算所需要的额外空间
2. 如果R轴无法全载，则需要根据可用的UB空间，结合二分累加的额外空间，重新计算出最大的并行度
在非全载模板下，二分累加的UB额外空间不会影响normalize+swish阶段一次可载入的R轴大小
*/
static void SetTilingKey4Regbase(const gert::TilingContext* context, uint64_t& maxReduceCount, uint64_t& ubRemain,
                                 bool& isReduceFullLoad, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint64_t ubSize = compileInfo->ubSizePlatForm;
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t gammaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, true);
    uint64_t betaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, true);
    uint64_t realNumPerCore = std::min(
        MAX_NUM_PER_CORE, static_cast<uint64_t>(std::max(tilingData.get_numPerCore(), tilingData.get_numLastCore())));
    uint64_t meanUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);
    uint64_t rstdUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);
    uint64_t quantScaleUbSize = CalcFoldQuantScaleUbSize(context, tilingData, blockSize);

    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    uint64_t meanUbExtraSize = 0, rstdUbExtraSize = 0, outDtypeSize = 0;
    uint64_t otherUbSize = AccumulateBaseOtherUb(context, tilingData, gammaUbSize, betaUbSize, meanUbSize, rstdUbSize,
                                                 quantScaleUbSize, realNumPerCore, xDtypeSize, blockSize,
                                                 meanUbExtraSize, rstdUbExtraSize, outDtypeSize);
    uint64_t xShapeSize = context->GetInputShape(INPUT_IDX_X)->GetStorageShape().GetShapeSize();
    if (xShapeSize == EMPTY_SHAPE_SIZE) {
        int64_t tilingKey = static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_EMPTY_TENSOR);
        tilingData.set_tilingKey(tilingKey);
        return;
    }

    bool hasOptional = context->GetOptionalInputDesc(INPUT_IDX_GAMMA) != nullptr ||
                       context->GetOptionalInputDesc(INPUT_IDX_BETA) != nullptr;
    uint64_t dichotomyAddExtraSize = GetDichotomyExtraSize(context, reduceCount);
    otherUbSize += dichotomyAddExtraSize;

    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;

    uint64_t gammaNewUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, false);
    uint64_t betaNewUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, false);
    if (TrySetFullLoad(context, tilingData, isReduceFullLoad, maxReduceCount, ubRemain, otherUbSize, gammaUbSize,
                       betaUbSize, gammaNewUbSize, betaNewUbSize, reduceCount, hasOptional, xDtypeSize)) {
        return;
    }
    gammaUbSize = gammaNewUbSize;
    betaUbSize = betaNewUbSize;
    SetGeneralizedOrPartial(context, tilingData, isReduceFullLoad, maxReduceCount, ubRemain, gammaUbSize, betaUbSize,
                            meanUbSize, rstdUbSize, meanUbExtraSize, rstdUbExtraSize, outDtypeSize,
                            dichotomyAddExtraSize, reduceCount, hasOptional, xDtypeSize, ubSize, blockSize);
}

static void SetDichotomyAddParams(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    uint64_t powerOfTwoForReduce = (1L << (ULONG_BIT_LEN - __builtin_clzl(reduceCount)));
    tilingData.set_powerOfTwoForReduce(powerOfTwoForReduce);
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
}

static void SetWelfordParallelN(const gert::TilingContext* context, uint64_t xDtypeSize, uint64_t ubRemain,
                                GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint32_t coeff = FLOAT32_BYTES / xDtypeSize;
    uint32_t totalNum = BUFFER_NUM * (coeff + 1);
    // parallelN 按 VL_FP32(=vectorLength/4=64)对齐: kernel Welford finalize 按 VL_FP32 整块读累加器,
    // 只按 block(16)对齐时 parallelN 非64倍数会让 finalize 读进相邻区(case00124 类)。
    uint32_t welfordBase = compileInfo->vectorLength / FLOAT32_BYTES;
    uint32_t maxParallelN = DownAlign((ubRemain / xDtypeSize) / totalNum, welfordBase);

    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    uint32_t ubCurUse = maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize +
                        maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    while (ubCurUse > ubRemain) {
        maxParallelN -= welfordBase;
        GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                              dichotomyAddLastNum);
        ubCurUse = maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize +
                   maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    }

    if (maxParallelN > tilingData.get_elemNum()) {
        maxParallelN = tilingData.get_elemNum();
    }
    GetDichotomyAddParams(context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize,
                          dichotomyAddLastNum);
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
    tilingData.set_parallelN(maxParallelN);
}

static void SetUbTiling4TwoPass(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                                uint64_t maxReduceCount, uint32_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t elemNum = tilingData.get_elemNum();
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t elemNumAlign = RoundUp(elemNum, blockSize / xDtypeSize);
    SetDichotomyAddParams(context, tilingData);
    if (elemNumAlign == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t count = maxReduceCount / elemNumAlign;
    uint64_t processSize = count * elemNumAlign;
    tilingData.set_processSize(processSize);
}

static void SetUbTiling4WelfordPerf(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                                    uint64_t maxReduceCount, uint32_t ubRemain, uint32_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    // Align 路(r_partial_load)按 VL_FP32 粒度读 reduce 轴,行/块对齐必须按 VL_FP32(非 block),否则 kernel
    // 过读越界(VEC_ERROR)。
    uint32_t reduceAlign = compileInfo->vectorLength / FLOAT32_BYTES;
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    uint64_t loopNum = 0;
    uint64_t loopTail = 0;
    uint64_t processSize = 0;
    uint64_t innerLoopNum = 0;
    uint64_t innerLoopTail = 0;
    uint64_t hwNum = tilingData.get_hwNum();
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t hwNumAlign = RoundUp(hwNum, reduceAlign);
    if (hwNumAlign == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t count = maxReduceCount / hwNumAlign;
    if (count >= 1) {
        loopNum = CeilDiv(tilingData.get_shapeD(), count);
        loopTail = (tilingData.get_shapeD() - (loopNum - 1) * count) * hwNumAlign;
        processSize = count * hwNumAlign;
        innerLoopNum = 1;
    } else {
        auto maxReduceCountDownAlign = DownAlign(maxReduceCount, reduceAlign);
        if (maxReduceCountDownAlign == 0) { // UB 不足以放下一行的对齐粒度, 交给出口统一拒收
            return;
        }
        innerLoopNum = CeilDiv(hwNum, maxReduceCountDownAlign);
        innerLoopTail = hwNum - maxReduceCountDownAlign * (innerLoopNum - 1);
        processSize = maxReduceCountDownAlign;
        loopNum = tilingData.get_shapeD();
        loopTail = 1;
    }
    tilingData.set_loopNum(loopNum);
    tilingData.set_loopTail(loopTail);
    tilingData.set_processSize(processSize);
    tilingData.set_innerLoopNum(innerLoopNum);
    tilingData.set_innerLoopTail(innerLoopTail);
}
// 迭代求解一次可载入的 count(gamma/beta per-range 常驻随 count 变化, 逐步减 count 直到 curUbSize<=ubRemain)。
// 逐行原样保留。返回最终 count, 并输出对应的 gammaRealSize/betaRealSize。
static uint64_t SolveWelfordGeneralizedCount(const gert::TilingContext* context,
                                             GroupNormSiluQuantRegbaseTilingData& tilingData, uint32_t ubRemain,
                                             uint32_t xDtypeSize, uint64_t hwNumAlign, uint64_t maxReduceCount,
                                             uint64_t& gammaRealSize, uint64_t& betaRealSize)
{
    if (xDtypeSize == 0 || hwNumAlign == 0) {
        return 0;
    }
    uint64_t count = maxReduceCount / hwNumAlign;
    gammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, count);
    betaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, count);
    uint64_t curUbSize = gammaRealSize + betaRealSize + count * hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    while (curUbSize > ubRemain && count >= 1) {
        count--;
        uint64_t tmpGammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, count);
        uint64_t tmpBetaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, count);
        curUbSize = tmpGammaRealSize + tmpBetaRealSize + count * hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    }
    return count;
}

// kernel 对 quantScale 按 per-range 常驻 rows(=processSize/hwNumAlign)个 fp32; 上游只按 shapeQuantScale 预留,
// 不足时按实际 用量补足差额(否则 dichotomyAddLocal 与 meanTensor 同地址、mean 输出被覆盖)。逐行原样保留。
static void AdjustUbForQuantScale(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                                  uint32_t& ubRemain, uint64_t processSize, uint64_t hwNumAlign, uint32_t blockSize)
{
    if (hwNumAlign == 0) {
        return;
    }
    uint64_t rowsForQuant = processSize / hwNumAlign;
    uint64_t quantScaleKernelSize = (rowsForQuant == 0) ?
                                        blockSize :
                                        RoundUp(rowsForQuant * FLOAT32_BYTES, static_cast<uint64_t>(blockSize));
    // 上游预留口径随 tilingKey 不同: 1120(SetPartialFallback)只预留 mean/rstd, quantScale 一个字节都没留,
    // 故要扣掉内核的全部用量; 1130 的上游已按整段 shapeQuantScale 预留, 只需补足差额。
    bool upfrontReservedFullQuantScale = tilingData.get_tilingKey() !=
                                         static_cast<int64_t>(
                                             GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_PARTIAL_LOAD_GENERALIZED);
    uint64_t quantScaleReserved = upfrontReservedFullQuantScale ?
                                      CalcQuantScaleUbSize(tilingData.get_shapeQuantScale(),
                                                           static_cast<uint64_t>(blockSize)) :
                                      0;
    if (quantScaleKernelSize > quantScaleReserved) {
        uint64_t quantScaleExtra = quantScaleKernelSize - quantScaleReserved;
        ubRemain = ubRemain > quantScaleExtra ? ubRemain - quantScaleExtra : 0;
    }
}

// 依据 count 是否 >=1 计算 loopNum/loopTail/processSize/innerLoopNum/innerLoopTail 并更新 ubRemain。逐行原样保留。
static void ComputeWelfordGeneralizedLoop(const gert::TilingContext* context,
                                          GroupNormSiluQuantRegbaseTilingData& tilingData, uint32_t& ubRemain,
                                          uint32_t xDtypeSize, uint32_t reduceAlign, uint64_t hwNum,
                                          uint64_t hwNumAlign, uint64_t count, uint64_t gammaRealSize,
                                          uint64_t betaRealSize, uint64_t maxReduceCount, uint64_t& loopNum,
                                          uint64_t& loopTail, uint64_t& processSize, uint64_t& innerLoopNum,
                                          uint64_t& innerLoopTail)
{
    if (xDtypeSize == 0 || hwNumAlign == 0) {
        return;
    }
    if (count >= 1) {
        loopNum = CeilDiv(tilingData.get_shapeD(), count);
        loopTail = (tilingData.get_shapeD() - (loopNum - 1) * count) * hwNumAlign;
        processSize = count * hwNumAlign;
        innerLoopNum = 1;
        // 下溢保护: ubRemain 是 uint32, 减到负数会翻成 40 亿级记账, 内核据此搬运即越界(issue #21 同类)
        ubRemain = ubRemain > gammaRealSize + betaRealSize ? ubRemain - gammaRealSize - betaRealSize : 0;
    } else {
        gammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 1);
        betaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 1);
        ubRemain = ubRemain > gammaRealSize + betaRealSize ? ubRemain - gammaRealSize - betaRealSize : 0;
        maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        uint64_t maxReduceCountDownAlign = DownAlign(maxReduceCount, reduceAlign);
        if (maxReduceCountDownAlign == 0) { // 同上, processSize 保持 0, 由出口统一拒收
            return;
        }
        innerLoopNum = CeilDiv(hwNum, maxReduceCountDownAlign);
        innerLoopTail = hwNum - maxReduceCountDownAlign * (innerLoopNum - 1);
        processSize = maxReduceCountDownAlign;
        loopNum = tilingData.get_shapeD();
        loopTail = 1;
    }
}

static void SetUbTiling4WelfordGeneralized(const gert::TilingContext* context,
                                           GroupNormSiluQuantRegbaseTilingData& tilingData, uint32_t ubRemain,
                                           uint32_t xDtypeSize)
{
    if (xDtypeSize == 0) {
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    // 本函数被 R_PARTIAL_LOAD_GENERALIZED(Align 路)与 R_FULL_LOAD(_GENERALIZED)(UnAlign 路)共用。
    // Align 路按 VL_FP32 粒度读 reduce 轴,行/块对齐必须按 VL_FP32,否则 kernel 过读越界(VEC_ERROR);UnAlign 路维持 block
    // 对齐。
    int64_t curTilingKey = tilingData.get_tilingKey();
    bool isPartialGeneralized = (curTilingKey ==
                                 static_cast<int64_t>(
                                     GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_PARTIAL_LOAD_GENERALIZED));
    uint32_t reduceAlign = isPartialGeneralized ? (compileInfo->vectorLength / FLOAT32_BYTES) :
                                                  (blockSize / xDtypeSize);
    uint64_t loopNumG = 0;
    uint64_t loopTailG = 0;
    uint64_t processSizeG = 0;
    uint64_t innerLoopNumG = 0;
    uint64_t innerLoopTailG = 0;
    uint64_t hwNumG = tilingData.get_hwNum();
    if (xDtypeSize == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t hwNumAlignG = RoundUp(hwNumG, reduceAlign);
    uint64_t maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
    if (hwNumAlignG == 0) {
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t gammaRealSize = 0;
    uint64_t betaRealSize = 0;
    uint64_t count = SolveWelfordGeneralizedCount(context, tilingData, ubRemain, xDtypeSize, hwNumAlignG,
                                                  maxReduceCount, gammaRealSize, betaRealSize);
    ComputeWelfordGeneralizedLoop(context, tilingData, ubRemain, xDtypeSize, reduceAlign, hwNumG, hwNumAlignG, count,
                                  gammaRealSize, betaRealSize, maxReduceCount, loopNumG, loopTailG, processSizeG,
                                  innerLoopNumG, innerLoopTailG);
    AdjustUbForQuantScale(context, tilingData, ubRemain, processSizeG, hwNumAlignG, blockSize);
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    tilingData.set_loopNum(loopNumG);
    tilingData.set_loopTail(loopTailG);
    tilingData.set_processSize(processSizeG);
    tilingData.set_innerLoopNum(innerLoopNumG);
    tilingData.set_innerLoopTail(innerLoopTailG);
}

static void SetUbTiling4Welford(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                                uint64_t maxReduceCount, uint64_t ubRemain, uint32_t xDtypeSize)
{
    if (tilingData.get_tilingKey() ==
        static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_PARTIAL_LOAD)) {
        return SetUbTiling4WelfordPerf(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    }
    return SetUbTiling4WelfordGeneralized(context, tilingData, ubRemain, xDtypeSize);
}

static void SetUbTiling4Regbase(const gert::TilingContext* context, uint64_t maxReduceCount, uint64_t ubRemain,
                                bool isReduceFullLoad, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    int32_t ubSize = compileInfo->ubSizePlatForm;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    tilingData.set_ubSize(ubSize);

    uint64_t xShapeSize = context->GetInputShape(INPUT_IDX_X)->GetStorageShape().GetShapeSize();
    if (xShapeSize == EMPTY_SHAPE_SIZE) {
        return;
    }

    if (!isReduceFullLoad) {
        SetUbTiling4Welford(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    } else {
        SetUbTiling4TwoPass(context, tilingData, maxReduceCount, xDtypeSize);
    }
}

static void SetTilingForRegbase(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    uint64_t maxReduceCount = 0;
    uint64_t ubRemain = 0;
    bool reduceFullLoad = false;
    SetTilingKey4Regbase(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
    SetUbTiling4Regbase(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
}

static void SetAttrParams(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    const float* epsilonPtr = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    const bool* activateSiluPtr = attrs->GetAttrPointer<bool>(INDEX_ACTIVATE_SILU);
    float eps = epsilonPtr == nullptr ? DEFAULT_EPS : *epsilonPtr;
    bool activateSilu = activateSiluPtr == nullptr ? DEFAULT_ACTIVATESILU : *activateSiluPtr;
    tilingData.set_numGroups(*numGroupsPtr);
    tilingData.set_epsilon(eps);
    tilingData.set_activateSilu(activateSilu);
}

static void SetTilingParams(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t hwNum = 1;
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tilingData.set_shapeC(xShape.GetDim(DIM_1));
    tilingData.set_shapeD(tilingData.get_shapeC() / tilingData.get_numGroups());
    tilingData.set_hwNum(hwNum);
    tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
    // quantScale: 长度 1(per-tensor) 或 shapeC(per-channel), 决定 kernel per-channel 分支
    uint64_t shapeQuantScale = 1;
    auto quantScaleShapePtr = context->GetOptionalInputShape(INPUT_IDX_QUANT_SCALE);
    if (quantScaleShapePtr != nullptr) {
        auto quantScaleShape = quantScaleShapePtr->GetStorageShape();
        if (quantScaleShape.GetDimNum() > 0 && quantScaleShape.GetShapeSize() > 1) {
            shapeQuantScale = static_cast<uint64_t>(quantScaleShape.GetShapeSize());
        }
    }
    tilingData.set_shapeQuantScale(shapeQuantScale);
}

static void SetBlockTiling(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    tilingData.set_numPerCore(CeilDiv(shapeN * tilingData.get_numGroups(), compileInfo->totalCoreNum));
    tilingData.set_realCoreNum(CeilDiv(shapeN * tilingData.get_numGroups(), tilingData.get_numPerCore()));
    tilingData.set_numLastCore(shapeN * tilingData.get_numGroups() -
                               tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    tilingData.set_shapeN(shapeN);
    tilingData.set_coresPerGroup(1); // 默认非 split-reduce; split-reduce 分支会覆盖
    tilingData.set_groupInvCnt(1.0f);
    uint64_t xShapeSize = xShape.GetShapeSize();
    if (xShapeSize == 0) {
        int64_t realCoreNum = tilingData.get_realCoreNum();
        tilingData.set_realCoreNum((realCoreNum == 0) ? compileInfo->totalCoreNum : realCoreNum);
    }
}

// 1140 内核的 UB 账,与 op_kernel/arch35/group_norm_silu_quant_regbase_split_reduce.h 的 InitUbBuffers 一一对应
// (该文件的缓冲增删必须同步改这里):
//   固定部分: inQue 2*TILE*sizeof(T1) + outQue 2*TILE*1 + (fbuf+rbuf+wbuf) 3*TILE*4 + reduf 32*4 + partBuf 32
//             + combBuf(coresPerGroup*2 个 float 对齐 + 32) + mrFBuf 2*VL_FP32*4 + mrOBuf 2*VL_FP32*sizeof(T1)
//   通道常驻: 仅在存在 gamma/beta 时, 每通道 (sizeof(T2)+4) 字节 x chanAlign(=CeilDiv(shapeD, coresPerGroup))
// issue #21: 原先没有这道校验, chanPerCore=169013 时通道常驻需 2.03MB 而 UB 只有 253952B → MTE2 写越界(errcode 82)。
static bool SplitReduceFitsUb(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData,
                              int64_t coresPerGroup)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    const uint64_t splitKernelTile = 4096; // = kernel 侧 GroupNormSiluQuantSplitReduce::TILE
    uint64_t ubSize = compileInfo->ubSizePlatForm;
    uint64_t blockSize = static_cast<uint64_t>(compileInfo->blockSizePlatform);
    uint64_t vlFp32 = static_cast<uint64_t>(compileInfo->vectorLength) / FLOAT32_BYTES;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    auto gammaDesc = context->GetOptionalInputDesc(INPUT_IDX_GAMMA);
    auto betaDesc = context->GetOptionalInputDesc(INPUT_IDX_BETA);
    uint64_t t2Size = xDtypeSize;
    if (gammaDesc != nullptr) {
        t2Size = ge::GetSizeByDataType(gammaDesc->GetDataType());
    } else if (betaDesc != nullptr) {
        t2Size = ge::GetSizeByDataType(betaDesc->GetDataType());
    }
    uint64_t fixedUb = splitKernelTile * xDtypeSize * DOUBLE_BUFFER + splitKernelTile * DOUBLE_BUFFER +
                       splitKernelTile * FLOAT32_BYTES * 3 + 32 * FLOAT32_BYTES + blockSize +
                       RoundUp(static_cast<uint64_t>(coresPerGroup) * 2 * FLOAT32_BYTES, blockSize) + blockSize +
                       2 * vlFp32 * FLOAT32_BYTES + 2 * vlFp32 * xDtypeSize;
    uint64_t chanPerCore = CeilDiv(static_cast<uint64_t>(tilingData.get_shapeD()),
                                   static_cast<uint64_t>(coresPerGroup));
    uint64_t chanAlign = chanPerCore < 8 ? 8 : chanPerCore;
    uint64_t chanResidentUb = 0;
    if (gammaDesc != nullptr) {
        chanResidentUb += chanAlign * (t2Size + FLOAT32_BYTES);
    }
    if (betaDesc != nullptr) {
        chanResidentUb += chanAlign * (t2Size + FLOAT32_BYTES);
    }
    return fixedUb + chanResidentUb <= ubSize;
}

// split-reduce(1140)判定:N*num_groups < 核数(A轴欠并行)且单组 reduce 很大时,按 shapeD 通道把每组切到多核。
// 返回 userWorkspace 字节(非 split 返回 0)。只覆盖 tilingKey/coresPerGroup/realCoreNum/ubSize,不碰其它路径。
static uint64_t MaybeSetSplitReduce(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint64_t totalCore = compileInfo->totalCoreNum;
    int64_t aAxis = tilingData.get_shapeN() * tilingData.get_numGroups(); // N*num_groups
    int64_t shapeD = tilingData.get_shapeD();
    int64_t elemNum = tilingData.get_elemNum();
    // 触发条件:A轴欠并行 + 每组 reduce 足够大(单核会很慢) + 通道可切
    const int64_t SPLIT_MIN_ELEM = 65536; // 单组元素数阈值,小于此单核已够快,不切
    if (aAxis <= 0 || (uint64_t)aAxis >= totalCore || shapeD < 2 || elemNum < SPLIT_MIN_ELEM) {
        return 0;
    }
    int64_t coresPerGroup = static_cast<int64_t>(totalCore) / aAxis;
    if (coresPerGroup > shapeD) {
        coresPerGroup = shapeD; // 不能超过通道数
    }
    if (coresPerGroup < 2) {
        return 0;
    }
    // 通道常驻的 gamma/beta 装不下 UB 时不能选 1140(内核会越界), 让它回落到按通道范围流式载入的路径。
    if (!SplitReduceFitsUb(context, tilingData, coresPerGroup)) {
        return 0;
    }
    tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_SPLIT_REDUCE));
    tilingData.set_coresPerGroup(coresPerGroup);
    tilingData.set_groupInvCnt(1.0f / static_cast<float>(elemNum)); // 1/(shapeD*hwNum)
    tilingData.set_realCoreNum(aAxis * coresPerGroup);
    tilingData.set_ubSize(compileInfo->ubSizePlatForm);
    // userWorkspace: 每核 2 个 float(sum,sumsq),按 32B 对齐
    uint64_t bytes = static_cast<uint64_t>(aAxis * coresPerGroup) * 2 * sizeof(float);
    return RoundUp(bytes, 32);
}

// 多小组 + tiny reduce:把"组"放到向量 lane,一条 VF 批量算多组,摊薄 per-group 固定开销。
// 与 split_reduce 互斥(它要 A 轴欠并行 + 大 reduce;这里要 A 轴已满并行 + 极小 reduce)。
// v1 仅接管最高价值的清晰子集:shapeD==1(每组 1 通道→gamma/beta/quantScale per-group)+ hwNum 小 + 大通道(当前走慢的
// 1130)。
static bool MaybeSetManyTinyGroups(const gert::TilingContext* context, GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    uint64_t totalCore = compileInfo->totalCoreNum;
    int64_t aAxis = tilingData.get_shapeN() * tilingData.get_numGroups(); // N*num_groups
    int64_t shapeD = tilingData.get_shapeD();
    int64_t hwNum = tilingData.get_hwNum();
    int64_t elemNum = tilingData.get_elemNum();
    const int64_t TINY_MAX_ELEM = 32; // 每组元素数上限(超过则 per-group 开销不再主导,原路更划算)
    // 守卫:A 轴已能填满核(多组)+ 每组 1 通道 + reduce 极小 + 大通道(现走 generalized 慢路)
    if (aAxis <= 0 || (uint64_t)aAxis < totalCore || shapeD != 1 || hwNum <= 0 || elemNum > TINY_MAX_ELEM) {
        return false;
    }
    if ((uint64_t)tilingData.get_shapeC() <= MAX_CHANNEL_SIZE) {
        return false; // 小通道本就 fold 批量,不慢,不接管
    }
    tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_MANY_TINY_GROUPS));
    tilingData.set_groupInvCnt(1.0f / static_cast<float>(elemNum)); // 1/hwNum(shapeD==1)
    tilingData.set_ubSize(compileInfo->ubSizePlatForm);
    // realCoreNum/numPerCore/numLastCore 沿用 SetBlockTiling 按 A 轴切好的值(每核一段连续组),无需 workspace
    return true;
}

inline static ge::graphStatus GroupNormSiluQuantSetTilingData(gert::TilingContext* context,
                                                              GroupNormSiluQuantRegbaseTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckEmptyTensorParams(const gert::TilingContext* context)
{
    auto yOutputShape = context->GetInputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutputShape);
    auto yShapeSize = yOutputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(yShapeSize != 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "y", std::to_string(yShapeSize),
                                                          "y must be an empty tensor when x is empty"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4GroupNormSiluQuantRegBase(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start running Tiling4GroupNormSiluQuant.");
    // check input && attrs params
    OP_CHECK_IF((CheckInputParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((CheckAttrParams(context) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "AttrParams is invalid."), return ge::GRAPH_FAILED);
    auto xInputShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShape);
    uint64_t xShapeSize = xInputShape->GetStorageShape().GetShapeSize();
    if (xShapeSize == 0 && CheckEmptyTensorParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    GroupNormSiluQuantRegbaseTilingData tilingData;
    SetAttrParams(context, tilingData);
    SetTilingParams(context, tilingData);
    // block tiling
    SetBlockTiling(context, tilingData);
    // ub tiling
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE;
    SetTilingForRegbase(context, tilingData);
    uint64_t splitUserWs = MaybeSetSplitReduce(context, tilingData); // 可能覆盖为 split-reduce(1140)
    if (splitUserWs == 0) {
        MaybeSetManyTinyGroups(context, tilingData); // 可能覆盖为 many-tiny-groups(1150),与 split 互斥,无 workspace
    }
    // UB 记账兜底: 1100/1110/1120/1130 都靠 processSize 驱动搬运, 解不出正值说明 UB 装不下该形状,
    // 必须显式拒收 —— 不能把 0 或下溢值下发给内核(issue #21 就是下溢值被下发后 UB 越界)。
    // 1140/1150 自带切分, 不读 processSize; 空 Tensor 走独立分支。
    int64_t finalKey = tilingData.get_tilingKey();
    bool keyUsesProcessSize = finalKey !=
                                  static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_EMPTY_TENSOR) &&
                              finalKey !=
                                  static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_R_SPLIT_REDUCE) &&
                              finalKey !=
                                  static_cast<int64_t>(GroupNormSiluQuantRegbaseTilingKey::TILINGKEY_MANY_TINY_GROUPS);
    OP_CHECK_IF(xShapeSize != 0 && keyUsesProcessSize && tilingData.get_processSize() <= 0,
                OP_LOGE(context->GetNodeName(),
                        "UB budget exhausted: cannot solve a UB tiling for shapeC=%ld, shapeD=%ld, hwNum=%ld "
                        "(tilingKey=%ld, UB=%ld B). Reduce channel/quantScale size or split the input.",
                        tilingData.get_shapeC(), tilingData.get_shapeD(), tilingData.get_hwNum(), finalKey,
                        tilingData.get_ubSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GroupNormSiluQuantSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GroupNormSiluQuantSetTilingData set tiling data fail."),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(),
            "tilingData is numGroups:%ld, hwNum:%ld, elemNum:%ld, shapeN:%ld, shapeC:%ld, shapeD:%ld, realCoreNum:%ld, \
            numPerCore:%ld, numLastCore:%ld, processSize:%ld, loopNum:%ld, loopTail:%ld, innerLoopNum:%ld, \
            innerLoopTail:%ld, tilingKey:%ld, epsilon:%f, activateSilu:%ld, parallelN:%ld, ubSize:%ld, dichotomyAddPower:%ld, \
            dichotomyAddK:%ld, dichotomyAddLastNum:%ld, powerOfTwoForReduce:%ld",
            tilingData.get_numGroups(), tilingData.get_hwNum(), tilingData.get_elemNum(), tilingData.get_shapeN(),
            tilingData.get_shapeC(), tilingData.get_shapeD(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
            tilingData.get_numLastCore(), tilingData.get_processSize(), tilingData.get_loopNum(),
            tilingData.get_loopTail(), tilingData.get_innerLoopNum(), tilingData.get_innerLoopTail(),
            tilingData.get_tilingKey(), tilingData.get_epsilon(), tilingData.get_activateSilu(),
            tilingData.get_parallelN(), tilingData.get_ubSize(), tilingData.get_dichotomyAddPower(),
            tilingData.get_dichotomyAddK(), tilingData.get_dichotomyAddLastNum(), tilingData.get_powerOfTwoForReduce());

    // block dim, tilingKey
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize + splitUserWs;

    return ge::GRAPH_SUCCESS;
}

// arch35 独立 TilingParse: 只在 ascend950 编译(TILING_DIR 路由), 恒 regbase。
static ge::graphStatus TilingParse4GroupNormSiluQuantRegBase(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GroupNormSiluQuantRegbaseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    compileInfo->isRegbase = 1;
    uint32_t vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF((vectorLength <= FLOAT32_BYTES), OP_LOGE(context->GetNodeName(), "vector length is invalid."),
                return ge::GRAPH_FAILED);
    compileInfo->vectorLength = vectorLength;
    int32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF((blockSize <= 0), OP_LOGE(context->GetNodeName(), "block size is invalid."), return ge::GRAPH_FAILED);
    compileInfo->blockSizePlatform = blockSize;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSiluQuant)
    .Tiling(Tiling4GroupNormSiluQuantRegBase)
    .TilingParse<GroupNormSiluQuantRegbaseCompileInfo>(TilingParse4GroupNormSiluQuantRegBase);

} // namespace optiling
