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
 * \file dynamic_hi_f4_quant_tiling.cpp
 * \brief
 */
#include "dynamic_hi_f4_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/dynamic_hi_f4_quant_tilingdata.h"
#include <cmath>
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/dynamic_hi_f4_quant_struct.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicHiF4QuantOp;
namespace optiling {
constexpr int64_t SECOND_LAST = -2;
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 1;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t SIXTEEN = 16;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_128 = 128;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t DIGIT_512 = 512;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t BLOCK_PER_GROUP = 1;
constexpr int64_t UINT32_BYTES_SIZE = 4;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t BYTES_OF_TMP_TYPE = 1;
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;
constexpr size_t MAX_DIM_NUM = 7;

constexpr int64_t RESERVED_UB_SIZE = 4096;

constexpr int64_t HIFP4_ONE_GROUP_INPUT_BYTES = 64 * 2;   // only fp16 bf16 now
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_BYTES = 32;      // fp4: 64 * 0.5
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES = 4; // scale is fp32
// CCE 移植后 -1 轴中间结果记账(每 group 92B):
// maxper8 16B + max64Raw 2B + E8 2B + EaRec 2B + e1p8 4B + e1p16 2B + tmpScale 64B
constexpr int64_t HIFP4_ONE_GROUP_CCE_MID_UB_RES_BYTES = 92;
// CCE 移植后 -1 轴队列深度(kernel 侧 InitBuffer 与此一一对应): 输入 4 深环 / y+maxper4 共槽 2 深环 /
// scale 银行 8 槽(2 bank x 4 tile)
constexpr int64_t HIFP4_NEG1_INPUT_QUE_DEPTH = 4;
constexpr int64_t HIFP4_NEG1_OUT_QUE_DEPTH = 2;
constexpr int64_t HIFP4_NEG1_SCALE_QUE_DEPTH = 8;
constexpr int64_t HIFP4_ONE_GROUP_CCE_UB_BYTES = HIFP4_NEG1_INPUT_QUE_DEPTH * HIFP4_ONE_GROUP_INPUT_BYTES +
                                                 HIFP4_NEG1_OUT_QUE_DEPTH * HIFP4_ONE_GROUP_OUTPUT_BYTES +
                                                 HIFP4_NEG1_SCALE_QUE_DEPTH * HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES +
                                                 HIFP4_ONE_GROUP_CCE_MID_UB_RES_BYTES; // 每 64 个输入元素在全部 UB
                                                                                       // 缓冲中占用的字节数, 共 700B
constexpr int64_t HIFP4_CCE_MAX_TILE_GROUP_NUM = 512;             // tile 上限, 对齐 cce 128x256 整块
constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256; // tile 粒度按 256 group 对齐

// -2 轴基本块 [64 行 × 128 列 × 2B] 的精确 UB 占用(kernel 侧 Init 记账与此一一对应):
// x 32768(DB) + y 8192(DB, fp4 两元素一字节装箱 0.5B/元素) + scale 前/后槽各 512(DB)
// + maxPer4 4096 + maxPer8 2048 + e8 256 + tmpScale 4096 = 52480B, 等价每列每 64 行 410B
constexpr int64_t HIFP4_NEG2_BASIC_BLOCK_UB_BYTES = 52480;
// 单 tile 基本块预算钳制: 全档位任务粒度统一(128/256/512 列档 4 块, 384 列档 3 块), 小矩阵多核均衡更好,
// 且 tiling 期望值对 UB 取值不敏感(960DT 实芯净值 376KB raw=7 与 UT mock 248KB raw=4 钳制后同为 4)
constexpr int64_t HIFP4_NEG2_MAX_BLOCKS_PER_TILE = 4;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT, Ops::NN::DT_HIFLOAT4_SCALE};

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus DynamicHiF4QuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    tilingParams.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context_->GetNodeName(), "The attr axis is %ld", tilingParams.axis);
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    tilingParams.axis = tilingParams.axis < 0 ? tilingParams.axis : tilingParams.axis - xShape.GetDimNum();
    OP_CHECK_IF(tilingParams.axis != -1 && tilingParams.axis != -2,
                OP_LOGE(context_->GetNodeName(), "The axis should be -1/-2, please check."), return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    int64_t scaleAlg = 0;
    if (attrScaleAlg != nullptr) {
        scaleAlg = *attrScaleAlg;
    }
    OP_CHECK_IF((scaleAlg != 0 && scaleAlg != 1),
                OP_LOGE(context_->GetNodeName(), "invalid scale_alg:%ld; scale_alg should be 0 or 1", scaleAlg),
                return ge::GRAPH_FAILED);
    tilingParams.scaleAlg = scaleAlg;
    OP_LOGD(context_->GetNodeName(), "The attr scale_alg is %ld", scaleAlg);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::CheckDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
                        Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Output y's data type only support HIFLOAT4, but y is [%s], please check.",
                Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output scale's data type only support FLOAT32 currently, but scale is [%s], please check.",
                        Ops::Base::ToString(scaleDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDimNum() < 2 || xShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "The shape is invalid, Input x rank[%ld] should be in [2, 7], please check.",
                static_cast<int64_t>(xShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
                OP_LOGE(context_->GetNodeName(),
                        "When output y's data type is HIFLOAT4, the last axis must be a multiple of 64, please check."),
                return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output y[%s] must be same with shape of input x[%s], please check.",
                        Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
                return ge::GRAPH_FAILED);

    auto ScaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ScaleShapePtr);
    auto ScaleShape = ScaleShapePtr->GetStorageShape();

    auto newScaleShape = xShape;
    int64_t lastDim = xShape.GetDim(xShape.GetDimNum() - 1);
    int64_t nonLastDim = xShape.GetDim(xShape.GetDimNum() - 2);
    if (tilingParams.axis == -1) {
        newScaleShape.SetDim(xShape.GetDimNum() - 1,
                             Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_64));
        newScaleShape.AppendDim(DIGIT_ONE);
        OP_CHECK_IF(newScaleShape != ScaleShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of output scale [%s] is incorrect, correct shape is [%s], please check.",
                            Shape2String(ScaleShape).c_str(), Shape2String(newScaleShape).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            lastDim % DIGIT_64 != 0,
            OP_LOGE(context_->GetNodeName(),
                    "When the quantization axis is the last axis, the size of the last axis must be a multiple of 64."
                    "The current last axis size is [%ld] please check.",
                    lastDim),
            return ge::GRAPH_FAILED);
    } else if (tilingParams.axis == SECOND_LAST) {
        newScaleShape.SetDim(xShape.GetDimNum() - DIGIT_TWO,
                             Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), DIGIT_64));
        newScaleShape.AppendDim(DIGIT_ONE);
        OP_CHECK_IF(newScaleShape != ScaleShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of output scale [%s] is incorrect, correct shape is [%s], please check.",
                            Shape2String(ScaleShape).c_str(), Shape2String(newScaleShape).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            lastDim % DIGIT_64 != 0 || nonLastDim % DIGIT_64 != 0,
            OP_LOGE(
                context_->GetNodeName(),
                "When the quantization axis is axis -2, the sizes of both axis -2 and the last axis must be multiples "
                "of 64."
                "The current axis -1 is [%ld] and axis -2 is [%ld], please check.",
                lastDim, nonLastDim),
            return ge::GRAPH_FAILED);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 合轴
ge::graphStatus DynamicHiF4QuantTiling::MergeAxis()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    for (size_t i = 0; i < xShape.GetDimNum() - DIGIT_TWO; i++) {
        tilingParams.dim0 *= xShape.GetDim(i);
    }
    tilingParams.dimNeg2 = xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO);
    tilingParams.dimNeg1 = xShape.GetDim(xShape.GetDimNum() - 1);

    return ge::GRAPH_SUCCESS;
}

// 切块: 横竖自适应 —— 优先横切(列向加宽吃行长突发), 行长不足再竖切(行向堆叠 64 行 scale 组)。
// B_eff = min(ubSize/52480, 4): 基本块预算统一钳制; W = min(ceil128(dimNeg1), 512, B_eff*128)
// (ceil 上取整保证 dimNeg1≤512 时单列块 + GM srcStride=0 纯连续流; 512 封顶为 MTE 单行 1KB 长突发;
// B_eff*128 防小 UB 平台超预算); G = B_eff/(W/128) 把剩余预算竖切堆组, 至少 1
void DynamicHiF4QuantTiling::SplitCore(int64_t blockSize)
{
    const int64_t blockBudget = std::min(tilingParams.ubSize / HIFP4_NEG2_BASIC_BLOCK_UB_BYTES,
                                         HIFP4_NEG2_MAX_BLOCKS_PER_TILE);
    tilingParams.blockW = std::min(std::min(Ops::Base::CeilDiv(tilingParams.dimNeg1, DIGIT_128) * DIGIT_128, DIGIT_512),
                                   blockBudget * DIGIT_128);
    tilingParams.groupPerUb = std::max(blockBudget / (tilingParams.blockW / DIGIT_128), DIGIT_ONE);
    tilingParams.splitBlockH = blockSize * tilingParams.groupPerUb;
    tilingParams.dimNeg2SplitBlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.splitBlockH);
    tilingParams.dimNeg1BlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockW);
    tilingParams.totalTaskNum = tilingParams.dim0 * tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
}

ge::graphStatus DynamicHiF4QuantTiling::SetTilingParamsNeg2()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling SetTilingParams.");

    MergeAxis();

    OP_CHECK_IF(
        tilingParams.ubSize < HIFP4_NEG2_BASIC_BLOCK_UB_BYTES,
        OP_LOGE(context_->GetNodeName(),
                "Ub size %ld is too small for hif4 quant axis -2, at least %ld bytes are required, please check.",
                tilingParams.ubSize, HIFP4_NEG2_BASIC_BLOCK_UB_BYTES),
        return ge::GRAPH_FAILED);

    tilingParams.blockSize = BLOCK_SIZE;
    SplitCore(tilingParams.blockSize);
    tilingParams.dimNeg2Tail = tilingParams.dimNeg2 % tilingParams.splitBlockH == 0 ?
                                   tilingParams.splitBlockH :
                                   tilingParams.dimNeg2 % tilingParams.splitBlockH;
    tilingParams.dimNeg1Tail = tilingParams.dimNeg1 % tilingParams.blockW == 0 ?
                                   tilingParams.blockW :
                                   tilingParams.dimNeg1 % tilingParams.blockW;

    tilingParams.dimNeg1IsPad = tilingParams.dimNeg1 % tilingParams.blockSize == 0 ? 0 : 1; // 0
    tilingParams.dimNeg1IsOdd = Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockSize) % DIGIT_TWO == 0 ?
                                    0 :
                                    1; // 尾轴scale奇数列补0 0
    tilingParams.dimNeg2IsOdd = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.blockSize) % DIGIT_TWO == 0 ?
                                    0 :
                                    1; // 非尾轴scale奇数行补0  0

    tilingParams.blockCountPerBatch = tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
    tilingParams.scale1ColCountPerBatch = Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockSize);
    tilingParams.scale2RowCountPerBatch = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.blockSize);

    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.totalTaskNum);
    tilingParams.blockPerHeadCore = Ops::Base::CeilDiv(tilingParams.totalTaskNum, tilingParams.totalCoreNum);
    tilingParams.blockPerTailCore = tilingParams.totalTaskNum / tilingParams.totalCoreNum;
    tilingParams.headCoreNum = tilingParams.totalTaskNum % tilingParams.totalCoreNum;

    return ge::GRAPH_SUCCESS;
}

int64_t DynamicHiF4QuantTiling::HiF4GetUbFormerNeg1()
{
    // CCE 移植后的 -1 轴固定缓冲结构记账(每 group 700B, kernel 侧 InitBuffer 与此一一对应):
    // 输入 4 深环 128B*4 + y/maxper4 共槽 2 深环 32B*2 + scale 银行 8 槽 4B*8 + 中间结果 92B;
    // 960DT 384KB UB: (384K-4K)/700=555, 256 group 对齐后取 512, 与 cce 128x256 整 tile 一致
    int64_t ubCap = (tilingParams.ubSize - RESERVED_UB_SIZE) / HIFP4_ONE_GROUP_CCE_UB_BYTES;
    int64_t ubFormer = ubCap / HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM *
                       HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM; // 256 group 对齐
    return std::min(ubFormer, HIFP4_CCE_MAX_TILE_GROUP_NUM);
}

ge::graphStatus DynamicHiF4QuantTiling::SetTilingParamsNeg1()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling SetTilingParamsNeg1.");
    const gert::StorageShape* xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    uint64_t totalshape = xShape.GetShapeSize();

    // -1 轴按 row-major 展平流切定长 tile(借鉴 cce :330-344): groupNum=全局总 group 数,
    // ubFormer=单 tile group 数(960DT 为 512), 全局尾 tile 的有效 group 数由 kernel 派生
    tilingParams.groupNum = static_cast<int64_t>(Ops::Base::CeilDiv(totalshape, static_cast<uint64_t>(DIGIT_64)));
    tilingParams.ubFormer = HiF4GetUbFormerNeg1();
    OP_CHECK_IF(tilingParams.ubFormer <= 0,
                OP_LOGE(context_->GetNodeName(),
                        "Ub size %ld is too small for hif4 quant, at least %ld bytes are required, please check.",
                        tilingParams.ubSize,
                        RESERVED_UB_SIZE + HIFP4_ONE_GROUP_CCE_UB_BYTES * HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM),
                return ge::GRAPH_FAILED);

    // 按 tile 均分核(借鉴 cce :334-344): 头部 headCoreNum 个核各多 1 个 tile, 各核 tile 区间连续衔接;
    // 小 shape(tile 数 < 核数)按 tile 数收缩用核, 与产品线行为一致
    tilingParams.totalTiles = Ops::Base::CeilDiv(tilingParams.groupNum, tilingParams.ubFormer);
    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.totalTiles);
    tilingParams.blockPerHeadCore = Ops::Base::CeilDiv(tilingParams.totalTiles, tilingParams.usedCoreNum);
    tilingParams.blockPerTailCore = tilingParams.totalTiles / tilingParams.usedCoreNum;
    tilingParams.headCoreNum = tilingParams.totalTiles % tilingParams.usedCoreNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling DoTiling.");

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
                return ge::GRAPH_FAILED);

    if (tilingParams.axis == -1) {
        OP_CHECK_IF(SetTilingParamsNeg1() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "The tiling params of axis -1 set failed."),
                    return ge::GRAPH_FAILED);
        SetTilingKey();
        SetTilingDataNeg1();
    } else {
        SetTilingParamsNeg2();
        SetTilingKey();
        SetTilingDataNeg2();
    }
    PrintTilingData();

    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void DynamicHiF4QuantTiling::SetTilingKey()
{
    if (tilingParams.axis == -1) {
        quantAxis_ = TPL_QUANT_TAIL_AXIS;
    } else {
        quantAxis_ = TPL_QUANT_NO_TAIL_AXIS;
    }
    tilingParams.tilingKey = GET_TPL_TILING_KEY(quantAxis_);
    context_->SetTilingKey(tilingParams.tilingKey);
}

void DynamicHiF4QuantTiling::SetTilingDataNeg1()
{
    // CCE 移植后 -1 轴仅传递 7 个字段: 满块/尾块循环数由 kernel 按 tile group 数以编译期常数派生,
    // 其余旧 -1 字段保持 0(tilingData 零初始化)
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.groupNum = tilingParams.groupNum;                 // 全局总 group 数
    tilingData.alignGroupNum = tilingParams.ubFormer;            // 单 tile group 数(256 对齐, 960DT=512)
    tilingData.blockPerHeadCore = tilingParams.blockPerHeadCore; // 满载核 tile 数
    tilingData.blockPerTailCore = tilingParams.blockPerTailCore; // 尾核 tile 数
    tilingData.headCoreNum = tilingParams.headCoreNum;           // 满载核数量
    tilingData.scaleAlg = tilingParams.scaleAlg;
}

void DynamicHiF4QuantTiling::SetTilingDataNeg2()
{
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.dim0 = tilingParams.dim0;
    tilingData.dimNeg2 = tilingParams.dimNeg2;
    tilingData.dimNeg1 = tilingParams.dimNeg1;
    tilingData.blockW = tilingParams.blockW;
    tilingData.tilingKey = tilingParams.tilingKey;
    tilingData.splitBlockH = tilingParams.splitBlockH;
    tilingData.dimNeg2Tail = tilingParams.dimNeg2Tail;
    tilingData.dimNeg1Tail = tilingParams.dimNeg1Tail;
    tilingData.dimNeg2SplitBlockNum = tilingParams.dimNeg2SplitBlockNum;
    tilingData.dimNeg1BlockNum = tilingParams.dimNeg1BlockNum;
    tilingData.blockPerHeadCore = tilingParams.blockPerHeadCore;
    tilingData.blockPerTailCore = tilingParams.blockPerTailCore;
    tilingData.headCoreNum = tilingParams.headCoreNum;
    tilingData.dimNeg2IsOdd = tilingParams.dimNeg2IsOdd;
    tilingData.dimNeg1IsOdd = tilingParams.dimNeg1IsOdd;
    tilingData.dimNeg1IsPad = tilingParams.dimNeg1IsPad;
    tilingData.blockCountPerBatch = tilingParams.blockCountPerBatch;
    tilingData.scale1ColCountPerBatch = tilingParams.scale1ColCountPerBatch;
    tilingData.scale2RowCountPerBatch = tilingParams.scale2RowCountPerBatch;
    tilingData.scaleAlg = tilingParams.scaleAlg;
}

void DynamicHiF4QuantTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "TilingData totalCoreNum: %ld, usedCoreNum: %ld, "
            "dim0: %ld, dimNeg2: %ld, dimNeg1: %ld, blockW: %ld, "
            "tilingKey: %ld, splitBlockH: %ld, dimNeg2Tail: %ld, "
            "dimNeg1Tail: %ld, dimNeg2SplitBlockNum: %ld, dimNeg1BlockNum: %ld, "
            "blockPerHeadCore: %ld, blockPerTailCore: %ld, headCoreNum: %ld, "
            "dimNeg2IsOdd: %ld, dimNeg1IsOdd: %ld, dimNeg1IsPad: %ld, blockCountPerBatch: %ld, "
            "scale1ColCountPerBatch: %ld, scale2RowCountPerBatch: %ld, "
            "groupNum: %ld, alignGroupNum: %ld, scaleAlg: %ld, ",
            tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.dim0, tilingData.dimNeg2, tilingData.dimNeg1,
            tilingData.blockW, tilingData.tilingKey, tilingData.splitBlockH, tilingData.dimNeg2Tail,
            tilingData.dimNeg1Tail, tilingData.dimNeg2SplitBlockNum, tilingData.dimNeg1BlockNum,
            tilingData.blockPerHeadCore, tilingData.blockPerTailCore, tilingData.headCoreNum, tilingData.dimNeg2IsOdd,
            tilingData.dimNeg1IsOdd, tilingData.dimNeg1IsPad, tilingData.blockCountPerBatch,
            tilingData.scale1ColCountPerBatch, tilingData.scale2RowCountPerBatch, tilingData.groupNum,
            tilingData.alignGroupNum, tilingData.scaleAlg);
}

static ge::graphStatus TilingForDynamicHiF4Quant(gert::TilingContext* context_)
{
    OP_LOGD("DynamicHiF4QuantTiling", "Enter TilingForDynamicHiF4QuantTiling");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DynamicHiF4QuantTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    optiling::DynamicHiF4QuantTiling hifloat4Tiling(context_);
    return hifloat4Tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicHiF4Quant(gert::TilingParseContext* context_)
{
    OP_LOGD("DynamicHiF4QuantTiling", "Enter TilingPrepareForDynamicHiF4QuantTiling");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DynamicHiF4QuantTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicHiF4Quant)
    .Tiling(TilingForDynamicHiF4Quant)
    .TilingParse<DynamicHiF4QuantCompileInfo>(TilingPrepareForDynamicHiF4Quant);
} // namespace optiling
