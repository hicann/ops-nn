/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_max_v2_tiling_arch35.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/quant_max_v2_tiling_data.h"
#include "../../op_kernel/arch35/quant_max_v2_struct.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace Ops::Base;
using namespace QuantMaxV2Op;
using namespace gert;

namespace optiling {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SCALE_INDEX = 1;
constexpr size_t INPUT_GROUP_LIST_INDEX = 2;
constexpr size_t ATTR_ROUND_MODE_INDEX = 0;
constexpr size_t ATTR_DST_TYPE_INDEX = 1;
constexpr size_t ATTR_DST_TYPE_MAX_INDEX = 2;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_AMAX_INDEX = 1;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 2;
constexpr size_t BYTES_OF_INPUT_SCALE_TYPE = 4;
constexpr size_t BYTES_OF_INPUT_GROUP_LIST_TYPE = 8;
constexpr size_t BYTES_OF_TMP_AMAX_TYPE = 4;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t VREG_BYTES = 256;
constexpr int64_t FLOAT_ALIGN_ELEM = 8; // groupStride 按 8 个 float 对齐。
constexpr int64_t MAX_GROUP_NUM = 384;
constexpr float DST_TYPE_MAX_LIMIT = 32768.0f;
constexpr size_t FIRST_DIM = 0;
constexpr int64_t DEFAULT_BASE_LEN = 128;
constexpr int64_t BUFF_NUM = 2;
constexpr size_t SYNC_RESERVE_BYTES = 1024 * 1024;
constexpr int64_t MERGE_SLAB_WORKSPACE_FLOATS = 8; // 分层归并的 [8][groupStride] 暂存区。
constexpr int64_t MIN_ELEMENTS_PER_VECTOR_CORE = 16384;

// tiling 主入口：平台参数 → 读取输入/输出/属性 → 合法性校验（dtype/attr/shape）→ 计算切分 → 写出 TilingData 与
// TilingKey。
ge::graphStatus QuantMaxV2Regbase::DoTiling()
{
    // 下层函数已通过 OP_LOGE_FOR_INVALID_* 上报具体错误；这里只记录并返回原错误码，避免重复上报。
    int32_t ret = static_cast<int32_t>(GetPlatform());
    OP_CHECK_IF(ret != 0,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "DoQuantMaxV2Tiling GetPlatform Failed, rc:%d.", ret),
                return static_cast<ge::graphStatus>(ret));
    ret = static_cast<int32_t>(GetOpParam());
    OP_CHECK_IF(ret != 0,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "DoQuantMaxV2Tiling GetOpParam Failed, rc:%d.", ret),
                return static_cast<ge::graphStatus>(ret));

    ret = static_cast<int32_t>(CalcTiling());
    OP_CHECK_IF(ret != 0,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "DoQuantMaxV2Tiling CalcTiling Failed, rc:%d.", ret),
                return static_cast<ge::graphStatus>(ret));
    CalcTilingKey();
    return WriteTilingData();
}

// 获取平台参数：AIV 核数与 UB 大小（经 PlatformAscendC）。
ge::graphStatus QuantMaxV2Regbase::GetPlatform()
{
    OP_LOGD(context_->GetNodeName(), "Enter arch3510 QuantMaxV2Tiling");
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((static_cast<int32_t>(coreNum) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF((static_cast<int64_t>(ubSize) <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    coreNum_ = static_cast<int64_t>(coreNum);
    ubSize_ = ubSize;

    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 校验 dtype：x/amax 取值与匹配关系、y 与 dstType 一致、scale 为 FP32、groupList 为 INT64。
ge::graphStatus QuantMaxV2Regbase::CheckDtype()
{
    auto xInputDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    xDtype_ = xInputDesc->GetDataType();
    OP_CHECK_IF(xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_FLOAT && xDtype_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                    "The dtype of x must be within the range [DT_FLOAT16, DT_FLOAT, DT_BF16]"),
                return ge::GRAPH_FAILED);

    auto scaleInputDesc = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX);
    hasScale_ = scaleInputDesc != nullptr;
    if (hasScale_) {
        scaleDtype_ = scaleInputDesc->GetDataType();
        OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale",
                                                          Ops::Base::ToString(scaleDtype_).c_str(),
                                                          "The dtype of scale must be FLOAT"),
                    return ge::GRAPH_FAILED);
    }
    auto groupListInputDesc = context_->GetOptionalInputDesc(INPUT_GROUP_LIST_INDEX);
    hasGroupList_ = groupListInputDesc != nullptr;
    if (hasGroupList_) {
        groupListDtype_ = groupListInputDesc->GetDataType();
        OP_CHECK_IF(groupListDtype_ != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "group_list",
                                                          Ops::Base::ToString(groupListDtype_).c_str(),
                                                          "The dtype of group_list must be INT64"),
                    return ge::GRAPH_FAILED);
    }
    auto yOutputDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputDesc);
    yDtype_ = yOutputDesc->GetDataType();
    OP_CHECK_IF(yDtype_ != ge::DT_HIFLOAT8 && yDtype_ != ge::DT_FLOAT8_E5M2 && yDtype_ != ge::DT_FLOAT8_E4M3FN,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str(),
                    "The dtype of y must be within the range [DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]"),
                return ge::GRAPH_FAILED);

    auto amaxOutputDesc = context_->GetOutputDesc(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutputDesc);
    amaxDtype_ = amaxOutputDesc->GetDataType();
    OP_CHECK_IF(
        amaxDtype_ != xDtype_,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "amax", Ops::Base::ToString(amaxDtype_).c_str(),
                                              "The dtype of amax must be the same as x"),
        return ge::GRAPH_FAILED);

    auto yScaleOutputDesc = context_->GetOutputDesc(OUTPUT_Y_SCALE_INDEX);
    // y_scale 为可选输出：静态模式不输出，提供时类型须为 FLOAT。
    if (yScaleOutputDesc != nullptr) {
        OP_CHECK_IF(yScaleOutputDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y_scale",
                                                          Ops::Base::ToString(yScaleOutputDesc->GetDataType()).c_str(),
                                                          "The dtype of y_scale must be FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    dtypeSizeX_ = ge::GetSizeByDataType(xDtype_);
    dtypeSizeY_ = ge::GetSizeByDataType(yDtype_);

    return ge::GRAPH_SUCCESS;
}

// 把字符串 roundMode 映射为枚举（rint/round/hybrid），非法值报错。
RoundMode QuantMaxV2Regbase::GetRoundMode(const std::string& roundMode)
{
    if (dstType_ == ge::DT_FLOAT8_E5M2 || dstType_ == ge::DT_FLOAT8_E4M3FN) {
        if (roundMode == "rint") {
            return RoundMode::MODE_RINT;
        }
        errorMsg_ = "round_mode only supports 'rint' for float8_e5m2/float8_e4m3fn.";
        return RoundMode::MODE_UNDEFINED;
    }

    if (dstType_ == ge::DT_HIFLOAT8) {
        if (roundMode == "round") {
            return RoundMode::MODE_ROUND;
        } else if (roundMode == "hybrid") {
            return RoundMode::MODE_HYBRID;
        }
        errorMsg_ = "round_mode only supports 'round' and 'hybrid' for hifloat8.";
        return RoundMode::MODE_UNDEFINED;
    }
    return RoundMode::MODE_UNDEFINED;
}

// 校验属性：dstType 取值 {34,35,36}、roundMode 与 dstType 匹配（HiF8=round/hybrid，FP8=rint）、dstTypeMax 为 [0,32768]
// 内有限值。
ge::graphStatus QuantMaxV2Regbase::CheckAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* roundMode = attrs->GetAttrPointer<char>(ATTR_ROUND_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, roundMode);
    // Int 属性占 8 字节，必须用 int64_t 读取，与算子定义和形状推导保持一致。
    const int64_t* dstType = attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstType);
    dstType_ = *dstType;

    if (dstType_ != ge::DT_HIFLOAT8 && dstType_ != ge::DT_FLOAT8_E5M2 && dstType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "dstType", ToString(static_cast<ge::DataType>(dstType_)).c_str(),
            "dstType is invalid, must be DT_HIFLOAT8/DT_FLOAT8_E5M2/DT_FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (dstType_ != yDtype_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dstType",
                                              ToString(static_cast<ge::DataType>(dstType_)).c_str(),
                                              "dstType must be equal to output y dtype");
        return ge::GRAPH_FAILED;
    }

    const float* dstTypeMax = attrs->GetAttrPointer<float>(ATTR_DST_TYPE_MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dstTypeMax);
    dstTypeMax_ = *dstTypeMax;
    OP_CHECK_IF(dstTypeMax_ < 0.0f || dstTypeMax_ > DST_TYPE_MAX_LIMIT || !std::isfinite(dstTypeMax_),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type_max",
                                                      std::to_string(dstTypeMax_).c_str(),
                                                      "dst_type_max must be finite and in [0.0, 32768.0]"),
                return ge::GRAPH_FAILED);
    // 与 golden 一致：dst_type_max 仅对 HIFLOAT8 生效；FP8 使用各自的最大有限值。静态模式直接使用输入 scale，不消费
    // dstTypeMax。
    if (dstType_ == ge::DT_FLOAT8_E5M2) {
        dstTypeMax_ = 57344.0f; // 1.75 * 2^15
    } else if (dstType_ == ge::DT_FLOAT8_E4M3FN) {
        dstTypeMax_ = 448.0f; // 1.75 * 2^8
    }
    std::string roundModeStr = roundMode;
    roundMode_ = GetRoundMode(roundModeStr);
    OP_CHECK_IF(
        (roundMode_ == RoundMode::MODE_UNDEFINED),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "roundMode", roundMode, errorMsg_.c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验 shape：x 为 2~8 维；scale/groupList 为 1 维且元素数=numGroups（1~384）；y 与 x 同形；amax/yScale
// 形状正确；两模式输入契约（动态缺 scale 合法、静态 scale 必给）。
ge::graphStatus QuantMaxV2Regbase::CheckShape(const gert::Shape& xShape, const gert::Shape* scaleShape,
                                              const gert::Shape* groupListShape, const gert::Shape& yShape,
                                              const gert::Shape& amaxShape, const gert::Shape* yScaleShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() < 2 || xShape.GetDimNum() > 8,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                         std::to_string(xShape.GetDimNum()).c_str(),
                                                         "The dim num of x must be in range [2, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y", Ops::Base::ToString(yShape).c_str(),
                                               "The shape of y must be the same as x"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape != nullptr && groupListShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "group_list",
                                                         std::to_string(groupListShape->GetDimNum()).c_str(),
                                                         "group_list must be rank one"),
                return ge::GRAPH_FAILED);
    const int64_t groups = groupListShape == nullptr ? 1 : groupListShape->GetDim(0);
    OP_CHECK_IF(groups < 1 || groups > MAX_GROUP_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "groups", std::to_string(groups).c_str(),
                                                      "group count must be in [1, 384]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleShape != nullptr && scaleShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "scale",
                                                         std::to_string(scaleShape->GetDimNum()).c_str(),
                                                         "scale must be rank one [num_groups]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleShape != nullptr && scaleShape->GetDim(0) != groups,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "scale",
                                                          std::to_string(scaleShape->GetDim(0)).c_str(),
                                                          "the size of scale must equal the group count"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(amaxShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "amax",
                                                         std::to_string(amaxShape.GetDimNum()).c_str(),
                                                         "amax must be rank one [num_groups]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(amaxShape.GetDim(0) != groups,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "amax",
                                                          std::to_string(amaxShape.GetDim(0)).c_str(),
                                                          "the size of amax must equal the group count"),
                return ge::GRAPH_FAILED);
    // 动态模式必须提供 y_scale；静态模式可以不提供。
    OP_CHECK_IF(
        yScaleShape == nullptr ? !hasScale_ :
                                 (yScaleShape->GetDimNum() != 1 || yScaleShape->GetDim(0) != (hasScale_ ? 0 : groups)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "y_scale",
                                              yScaleShape == nullptr ? "nullptr" : "[wrong size]",
                                              "y_scale must be [num_groups] in dynamic mode and absent in static mode"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 记录参与 tiling 的输入 shape（合并到 xInputShape_ 供切分使用）。
void QuantMaxV2Regbase::MergeInputShape(const gert::Shape& input)
{
    int64_t shape0 = 1;
    int64_t shape1 = 1;

    for (size_t idx = 1; idx < input.GetDimNum(); ++idx) {
        shape1 = shape1 * input.GetDim(idx);
    }

    dim1_ = shape1;

    for (size_t idx = 0; idx < input.GetDimNum(); ++idx) {
        shape0 = shape0 * input.GetDim(idx);
    }

    xInputShape_.SetDimNum(1);
    xInputShape_.SetDim(FIRST_DIM, shape0);
    OP_LOGI(context_->GetNodeName(), "merged shape:%ld", shape0);
}

// 读取算子输入/输出/属性：dtype、shape、roundMode/dstType/dstTypeMax，并标记 hasScale/hasGroupList 场景。
ge::graphStatus QuantMaxV2Regbase::GetOpParam()
{
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    const gert::Shape& xInputShape = Ops::Base::EnsureNotScalar(xInput->GetStorageShape());

    auto scaleInput = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    auto groupListInput = context_->GetOptionalInputShape(INPUT_GROUP_LIST_INDEX);
    const gert::Shape* scaleInputShape = scaleInput == nullptr ? nullptr : &scaleInput->GetStorageShape();
    const gert::Shape* groupListInputShape = groupListInput == nullptr ? nullptr : &groupListInput->GetStorageShape();

    auto yOutput = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutput);
    const gert::Shape& yOutputShape = Ops::Base::EnsureNotScalar(yOutput->GetStorageShape());

    auto amaxOutput = context_->GetOutputShape(OUTPUT_AMAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, amaxOutput);
    const gert::Shape& amaxOutputShape = Ops::Base::EnsureNotScalar(amaxOutput->GetStorageShape());
    auto yScaleOutput = context_->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    const gert::Shape* yScaleOutputShape = yScaleOutput == nullptr ?
                                               nullptr :
                                               &Ops::Base::EnsureNotScalar(yScaleOutput->GetStorageShape());

    int64_t xSizeNum = xInputShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        actualCoreNum_ = 0;
    }

    OP_CHECK_IF((CheckDtype() != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "check input/output dtype failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckAttrs() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "op attrs is invalid."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((CheckShape(xInputShape, scaleInputShape, groupListInputShape, yOutputShape, amaxOutputShape,
                            yScaleOutputShape) != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "input/output shape is invalid."), return ge::GRAPH_FAILED);

    MergeInputShape(xInputShape);

    return ge::GRAPH_SUCCESS;
}

// 计算核内 tile 大小 ubFactor（每核把分到的元素段再切成 ubFactor 大小的块，逐块搬入 UB 计算）。
// 主公式：ubFactor = ⌊(availableUb - tmpBuffer) / ubPerElement⌋ 向下对齐到 alignElem。
//   availableUb  = ubSize - reserveUb（UB 总量扣平台保留 2KB）；
//   ubPerElement = 4*sizeof(x) + yNum*sizeof(y)，一个元素在 UB 里的输入×缓冲数 + 输出×缓冲数开销，
//                  yNum 静态为 2（X4/Y2）、动态为 3（X4/Y3），分别计算避免静态多预留输出缓冲；
//   alignElem    = lcm(32/sizeof(x), 32/sizeof(y))，x、y 同时 32B 对齐的最小元素数。
// tmpBuffer（常驻缓冲逐项）：
//   scaleSize（G×4B 对齐 32B）+ groupListSize（G×8B 对齐 32B，分组才有）+ maxSize（G×4B 对齐 32B，
//   本核逐组最大值）+ groupMaxSize（跨核归并矩阵 = mergeRows×groupStride×4B + 256B VF 余量，
//   groupStride = G 向上对齐到 8；mergeRows 静态且 G>=64 时取 max(ceil(cores/8), 8)——分层归并
//   一阶段每核只读片段行（ceil(cores/8)），二阶段归并核读全部有效片段行（<= 8 片），队列须同时
//   容纳两者；否则取全核数）+ amaxSize（G×4B 对齐 32B）+（动态模式再加一份 scaleSize 作 y_scale
//   输出缓冲）。
// 三个分支修正：
//   1) 静态 G>=64 分层：mergeRows 用 max(ceil(cores/8), 8) 而非全核数，归并队列按实际读取行数收窄
//      （g384 静态由此多出约 50% 空间给 tile）；下界 8 保证二阶段归并核读全部片段行不越界；
//   2) 动态 bf16：ubFactor 封顶 8192（限制单次 MTE2 搬运长度，4 输入缓冲覆盖搬运延迟），并进入
//      不动点迭代——tile 变小则 bf16 位级最大值 16 位暂存区 tileBufferSize =
//      ceil(maxTileCount/128)×128×2B（maxTileCount = ceil(blockFactor/ubFactor) + G 段数）变大，
//      扣除后再算，直到 nextUbFactor == ubFactor 收敛（<=32 轮）；
//   3) 对齐取整在初值与迭代每轮均执行。
// 示例（静态 bf16 G=384、64 核）：tmpBuffer≈19.8KB、ubPerElement=10B，ubFactor≈(233K-19.8K)/10≈22.9K。
ge::graphStatus QuantMaxV2Regbase::CalcUbFactor()
{
    int64_t availableUb = static_cast<int64_t>(ubSize_) - reserveUb_;
    // 可选输入的形状指针和描述符指针相互独立，解引用前单独检查。
    auto groupListShape = hasGroupList_ ? context_->GetOptionalInputShape(INPUT_GROUP_LIST_INDEX) : nullptr;
    OP_CHECK_IF(hasGroupList_ && groupListShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "group_list shape is nullptr."), return ge::GRAPH_FAILED);
    int64_t numGroupsLocal = groupListShape != nullptr ? groupListShape->GetStorageShape().GetDim(0) : 1;
    numGroups_ = numGroupsLocal;

    int64_t scaleSize = ((numGroupsLocal * BYTES_OF_INPUT_SCALE_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t groupListSize = ((numGroupsLocal * BYTES_OF_INPUT_GROUP_LIST_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) *
                            ALIGN_BYTES;
    int64_t maxSize = ((numGroupsLocal * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t groupStride = ((numGroupsLocal + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM) * FLOAT_ALIGN_ELEM;
    // 静态分组的分层归并最多读取 ceil(cores/8) 行；动态归并仍读取完整矩阵。UB
    // 预算与各模板实际分配保持一致，核数保守取总核数。
    constexpr int64_t MERGE_SLAB_COUNT_HOST = 8;
    constexpr int64_t MERGE_SLAB_MIN_GROUPS_HOST = 64;
    const bool mergeHierarchicalHost = hasScale_ && numGroupsLocal >= MERGE_SLAB_MIN_GROUPS_HOST &&
                                       coreNum_ > MERGE_SLAB_COUNT_HOST;
    // 分层归并的归并队列必须同时容纳两个阶段的读取行数：一阶段每核读 ceil(cores/8) 行，
    // 二阶段归并核读 ceil(cores/ceil(cores/8)) 行（恒 <= MERGE_SLAB_COUNT）。取两者上界，
    // 与 kernel Init 的分配公式保持同源。
    const int64_t slabRowsHost = (coreNum_ + MERGE_SLAB_COUNT_HOST - 1) / MERGE_SLAB_COUNT_HOST;
    const int64_t mergeRows = mergeHierarchicalHost ? std::max(slabRowsHost, MERGE_SLAB_COUNT_HOST) : coreNum_;
    int64_t groupMaxSize = mergeRows * groupStride * BYTES_OF_TMP_AMAX_TYPE + VREG_BYTES;
    int64_t amaxSize = ((numGroupsLocal * BYTES_OF_TMP_AMAX_TYPE + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    int64_t tmpBuffer = scaleSize + groupListSize + maxSize + groupMaxSize + amaxSize + (hasScale_ ? 0 : scaleSize);

    // 动态模板使用 4 个输入、3 个输出缓冲；静态模板使用 4 个输入、2
    // 个输出缓冲，延后一块写回。分别计算预算，避免静态路径多预留一个输出缓冲。
    const int64_t xBufferNum = 4;
    const int64_t yBufferNum = hasScale_ ? 2 : 3;
    const int64_t ubPerElement = xBufferNum * dtypeSizeX_ + yBufferNum * dtypeSizeY_;
    OP_CHECK_IF((tmpBuffer >= availableUb), OP_LOGE(context_->GetNodeName(), "UB allocation overflow."),
                return ge::GRAPH_FAILED);
    ubFactor_ = (availableUb - tmpBuffer) / ubPerElement;
    int64_t alignElemX = ALIGN_BYTES / dtypeSizeX_;
    int64_t alignElemY = ALIGN_BYTES / dtypeSizeY_;
    int64_t alignElem = std::lcm(alignElemX, alignElemY);
    ubFactor_ = (ubFactor_ / alignElem) * alignElem;

    bf16TileMaxBufferSize_ = 0;
    // 仅动态 bf16 走位级暂存链（首遍逐 tile 16 位最大值），需要封顶 ubFactor 并联立求解暂存区；
    // 静态路与其它 dtype 无此链，ubFactor 维持通用预算。
    if (!hasScale_ && xDtype_ == ge::DT_BF16) {
        // 限制单次 MTE2 搬运长度，使四个输入缓冲覆盖搬运延迟；G=1 时组段额外开销退化为一个尾块，与 kernel
        // 暂存区分配一致。
        constexpr int64_t BF16_DYNAMIC_TILE_CAP = 8192;
        ubFactor_ = std::min(ubFactor_, BF16_DYNAMIC_TILE_CAP);
        const int64_t b16VectorElements = VREG_BYTES / sizeof(uint16_t);
        for (int32_t iter = 0; iter < 32; ++iter) {
            // blockFactor_ 已是单核分配到的展平元素个数。
            const int64_t maxCoreElements = blockFactor_;
            const int64_t maxGroupSegments = (numGroups_ < blockFactor_) ? numGroups_ : blockFactor_;
            const int64_t maxTileCount = (maxCoreElements + ubFactor_ - 1) / ubFactor_ + maxGroupSegments;
            int64_t tileCapacity = ((maxTileCount + b16VectorElements - 1) / b16VectorElements) * b16VectorElements;
            if (tileCapacity == 0) {
                tileCapacity = b16VectorElements;
            }
            const int64_t tileBufferSize = tileCapacity * sizeof(uint16_t);
            const int64_t nextUbFactor = std::min(
                ((availableUb - tmpBuffer - tileBufferSize) / ubPerElement / alignElem) * alignElem,
                BF16_DYNAMIC_TILE_CAP);
            OP_CHECK_IF((nextUbFactor <= 0),
                        OP_LOGE(context_->GetNodeName(),
                                "UB allocation overflow for BF16 scratch: ubSize=%lu availableUb=%ld "
                                "tmpBuffer=%ld ubPerElement=%ld ubFactor=%ld maxCoreElements=%ld "
                                "maxTileCount=%ld tileBufferSize=%ld nextUbFactor=%ld",
                                ubSize_, availableUb, tmpBuffer, ubPerElement, ubFactor_, maxCoreElements, maxTileCount,
                                tileBufferSize, nextUbFactor),
                        return ge::GRAPH_FAILED);
            if (nextUbFactor == ubFactor_) {
                bf16TileMaxBufferSize_ = tileBufferSize;
                break;
            }
            ubFactor_ = nextUbFactor;
        }
        OP_CHECK_IF((bf16TileMaxBufferSize_ == 0),
                    OP_LOGE(context_->GetNodeName(), "Failed to size BF16 tile max scratch."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 切分计算：核间按元素均分（blockFactor/blockTailCoreNum/actualCoreNum），核内按 ubFactor 切 tile，并派生 per-tensor
// 调度 8 字段。
ge::graphStatus QuantMaxV2Regbase::CalcTiling()
{
    int64_t shape = xInputShape_.GetDim(FIRST_DIM);
    // 小张量减少启动核数以降低初始化和跨核归并开销；必须在分核和 UB/workspace 计算前选择，使所有使用方采用同一核数。
    const int64_t workCores = shape / MIN_ELEMENTS_PER_VECTOR_CORE + (shape % MIN_ELEMENTS_PER_VECTOR_CORE != 0);
    coreNum_ = std::min(coreNum_, std::max<int64_t>(1, workCores));
    int64_t dtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF((dtypeSize <= 0), OP_LOGE(context_->GetNodeName(), "dtypeSize is invalid: %ld", dtypeSize),
                return ge::GRAPH_FAILED);
    if (shape == 0) {
        actualCoreNum_ = 0;
        blockFactor_ = 0;
        blockTailFactor_ = 0;
        blockTailCoreNum_ = 0;
        return CalcUbFactor();
    }
    actualCoreNum_ = std::min(shape, coreNum_);
    blockFactor_ = CeilDiv(shape, actualCoreNum_);
    blockTailCoreNum_ = (actualCoreNum_ * blockFactor_) - shape;
    if (blockTailCoreNum_ == 0) {
        blockTailFactor_ = blockFactor_;
    } else {
        blockTailFactor_ = blockFactor_ - 1;
    }

    OP_CHECK_IF((CalcUbFactor() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalcUBFactor failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 生成 TilingKey：roundMode(2bit)×quantMode(1bit)×hasGroup(1bit)，选择编译期 kernel 变体。
void QuantMaxV2Regbase::CalcTilingKey()
{
    uint32_t roundModeKey = static_cast<uint32_t>(roundMode_);
    const uint32_t quantModeKey = hasScale_ ? TPL_QUANT_MODE_STATIC : TPL_QUANT_MODE_DYNAMIC;
    // HasGroup 在编译期选择模板特化；不分组路径不分配组列表，也不执行组循环。
    const uint32_t hasGroupKey = hasGroupList_ ? TPL_GROUP_MODE_GROUP : TPL_GROUP_MODE_TENSOR;
    tilingKey_ = GET_TPL_TILING_KEY(roundModeKey, quantModeKey, hasGroupKey);
    OP_LOGD(context_->GetNodeName(), "roundMode is %ld", roundMode_);
    context_->SetTilingKey(tilingKey_);
}

// 把全部切分参数写入 TilingData 结构下发 device 侧。
ge::graphStatus QuantMaxV2Regbase::WriteTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum:%ld, tilingKey:%lu", coreNum_, tilingKey_);
    context_->SetBlockDim(coreNum_);
    context_->SetScheduleMode(1);
    // amax 必选，始终分配跨核归并 workspace；静态分层归并在核间矩阵后追加 [8][groupStride] 暂存区。
    context_->GetWorkspaceSizes(
        1)[0] = sysWorkspaceSize_ + SYNC_RESERVE_BYTES +
                coreNum_ * ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(float) +
                MERGE_SLAB_WORKSPACE_FLOATS *
                    ((numGroups_ + FLOAT_ALIGN_ELEM - 1) / FLOAT_ALIGN_ELEM * FLOAT_ALIGN_ELEM) * sizeof(float);

    QuantMaxV2TilingData* tilingData_ = context_->GetTilingData<QuantMaxV2TilingData>();

    OP_LOGD(context_->GetNodeName(),
            "totalCoreNum:%ld, actualCoreNum:%ld, blockTailCoreNum_:%ld, blockFactor:%ld, blockTailFactor:%ld, "
            "ubFactor:%ld, dim1:%ld, numGroups:%ld",
            coreNum_, actualCoreNum_, blockTailCoreNum_, blockFactor_, blockTailFactor_, ubFactor_, dim1_, numGroups_);
    tilingData_->totalCoreNum = coreNum_;
    tilingData_->actualCoreNum = actualCoreNum_;
    tilingData_->blockTailCoreNum = blockTailCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->blockTailFactor = blockTailFactor_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->dim1 = dim1_;
    tilingData_->numGroups = numGroups_;
    tilingData_->coreDivReciprocal = actualCoreNum_ > 1 ? static_cast<uint32_t>((uint64_t{1} << 32) / actualCoreNum_) :
                                                          0;
    tilingData_->tileDivReciprocal = ubFactor_ > 1 ? static_cast<uint32_t>((uint64_t{1} << 32) / ubFactor_) : 0;
    tilingData_->dstTypeMax = dstTypeMax_;
    tilingData_->tensorBlockLoopNum = 0;
    tilingData_->tensorBlockLoopTail = 0;
    tilingData_->tensorBlockTileCount = 0;
    tilingData_->tensorTailLoopNum = 0;
    tilingData_->tensorTailLoopTail = 0;
    tilingData_->tensorTailTileCount = 0;
    tilingData_->tensorTailCoreStart = 0;
    tilingData_->tensorBf16TileMaxBufferSize = 0;
    if (!hasScale_ && !hasGroupList_) {
        OP_CHECK_IF((ubFactor_ <= 0), OP_LOGE(context_->GetNodeName(), "Invalid tensor tile size."),
                    return ge::GRAPH_FAILED);
        // 用商和余数计算向上取整，避免加法溢出；大规格排程保留 64 位计算。
        tilingData_->tensorBlockLoopNum = blockFactor_ / ubFactor_;
        tilingData_->tensorBlockLoopTail = blockFactor_ % ubFactor_;
        tilingData_->tensorBlockTileCount = tilingData_->tensorBlockLoopNum + (tilingData_->tensorBlockLoopTail != 0);
        tilingData_->tensorTailLoopNum = blockTailFactor_ / ubFactor_;
        tilingData_->tensorTailLoopTail = blockTailFactor_ % ubFactor_;
        tilingData_->tensorTailTileCount = tilingData_->tensorTailLoopNum + (tilingData_->tensorTailLoopTail != 0);
        tilingData_->tensorTailCoreStart = (actualCoreNum_ - blockTailCoreNum_) * blockFactor_;
        // CalcUbFactor 已求解 BF16 暂存区与 tile 大小的相互约束。
        tilingData_->tensorBf16TileMaxBufferSize = bf16TileMaxBufferSize_;
    }

    return ge::GRAPH_SUCCESS;
}

// tiling 注册入口（IMPL_OP_OPTILING 绑定）：构造 QuantMaxV2Regbase 并执行 DoTiling。
static ge::graphStatus TilingForQuantMaxV2(gert::TilingContext* context)
{
    OP_LOGD("QuantMaxV2Tiling", "Enter TilingForQuantMaxV2Tiling");

    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantMaxV2Tiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    QuantMaxV2Regbase tilingOp(context);
    return tilingOp.DoTiling();
}

static ge::graphStatus TilingPrepareForQuantMaxV2(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantMaxV2Tiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<QuantMaxV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    // 每次从 TilingContext 读取平台参数；不在 CompileInfo 缓存，避免异构设备部署使用旧值。
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(QuantMaxV2).Tiling(TilingForQuantMaxV2).TilingParse<QuantMaxV2CompileInfo>(TilingPrepareForQuantMaxV2);
} // namespace optiling
