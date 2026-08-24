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
 * \file arch35/bn3d_training_reduce_tiling_base_arch35.cpp
 * \brief 平台信息获取、dtype / format / shape 校验，以及 R1-A-R0 归一化。
 */
#include <vector>
#include <algorithm>
#include "bn3d_training_reduce_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_MIN_DIM_NUM = 2;
constexpr int64_t NCDHW_MAX_DIM_NUM = 5;
constexpr int64_t NDC1HWC0_DIM_NUM = 6;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t FIRST_REDUCE_DIM = 2;

// NDC1HWC0 = [N, D, C1, H, W, C0]
constexpr int64_t NDC1HWC0_N_IDX = 0;
constexpr int64_t NDC1HWC0_D_IDX = 1;
constexpr int64_t NDC1HWC0_C1_IDX = 2;
constexpr int64_t NDC1HWC0_H_IDX = 3;
constexpr int64_t NDC1HWC0_W_IDX = 4;
constexpr int64_t NDC1HWC0_C0_IDX = 5;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT, ge::DataType::DT_BF16};

// 带溢出检查的 64 位乘法：溢出返回 false，绝不静默回绕。
inline bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& out) { return !__builtin_mul_overflow(lhs, rhs, &out); }
} // namespace

namespace optiling {
ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const BN3DTrainingReduceCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    vlfp32_ = compileInfoPtr->vectorLength / sizeof(float);
    ubBlockSize_ = compileInfoPtr->ubBlockSize;
    vectorLength_ = compileInfoPtr->vectorLength;

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

// ★动态图下 GE 会给 storage shape 左补前导 1，把通道轴挤走★
//
// 实测（GEIR 动态图，-d）：origin shape [1, 32700, 1] 的 storage shape 变成
// [1, 1, 1, 32700, 1]——前面补了两个 1 凑到 NCDHW 的 5 维。NCDHW 的通道轴钉在 dim1，
// 左补之后 32700 从 dim1 挪到了 dim3，于是 A 被读成 1、R0 被读成 32700，整张量被
// 当成单个通道归约（实测 输出[0] = 全张量和）。静态图下 storage == origin，两者恰好
// 一致，所以这条路一直没暴露；而 InferShape 用的是 origin format + origin shape，
// 判定输出仍是 [32700]——两处真值源就此劈叉。
//
// 还原条件写得很保守，只认"纯前导 1 的补齐"这一种情形，其余一律原样返回：
//   1) storage 的秩确实比 origin 大（真的补过）
//   2) 多出来的前导维全是 1（不携带数据，去掉不改变内存布局）
//   3) 去掉之后逐维等于 origin shape（确认就是同一个张量，不是别的变换）
// 三条同时成立才还原。因此：
//   * NDC1HWC0 不受影响——storage 与 origin 同为 6 维，条件 1 不成立；
//   * origin 为 NDHWC（通道在末维、GE 会转置）的情形也不受影响——NDHWC 恒 5 维、
//     storage 亦 5 维，条件 1 同样不成立。
gert::Shape BN3DTrainingReduceRegbaseTilingBase::StripLeadingPad(const gert::Shape& storageShape,
                                                                 const gert::Shape& originShape) const
{
    const size_t storageRank = storageShape.GetDimNum();
    const size_t originRank = originShape.GetDimNum();
    // origin 为空（秩 0）时无从判断补了什么，直接放弃还原。缺了这条早退，全 1 的退化
    // shape（如 [1,1]）会同时满足「前导全为 1」与「逐维比较循环不执行」，被还原成空 shape。
    // 当前两种测试模式都取不到空 origin（TTK kernel 模式按 storage 填 origin，geir 模式的
    // CSV 显式带 input_ori_shapes），属防御性早退。
    if (originRank == 0 || storageRank <= originRank) {
        return storageShape;
    }
    const size_t padCount = storageRank - originRank;
    for (size_t i = 0; i < padCount; ++i) {
        if (storageShape.GetDim(i) != 1) {
            return storageShape;
        }
    }
    for (size_t i = 0; i < originRank; ++i) {
        if (storageShape.GetDim(padCount + i) != originShape.GetDim(i)) {
            return storageShape;
        }
    }
    OP_LOGD(context_->GetNodeName(), "storage shape %s is origin %s left-padded by %zu, stripping",
            ToString(storageShape).c_str(), ToString(originShape).c_str(), padCount);
    return originShape;
}

ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BN3DTrainingReduce", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    xStorageShape_ = StripLeadingPad(xShape->GetStorageShape(), xShape->GetOriginShape());
    OP_CHECK_IF(CheckShapeAllNotNegative(xStorageShape_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Not supported shape info."), return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    dataType_ = xDesc->GetDataType();
    originFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetOriginFormat()));

    OP_CHECK_IF(CheckDtypeValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Not supported datatype info."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseShapeByFormat() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Failed to parse shape by format."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 把 storage shape 归一化为 R1-A-R0，两种布局共用同一个 GM 下标模型
//   idx(r1, a, r0) = r1 * (A * R0) + a * R0 + r0
//
//   * channel-first（NCDHW / NCHW）：R1 = dim0（N），A = dim1（C），
//     R0 = product(dim2:)（rank 2 时空乘积为 1）。归约轴是除 dim1 外的全部轴。
//   * C0 打包（NDC1HWC0 = [N,D,C1,H,W,C0]）：R1 = N * D，A = C1，R0 = H * W * C0。
//     归约轴是 N、D、H、W，保留 C1 与 C0。
// 两者的通道语义都与 canndev 对应 format 分支一致。
ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::ParseShapeByFormat()
{
    const int64_t xDimNum = static_cast<int64_t>(xStorageShape_.GetDimNum());
    auto storageFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context_->GetInputDesc(INPUT_X_INDEX)->GetStorageFormat()));

    if (storageFormat == FORMAT_NDC1HWC0) {
        return ParseNdc1hwc0Shape(xDimNum);
    }
    if (storageFormat == FORMAT_NCHW || storageFormat == FORMAT_NCDHW) {
        return ParseChannelFirstShape(xDimNum, storageFormat);
    }
    // ND / NDHWC 不是受支持的 storage format。
    OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ToString(storageFormat).c_str(),
                               "NCDHW, NCHW or NDC1HWC0");
    return ge::GRAPH_FAILED;
}

// channel-first（NCDHW / NCHW）：R1 = dim0（N），A = dim1（C），R0 = product(dim2:)。
// 两种 format 只有允许的 rank 不同，归一化过程完全一致。
ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::ParseChannelFirstShape(int64_t xDimNum, ge::Format storageFormat)
{
    if (storageFormat == FORMAT_NCHW) {
        OP_CHECK_IF(
            xDimNum != NCHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                     "The shape dim of input x must be 4 when the format of x is NCHW"),
            return ge::GRAPH_FAILED);
    } else if (storageFormat == FORMAT_NCDHW) {
        OP_CHECK_IF(xDimNum < NCDHW_MIN_DIM_NUM || xDimNum > NCDHW_MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be in the range of [2, 5] when the format of x is NCDHW"),
                    return ge::GRAPH_FAILED);
    }

    r1_ = xStorageShape_.GetDim(DIM_0);
    a_ = xStorageShape_.GetDim(DIM_1);

    // 有效逻辑通道数为 0：产出两个空输出，不启动归约 Kernel（其余维允许为 0）。
    if (a_ == 0) {
        isEmptyChannel_ = true;
        r0_ = 0;
        return ge::GRAPH_SUCCESS;
    }

    // C > 0 时，N 及全部空间归约维必须为正，否则归约集合为空，属于不支持的输入。
    OP_CHECK_IF(r1_ <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                    "The N-dimension of input x must be positive when the C-dimension is not 0"),
                return ge::GRAPH_FAILED);

    r0_ = 1;
    for (int64_t i = FIRST_REDUCE_DIM; i < xDimNum; ++i) {
        const int64_t dim = xStorageShape_.GetDim(i);
        OP_CHECK_IF(dim <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                        "Every spatial reduction dimension of input x must be positive when the C-dimension is not 0"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            !CheckedMul(r0_, dim, r0_),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                                                  "The product of spatial dimensions of input x overflows"),
            return ge::GRAPH_FAILED);
    }

    // 元素总数必须在 64 位内可表达，避免后续字节数 / 偏移计算回绕。
    int64_t totalElements = 0;
    OP_CHECK_IF(!CheckedMul(r1_, a_, totalElements) || !CheckedMul(totalElements, r0_, totalElements),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                                                      "The total element count of input x overflows int64"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// NDC1HWC0 = [N, D, C1, H, W, C0] → R1 = N * D，A = C1，R0 = H * W * C0。
// 固定 rank 6，且 C0 必须整除 VL_FP32（Kernel 靠 lane 下标 % C0 定位 c0 通道），
// 后者依赖平台信息，故在 DoOpTiling 中校验——GetShapeAttrsInfo 早于 GetPlatformInfo 执行。
ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::ParseNdc1hwc0Shape(int64_t xDimNum)
{
    OP_CHECK_IF(
        xDimNum != NDC1HWC0_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                 "The shape dim of input x must be 6 when the format of x is NDC1HWC0"),
        return ge::GRAPH_FAILED);

    const int64_t dimN = xStorageShape_.GetDim(NDC1HWC0_N_IDX);
    const int64_t dimD = xStorageShape_.GetDim(NDC1HWC0_D_IDX);
    const int64_t dimC1 = xStorageShape_.GetDim(NDC1HWC0_C1_IDX);
    const int64_t dimH = xStorageShape_.GetDim(NDC1HWC0_H_IDX);
    const int64_t dimW = xStorageShape_.GetDim(NDC1HWC0_W_IDX);
    const int64_t dimC0 = xStorageShape_.GetDim(NDC1HWC0_C0_IDX);

    a_ = dimC1;
    // C1 == 0 或 C0 == 0：有效逻辑通道数为 0，产出两个空输出且不启动归约 Kernel。
    if (dimC1 == 0 || dimC0 == 0) {
        isEmptyChannel_ = true;
        r1_ = 0;
        r0_ = 0;
        c0_ = 0;
        a_ = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 通道数非 0 时，全部归约维必须为正，否则归约集合为空，属于不支持的输入。
    OP_CHECK_IF(dimN <= 0 || dimD <= 0 || dimH <= 0 || dimW <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                    "The N/D/H/W dimensions of input x must be positive when the C1/C0 dimensions are not 0"),
                return ge::GRAPH_FAILED);

    c0_ = dimC0;

    OP_CHECK_IF(!CheckedMul(dimN, dimD, r1_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                                                      "N * D overflows int64"),
                return ge::GRAPH_FAILED);

    int64_t spatial = 0;
    OP_CHECK_IF(!CheckedMul(dimH, dimW, spatial) || !CheckedMul(spatial, dimC0, r0_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                                                      "H * W * C0 overflows int64"),
                return ge::GRAPH_FAILED);

    int64_t totalElements = 0;
    OP_CHECK_IF(!CheckedMul(r1_, a_, totalElements) || !CheckedMul(totalElements, r0_, totalElements),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape_).c_str(),
                                                      "The total element count of input x overflows int64"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::CheckDtypeValid()
{
    OP_CHECK_IF(std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), dataType_) == DTYPE_LIST.end(),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(dataType_).c_str(),
                                          "FLOAT, FLOAT16 or BFLOAT16"),
                return ge::GRAPH_FAILED);

    // 两个输出恒 fp32，与输入 dtype 无关；且必须分别校验，不能只看其中一个。
    auto sumDesc = context_->GetOutputDesc(OUTPUT_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumDesc);
    OP_CHECK_IF(
        sumDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "sum", ToString(sumDesc->GetDataType()).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);

    auto squareSumDesc = context_->GetOutputDesc(OUTPUT_SQUARE_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, squareSumDesc);
    OP_CHECK_IF(squareSumDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "square_sum",
                                          ToString(squareSumDesc->GetDataType()).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::CheckShapeAllNotNegative(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(shape).c_str(),
                                                          "Input x has negative axes"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus BN3DTrainingReduceRegbaseTilingBase::GetWorkspaceSize()
{
    // DENSE_CHANNEL 路线通道独占，无跨核归并，故不需要算子私有 workspace，
    // 只申请平台必需的 system workspace。
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
