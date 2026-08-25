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
 * \file bn3d_training_reduce_tiling.h
 * \brief
 */

#ifndef BN3D_TRAINING_REDUCE_TILING_H
#define BN3D_TRAINING_REDUCE_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../op_kernel/arch35/bn3d_training_reduce_tiling_data.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
static constexpr uint64_t BN3D_TRAINING_REDUCE_DENSE_CHANNEL_PRIORITY = 9000;

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t OUTPUT_SUM_INDEX = 0;
constexpr uint32_t OUTPUT_SQUARE_SUM_INDEX = 1;

struct BN3DTrainingReduceCompileInfo {
    uint64_t coreNum;      // AIV 核数
    uint64_t ubSize;       // UB 空间
    uint32_t vectorLength; // 向量寄存器字节宽度
    uint64_t ubBlockSize;  // 32B，UB 的字节对齐单位
};

class BN3DTrainingReduceRegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit BN3DTrainingReduceRegbaseTilingBase(gert::TilingContext* context)
        : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        r1_ = 0;
        a_ = 0;
        r0_ = 0;
        c0_ = 0;
        vlfp32_ = 0;
        vectorLength_ = 0;
        ubBlockSize_ = 0;
        isEmptyChannel_ = false;
        blockNum_ = 1;
        // 这三个同样是跨次复用的成员：GetShapeAttrsInfo 在取到它们之前就可能
        // 提前失败返回（如 xShape 判空），不清零会把上一次的值留给下一次调用。
        dataType_ = ge::DT_UNDEFINED;
        originFormat_ = ge::FORMAT_ND;
        xStorageShape_ = gert::Shape{};
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;

    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckShapeAllNotNegative(const gert::Shape& shape);
    // 动态图下 GE 会给 storage shape 左补前导 1 把通道轴挤走；仅当"多出的前导维全为 1
    // 且去掉后逐维等于 origin"时还原成 origin shape，其余原样返回。详见 .cpp 的说明。
    gert::Shape StripLeadingPad(const gert::Shape& storageShape, const gert::Shape& originShape) const;
    // 按 storage format 把 storage shape 归一化为 R1-A-R0（只做分发）。
    ge::graphStatus ParseShapeByFormat();
    // channel-first（NCDHW / NCHW）的归一化分支。
    ge::graphStatus ParseChannelFirstShape(int64_t xDimNum, ge::Format storageFormat);
    // NDC1HWC0 [N,D,C1,H,W,C0] 的归一化分支。
    ge::graphStatus ParseNdc1hwc0Shape(int64_t xDimNum);

protected:
    int64_t r1_{0}; // NCDHW：N；NDC1HWC0：N * D
    int64_t a_{0};  // NCDHW：C；NDC1HWC0：C1
    int64_t r0_{0}; // NCDHW：product(dim2:)；NDC1HWC0：H * W * C0
    // 0：每通道归约成 1 个标量（NCDHW / NCHW）；
    // > 0：C0 打包布局，每通道归约成 c0_ 个标量（NDC1HWC0）。
    int64_t c0_{0};
    int64_t vlfp32_{0};
    int64_t vectorLength_{0};
    int64_t ubBlockSize_{0};

    int64_t blockNum_{1}; // 实际下发的 blockDim

    // 有效逻辑通道数为 0：产出两个空输出且不启动归约 Kernel。
    bool isEmptyChannel_{false};

    ge::DataType dataType_{ge::DT_UNDEFINED};
    ge::Format originFormat_{ge::FORMAT_ND};
    gert::Shape xStorageShape_;
};

class BN3DTrainingReduceDenseChannelTiling : public BN3DTrainingReduceRegbaseTilingBase {
public:
    explicit BN3DTrainingReduceDenseChannelTiling(gert::TilingContext* context)
        : BN3DTrainingReduceRegbaseTilingBase(context)
    {}
    ~BN3DTrainingReduceDenseChannelTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BN3DTrainingReduceRegbaseTilingBase::Reset(context);
        td_ = BN3DTrainingReduceDenseChannelTilingData{};
        splitReduce_ = false;
    }

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus SetEmptyTilingData();
    ge::graphStatus ValidateAndNormalizeShape();
    ge::graphStatus BuildDenseTilingData();
    // 在给定 UB 预算下求解 nTile / sub-R 分块，返回 false 表示无可行解。
    bool SolveUbSplit(int64_t r0Align, int64_t elemSize, int64_t ubBudget);
    bool SolveUbSplitWithSlots(int64_t r0Align, int64_t elemSize, int64_t ubBudget, int64_t accSlots);
    int64_t PickAccSlots(int64_t numSteps, int64_t totalChain, int64_t ubBudget) const;
    int64_t AccBytes(int64_t accSlots) const;

    BN3DTrainingReduceDenseChannelTilingData td_{};
    bool splitReduce_{false};
};

} // namespace optiling
#endif // BN3D_TRAINING_REDUCE_TILING_H
