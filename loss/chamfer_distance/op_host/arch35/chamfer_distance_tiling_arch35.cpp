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
 * \file chamfer_distance_tiling_arch35.cpp
 * \brief
 */
#include <cstring>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/chamfer_distance_tiling_data.h"

namespace optiling {
using namespace Ops::Base;

namespace {
constexpr uint32_t INPUT_XYZ1_IDX = 0;
constexpr uint32_t INPUT_XYZ2_IDX = 1;

constexpr int64_t DTYPE_LEN_B16 = 2; // float16 / bfloat16
constexpr int64_t DTYPE_LEN_FP32 = 4;

constexpr size_t XYZ_DIM_NUM = 3;
constexpr size_t DIM_COORD = 0; // 首维固定为 2(x/y 两个坐标平面)
constexpr size_t DIM_B = 1;
constexpr size_t DIM_N = 2;
constexpr int64_t COORD_NUM = 2;

// 被查集合每个点在 UB 里的占用: 原始 x/y(输入 dtype, fp16/bf16 在 VF 内随路转 fp32, 不另占缓冲)
// + 距离缓冲 + ReduceMin 的 work 缓冲, 两者均为 fp32
constexpr int64_t DIST_BUF_PER_POINT = DTYPE_LEN_FP32; // 段内平方距离
constexpr int64_t WORK_BUF_PER_POINT = DTYPE_LEN_FP32; // ReduceMin work tensor
// 每个查询点的跨段累加器: bestVal | bestIdx, 各按一个向量宽度驻留
constexpr int64_t ACC_BUF_NUM = 2;
// SIMD/SIMT 共用的 dcache, 与参照算子一致预留
constexpr uint64_t SIMD_SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
} // namespace

class ChamferDistanceTiling {
public:
    explicit ChamferDistanceTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CalUbSplit();
    void CalCoreSplit();
    void PrintTilingData();

    gert::TilingContext* context_ = nullptr;
    ChamferDistanceArch35TilingData* tilingData_{nullptr};

    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vlFp32_ = 0;
    int64_t xyzDtypeLen_ = DTYPE_LEN_FP32;
    size_t sysWorkspaceSize_ = 0;
};

ge::graphStatus ChamferDistanceTiling::CheckInput()
{
    auto xyz1Desc = context_->GetInputDesc(INPUT_XYZ1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xyz1Desc);
    auto xyz1Dtype = xyz1Desc->GetDataType();
    OP_CHECK_IF((xyz1Dtype != ge::DT_FLOAT && xyz1Dtype != ge::DT_FLOAT16 && xyz1Dtype != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "xyz1", ToString(xyz1Dtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);
    // dtype 只影响 UB 记账; 内核实例由 def 的 dtype profile 驱动(DTYPE_XYZ1), 不进 TilingKey
    xyzDtypeLen_ = (xyz1Dtype == ge::DT_FLOAT) ? DTYPE_LEN_FP32 : DTYPE_LEN_B16;

    auto xyz2Desc = context_->GetInputDesc(INPUT_XYZ2_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xyz2Desc);
    OP_CHECK_IF((xyz2Desc->GetDataType() != xyz1Dtype),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "xyz2", ToString(xyz2Desc->GetDataType()).c_str(),
                                          "the same as xyz1 dtype"),
                return ge::GRAPH_FAILED);

    auto xyz1Shape = context_->GetInputShape(INPUT_XYZ1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xyz1Shape);
    auto storageShape = xyz1Shape->GetStorageShape();
    OP_CHECK_IF(
        (storageShape.GetDimNum() != XYZ_DIM_NUM),
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "xyz1", std::to_string(storageShape.GetDimNum()), "3"),
        return ge::GRAPH_FAILED);
    // 首维是坐标轴(x/y 两个平面), 不是点数; 见 01 §6.1
    OP_CHECK_IF((storageShape.GetDim(DIM_COORD) != COORD_NUM),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "xyz1",
                                                      Ops::Base::ToString(storageShape).c_str(),
                                                      "dim 0 must be 2 (x/y coordinate planes)"),
                return ge::GRAPH_FAILED);

    auto xyz2Shape = context_->GetInputShape(INPUT_XYZ2_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xyz2Shape);
    OP_CHECK_IF((xyz2Shape->GetStorageShape() != storageShape),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "xyz2",
                                                      Ops::Base::ToString(xyz2Shape->GetStorageShape()).c_str(),
                                                      "must be the same as xyz1 shape"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ChamferDistanceTiling::CalCoreSplit()
{
    int64_t taskNum = tilingData_->taskNum;
    if (taskNum <= 0) {
        tilingData_->realCoreNum = 0;
        tilingData_->tasksPerCore = 0;
        tilingData_->tailTasks = 0;
        return;
    }
    int64_t useCoreNum = taskNum > coreNum_ ? coreNum_ : taskNum;
    int64_t tasksPerCore = Ops::Base::CeilDiv(taskNum, useCoreNum);
    int64_t realCoreNum = Ops::Base::CeilDiv(taskNum, tasksPerCore);
    tilingData_->tasksPerCore = tasksPerCore;
    tilingData_->realCoreNum = realCoreNum;
    tilingData_->tailTasks = taskNum - tasksPerCore * (realCoreNum - 1);
}

ge::graphStatus ChamferDistanceTiling::CalUbSplit()
{
    int64_t n = tilingData_->n;
    if (tilingData_->taskNum <= 0 || n <= 0) {
        tilingData_->colsPerChunk = vlFp32_;
        tilingData_->chunkNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 每个被查点的 UB 占用 + 每个查询点的跨段累加器(按向量宽度常驻)
    int64_t bytesPerPoint = COORD_NUM * xyzDtypeLen_ + DIST_BUF_PER_POINT + WORK_BUF_PER_POINT;
    int64_t reserve = ACC_BUF_NUM * vlFp32_ * DTYPE_LEN_FP32;
    int64_t usable = static_cast<int64_t>(ubSize_) - reserve;
    OP_CHECK_IF((usable <= bytesPerPoint * vlFp32_),
                OP_LOGE(context_->GetNodeName(), "ub size %lu is too small for one vector of points", ubSize_),
                return ge::GRAPH_FAILED);

    int64_t budget = usable / bytesPerPoint;
    int64_t colsPerChunk = budget / vlFp32_ * vlFp32_; // 向下取 VL 对齐
    int64_t alignN = Ops::Base::CeilAlign(n, vlFp32_);
    if (colsPerChunk >= alignN) {
        colsPerChunk = alignN; // 单段全载
    }
    tilingData_->colsPerChunk = colsPerChunk;
    tilingData_->chunkNum = Ops::Base::CeilDiv(n, colsPerChunk);
    return ge::GRAPH_SUCCESS;
}

void ChamferDistanceTiling::PrintTilingData()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "ChamferDistance tiling: b=%ld n=%ld taskNum=%ld realCoreNum=%ld", tilingData_->b, tilingData_->n,
            tilingData_->taskNum, tilingData_->realCoreNum);
    OP_LOGD(nodeName, "ChamferDistance tiling: tasksPerCore=%ld tailTasks=%ld colsPerChunk=%ld chunkNum=%ld",
            tilingData_->tasksPerCore, tilingData_->tailTasks, tilingData_->colsPerChunk, tilingData_->chunkNum);
}

ge::graphStatus ChamferDistanceTiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, aivCoreNum %ld", coreNum_),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    sysWorkspaceSize_ = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    OP_CHECK_IF((ubSize_ <= SIMD_SIMT_DCACHE_SIZE),
                OP_LOGE(context_->GetNodeName(), "GetHardwareInfo failed, ubSize %lu", ubSize_),
                return ge::GRAPH_FAILED);
    ubSize_ -= SIMD_SIMT_DCACHE_SIZE;
    vlFp32_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)) / DTYPE_LEN_FP32;
    OP_CHECK_IF((vlFp32_ <= 0), OP_LOGE(context_->GetNodeName(), "GetVRegSize failed"), return ge::GRAPH_FAILED);

    tilingData_ = context_->GetTilingData<ChamferDistanceArch35TilingData>();
    OP_CHECK_IF((tilingData_ == nullptr), OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ChamferDistanceArch35TilingData), 0,
                          sizeof(ChamferDistanceArch35TilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChamferDistanceTiling::DoTiling()
{
    OP_CHECK_IF((CheckInput() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CheckInput failed"),
                return ge::GRAPH_FAILED);

    auto xyz1Shape = context_->GetInputShape(INPUT_XYZ1_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xyz1Shape);
    auto storageShape = xyz1Shape->GetStorageShape();
    tilingData_->b = storageShape.GetDim(DIM_B);
    tilingData_->n = storageShape.GetDim(DIM_N);
    tilingData_->taskNum = tilingData_->b * tilingData_->n;

    CalCoreSplit();
    OP_CHECK_IF((CalUbSplit() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "CalUbSplit failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();

    context_->SetTilingKey(0); // 单一算法路径; dtype 由 def profile 驱动, 不进入 TilingKey
    // 空 tensor 时不进核, 但 blockDim 必须合法
    context_->SetBlockDim(tilingData_->realCoreNum > 0 ? tilingData_->realCoreNum : 1);
    context_->SetLocalMemorySize(ubSize_);
    // 本算子不使用系统 workspace(无跨核同步、无中间落盘), 只按平台要求预留框架自用部分
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = sysWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ChamferDistance(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ChamferDistanceTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ChamferDistance init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ChamferDistance do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4ChamferDistance(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("ChamferDistance", "Tiling parse context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct ChamferDistanceCompileInfo {};

IMPL_OP_OPTILING(ChamferDistance)
    .Tiling(Tiling4ChamferDistance)
    .TilingParse<ChamferDistanceCompileInfo>(TilingParse4ChamferDistance);
} // namespace optiling
