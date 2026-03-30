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
 * \file scatter_update_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_UPDATE_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_UPDATE_TILING_H_
#pragma once

#include <cstdint>
#include <string>
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "index/scatter_update/op_kernel/arch35/scatter_update_struct.h" 

namespace optiling
{
struct ScatterUpdateCompileInfo {
  int64_t core_num{1};
  int64_t ub_size{1};
  bool is_950{false};
};

using namespace Ops::NN::Optiling;
class ScatterUpdateTiling : public TilingBaseClass
{
public:
    explicit ScatterUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context){}

protected:
    constexpr static uint64_t BASE_BLOCK_SIZE {8192};
    constexpr static uint64_t MOVE_ALIGN_LIMIT_BYTE {256};
    constexpr static uint64_t BASE_BLOCK_COPY_ALIGN {2048};
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckUpdatesShape(const gert::Shape& varShape, const gert::Shape& indicesShape,
                                    const gert::Shape& updatesShape);
    void CalcKernelParam();
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd, uint64_t updatesSize, uint64_t reserveSize);
    void ClacColUbParam(uint64_t blockColNum);
    void ClacRowUbParam(uint64_t processRowPerub, uint64_t blockColNum);
    void ProcessSimdSort();
    void ProcessSimdNonSort(uint64_t existNodeSize);
    void DoMaskSimdTiling();
    void DoSimtTiling();
    void DoDeterministicTiling();
    void CalcMask();
    uint64_t CalcIndicesUbFactor();
    uint64_t CalcSimtIndicesUbFactor();
    ge::graphStatus GetCastType();

private:
    uint64_t varSize_ = 0;
    uint64_t indicesSize_ = 0;
    uint64_t normBlockIndices_ = 0;
    uint64_t tailBlockIndices_ = 0;
    uint64_t indicesFactor_ = 0;
    uint64_t normBlockLoop_ = 0;
    uint64_t tailBlockLoop_ = 0;
    uint64_t normBlockTail_ = 0;
    uint64_t tailBlockTail_ = 0;
    uint64_t varTypeSize_ = 0;
    uint32_t isUpdateScalar_ = 0;
    uint64_t totalCoreNum_ = 1;
    uint64_t usedCoreNum_ = 0;
    uint64_t varShape_[2] = {1, 1};
    uint64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;
    bool isSimt_ = false;
    uint64_t isSort_ = 0;
    uint64_t isMask_ = 0;
    uint64_t updatesSize_ = 0;
    uint64_t updateShape_[2] = {1, 1};
    uint64_t indicesDtypeSize_ = 0;
    uint64_t normBlockColNum_ = 0;
    uint64_t normBlockRowNum_ = 0;
    uint64_t tailBlockColNum_ = 0;
    uint64_t tailBlockRowNum_ = 0;
    uint64_t normNeedSplitRow_ = 0;
    uint64_t tailNeedSplitRow_ = 0;
    uint64_t processRowPerUb_ = 0;
    uint64_t processColNum_ = 0;
    uint64_t rowLoopByUb_ = 0;
    uint64_t processRowTail_ = 0;
    uint64_t indicesUbFactor_ = 0;
    uint64_t updateUbSize_ = 0;
    uint64_t processColPerUb_ = 0;
    uint64_t colLoopByUb_ = 0;
    uint64_t processColTail_ = 0;
    uint64_t templateKey_ = 0;
    uint64_t indicesBatchCopySizeAlign_ = 0;
    uint64_t varStride_ = 0;
    uint64_t isDeterministic_ = 0;
    uint64_t deterministicMode_ = 0;
    uint64_t isDeterministicSplitCol_ = 0;
    uint64_t updateColUbFactor_ = 0;
    uint64_t indicesLoopSize_ = 0;
    uint64_t indicesTailLoopNum_ = 0;
    uint64_t updatesNormBlockLoop_ = 0;
    uint64_t updatesTailBlockLoop_ = 0;
    uint64_t updatesNormBlockTailLoopSize_ = 0;
    uint64_t updatesTailBlockTailLoopSize_ = 0;
    uint64_t maskNormBlockLen_ = 0;
    uint64_t maskTailBlockLen_ = 0;
    bool isIndicesSizeInt64_ = false;
    uint64_t indicesCastMode_ = 0;  // 0: 不Cast；1：int32 Cast int16；2：int64 Cast int32；3：int64 Cast int16
    uint64_t indicesCastDtypeSize_ = 0;

    ge::DataType varDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesCastDtype_ = ge::DT_UNDEFINED;
    const char* opName = "ScatterUpdate";

    uint64_t rowTotalNum_ {0};
    uint64_t colTotalNum_ {0};
    uint64_t rowNormalNum_ {0};
    uint64_t colNormalNum_ {0};
    uint64_t rowTailNum_ {0};
    uint64_t colTailNum_ {0};
    uint64_t rowTileNum_ {0};// 行切片数量
    uint64_t colTileNum_ {0};// 列切片数量
    ge::DataType dataType_ {ge::DataType::DT_FLOAT};

    void AutoTiling();
    std::set<uint64_t> FindUniqueCut();
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_UPDATE_TILING_H_