/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_tiling_base.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ADD_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ADD_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"

namespace optiling
{

// ///////////////////////////////////
// tilingdata define
// ///////////////////////////////////
BEGIN_TILING_DATA_DEF(ScatterAddTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, varShape);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, indicesSize);

TILING_DATA_FIELD_DEF(uint64_t, indicesUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, indicesLoopSize);
TILING_DATA_FIELD_DEF(uint64_t, indicesTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesLoopSize);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreIndicesTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, updatesUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, updatesLoopSize);
TILING_DATA_FIELD_DEF(uint64_t, updatesTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, copyCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleVar);
TILING_DATA_FIELD_DEF(uint64_t, atomicAddCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleIndices);
TILING_DATA_FIELD_DEF(uint64_t, postAxisSize);

TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleCol);
TILING_DATA_FIELD_DEF(uint64_t, logicCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, isDeterministic);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreHandleCol);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreColsLoopSize);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreColsTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(uint64_t, perCoreHandleRows); //反量化按var[0]分核或indices分核
TILING_DATA_FIELD_DEF(uint64_t, tailCoreHandleRows);
TILING_DATA_FIELD_DEF(uint64_t, rowsInUb);
TILING_DATA_FIELD_DEF(uint64_t, deQuantizeCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, isDeterminTemplate);
TILING_DATA_FIELD_DEF(uint64_t, dequantizeBranch);

// simt排序
TILING_DATA_FIELD_DEF(uint64_t, normBlockIndices);
TILING_DATA_FIELD_DEF(uint64_t, indicesFactor);
TILING_DATA_FIELD_DEF(uint64_t, normBlockLoop);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockLoop);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockTail);
TILING_DATA_FIELD_DEF(uint64_t, sortCoreNum);

// simd排序+非排序
TILING_DATA_FIELD_DEF(uint64_t, rowTileNum);    // 行切分份数
TILING_DATA_FIELD_DEF(uint64_t, colTileNum);    // 列切分份数
TILING_DATA_FIELD_DEF(uint64_t, normBlockRow);  // 整核分块行数
TILING_DATA_FIELD_DEF(uint64_t, tailBlockRow);  // 行尾核分块行数
TILING_DATA_FIELD_DEF(uint64_t, normBlockCol);  // 整核分块列数
TILING_DATA_FIELD_DEF(uint64_t, tailBlockCol);  // 列尾核分块列数
TILING_DATA_FIELD_DEF(uint64_t, ubFactorRow);   // UB每次循环搬运的行数
TILING_DATA_FIELD_DEF(uint64_t, ubFactorCol);   // UB每次循环搬运的列数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAdd, ScatterAddTilingData)

ge::graphStatus ScatterAddTilingForAscendC(gert::TilingContext* context);

class ScatterAddTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit ScatterAddTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus ScatterAddTilingForAscendC(gert::TilingContext* context);
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus TilingCopyCompute();
    ge::graphStatus TilingSimtSort();
    ge::graphStatus TilingSimdCompute();
    std::set<uint64_t> FindUniqueCut(uint64_t usedCoreNum);
    std::tuple<uint64_t, uint64_t> SimdTiling(uint64_t usedCoreNum, uint64_t colNumAlign, uint64_t colLimitSize, bool colTileNumMin=false);
    void DoBlockTiling(uint64_t baseCol);
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd);
    ge::graphStatus TilingSimdSupportAtomicAddSortCompute();
    ge::graphStatus TilingSimdSupportAtomicAddCompute();
    ge::graphStatus TilingSimdNotSupportAtomicAddCompute(bool supportAtomicAdd);
    ge::graphStatus ScatterAddDeterministicTiling();
    ge::graphStatus getRestAvailableSize(uint64_t sampleNum, uint64_t valueTypeBytes, uint64_t originalSize,
        uint64_t postAxisSize, ge::DataType idType);
    uint64_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
    void DumpTilingInfo() override;
    void SetTilingData();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckUpdatesShape(const gert::Shape& varShape, const gert::Shape& indicesShape,
                                    const gert::Shape& updatesShape);
    ge::graphStatus GetCastType();

private:
    uint64_t blockFactor_ = 0;
    uint64_t tailBlockFactor_ = 1;
    uint64_t ubFactor_ = 0;
    uint64_t tailUbFactor_ = 0;
    uint64_t tailCoreTailUbFactor_ = 0;
    uint64_t varSize_ = 0;
    uint64_t indicesNum_ = 0;
    uint64_t varTypeSize_ = 0;
    uint32_t isUpdateScalar_ = 0;
    uint64_t totalCoreNum_ = 1;
    uint64_t castTypeSize_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t varShape_[2] = {1, 1};
    uint64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;
    bool isSimt_ = false;
    uint64_t isSort_ = 0;
    uint64_t perCoreHandleVar_ = 0;
    uint64_t copyCoreNum_ = 0;
    uint64_t indicesCastMode_ = 0;  // 0: 不Cast；1：int32 Cast int16；2：int64 Cast int32；3：int64 Cast int16

    uint64_t atomicAddCoreNum_ = 0;
    uint64_t perCoreHandleIndices_ = 0;
    uint64_t indicesUbFactor_ = 0;
    uint64_t indicesLoopSize_ = 0;
    uint64_t indicesTailUbFactor_ = 0;
    uint64_t tailCoreIndicesLoopSize_ = 0;
    uint64_t tailCoreIndicesTailUbFactor_ = 0;
    uint64_t updatesUbFactor_ = 0;
    uint64_t updatesLoopSize_ = 0;
    uint64_t updatesTailUbFactor_ = 0;
    uint64_t postAxisSize_ = 0;

    uint64_t updatesSize_ = 0;
    uint64_t updatesDtypeSize_ = 0;
    uint64_t indicesDtypeSize_ = 0;
    uint64_t indicesCastDtypeSize_ = 0;
    uint64_t templateKey_ = 0;

    uint64_t perCoreHandleCol_ = 0;
    uint64_t isDeterministic_ = 0;
    uint64_t logicCoreNum_ = 0;
    uint64_t tailCoreHandleCol_ = 0;
    uint64_t tailCoreColsLoopSize_ = 0;
    uint64_t tailCoreColsTailUbFactor_ = 0;
    uint64_t deQuantizeCoreNum_ = 0;
    uint64_t perCoreHandleRows_ = 0;
    uint64_t tailCoreHandleRows_ = 0;
    uint64_t rowsInUb_ = 0;
    uint64_t postVarAlignSize_ = 0;
    uint64_t isDeterminTemplate_ = 0;
    uint64_t dequantizeBranch_ = 0;

    uint64_t normBlockIndices_ = 0;
    uint64_t indicesFactor_ = 0;
    uint64_t normBlockLoop_ = 0;
    uint64_t tailBlockLoop_ = 0;
    uint64_t tailBlockTail_ = 0;

    uint64_t rowTileNum_ = 0;
    uint64_t colTileNum_ = 0;
    uint64_t normBlockRow_ = 0;
    uint64_t tailBlockRow_ = 0;
    uint64_t normBlockCol_ = 0;
    uint64_t tailBlockCol_ = 0;
    uint64_t ubFactorRow_ = 0;
    uint64_t ubFactorCol_ = 0;

    ge::DataType varDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesCastDtype_ = ge::DT_UNDEFINED;
    const char* opName = "ScatterAdd";
    ScatterAddTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ADD_TILING_H_