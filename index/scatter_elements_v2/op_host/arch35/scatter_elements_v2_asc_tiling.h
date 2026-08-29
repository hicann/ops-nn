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
 * \file scatter_elements_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "../../../sort_lib/op_host/arch35/sort_lib_tiling.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
constexpr int16_t TILING_ARRAY_LEN = 7;

BEGIN_TILING_DATA_DEF(ScatterElementsV2SortTilingData)
TILING_DATA_FIELD_DEF(int64_t, indicesTotalNum);  // 索引总数 N（排序模板切核依据）
TILING_DATA_FIELD_DEF(int64_t, keySize);          // Sort key 字节宽 2/4/8
TILING_DATA_FIELD_DEF(int64_t, permSize);         // Sort perm(索引) 字节宽 4=int32 / 8=int64
TILING_DATA_FIELD_DEF(int32_t, countMode);        // 0=uint32, 1=int64 计数（SortInvoke CountT 运行时 dispatch）
TILING_DATA_FIELD_DEF(int32_t, shapeMode);        // 0=SAME, 1=SUBSET
TILING_DATA_FIELD_DEF(int32_t, dimNormalized);    // 归一化 scatter 轴
TILING_DATA_FIELD_DEF(uint32_t, sortUsedCoreNum); // 排序模板实际用核数（SortLib coreNumNeed）
// SortLib 分块参数
TILING_DATA_FIELD_DEF(uint32_t, numTileData);
TILING_DATA_FIELD_DEF(uint32_t, tileCount);
TILING_DATA_FIELD_DEF(uint32_t, activeCores);
TILING_DATA_FIELD_DEF(uint32_t, tmpUbSize);
TILING_DATA_FIELD_DEF(uint32_t, isSingleCore);
// GM workspace 偏移
TILING_DATA_FIELD_DEF(uint64_t, wsLinearIdxOff);
TILING_DATA_FIELD_DEF(uint64_t, wsSortedOff);
TILING_DATA_FIELD_DEF(uint64_t, wsPermOff);
TILING_DATA_FIELD_DEF(uint64_t, wsSrcPosOff);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2SortTilingDataOp, ScatterElementsV2SortTilingData)

BEGIN_TILING_DATA_DEF(ScatterElementsV2AscTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, dataStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, indicesStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, updatesStride);
TILING_DATA_FIELD_DEF(int64_t, loopLength);
TILING_DATA_FIELD_DEF(int64_t, allAxis);
TILING_DATA_FIELD_DEF(int64_t, dataAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesAxis);
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, midAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, indicesUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesNormBlockData);
TILING_DATA_FIELD_DEF(int64_t, indicesTailBlockData);
TILING_DATA_FIELD_DEF(int64_t, baseS);
TILING_DATA_FIELD_DEF(int64_t, baseA);
TILING_DATA_FIELD_DEF(int64_t, isDeterministic);
TILING_DATA_FIELD_DEF(int16_t, rank);
TILING_DATA_FIELD_DEF(int16_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, sortSharedBufSize);
// === 排序模板 tilingdata（嵌套 struct，字段定义见 ScatterElementsV2SortTilingData）
TILING_DATA_FIELD_DEF_STRUCT(ScatterElementsV2SortTilingData, sortTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2, ScatterElementsV2AscTilingData)

class ScatterElementsV2AscTiling : public TilingBaseClass {
public:
    explicit ScatterElementsV2AscTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckXDtype(const ge::DataType dtype);
    bool CompareShape(const gert::Shape& shape1, const gert::Shape& shape2, int16_t dim = -1);
    void ComputeShape(const gert::Shape& dataShape, const gert::Shape& indicesShape, const gert::Shape& updatesShape);
    uint64_t GetStride(const std::vector<uint64_t>& shapeList, int16_t start);
    void ComputeStride();
    void GetCastTypeSize();
    void CombineIndicesAxis();
    uint32_t GetMaxSortTmpBuf(int64_t sortDim);
    int64_t CalBestBaseSize(int64_t baseXoStart, int64_t baseXoEnd);
    // 排序模板准入：dtype 白名单 + 索引轴主导 + 整型/浮点各自的形状条件 + index-count 切核收益
    // aAxisCoreNum：DoOpTiling 先行算出的实际 A 轴切核核数 indicesUsedCoreNum_
    bool IsSortTemplateAdmitted(int64_t aAxisCoreNum) const;
    bool IsSortAdmittedInt() const;
    bool IsSortAdmittedFloat(int64_t aAxisCoreNum) const;

private:
    int16_t dim_ = 0;
    int16_t rank_ = 1;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t loopLength_ = 0;
    int64_t allAxis_ = 1;
    int64_t dataAxis_ = 1;
    int64_t updatesAxis_ = 1;
    int64_t castTypeSize_ = 0;
    int64_t isDeterministic_ = 0;     // 原确定性模板语义开关（add 且 dtype∈SCAT_ELE_ADD_DETERM_DTYPE）
    int64_t isSortDeterministic_ = 0; // 排序模板 dtype 资格（add 且 ∈SORT_DETERM / none 且 int），

    int64_t preAxis_ = 1;
    int64_t midAxis_ = 1;
    int64_t afterAxis_ = 1;
    int64_t indicesUsedCoreNum_ = 0;
    int64_t indicesNormBlockData_ = 0;
    int64_t indicesTailBlockData_ = 0;
    int64_t baseS_ = 1;
    int64_t baseA_ = 1;
    int64_t indicesTypeSize_ = 0;
    int64_t ubBlockSize_ = 0;
    uint64_t reduction_ = 0;
    uint64_t typeSize_ = 0;
    uint64_t tilingKey_ = 0;
    uint32_t sortSharedBufSize_ = 0;
    // === 排序模板 host 侧内部状态 ===
    bool isSortDeterm_ = false;                // 排序模板启用标志（准入通过且 SortLib tiling 成功）
    int32_t shapeMode_ = 0;                    // 0=SAME, 1=SUBSET
    ge::DataType keyDtype_ = ge::DT_UNDEFINED; // Sort key dtype
    int32_t countMode_ = 0;                    // 0=uint32, 1=int64 计数
    int64_t indicesTotalNum_ = 0;              // 索引总数（排序模板切核依据）
    std::vector<uint64_t> dataShapeVec_;       // 原始 data 各维大小（SUBSET/size 判定用）
    std::vector<uint64_t> indicesShapeVec_;    // 原始 indices 各维大小
    std::vector<uint64_t> updatesShapeVec_;    // 原始 updates 各维大小
    int64_t keySize_ = 0;                      // Sort key 字节宽 2/4/8
    int64_t permSize_ = 4;                     // Sort perm 字节宽 4=int32 / 8=int64
    int64_t sortUsedCoreNum_ = 0;              // 排序模板实际用核数（SortLib coreNumNeed）
    int64_t multiSortWsBytes_ = 0;             // SortLib workspaceBytes（单核=0）
    uint64_t wsLinearIdxOff_ = 0;
    uint64_t wsSortedOff_ = 0;
    uint64_t wsPermOff_ = 0;
    uint64_t wsSrcPosOff_ = 0;
    uint64_t wsUserSize_ = 0;         // 排序模板 userWs 总字节（GetWorkspaceSize 叠加）
    SortLib::SortTilingResult sortR_; // SortLib tiling 结果缓存（供 GetWorkspaceSize/PostTiling 读）
    std::vector<uint64_t> dataCurSize_;
    std::vector<uint64_t> indicesCurSize_;
    std::vector<uint64_t> updatesCurSize_;
    uint64_t dataStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};
    uint64_t indicesStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};
    uint64_t updatesStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};

    ge::DataType dtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    ScatterElementsV2AscTilingData tilingData_;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_
