/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_with_dual_axis_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicMxQuantWithDualAxisTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);           // 总核数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);            // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, roundMode);              // 数据类型转换的模式
TILING_DATA_FIELD_DEF(int64_t, dstType);                // 输出y的数据类型
TILING_DATA_FIELD_DEF(int64_t, scaleAlg);               // CuBlas实现或OCP实现，默认OCP实现
TILING_DATA_FIELD_DEF(int64_t, blockSize);              //
TILING_DATA_FIELD_DEF(int64_t, dim0);                   //
TILING_DATA_FIELD_DEF(int64_t, dimNeg2);                //
TILING_DATA_FIELD_DEF(int64_t, dimNeg1);                //
TILING_DATA_FIELD_DEF(int64_t, blockW);                 // 所切基本块的宽
TILING_DATA_FIELD_DEF(int64_t, splitBlockH);            // 所切基本块的高
TILING_DATA_FIELD_DEF(int64_t, tilingKey);              //
TILING_DATA_FIELD_DEF(int64_t, dimNeg2Tail);            // -2轴方向尾块
TILING_DATA_FIELD_DEF(int64_t, dimNeg1Tail);            // -1轴方向尾块
TILING_DATA_FIELD_DEF(int64_t, dimNeg2SplitBlockNum);   // -2轴切分基本块的个数
TILING_DATA_FIELD_DEF(int64_t, dimNeg1BlockNum);        // 尾轴切分基本块的个数
TILING_DATA_FIELD_DEF(int64_t, blockPerHeadCore);       // 正常核计算的task数
TILING_DATA_FIELD_DEF(int64_t, blockPerTailCore);       // 尾核计算的task数
TILING_DATA_FIELD_DEF(int64_t, headCoreNum);            // 正常核个数
TILING_DATA_FIELD_DEF(int64_t, dimNeg2IsOdd);           // 量化轴block数是否是奇数
TILING_DATA_FIELD_DEF(int64_t, dimNeg1IsOdd);           // 尾轴block数是否为奇数
TILING_DATA_FIELD_DEF(int64_t, dimNeg1IsPad);           // 尾轴是否需要32对齐
TILING_DATA_FIELD_DEF(int64_t, blockCountPerBatch);     // 一个batch轴切分块数
TILING_DATA_FIELD_DEF(int64_t, scale1ColCountPerBatch); // 一个batch轴-1轴的scale列数
TILING_DATA_FIELD_DEF(int64_t, scale2RowCountPerBatch); // 一个batch轴-2轴的scale的行数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicMxQuantWithDualAxis, DynamicMxQuantWithDualAxisTilingData)

struct DynamicMxQuantWithDualAxisCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicMxQuantWithDualAxisTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t roundMode;
    int64_t dstType{0};
    int64_t scaleAlg{0};
    int64_t blockSize{0};
    int64_t blockW{0};
    int64_t splitBlockH{0};
    int64_t dim0{1};
    int64_t dimNeg2{1};
    int64_t dimNeg1{1};
    int64_t dimNeg2SplitBlockNum{0};
    int64_t dimNeg1BlockNum{0};
    int64_t dimNeg2Tail{0};
    int64_t dimNeg1Tail{0};
    int64_t groupPerUb{0};
    int64_t totalTaskNum{0};
    int64_t blockPerHeadCore{0};
    int64_t blockPerTailCore{0};
    int64_t headCoreNum{0};
    int64_t dimNeg2IsOdd{0};
    int64_t dimNeg1IsOdd{0};
    int64_t dimNeg1IsPad{0};
    int64_t tilingKey{0};
    int64_t blockCountPerBatch{0};
    int64_t scale1ColCountPerBatch{0};
    int64_t scale2RowCountPerBatch{0};
    int64_t workspaceSize{0};
};

enum class RoundModeList
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class DynamicMxQuantWithDualAxisTiling {
public:
    explicit DynamicMxQuantWithDualAxisTiling(gert::TilingContext* context) : context_(context)
    {}
    ~DynamicMxQuantWithDualAxisTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus MergeAxis();
    ge::graphStatus SetTilingParams();

    void SetTilingKey();
    void SetTilingData();
    void PrintTilingData();
    void SplitCore(int64_t blockW, int64_t blockSize);

    RoundModeList GetRoundMode(const std::string& roundMode);

private:
    uint64_t roundMode_ = 0;
    uint64_t scaleAlg_ = 0;
    uint64_t mode_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicMxQuantWithDualAxisTilingData tilingData;
    DynamicMxQuantWithDualAxisTilingParam tilingParams;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H