/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_sum_tiling.h
 * \brief
 */
#ifndef __OP_HOST_QUANT_MATMUL_REDUCE_SUM_TILING_H__
#define __OP_HOST_QUANT_MATMUL_REDUCE_SUM_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

BEGIN_TILING_DATA_DEF(QuantMatmulReduceSumParams)
    TILING_DATA_FIELD_DEF(uint32_t, batchNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
    TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
    TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
    TILING_DATA_FIELD_DEF(uint32_t, isPertoken);
    TILING_DATA_FIELD_DEF(uint32_t, isDetermine);
    TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantMatmulReduceSumParamsOp, QuantMatmulReduceSumParams)

BEGIN_TILING_DATA_DEF(QuantMatmulReduceSumTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(QuantMatmulReduceSumParams, qbmmReduceSumParams);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantMatmulReduceSum, QuantMatmulReduceSumTilingData)
REGISTER_TILING_DATA_CLASS(QuantMatmulReduceSumTilingDataOp, QuantMatmulReduceSumTilingData)

struct QuantMatmulReduceSumCompileInfo {
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l2Size{0};
    uint64_t l0ASize{0};
    uint64_t l0BSize{0};
    uint64_t l0CSize{0};
    uint64_t aicNum{0};
    uint64_t cubeFreq{0};
};

struct QuantMatmulReduceSumInfo {
public:
    int64_t usedCoreNum = 0;
    int64_t mSize = 0;
    int64_t kSize = 0;
    int64_t nSize = 0;
    int64_t batchNum = 0;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseK = 0;
    bool initFlag = false;  // 避免重复解析flag
    bool transA = false;
    bool transB = false;
    bool isPertoken = false;
    ge::DataType aDtype = ge::DT_INT8;
    ge::DataType bDtype = ge::DT_INT8;
    ge::DataType cDtype = ge::DT_INT32;
    ge::DataType yDtype = ge::DT_BF16;
    ge::DataType x1scaleDtype = ge::DT_FLOAT;
    ge::DataType x2scaleDtype = ge::DT_BF16;
    const char *opName = "QuantMatmulReduceSum";
};

class QuantMatmulReduceSumTiling : public TilingBaseClass {
public:
    explicit QuantMatmulReduceSumTiling(gert::TilingContext* context) : TilingBaseClass(context){
        InitCompileInfo();
    };
    ~QuantMatmulReduceSumTiling() override = default;

protected:
    bool IsCapable() override {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeAttrs();
    ge::graphStatus AnalyzeShapes();
    void PrintTilingData();
    ge::graphStatus CalUbSize();
    bool SetMatmulTiling();
    bool SetQbmmTiling();
    
    QuantMatmulReduceSumTilingData tilingData_;
    QuantMatmulReduceSumInfo inputParams_;
    bool isUbQuant_ = false;
    QuantMatmulReduceSumCompileInfo compileInfo_;
    size_t tilingDataSize_ = 0;
};
}  // namespace optiling
#endif  // __OP_HOST_QUANT_MATMUL_REDUCE_SUM_TILING_H__
