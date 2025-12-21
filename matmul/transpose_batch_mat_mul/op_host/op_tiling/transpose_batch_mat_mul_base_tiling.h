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
 * \file transpose_batch_mat_mul_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_BASE_TILING_H__
#define __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_BASE_TILING_H__

#include "transpose_batch_mat_mul_tiling.h"
#include "tiling_base/tiling_base.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {
namespace transpose_batch_mat_mul {
    struct BatchShapeInfo {
        uint64_t batchA = 1;
        uint64_t batchA0 = 1;
        uint64_t batchA1 = 1;
        uint64_t batchA2 = 1;
        uint64_t batchA3 = 1;
        uint64_t batchB = 1;
        uint64_t batchB0 = 1;
        uint64_t batchB1 = 1;
        uint64_t batchB2 = 1;
        uint64_t batchB3 = 1;
        uint64_t batchC = 1;
        uint64_t batchC0 = 1;
        uint64_t batchC1 = 1;
        uint64_t batchC2 = 1;
        uint64_t batchC3 = 1;
        bool biasWithBatch = false;
    };

enum class TilingCalcSelect : uint8_t //选择不同的计算Tiling的方法
{
    ALL = 0,
    COMMON = 1
};

class TransposeBatchMatMulBaseTiling : public matmul_v3::MatmulV3BaseTiling {
public:
    explicit TransposeBatchMatMulBaseTiling(gert::TilingContext* context)
       : MatmulV3BaseTiling(context, &tbmmTilingDataSelf_.matmulTiling) , tbmmTilingData_(tbmmTilingDataSelf_){
    }

    TransposeBatchMatMulBaseTiling(gert::TilingContext* context, TBMMTilingData &tbmmTilingData,
        TilingCalcSelect tilingSelect = TilingCalcSelect::COMMON)
       : MatmulV3BaseTiling(context, &tbmmTilingData.matmulTiling), tbmmTilingData_(tbmmTilingData) {
        tilingSelect_ = tilingSelect;
    }

    ~TransposeBatchMatMulBaseTiling() override {
    }

protected:
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override; //4
    // 6、保存Tiling数据
    ge::graphStatus PostTiling() override; //6
    // 7、计算TilingKey
    uint64_t GetTilingKey() const override; //7

    ge::graphStatus CheckArgs() override;
    ge::graphStatus GetArgs() override;
    ge::graphStatus GetShape();
    ge::graphStatus GetShapeMKN(const gert::Shape &aShape, const gert::Shape &bShape);
    ge::graphStatus GetShapeBatch(const gert::Shape &aShape, const gert::Shape &bShape);
    ge::graphStatus GetShapeBias();

    bool CheckBMMTilingDataIsVaild() const;
    void DoCommonTiling();
    TBMMTilingData &tbmmTilingData_;
    BatchShapeInfo batchInfo_;
    TilingCalcSelect tilingSelect_ = TilingCalcSelect::ALL;

private:
    const gert::ContinuousVector *aPermList_ = nullptr;
    const gert::ContinuousVector *bPermList_ = nullptr;
    uint64_t transA_;
    uint64_t transB_;
    int32_t batchSplitFactor_ = 1;
    TBMMTilingData tbmmTilingDataSelf_{};
    uint64_t aBatchDimAll_{1};
    uint64_t bBatchDimAll_{1};
    uint64_t cBatchDimAll_{1};
};
}
}
#endif // __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_BASE_TILING_H__
