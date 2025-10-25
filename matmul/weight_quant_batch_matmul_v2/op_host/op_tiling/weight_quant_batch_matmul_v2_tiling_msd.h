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
 * \file weight_quant_batch_matmul_v2_tiling_msd.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (29) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H

#include "weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2MsdTilingData)
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint16_t, v1BaseM);
TILING_DATA_FIELD_DEF(uint16_t, preloadTimes);
TILING_DATA_FIELD_DEF(uint16_t, taskNSize);
TILING_DATA_FIELD_DEF(uint16_t, taskSingleCoreNSize);
TILING_DATA_FIELD_DEF(uint16_t, postProcessBaseM);
TILING_DATA_FIELD_DEF(uint16_t, postProcessSingleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, preProcessUsedVecNum);
TILING_DATA_FIELD_DEF(uint32_t, v1BaseK);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupPack);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365332066075393, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365057188168449, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365332602946305, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365057725039361, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367531089330945, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367256211424001, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367531626201857, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367256748294913, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356535973053185, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356537046795009, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356261095146241, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356262168888065, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356536509924097, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_356261632017153, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_363133579690753, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_360934556435201, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_362858701783809, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_360659678528257, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_358734996308737, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_358460118401793, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_358735533179649, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_358460655272705, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367514983862529, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367514982813953, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365315960606977, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367240105955585, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_367240104907009, WeightQuantBatchMatmulV2MsdTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_365041082700033, WeightQuantBatchMatmulV2MsdTilingData)

class WeightQuantBatchMatmulV2Msd : public WeightQuantBatchMatmulV2Tiling
{
public:
    explicit WeightQuantBatchMatmulV2Msd(gert::TilingContext* context) : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2Msd() override = default;

protected:
    uint32_t order_ = 2; // 展开的阶数
    uint32_t blkDim_ = 0;
    bool splitKFlag_;
    bool highPrecision_;
    uint64_t cubeBaseN_;
    std::unique_ptr<WeightQuantBatchMatmulV2MsdTilingData> tilingData_;

    void Reset();
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoMSDGeneralOpTiling();
    ge::graphStatus DoOpTiling() override;
    uint64_t SplitKByKBlock(uint64_t kBlockNum) const;
    ge::graphStatus DoMSDGroupSplitKOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    bool CheckCacheTiling();
    bool CheckInt4MatmulTiling() const;
    bool CheckInt8MatmulTiling(uint64_t singleCoreNCalc) const;
    bool InvokeCacheTiling();
    bool GetMatMulTiling();
    void ReviseMMTiling() const;
    bool GetTilingFromCache();
    uint64_t GetInnerPreciseTilingKey() const;
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_H