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
 * \file weight_quant_batch_matmul_v2_weight_nz_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (29) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_WEIGHT_NZ_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_WEIGHT_NZ_TILING_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "op_cache_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2NzTilingData)
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint16_t, AL1Pingpong);
TILING_DATA_FIELD_DEF(uint16_t, BL1Pingpong);
TILING_DATA_FIELD_DEF(uint64_t, kAlign);
TILING_DATA_FIELD_DEF(uint64_t, nAlign);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, mAubSize);
TILING_DATA_FIELD_DEF(uint64_t, kAubSize);
TILING_DATA_FIELD_DEF(uint64_t, nBubSize);
TILING_DATA_FIELD_DEF(uint64_t, kBubSize);
TILING_DATA_FIELD_DEF(uint64_t, mCubSize);
TILING_DATA_FIELD_DEF(uint64_t, nCubSize);
TILING_DATA_FIELD_DEF(uint64_t, mAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, nBL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kBL1Size);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF(uint64_t, batchX0);
TILING_DATA_FIELD_DEF(uint64_t, batchX1);
TILING_DATA_FIELD_DEF(uint64_t, batchX2);
TILING_DATA_FIELD_DEF(uint64_t, batchX3);
TILING_DATA_FIELD_DEF(uint64_t, batchWeight0);
TILING_DATA_FIELD_DEF(uint64_t, batchWeight1);
TILING_DATA_FIELD_DEF(uint64_t, batchWeight2);
TILING_DATA_FIELD_DEF(uint64_t, batchWeight3);
TILING_DATA_FIELD_DEF(uint64_t, batchY0);
TILING_DATA_FIELD_DEF(uint64_t, batchY1);
TILING_DATA_FIELD_DEF(uint64_t, batchY2);
TILING_DATA_FIELD_DEF(uint64_t, batchY3);
TILING_DATA_FIELD_DEF(uint32_t, biasWithBatch);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13195213800193, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13195482235649, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13196287542017, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13196555977473, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13197361283841, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13197629719297, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13470091707137, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13470360142593, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13471165448961, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13471433884417, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13472239190785, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_13472507626241, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139095120642817, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139095389078273, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139096194384641, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139096462820097, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139097268126465, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139097536561921, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139369998549761, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139370266985217, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139371072291585, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139371340727041, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139372146033409, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1139372414468865, WeightQuantBatchMatmulV2NzTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2NzTilingDataOp, WeightQuantBatchMatmulV2NzTilingData)

class MmTilingInput
{
public:
    matmul_tiling::TPosition aPosition;
    matmul_tiling::TPosition bPosition;
    matmul_tiling::TPosition cPosition;
    matmul_tiling::TPosition biasPosition;
    matmul_tiling::CubeFormat aFormat = matmul_tiling::CubeFormat::NZ;
    matmul_tiling::CubeFormat bFormat = matmul_tiling::CubeFormat::NZ;
    matmul_tiling::CubeFormat cFormat = matmul_tiling::CubeFormat::NZ;
    matmul_tiling::CubeFormat biasFormat = matmul_tiling::CubeFormat::NZ;
    matmul_tiling::DataType aDtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bDtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType biasDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType cDtype = matmul_tiling::DataType::DT_FLOAT16;
};

class WeightQuantBatchMatmulV2WeightNz : public TilingBaseClass
{
public:
    explicit WeightQuantBatchMatmulV2WeightNz(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    explicit WeightQuantBatchMatmulV2WeightNz(gert::TilingContext* context, WeightQuantBatchMatmulV2NzTilingData* out)
        : TilingBaseClass(context)
    {
        Reset();
        tilingData_ = out;
        InitCompileInfo();
    }
    ~WeightQuantBatchMatmulV2WeightNz() override = default;
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    enum class FullLoadMode
    {
        NONE_AB_K = 0,
        FULL_AKL1,
        FULL_BKL1,
        FULL_K,
        FULL_K_REORDER_MN
    };

    ge::Format aFormat;
    ge::Format bFormat;

    void SetMatmulTiling();
    void GetubFactorTiling();
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    bool GetMmTilingInput(MmTilingInput& mmTilingInput);
    bool CheckUBSize();
    bool GetMatMulTiling();
    ge::graphStatus CheckContext() const;
    bool AnalyzeDtype();
    bool AnalyzeBiasDtype(const gert::CompileTimeTensorDesc* biasDesc);
    bool AnalyzeAntiQuantDtype(
        ge::DataType antiQuantScaleDtype, const gert::CompileTimeTensorDesc* antiQuantOffsetDesc) const;
    bool AnalyzeAttrs();
    bool AnalyzeInputs();
    bool AnalyzeInputShape(
        const gert::StorageShape* xShape, const gert::StorageShape* weightShape, const gert::StorageShape* outShape);
    bool AnalyzeAntiQuantShape(
        const gert::StorageShape* antiQuantScaleShape, const gert::StorageShape* antiQuantOffsetShape);
    bool AnalyzeBiasShape(const gert::StorageShape* outShape);
    bool SetAntiQuantType(const gert::StorageShape* antiQuantScaleShape);
    void Convert2AscendCTiling(const CacheTilingData& tbeTiling, TCubeTiling& matmulTiling);
    void SetAscendCTiling(TCubeTiling& matmulTiling);
    MatrixTraverse GetIteratorOrder(
        const CacheTilingData& tbeTiling, int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK) const;
    void GetBaseMKNByTrans(matmul_tiling::MatmulApiTiling& mmTiling) const;
    bool GetLoopOrder();

    // 判断A/B在L1是否开pingpong
    void GetL1Pingpong();
    void GetL1tiling();
    uint64_t GetBubSize(uint64_t bubN, uint64_t bubD) const;
    uint64_t GetAubSize(uint64_t aubN, uint64_t aubD) const;
    uint64_t CalBubFactorTiling(uint64_t bubCanUseUbSize);
    uint64_t CalAubFactorTiling(uint64_t aubCanUseUbSize);
    uint64_t CalCubFactorTiling(uint64_t cubNz2NdCanUseSize);
    void PrintCVTilingData(bool debugLevel);
    ge::graphStatus PostTiling() override;
    WeightQuantBatchMatmulInfo inputParams_;
    std::unique_ptr<WeightQuantBatchMatmulV2NzTilingData> tilingDataManager_;
    std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo> compileInfoPtr_;
    WeightQuantBatchMatmulV2NzTilingData* tilingData_ = nullptr;
    const char* opName_;
    bool InvokeCacheTiling();
    virtual bool GetTilingFromCache();
    void Reset();
    void InitCompileInfo();

private:
    int32_t cubeBaseM_ = -1;
    int32_t cubeBaseN_ = -1;
    int32_t defaultValue_ = -1;
    int32_t cubeSingleMinK_ = -1;
    FullLoadMode L1FullloadMode_ = FullLoadMode::NONE_AB_K;
    uint64_t weightBlockAlignSize_ = 32;
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H