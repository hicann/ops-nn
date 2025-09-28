/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// mc2编译依赖
// 
// mc2编译依赖
/*!
 * \file quant_batch_matmul_v3_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V3_TILING_H
#define QUANT_BATCH_MATMUL_V3_TILING_H

#include "quant_batch_matmul_v3_tiling_base.h"

namespace optiling {

// QuantBatchMatmulV3Tiling set QuantBatchMatmulV3Params tilingData, mc2 calls QuantBatchMatmulV3Tiling DoLibApiTiling
BEGIN_TILING_DATA_DEF(QuantBatchMatmulV3Params)
    TILING_DATA_FIELD_DEF(uint32_t, batchA);
    TILING_DATA_FIELD_DEF(uint32_t, batchB);
    TILING_DATA_FIELD_DEF(uint32_t, batchC);
    TILING_DATA_FIELD_DEF(uint32_t, batchA1);
    TILING_DATA_FIELD_DEF(uint32_t, batchA2);
    TILING_DATA_FIELD_DEF(uint32_t, batchA3);
    TILING_DATA_FIELD_DEF(uint32_t, batchA4);
    TILING_DATA_FIELD_DEF(uint32_t, batchB1);
    TILING_DATA_FIELD_DEF(uint32_t, batchB2);
    TILING_DATA_FIELD_DEF(uint32_t, batchB3);
    TILING_DATA_FIELD_DEF(uint32_t, batchB4);
    TILING_DATA_FIELD_DEF(uint32_t, batchC1);
    TILING_DATA_FIELD_DEF(uint32_t, batchC2);
    TILING_DATA_FIELD_DEF(uint32_t, batchC3);
    TILING_DATA_FIELD_DEF(uint32_t, batchC4);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, isPerTensor);
    TILING_DATA_FIELD_DEF(uint32_t, isPertoken);
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleScale);
    TILING_DATA_FIELD_DEF(uint32_t, biasThreeDim);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalcM);
    TILING_DATA_FIELD_DEF(uint32_t, ubCalcN);
    TILING_DATA_FIELD_DEF(uint32_t, needUbBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, realSingleCoreM);
    TILING_DATA_FIELD_DEF(uint32_t, realSingleCoreN);
    TILING_DATA_FIELD_DEF(uint32_t, biasDtype); //代替原来的isBiasBf16
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
    TILING_DATA_FIELD_DEF(uint32_t, isMClash);
    TILING_DATA_FIELD_DEF(uint32_t, isNClash);
    TILING_DATA_FIELD_DEF(uint32_t, groupSizeM);
    TILING_DATA_FIELD_DEF(uint32_t, groupSizeN);
    TILING_DATA_FIELD_DEF(uint32_t, groupSizeK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3ParamsOp, QuantBatchMatmulV3Params)

BEGIN_TILING_DATA_DEF(L2cacheTileParam)
    TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
    TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
    TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
    TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
    TILING_DATA_FIELD_DEF(uint32_t, calOrder);
    TILING_DATA_FIELD_DEF(uint32_t, isBasicTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheTileParamOp, L2cacheTileParam)

BEGIN_TILING_DATA_DEF(SlidingWindowParam)
    TILING_DATA_FIELD_DEF(uint32_t, mTailTile);
    TILING_DATA_FIELD_DEF(uint32_t, nTailTile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SlidingWindowParamOp, SlidingWindowParam)

BEGIN_TILING_DATA_DEF(QuantBatchMatmulV3TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(QuantBatchMatmulV3Params, params);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF_STRUCT(L2cacheTileParam, tileL2cacheTiling);
    TILING_DATA_FIELD_DEF_STRUCT(SlidingWindowParam, adaptiveSlidingWin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3, QuantBatchMatmulV3TilingData)
REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3TilingDataOp, QuantBatchMatmulV3TilingData)

class QuantBatchMatmulV3Tiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit QuantBatchMatmulV3Tiling(gert::TilingContext *context);
    QuantBatchMatmulV3Tiling(gert::TilingContext *context, QuantBatchMatmulV3TilingData *out);
    ~QuantBatchMatmulV3Tiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();
    bool CheckDtypeOnOnlyL0c2ub() const;
    bool CheckDtypeOnOnlyL0c2out() const;
    bool CheckDtypeOnOnlyL0c2outForSupportedList() const;
    bool CheckDtypeOnOnlyL0c2outForA4W4() const;
    bool CheckDtypeOnOnlyL0c2outForPertoken() const;
    bool CheckDtypeOnOnlyL0c2outForX1NZ() const;
    bool CheckDtypeOnOnlyL0c2outForUnclassified() const;
    uint64_t GetTilingKey(bool isBasicTiling) const;
    bool GetUbDequantExtreSpace() override;
    ge::graphStatus CalcUbTiling() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const override;
    bool SetMatmulTilingFromTbeTiling();
    bool GetTbeTiling();
    void ProcessMSmall();
    int32_t GetIteratorOrder();
    void PrintTilingData();
    void PrintTbeTiling();
    void PrintTilingParams() const;

    ge::graphStatus CalcPertokenOptUbTiling();
    ge::graphStatus CalcUbTiling(uint32_t baseN, uint32_t baseM);
    void SpiltSingleCore(int32_t &singleCoreM, int32_t &singleCoreN);
    void SpiltForWorkSpaceLimit(int32_t singleCoreM, int32_t singleCoreN, int32_t blockDim);
    bool SetBlockDimsAndSingleCore(TCubeTiling &mt);
    bool CalcUsedL1AndUBSize(int32_t aL1Size, int32_t bL1Size, bool &fallback);
    bool CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const;
    ge::graphStatus InitTilingData(matmul_tiling::MatmulApiTilingBase &mm, bool fallback = false);
    void Int4LowerAxisAlign(uint64_t &baseM, uint64_t &baseN) const;
    uint64_t CalcL1SizeForBiasAndScale();
    int32_t CalcND2NZSpace() const;
    void ConstructCacheParams(BatchmatmulCompileParas &compileParams, BatchmatmulRunParas &runParams) const;
    void ModifyCacheParams(BatchmatmulRunParas &runParams) const;
    bool NeedAtomiClean() const;
    
    bool CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const;
    
    bool CheckShapeInRangeForOptionalInputs(const gert::StorageShape* biasShape,
                                            const gert::StorageShape* pertokenShape) const;
    bool BiasShapeCheck(const gert::Shape &biasShape) const;
    uint64_t GetTotalSize(uint64_t m, uint64_t k, uint64_t n) const;
    
    uint32_t GetABankConflictSize();
    void UpdateSmallMTbeTiling();
    void UpdateSmallMTbeTiling(uint64_t baseM, uint64_t baseN, uint64_t baseK);
    void SetQuantBatchMatmulRunParas(QuantBatchMatmulRunParas& runParams, const optiling::QuantBatchMatmulInfo& inputParams);
    // 新增数据成员请注意：如果是在GetShapeAttrsInfo函数过程中获取的，请放到QuantBatchMatmulInfo结构体中，或者保证在DoOpTiling赋值
    QuantBatchMatmulV3TilingData tilingDataSelf_;
    QuantBatchMatmulV3TilingData &tilingData_;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_TILING_H