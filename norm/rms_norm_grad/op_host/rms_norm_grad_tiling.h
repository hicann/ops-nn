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
 * \file rms_norm_grad_tiling.h
 * \brief RmsNormGrad tiling file
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H

#include "op_common/log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RmsNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, data_type);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_split_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_loop);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_loop);
TILING_DATA_FIELD_DEF(uint32_t, fixed_output);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGrad, RmsNormGradTilingData);

BEGIN_TILING_DATA_DEF(RmsNormGradRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDx);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint32_t, vlFp32);
TILING_DATA_FIELD_DEF(uint32_t, isFullLoad);
TILING_DATA_FIELD_DEF(uint32_t, isMultiColset);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNumDG);
TILING_DATA_FIELD_DEF(int64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(int64_t, colsLastCoreDG);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, blockFactorDx);
TILING_DATA_FIELD_DEF(int64_t, bodyPart);
TILING_DATA_FIELD_DEF(int64_t, colsPerTailCoreDG);
TILING_DATA_FIELD_DEF(int64_t, rowsPerUBDG);
TILING_DATA_FIELD_DEF(int32_t, powerofTwoValueDG);
TILING_DATA_FIELD_DEF(int32_t, rowsTailDG);
TILING_DATA_FIELD_DEF(int32_t, totalBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, mainBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, tailBlockCountwithPadDG);
TILING_DATA_FIELD_DEF(int32_t, powerOfTwoBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, tailBlockCountWithoutPadDG);
TILING_DATA_FIELD_DEF(int32_t, binaryAddKDG);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7000, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7001, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7010, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7011, RmsNormGradRegbaseTilingData);

BEGIN_TILING_DATA_DEF(RmsNormGradEmptyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(uint64_t, cols);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint64_t, colsPerUBDG);
TILING_DATA_FIELD_DEF(uint64_t, coreUbBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, tailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreTailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, colsLastCoreDG);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNormGrad_8000, RmsNormGradEmptyTilingData);

class RmsNormGradRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormGradRegbaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t ubSize_;
    uint32_t blockSize_;
    uint32_t vecRegSize_;
    uint32_t vlFp32_;
    uint32_t aivCoreNum_;
    uint32_t usedCoreNumDx_;
    int64_t rows_; // A轴
    int64_t cols_; // R轴
    int64_t blockFactorDx_;
    int64_t bodyPart_;
    int64_t colsPerCore_;
    int64_t rowsPerCore_;
    // params for dgamma
    int64_t usedCoreNumDG_;
    int64_t colsPerCoreDG_;
    int64_t colsPerTailCoreDG_;
    int64_t binaryAddNumDG_;
    int64_t colsPerLoopDG_;
    int64_t rowsPerUBDG_;
    int64_t cols_sets_;
    int64_t colsPerUBDG_;
    uint32_t isFullLoadDG_;
    uint32_t isMultiColset_;
    int64_t colsLastCoreDG_;
    int64_t isPowerofTwoDG_;
    int32_t powerofTwoValueDG_;
    int32_t rowsTailDG_;
    int32_t maxRowsNumDG_;
    int32_t totalBlockCountDG_;
    int32_t mainBlockCountDG_;
    int32_t tailBlockCountwithPadDG_;
    int32_t powerOfTwoBlockCountDG_;
    int32_t tailBlockCountWithoutPadDG_;
    int32_t binaryAddKDG_;
    int64_t tailCoreNumDG_;

    ge::DataType dataType_;
    uint32_t tilingKey_;
    RmsNormGradRegbaseTilingData tilingData_;

    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtypeAndFormat();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape);
    void CalcUsedCoreNumGamma();
    ge::graphStatus CalcUbBufferSizeDgamma();
    ge::graphStatus CalcTilingDataDgamma();
    ge::graphStatus CalcTilingDataDx();
    int32_t CalcRowsTails();
    int64_t GetSizeOfBlockAlign(int64_t nonAlignSize);
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    void LogTilingResult();
};

class RmsNormGradEmptyTiling : public Ops::NN::Optiling::TilingBaseClass  {
public:
    explicit RmsNormGradEmptyTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;

    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:

    uint32_t aivCoreNum_;
    uint64_t rows_; 
    uint64_t cols_; 
    uint64_t usedCoreNumDG_;
    uint64_t colsPerCoreDG_;
    uint64_t colsPerUBDG_;
    uint64_t tailUbCols_;
    uint64_t lastCoreBlockCount_;
    uint64_t lastCoreTailUbCols_;
    uint64_t coreUbBlockCount_;
    uint64_t colsLastCoreDG_;
    uint64_t ubSize_;

    uint32_t tilingKey_;
    RmsNormGradEmptyTilingData tilingData_;

    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtypeAndFormat();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape);
    ge::graphStatus CalcTilingDataDgamma();
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    void CalcUsedCoreNumGamma();
    void LogTilingResult();
};


} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
