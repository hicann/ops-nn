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
 * \file rms_norm_grad_quant_tiling.h
 * \brief RmsNormGradQuant tiling file
 */
#ifndef RMS_NORM_GRAD_QUANT_TILING_H
#define RMS_NORM_GRAD_QUANT_TILING_H

#include "op_common/log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "../op_kernel/arch35/rms_norm_grad_quant_tiling_data.h"
#include "../op_kernel/arch35/rms_norm_grad_quant_tiling_key.h"

namespace optiling {
namespace rms_norm_grad_quant {

enum class ComputeModeDx : uint64_t {
    FULL_LOAD = 0,
    SPLIT_D = 1,
};

enum class ComputeModeDgamma : uint64_t {
    FULL_LOAD = 0,
    BIG_M = 1,
    WITH_LARGE_ROWS = 2,
    EMPTY = 3,
};

enum class ComputeModeOffsetX : uint64_t {
    WITH_OFFSET_X = 0,
    WITHOUT_OFFSET_X = 1,
};

enum class ComputeModeDivMode : uint64_t {
    DIV_MODE = 0,
    NOT_DIV_MODE = 1,
};

class RmsNormGradQuantTilingKey {
public:
    RmsNormGradQuantTilingKey& SetComputeModeDx(ComputeModeDx computeModeDx)
    {
        computeModeDx_ = computeModeDx;
        return *this;
    }

    RmsNormGradQuantTilingKey& SetComputeModeDgamma(ComputeModeDgamma computeModeDgamma)
    {
        computeModeDgamma_ = computeModeDgamma;
        return *this;
    }

    RmsNormGradQuantTilingKey& SetComputeModeOffsetX(ComputeModeOffsetX computeModeOffsetX)
    {
        computeModeOffsetX_ = computeModeOffsetX;
        return *this;
    }

    RmsNormGradQuantTilingKey& SetComputeModeDivMode(ComputeModeDivMode computeModeDivMode)
    {
        computeDivMode_ = computeModeDivMode;
        return *this;
    }

    uint64_t GetTilingKey() const
    {
        return GET_TPL_TILING_KEY(static_cast<uint64_t>(computeModeDx_), static_cast<uint64_t>(computeModeDgamma_),
                                  static_cast<uint64_t>(computeModeOffsetX_), static_cast<uint64_t>(computeDivMode_));
    }

private:
    ComputeModeDx computeModeDx_ = ComputeModeDx::FULL_LOAD;
    ComputeModeDgamma computeModeDgamma_ = ComputeModeDgamma::FULL_LOAD;
    ComputeModeOffsetX computeModeOffsetX_ = ComputeModeOffsetX::WITH_OFFSET_X;
    ComputeModeDivMode computeDivMode_ = ComputeModeDivMode::DIV_MODE;
};
} // namespace rms_norm_grad_quant

class RmsNormGradQuantRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormGradQuantRegbaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CalcTilingDataDx();

protected:
    uint64_t ubSize_{0};
    uint32_t blockSize_{0};
    uint32_t vecRegSize_{0};
    uint32_t vlFp32_{0};
    uint32_t aivCoreNum_{0};
    uint32_t usedCoreNumDx_{0};
    int64_t rows_{0}; // A轴
    int64_t cols_{0}; // R轴
    int64_t blockFactorDx_{0};
    int64_t bodyPart_{0};
    int64_t colsPerCore_{0};
    int64_t rowsPerCore_{0};
    // params for dgamma
    int64_t usedCoreNumDG_{0};
    int64_t colsPerCoreDG_{0};
    int64_t colsPerTailCoreDG_{0};
    int64_t binaryAddNumDG_{0};
    int64_t colsPerLoopDG_{0};
    int64_t rowsPerUBDG_{0};
    int64_t cols_sets_{0};
    int64_t colsPerUBDG_{0};
    uint32_t isFullLoadDG_{0};
    uint32_t isMultiColset_{0};
    int64_t colsLastCoreDG_{0};
    int64_t isPowerofTwoDG_{0};
    int32_t powerofTwoValueDG_{0};
    int32_t rowsTailDG_{0};
    int32_t maxRowsNumDG_{0};
    int32_t totalBlockCountDG_{0};
    int32_t mainBlockCountDG_{0};
    int32_t tailBlockCountwithPadDG_{0};
    int32_t powerOfTwoBlockCountDG_{0};
    int32_t tailBlockCountWithoutPadDG_{0};
    int32_t binaryAddKDG_{0};
    int64_t tailCoreNumDG_{0};
    // quant
    rms_norm_grad_quant::ComputeModeOffsetX hasOffsetX_;
    rms_norm_grad_quant::ComputeModeDivMode divMode_;
    int64_t quantMode_{0};
    // compuet_mode
    rms_norm_grad_quant::ComputeModeDx computeModeDx_;
    rms_norm_grad_quant::ComputeModeDgamma computeModeDgamma_;

    ge::DataType dyDtype_{ge::DataType::DT_FLOAT};
    ge::DataType gammaDtype_{ge::DataType::DT_FLOAT};
    uint32_t tilingKey_{0};
    RmsNormGradQuantRegbaseTilingData tilingData_;
    ge::graphStatus CheckShapeBeSameWithOne(gert::Shape& shape);
    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckShapeDimNum(gert::Shape& shape, int64_t minDimNum, int64_t maxDimNum, const char* name);
    ge::graphStatus CheckShapePrefixMatch(gert::Shape& prefixShape, gert::Shape& fullShape, const char* prefixName,
                                          const char* fullName);
    ge::graphStatus CheckShapeSuffixMatch(gert::Shape& suffixShape, gert::Shape& fullShape, const char* suffixName,
                                          const char* fullName);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtype();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape);
    void CalcUsedCoreNumGamma();
    ge::graphStatus CalcUbBufferSizeDgamma();
    ge::graphStatus CalcTilingDataDgamma();
    int32_t CalcRowsTails();
    int64_t GetSizeOfBlockAlign(int64_t nonAlignSize);
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    void LogTilingResult();
};

class RmsNormGradQuantEmptyTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormGradQuantEmptyTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

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
    uint32_t aivCoreNum_{0};
    uint64_t rows_{0};
    uint64_t cols_{0};
    uint64_t usedCoreNumDG_{0};
    uint64_t colsPerCoreDG_{0};
    uint64_t colsPerUBDG_{0};
    uint64_t tailUbCols_{0};
    uint64_t lastCoreBlockCount_{0};
    uint64_t lastCoreTailUbCols_{0};
    uint64_t coreUbBlockCount_{0};
    uint64_t colsLastCoreDG_{0};
    uint64_t ubSize_{0};

    uint32_t tilingKey_{0};
    RmsNormGradQuantRegbaseEmptyTilingData tilingData_;

    rms_norm_grad_quant::ComputeModeOffsetX hasOffsetX_;
    rms_norm_grad_quant::ComputeModeDivMode divMode_;
    rms_norm_grad_quant::ComputeModeDx computeModeDx_;
    rms_norm_grad_quant::ComputeModeDgamma computeModeDgamma_;
    int64_t quantMode_{0};

    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtype();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& gammaShape);
    ge::graphStatus CalcTilingDataDgamma();
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    ge::graphStatus CalcUsedCoreNumGamma();
    void LogTilingResult();
};

class RmsNormGradQuantBigMTiling : public RmsNormGradQuantRegbaseTiling {
public:
    explicit RmsNormGradQuantBigMTiling(gert::TilingContext* context) : RmsNormGradQuantRegbaseTiling(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    RmsNormGradQuantRegbaseBigMTilingData tilingData_;

    ge::graphStatus DgammaDoTiling();
    ge::graphStatus DgammaDoTilingStg0();
    ge::graphStatus DgammaDoTilingStg1();

    int64_t GetCacheID(const int64_t idx);
    int64_t FindNearestPower2(const int64_t value);

    int64_t usedCoreNumDgamma_{0};
};
} // namespace optiling
#endif // RMS_NORM_GRAD_QUANT_TILING_H
