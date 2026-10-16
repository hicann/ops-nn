/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_HI_F4_QUANT_TILING_H
#define RMS_NORM_HI_F4_QUANT_TILING_H

#include <cmath>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../op_kernel/arch92/rms_norm_hi_f4_quant_tiling_data.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace Ops::NN::Optiling;

namespace optiling {
constexpr uint64_t X_INDEX = 0;
constexpr uint64_t GAMMA_INDEX = 1;
constexpr uint64_t BETA_INDEX = 2;

constexpr uint64_t Y_INDEX = 0;
constexpr uint64_t SCALE_INDEX = 1;
constexpr uint64_t RSTD_INDEX = 2;

constexpr uint64_t EPS_ATTR_INDEX = 0;
constexpr uint64_t OUTPUT_RSTD_ATTR_INDEX = 1;
constexpr uint64_t SCALE_ALG_ATTR_INDEX = 2;

constexpr uint32_t ULONG_BIT_LEN = 64;
constexpr uint32_t MAX_DIM_CNT = 7;
constexpr uint32_t HIF4_BLOCK_SIZE = 64;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t B16_SIZE = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint64_t ALIGN_FACTOR_512 = 512;
constexpr uint64_t COL_ALIGN_NUM = 64;
constexpr uint32_t UB_RESERVE_FOR_RSTD_ALIGN = 1024;
constexpr uint64_t ARND_REDUCE_EMPTY_PRIORITY = 500;
constexpr uint64_t ARND_R_FULL_LOAD_PRIORITY = 1000;
constexpr uint64_t ARND_SPLIT_D_PRIORITY = 2000;
constexpr uint64_t FULL_LOAD_R_MAX = 16384;

struct RmsNormHiF4QuantCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
};

class RmsNormHiF4QuantRegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormHiF4QuantRegbaseTilingBase(gert::TilingContext* tilingContext)
        : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~RmsNormHiF4QuantRegbaseTilingBase() override {}

    void Reset(gert::TilingContext* context) override { TilingBaseClass::Reset(context); }

    const string nodeName = "RmsNormHiF4QuantRegbaseTiling";

    ge::graphStatus CheckShapeNull();
    ge::graphStatus CheckOptionalInput();
    ge::graphStatus CheckInputShapeDim();
    ge::graphStatus CheckInputShapeValue();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckOutputDtype();
    ge::graphStatus CheckOutputShapeValue();
    ge::graphStatus SetInputParams();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;

protected:
    uint64_t totalCoreNum_{0};
    uint64_t maxUbSize_{0};
    uint64_t vecLengthFP32_{0};
    uint64_t ubBlockSize_{0};

    uint64_t numRow_{0};
    uint64_t numCol_{0};
    uint64_t numColAlign_{0};

    uint64_t xDtypeSize_{2};
    uint64_t gammaDtypeSize_{2};
    ge::DataType xDtype_{Ops::NN::DT_FLOAT16};
    ge::DataType gammaDtype_{Ops::NN::DT_FLOAT16};

    uint64_t blockFactor_{0};
    uint64_t rowFactor_{0};
    uint64_t usedCoreNum_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    uint64_t binAddQuotient_{0};

    float epsilon_{1e-6};
    float avgFactor_{0.0};
    int64_t scaleAlg_{0};

    uint64_t mxBlockSize_{HIF4_BLOCK_SIZE};
    uint64_t blockNumInColAxis_{0};
    uint64_t mxScaleSize_{0};

    uint32_t betaFlag_{0};
    uint32_t rstdFlag_{0};
    bool gammaIsFp32_{false};
};

class RmsNormHiF4QuantRFullLoadTiling : public RmsNormHiF4QuantRegbaseTilingBase {
public:
    explicit RmsNormHiF4QuantRFullLoadTiling(gert::TilingContext* context) : RmsNormHiF4QuantRegbaseTilingBase(context)
    {}
    ~RmsNormHiF4QuantRFullLoadTiling() override = default;
    void Reset(gert::TilingContext* context) override { RmsNormHiF4QuantRegbaseTilingBase::Reset(context); }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    uint64_t CalUBTotalSize();
    uint64_t CalUBTotalSizeWithRowFactor(uint64_t rowFactor);
    void SetTilingData();
    void PrintTilingData();

private:
    RmsNormHiF4QuantTilingData tilingData;
};

class RmsNormHiF4QuantSplitDTiling : public RmsNormHiF4QuantRegbaseTilingBase {
public:
    explicit RmsNormHiF4QuantSplitDTiling(gert::TilingContext* context) : RmsNormHiF4QuantRegbaseTilingBase(context) {}
    ~RmsNormHiF4QuantSplitDTiling() override = default;
    void Reset(gert::TilingContext* context) override { RmsNormHiF4QuantRegbaseTilingBase::Reset(context); }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    int64_t GetPowerSplit(uint64_t numN);
    int64_t GetCacheId(int64_t idx);
    uint64_t GetMaxBaseN(uint64_t initialN);
    uint64_t CalUBTotalSize(uint64_t candidateN, uint64_t powerSplit);
    void SetTilingData();
    void PrintTilingData();

private:
    RmsNormHiF4QuantSplitDTilingData tilingData;
    uint64_t baseN_{HIF4_BLOCK_SIZE};
    uint64_t baseM_{128};
    uint64_t baseNBlockSize_{0};
    uint64_t nUbLoops_{0};
    uint64_t powerSplit_{0};
    uint64_t mainFoldCount_{0};
    uint64_t foldTail_{0};
    uint64_t mLastCore_{0};
};

class RmsNormHiF4QuantReduceEmptyTiling : public RmsNormHiF4QuantRegbaseTilingBase {
public:
    explicit RmsNormHiF4QuantReduceEmptyTiling(gert::TilingContext* context)
        : RmsNormHiF4QuantRegbaseTilingBase(context)
    {}
    ~RmsNormHiF4QuantReduceEmptyTiling() override = default;
    void Reset(gert::TilingContext* context) override { RmsNormHiF4QuantRegbaseTilingBase::Reset(context); }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    RmsNormHiF4QuantReduceEmptyTilingData td_;
};

} // namespace optiling

#endif
