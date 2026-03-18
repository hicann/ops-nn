/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ada_layer_norm_grad_tiling.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_TILING_H
#define ADA_LAYER_NORM_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"



namespace optiling
{
constexpr uint64_t LNG_TEMPLATE_KEY_WEIGHT = 100;
constexpr uint64_t LNG_DETERMINISTIC_KEY_WEIGHT = 10;
constexpr uint64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t HALF_SIZE = 2;
const uint64_t ULONG_BIT_LEN = 64;

BEGIN_TILING_DATA_DEF(AdaLayerNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad, AdaLayerNormGradTilingData)

// AdaLayerNormGradTilingDataWorkspace
BEGIN_TILING_DATA_DEF(AdaLayerNormGradTilingDataWorkspace)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seq);
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubLoop);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, colAlignM);
TILING_DATA_FIELD_DEF(int64_t, colAlignV);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_201, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_202, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_203, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_204, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_205, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_211, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_212, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_213, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_214, AdaLayerNormGradTilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_215, AdaLayerNormGradTilingDataWorkspace)

// AdaLayerNormGradTilingDataCommon
BEGIN_TILING_DATA_DEF(AdaLayerNormGradTilingDataCommon)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seq);
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, colAlignM);
TILING_DATA_FIELD_DEF(int64_t, colAlignV);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, lastRBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, nlastRBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, lastBrcbBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, blockFormerScaleBufferBytes);//每个block计算batch数的scale行
TILING_DATA_FIELD_DEF(int64_t, wholeBufferElemNums);
TILING_DATA_FIELD_DEF(int64_t, blockFormerScaleBufferElemNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_401, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_402, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_403, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_404, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_405, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_411, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_412, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_413, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_414, AdaLayerNormGradTilingDataCommon)
REGISTER_TILING_DATA_CLASS(AdaLayerNormGrad_415, AdaLayerNormGradTilingDataCommon)

// TilingKey生成方式：LNGTemplateKey * 100 + isDeterministicKey * 10 + dtypeKey
enum class LNGDtypeKey : int {
    FLOAT_FLOAT = 1,
    FLOAT16_FLOAT16 = 2,
    FLOAT16_FLOAT = 3,
    BFLOAT16_BFLOAT16 = 4,
    BFLOAT16_FLOAT = 5
};

enum class LNGTemplateKey : int {
    SINGEL_READ = 1,
    WORKSPACE = 2,
    TRANSPOSE = 3,
    COMMON = 4,
    RECOMPUTE = 5,
    GROUPED_REDUCE_BIG_M = 6,
    GROUPED_REDUCE_BIG_N = 7
};

struct ParamsAdaLayerNormGrad {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    uint64_t batchSize = 1;
    uint64_t seqSize = 1;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    uint64_t colAlign = 1;
    ge::DataType dyDtype;
    ge::DataType xDtype;
    ge::DataType rstdDtype;
    ge::DataType meanDtype;
    ge::DataType scaleDtype;
    ge::DataType gammaDtype;
    ge::DataType betaDtype;
    ge::DataType dxDtype;
    ge::DataType dgammaDtype;
    ge::DataType dbetaDtype;
    ge::DataType dscaleDtype;
    ge::DataType dshiftDtype;

    uint64_t isDeterministicKey;
    LNGDtypeKey dtypeKey;
    bool isRegBase = false;
    bool pdxIsRequire = true;
    bool pdgammaIsRequire = true;
    bool pdbetaIsRequire = true;
    bool pdscaleIsRequire = true;
    bool pdshiftIsRequire = true;
};

struct AdaLayerNormGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase = false;
};

class AdaLayerNormGradTilingBase : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit AdaLayerNormGradTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }
    ~AdaLayerNormGradTilingBase() override = default;
    ParamsAdaLayerNormGrad commonParams;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    int64_t FindNearestPower2(const int64_t value);
    int64_t GetCacheID(const int64_t idx);
};

class AdaLayerNormGradWorkspaceTiling : public AdaLayerNormGradTilingBase
{
public:
    explicit AdaLayerNormGradWorkspaceTiling(gert::TilingContext* context) : AdaLayerNormGradTilingBase(context)
    {
    }
    ~AdaLayerNormGradWorkspaceTiling() override = default;
    AdaLayerNormGradTilingDataWorkspace td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class AdaLayerNormGradCommonTiling : public AdaLayerNormGradTilingBase
{
public:
    explicit AdaLayerNormGradCommonTiling(gert::TilingContext* context) : AdaLayerNormGradTilingBase(context)
    {
    }
    ~AdaLayerNormGradCommonTiling() override = default;
    AdaLayerNormGradTilingDataCommon td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    int64_t CalculateUbFormer();

private:
    int64_t batch_{-1};
    int64_t seq_{-1};
    int64_t row_{-1};
    int64_t col_{-1};
    // for vector
    int64_t colAlignV_{-1};
    // for mte
    // colAlignM >= colAlignV
    int64_t colAlignM_{-1};
};

}  // namespace optiling

namespace ops
{
template <typename T>
inline auto CeilAlign(T num1, T num2) -> T
{
    return Ops::Base::CeilAlign(num1, num2);
}

template <typename T>
inline auto FloorAlign(T num1, T num2) -> T
{
    return Ops::Base::FloorAlign(num1, num2);
}

template <typename T>
inline auto FloorDiv(T num1, T num2) -> T
{
    return Ops::Base::FloorDiv(num1, num2);
}

template <typename T>
inline auto CeilDiv(T num1, T num2) -> T
{
    return Ops::Base::CeilDiv(num1, num2);
}
}  // namespace ops
#endif  // ADA_LAYER_NORM_GRAD_TILING_H