/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_TILING_H
#define LAYER_NORM_GRAD_TILING_H

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

BEGIN_TILING_DATA_DEF(LayerNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGrad, LayerNormGradTilingData)

// LayerNormGradTilingDataRecompute
BEGIN_TILING_DATA_DEF(LayerNormGradTilingDataRecompute)
TILING_DATA_FIELD_DEF(int64_t, row);
  TILING_DATA_FIELD_DEF(int64_t, col);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainBlockFactor);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNloopMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNtailMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNloopTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNtailTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMtail);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaBasicBlockLoop);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainFoldCount);
  TILING_DATA_FIELD_DEF(int64_t, backwardMainBlockFactor);
  TILING_DATA_FIELD_DEF(int64_t, backwardMainBlockCount);
  TILING_DATA_FIELD_DEF(int64_t, backwardTailBlockCount);
  TILING_DATA_FIELD_DEF(int64_t, backwardTailBlockFactor);
  TILING_DATA_FIELD_DEF(int64_t, backwardMLoopMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardMLoopTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardMLoopTailTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardMTailTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardNLoopMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardNTotalLoopMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardNLoopTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardBasicBlockLoop);
  TILING_DATA_FIELD_DEF(int64_t, backwardMainFoldCount);
  TILING_DATA_FIELD_DEF(int64_t, backwardNfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, backwardMfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, backwardCeilVLCount);
  TILING_DATA_FIELD_DEF(int64_t, backwardFoldPoint);
  TILING_DATA_FIELD_DEF(int64_t, backwardFoldSize);
  TILING_DATA_FIELD_DEF(int32_t, gammaBetaBlockDim);
  TILING_DATA_FIELD_DEF(int32_t, gammaBetaCacheBufferCount);
  TILING_DATA_FIELD_DEF(int32_t, gammaBetaResultCacheID);
  TILING_DATA_FIELD_DEF(int32_t, gammaBetaNfactor);
  TILING_DATA_FIELD_DEF(int32_t, gammaBetaMfactor);
  TILING_DATA_FIELD_DEF(int32_t, backwardBlockDim);
  TILING_DATA_FIELD_DEF(int32_t, backwardMfactor);
  TILING_DATA_FIELD_DEF(int32_t, backwardNfactor);
  TILING_DATA_FIELD_DEF(int32_t, backwardCacheBufferCountMain);
  TILING_DATA_FIELD_DEF(int32_t, backwardResultCacheIDMain);

  TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(LayerNormGrad_500, LayerNormGradTilingDataRecompute)


// LayerNormGradTilingDataGroupedReduceBigM
BEGIN_TILING_DATA_DEF(LayerNormGradTilingDataGroupedReduceBigM)
  TILING_DATA_FIELD_DEF(int64_t, row);
  TILING_DATA_FIELD_DEF(int64_t, col);

  TILING_DATA_FIELD_DEF(int64_t, gammaBetaUsableBlocks);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMPerBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMReminder);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNloop);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNtail);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMToProcessMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMLoopMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMTotalLoopMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMTailMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaBasicBlockLoopMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMainFoldCountMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaCacheBufferCountMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaResultCacheIDMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMToProcessTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMLoopTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMTotalLoopTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMTailTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaBasicBlockLoopTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaMainFoldCountTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaCacheBufferCountTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammabetaResultCacheIDTailBlock);

  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMTailStg2);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMBasicBlockLoopStg2);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMMainFoldCountStg2);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMResultCacheIDStg2);

  TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGrad_600, LayerNormGradTilingDataGroupedReduceBigM)

// LayerNormGradTilingDataGroupedReduceBigN
BEGIN_TILING_DATA_DEF(LayerNormGradTilingDataGroupedReduceBigN)
  TILING_DATA_FIELD_DEF(int64_t, row);
  TILING_DATA_FIELD_DEF(int64_t, col);
  // pdgamma, pdbeta
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainBlockFactor);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaBlockDim);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNloopMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNtailMainBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNloopTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNtailTailBlock);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMtail);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaBasicBlockLoop);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainFoldCount);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaCacheBufferCount);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaResultCacheID);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaNfactor);
  TILING_DATA_FIELD_DEF(int64_t, gammaBetaMfactor);
  // pdx
  TILING_DATA_FIELD_DEF(int64_t, backwardBlockDim);
  TILING_DATA_FIELD_DEF(int64_t, backwardNPerBlock);
  TILING_DATA_FIELD_DEF(int64_t, backwardNRem);
  TILING_DATA_FIELD_DEF(int64_t, nToProcessMain);
  TILING_DATA_FIELD_DEF(int64_t, nToProcessTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardMTotalLoop);
  TILING_DATA_FIELD_DEF(int64_t, backwardMtail);
  TILING_DATA_FIELD_DEF(int64_t, backwardNloopMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardNtailMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardBasicBlockLoopMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardMainFoldCountMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardNfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, backwardMfactor);
  TILING_DATA_FIELD_DEF(int64_t, backwardMfactorBlockAligned);
  TILING_DATA_FIELD_DEF(int64_t, backwardCacheBufferCountMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardResultCacheIDMain);
  TILING_DATA_FIELD_DEF(int64_t, backwardNloopTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardNtailTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardBasicBlockLoopTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardMainFoldCountTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardCacheBufferCountTail);
  TILING_DATA_FIELD_DEF(int64_t, backwardResultCacheIDTail);

  TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
  TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
  TILING_DATA_FIELD_DEF(float, epsilon);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGrad_700, LayerNormGradTilingDataGroupedReduceBigN)

// TilingKey生成方式：LNGTemplateKey * 100 + isDeterministicKey * 10 + dtypeKey
enum class LNGDtypeKey : int {
    FLOAT_FLOAT = 1,
    FLOAT16_FLOAT16 = 2,
    FLOAT16_FLOAT = 3,
    BFLOAT16_BFLOAT16 = 4,
    BFLOAT16_FLOAT = 5
};

enum class LNGTemplateKey : int {
    RECOMPUTE = 5,
    GROUPED_REDUCE_BIG_M = 6,
    GROUPED_REDUCE_BIG_N = 7
};

struct ParamsLayerNormGrad {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    uint64_t colAlign = 1;
    ge::DataType dyDtype;
    ge::DataType xDtype;
    ge::DataType varianceDtype;
    ge::DataType meanDtype;
    ge::DataType gammaDtype;
    ge::DataType dxDtype;
    ge::DataType dgammaDtype;
    ge::DataType dbetaDtype;
    uint64_t isDeterministicKey;
    LNGDtypeKey dtypeKey;
    bool isRegBase = false;
    float epsilon = 1e-5;
    bool pdxIsRequire = true;
    bool pdgammaIsRequire = true;
    bool pdbetaIsRequire = true;
};

struct LayerNormGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase = false;
};

class LayerNormGradTilingBase : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit LayerNormGradTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }
    ~LayerNormGradTilingBase() override = default;
    ParamsLayerNormGrad commonParams;

protected:
    ge::graphStatus InputDtypeCheck(
        ge::DataType dyDtype, ge::DataType xDtype, ge::DataType varianceDtype,
        ge::DataType meanDtype, ge::DataType gammaDtype);
    bool CheckShapeSame(
        const size_t leftIndex, const size_t rightIndex, const bool isLeftInput, const bool isRightInput);
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

class LayerNormGradRecomputeTiling : public LayerNormGradTilingBase
{
public:
    explicit LayerNormGradRecomputeTiling(gert::TilingContext* context) : LayerNormGradTilingBase(context)
    {
    }
    ~LayerNormGradRecomputeTiling() override = default;
    LayerNormGradTilingDataRecompute td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus GammaBetaKernelTiling();
    ge::graphStatus BackwardKernelTiling();
};

class LayerNormGradGroupedReduceBigMTiling : public LayerNormGradTilingBase
{
public:
    explicit LayerNormGradGroupedReduceBigMTiling(gert::TilingContext* context) : LayerNormGradTilingBase(context)
    {
    }
    ~LayerNormGradGroupedReduceBigMTiling() override = default;
    LayerNormGradTilingDataGroupedReduceBigM td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus GammaBetaKernelTiling();
    ge::graphStatus BackwardKernelTiling();
};

class LayerNormGradGroupedReduceBigNTiling : public LayerNormGradTilingBase
{
public:
    explicit LayerNormGradGroupedReduceBigNTiling(gert::TilingContext* context) : LayerNormGradTilingBase(context)
    {
    }
    ~LayerNormGradGroupedReduceBigNTiling() override = default;
    LayerNormGradTilingDataGroupedReduceBigN td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus GammaBetaKernelTiling();
    ge::graphStatus BackwardKernelTiling();
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
#endif  // LAYER_NORM_GRAD_TILING_H
