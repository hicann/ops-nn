/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 *
 *
 *
 *
 *
 *
 *
 */

/* !
 * \file layer_norm_grad_v3_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_TILING_H
#define LAYER_NORM_GRAD_V3_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
constexpr uint64_t LNG_TEMPLATE_KEY_WEIGHT = 100;
constexpr uint64_t LNG_DETERMINISTIC_KEY_WEIGHT = 10;
constexpr uint64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t HALF_SIZE = 2;

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3, LayerNormGradV3TilingData)

// LayerNormGradV3TilingDataTranspose
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataTranspose)
TILING_DATA_FIELD_DEF(uint64_t, row);                    // 输入tensor的行
TILING_DATA_FIELD_DEF(uint64_t, col);                    // 输入tensor的列，即reduce的轴
TILING_DATA_FIELD_DEF(uint64_t, blockDim);               // 实际使用的core数量
TILING_DATA_FIELD_DEF(uint64_t, blockFormer);            // 整核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, blockTail);              // 尾核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubFormer);               // ub整循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfFormerBlock);    // 整核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfTailBlock);      // 尾核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfFormerBlock);    // 整核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfTailBlock);      // 尾核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, bFormer);                // ubFormer借轴大小，ubFormer->16*bFormer
TILING_DATA_FIELD_DEF(uint64_t, dichotomizeAddDiffSize); // row与小于row的最近二次幂的差值
TILING_DATA_FIELD_DEF(uint64_t, deterministicComputeWspSize); // 确定性计算需要的pdGamma或pdBeta workspace size大小
TILING_DATA_FIELD_DEF(float, coefficient);                    // 1/col
TILING_DATA_FIELD_DEF(float, placeHolder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_301, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_302, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_303, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_304, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_305, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_311, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_312, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_313, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_314, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_315, LayerNormGradV3TilingDataTranspose)

// LayerNormGradV3TilingDataWorkspace
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataWorkspace)
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

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_201, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_202, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_203, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_204, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_205, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_211, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_212, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_213, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_214, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_215, LayerNormGradV3TilingDataWorkspace)

// LayerNormGradV3TilingDataSingleRead
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataSingleRead)
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
TILING_DATA_FIELD_DEF(int64_t, bufferElemNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_101, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_102, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_103, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_104, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_105, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_111, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_112, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_113, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_114, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_115, LayerNormGradV3TilingDataSingleRead)

// LayerNormGradV3TilingDataCommon
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataCommon)
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
TILING_DATA_FIELD_DEF(int64_t, wholeBufferElemNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_401, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_402, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_403, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_404, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_405, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_411, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_412, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_413, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_414, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_415, LayerNormGradV3TilingDataCommon)

// LayerNormGradV3TilingDataRecompute
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataRecompute)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);

TILING_DATA_FIELD_DEF(int64_t, gammaBetaMfactor);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaNfactor);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaBlockDim);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainBlockCount);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaTailBlockCount);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaTailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMloop);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMTotalLoop);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMtail);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaBasicBlockLoop);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaNfactorBlockAligned);
TILING_DATA_FIELD_DEF(int64_t, gammaBetaMfactorBlockAligned);

TILING_DATA_FIELD_DEF(int64_t, backwardMfactor);
TILING_DATA_FIELD_DEF(int64_t, backwardNfactor);
TILING_DATA_FIELD_DEF(int64_t, backwardMainBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, backwardBlockDim);
TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_501, LayerNormGradV3TilingDataRecompute)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_502, LayerNormGradV3TilingDataRecompute)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_503, LayerNormGradV3TilingDataRecompute)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_504, LayerNormGradV3TilingDataRecompute)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_505, LayerNormGradV3TilingDataRecompute)

// LayerNormGradV3TilingDataGroupedReduceBigM
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataGroupedReduceBigM)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_601, LayerNormGradV3TilingDataGroupedReduceBigM)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_602, LayerNormGradV3TilingDataGroupedReduceBigM)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_603, LayerNormGradV3TilingDataGroupedReduceBigM)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_604, LayerNormGradV3TilingDataGroupedReduceBigM)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_605, LayerNormGradV3TilingDataGroupedReduceBigM)

// LayerNormGradV3TilingDataGroupedReduceBigN
BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataGroupedReduceBigN)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int32_t, pdxIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdgammaIsRequire);
TILING_DATA_FIELD_DEF(int32_t, pdbetaIsRequire);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_701, LayerNormGradV3TilingDataGroupedReduceBigN)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_702, LayerNormGradV3TilingDataGroupedReduceBigN)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_703, LayerNormGradV3TilingDataGroupedReduceBigN)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_704, LayerNormGradV3TilingDataGroupedReduceBigN)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_705, LayerNormGradV3TilingDataGroupedReduceBigN)

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

struct ParamsLayerNormGradV3 {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    uint64_t colAlign = 1;
    ge::DataType dyDtype;
    ge::DataType gammaDtype;
    uint64_t isDeterministicKey;
    LNGDtypeKey dtypeKey;
    bool isRegBase = false;
    bool pdxIsRequire = true;
    bool pdgammaIsRequire = true;
    bool pdbetaIsRequire = true;
};

struct LayerNormGradV3CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isRegBase = false;
};

class LayerNormGradV3TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit LayerNormGradV3TilingBase(gert::TilingContext *context) : Ops::NN::Optiling::TilingBaseClass(context) {}
    ~LayerNormGradV3TilingBase() override = default;
    ParamsLayerNormGradV3 commonParams;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class LayerNormGradV3WorkspaceTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3WorkspaceTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3WorkspaceTiling() override = default;
    LayerNormGradV3TilingDataWorkspace td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class LayerNormGradV3SingleReadTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3SingleReadTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3SingleReadTiling() override = default;
    LayerNormGradV3TilingDataSingleRead td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    // for vector
    int64_t colAlignV;
    // for mte
    // colAlignM >= colAlignV
    int64_t colAlignM;
    bool dbFlag;
};

class LayerNormGradV3TransposeTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3TransposeTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3TransposeTiling() override = default;
    LayerNormGradV3TilingDataTranspose td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    uint64_t CalcBorrowFactor(uint64_t oriFactor);
    uint32_t FindDichotomizeAddDiffSize();
};

class LayerNormGradV3CommonTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3CommonTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3CommonTiling() override = default;
    LayerNormGradV3TilingDataCommon td_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    int64_t CalculateUbFormer();

private:
    int64_t row_{ -1 };
    int64_t col_{ -1 };
    // for vector
    int64_t colAlignV_{ -1 };
    // for mte
    // colAlignM >= colAlignV
    int64_t colAlignM_{ -1 };
};

class LayerNormGradV3RecomputeTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3RecomputeTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3RecomputeTiling() override = default;
    LayerNormGradV3TilingDataRecompute td_;

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

class LayerNormGradV3GroupedReduceBigMTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3GroupedReduceBigMTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3GroupedReduceBigMTiling() override = default;
    LayerNormGradV3TilingDataGroupedReduceBigM td_;

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

class LayerNormGradV3GroupedReduceBigNTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3GroupedReduceBigNTiling(gert::TilingContext *context) : LayerNormGradV3TilingBase(context) {}
    ~LayerNormGradV3GroupedReduceBigNTiling() override = default;
    LayerNormGradV3TilingDataGroupedReduceBigN td_;

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

} // namespace optiling
#endif // LAYER_NORM_GRAD_V3_TILING_H
