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
 * \file layer_norm_v3_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_V3_TILING_H
#define LAYER_NORM_V3_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling {
constexpr uint64_t LN_TEMPLATE_KEY_WEIGHT = 100;

BEGIN_TILING_DATA_DEF(LayerNormV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV3, LayerNormV3TilingData)
REGISTER_TILING_DATA_CLASS(LayerNorm, LayerNormV3TilingData)

BEGIN_TILING_DATA_DEF(LayerNormV3TilingDataRegBaseNoReduce)
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactorAlignB32);
TILING_DATA_FIELD_DEF(int32_t, formerBlockUbLoops);
TILING_DATA_FIELD_DEF(int32_t, tailBlockUbLoops);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int8_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int8_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV3_600, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNormV3_610, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNormV3_611, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNormV3_620, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNormV3_622, LayerNormV3TilingDataRegBaseNoReduce)

REGISTER_TILING_DATA_CLASS(LayerNorm_600, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNorm_610, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNorm_611, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNorm_620, LayerNormV3TilingDataRegBaseNoReduce)
REGISTER_TILING_DATA_CLASS(LayerNorm_622, LayerNormV3TilingDataRegBaseNoReduce)

BEGIN_TILING_DATA_DEF(LayerNormV3TilingDataRegBaseTwoPass)
TILING_DATA_FIELD_DEF(int64_t, r);
TILING_DATA_FIELD_DEF(int64_t, rAlign);
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aFactor);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(int64_t, tmpBufferSize);
TILING_DATA_FIELD_DEF(int64_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int64_t, nullptrBeta);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layerNormTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV3_300, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV3_310, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV3_311, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV3_320, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV3_322, LayerNormV3TilingDataRegBaseTwoPass)

REGISTER_TILING_DATA_CLASS(LayerNorm_300, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNorm_310, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNorm_311, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNorm_320, LayerNormV3TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNorm_322, LayerNormV3TilingDataRegBaseTwoPass)

BEGIN_TILING_DATA_DEF(LayerNormV3TilingDataWelford)
TILING_DATA_FIELD_DEF(int64_t, M);                  // 输入tensor的行
TILING_DATA_FIELD_DEF(int64_t, N);                  // 输入tensor的列，即reduce的轴
TILING_DATA_FIELD_DEF(int64_t, rAlign);             // r对齐的大小
TILING_DATA_FIELD_DEF(int64_t, numBlocks);           // 实际使用的core数量
TILING_DATA_FIELD_DEF(int64_t, mainBlockCount);     // 整核的数量
TILING_DATA_FIELD_DEF(int64_t, mainBlockFactor);    // 整核处理的row大小
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);    // 尾核处理的row大小
TILING_DATA_FIELD_DEF(int64_t, tileLength);         // tile块的元素个数
TILING_DATA_FIELD_DEF(int64_t, welfordTempSize);    // welford临时buffer的大小
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTimes); // welford update的次数
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTail);  // welford update的尾数
TILING_DATA_FIELD_DEF(int64_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int64_t, nullptrBeta);
TILING_DATA_FIELD_DEF(int64_t, apiTempBufferSize);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV3_400, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV3_410, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV3_411, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV3_420, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV3_422, LayerNormV3TilingDataWelford)

REGISTER_TILING_DATA_CLASS(LayerNorm_400, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNorm_410, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNorm_411, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNorm_420, LayerNormV3TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNorm_422, LayerNormV3TilingDataWelford)

BEGIN_TILING_DATA_DEF(LayerNormV3TilingDataRegBaseTwoPassPerf)
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactorAlignB32);
TILING_DATA_FIELD_DEF(int32_t, r);
TILING_DATA_FIELD_DEF(int32_t, rAlign);
TILING_DATA_FIELD_DEF(int32_t, formerBlockUbLoops);
TILING_DATA_FIELD_DEF(int32_t, tailBlockUbLoops);
TILING_DATA_FIELD_DEF(int32_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int8_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int8_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV3_500, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV3_510, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV3_511, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV3_520, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV3_522, LayerNormV3TilingDataRegBaseTwoPassPerf)

REGISTER_TILING_DATA_CLASS(LayerNorm_500, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNorm_510, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNorm_511, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNorm_520, LayerNormV3TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNorm_522, LayerNormV3TilingDataRegBaseTwoPassPerf)

struct ParamsLayerNomrV3 {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
    int64_t blockSize = 0;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    float eps = 0;
    float coefficient = 0;
    uint64_t rowAlign;
    uint64_t gammaNullPtr;
    uint64_t betaNullPtr;
    uint64_t meanAndRstdNullPtr = 0;
    ge::DataType tensorDtype;
    ge::DataType paramDtype;
    int64_t dtypeKey = 0;
    bool isAscend310P = false;
    bool isRegBase = false;
    int64_t vlFp32 = 0;
    bool isV1 = false;
};

enum class LayerNormV3TilingKey : int64_t
{
    // FLOAT32/FLOAT16/BFLOAT16 -- 0/1/2
    // Regbase no reduce
    LAYER_NORM_REGBASE_NO_REDUCE_FLOAT32_FLOAT32 = 600,
    LAYER_NORM_REGBASE_NO_REDUCE_FLOAT16_FLOAT32 = 610,
    LAYER_NORM_REGBASE_NO_REDUCE_FLOAT16_FLOAT16 = 611,
    LAYER_NORM_REGBASE_NO_REDUCE_BFLOAT16_FLOAT32 = 620,
    LAYER_NORM_REGBASE_NO_REDUCE_BFLOAT16_BFLOAT16 = 622,
    // Regbase two pass
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT32_FLOAT32 = 300,
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT32 = 310,
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT16 = 311,
    LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_FLOAT32 = 320,
    LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_BFLOAT16 = 322,
    // Regbase two pass perf
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT32_FLOAT32 = 500,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT32 = 510,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT16 = 511,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_FLOAT32 = 520,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_BFLOAT16 = 522,
};

enum class LNTemplateKey : int
{
    SINGEL_READ = 1,
    TRANSPOSE = 2,
    SINGEL_READ_REGBASE = 3,
    WELFORD = 4
};

struct LayerNormV3CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
    bool isRegBase = false;
    uint32_t vectorLength = 0;
    uint64_t blockSize = 0;
};

class LayerNormV3TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit LayerNormV3TilingBase(gert::TilingContext* context_) : Ops::NN::Optiling::TilingBaseClass(context_)
    {}
    ~LayerNormV3TilingBase() override = default;
    ParamsLayerNomrV3 commonParams;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    ge::graphStatus InputDtypeCheck(ge::DataType xDtype, ge::DataType gammaDtype, ge::DataType betaDtype);
    bool isFloatDtype(ge::DataType dtype);
    int64_t GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype);
    ge::graphStatus InputShapeAndAxisCheck(
        const gert::Shape& xShape, const gert::Shape& gammaShape, const gert::Shape& betaShape, int64_t& beginNormAxis,
        int64_t& beginParamsAxis);
    bool isIndexValid(const gert::Shape& xShape, int64_t beginAxis);
    ge::graphStatus GetCommonPlatformInfo(const LayerNormV3CompileInfo* compileInfo);
};

class LayerNormV3RegBaseTwoPassTiling : public LayerNormV3TilingBase {
public:
    explicit LayerNormV3RegBaseTwoPassTiling(gert::TilingContext* context_) : LayerNormV3TilingBase(context_)
    {}
    ~LayerNormV3RegBaseTwoPassTiling() override = default;
    LayerNormV3TilingDataRegBaseTwoPass td_;

protected:
    bool CanFitInBuffer(int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse);
    bool CanFitInBuffer(
        int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse, int64_t xElemSize);
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t binaryAddQuotient;
    int64_t blockNum;
};

class LayerNormV3WelfordTiling : public LayerNormV3TilingBase {
public:
    explicit LayerNormV3WelfordTiling(gert::TilingContext* context_) : LayerNormV3TilingBase(context_)
    {}
    ~LayerNormV3WelfordTiling() override = default;
    LayerNormV3TilingDataWelford td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    bool IsValidTileLength(int64_t tileLength);
};

class LayerNormV3RegBaseTwoPassPerfTiling : public LayerNormV3TilingBase {
public:
    explicit LayerNormV3RegBaseTwoPassPerfTiling(gert::TilingContext* context_) : LayerNormV3TilingBase(context_)
    {}
    ~LayerNormV3RegBaseTwoPassPerfTiling() override = default;
    LayerNormV3TilingDataRegBaseTwoPassPerf td_;

protected:
    int64_t GetUBCanUseSize();
    int64_t GetRowWeight();
    bool CanFitInBuffer(int64_t curA);
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t blockNum_;
};

class LayerNormV3RegBaseNoReduceTiling : public LayerNormV3TilingBase {
public:
    explicit LayerNormV3RegBaseNoReduceTiling(gert::TilingContext* context_) : LayerNormV3TilingBase(context_)
    {}
    ~LayerNormV3RegBaseNoReduceTiling() override = default;
    LayerNormV3TilingDataRegBaseNoReduce td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t blockNum_;
};

ge::graphStatus Tiling4LayerNormV3ForAscendC(gert::TilingContext* context);
ge::graphStatus TilingPrepare4LayerNormV3ForAscendC(
    gert::TilingParseContext* context, LayerNormV3CompileInfo& regbaseCompileInfo);
ge::graphStatus TilingPrepare4LayerNormV3CompileInfo(
    gert::TilingParseContext* context, LayerNormV3CompileInfo* compileInfo);
} // namespace optiling

#endif // LAYER_NORM_V3_TILING_H