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
 * \file quant_batch_matmul_v4_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_TILING_H
#define QUANT_BATCH_MATMUL_V4_TILING_H

#include <cstdint>
#include <vector>

#include "tiling_base/tiling_templates_registry.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "tiling/tiling_api.h"
#include "op_cache_tiling.h"
#include "register/tilingdata_base.h"

#include "quant_batch_matmul_v4_basic_block_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_tool.h"
#include "../quant_batch_matmul_v4_compile_info.h"
namespace optiling {
using matmul_tiling::MatrixTraverse;
using namespace matmul_v4;
using Ops::NN::Optiling::TilingBaseClass;
namespace matmul_v4 {
// dim index
constexpr size_t DIM_INDEX_0 = 0;
constexpr size_t DIM_INDEX_1 = 1;
constexpr size_t DIM_INDEX_2 = 2;
constexpr size_t DIM_INDEX_3 = 3;
// input index
constexpr size_t X1_INDEX = 0UL;
constexpr size_t X2_INDEX = 1UL;
constexpr size_t BIAS_INDEX = 2UL;
constexpr size_t X1_SCALE_INDEX = 3UL;
constexpr size_t X2_SCALE_INDEX = 4UL;
constexpr size_t X2_OFFSET_INDEX = 7UL;
constexpr size_t Y_SCALE_INDEX = 5UL;
constexpr size_t Y_OFFSET_INDEX = 8UL;
// output index
constexpr size_t Y_OUTPUT_INDEX = 0UL;
// attr index
constexpr size_t TRANSPOSE_X1_INDEX = 2UL;
constexpr size_t TRANSPOSE_X2_INDEX = 3UL;
constexpr size_t GROUP_SIZE_INDEX = 4UL;
// valid dim
constexpr size_t VALID_INPUT_DIM_NUM = 2UL;
constexpr size_t MATMUL_SHAPE_DIM_NUM = 2UL;
constexpr size_t VALID_X1_SCALE_DIM_NUM = 2UL;
constexpr size_t VALID_X2_SCALE_DIM_NUM = 2UL;
constexpr size_t VALID_WEIGHT_NZ_DIM_NUM = 4UL;
constexpr size_t VALID_BIAS_MIN_DIM = 1;
constexpr size_t VALID_BIAS_MAX_DIM = 2;
constexpr uint64_t VEC_INNER_AXIS_ALIGN_UINT = 128UL;
constexpr uint64_t MAX_SHAPE_DIM = 0x7fffffffUL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr int32_t BASIS_PRIORITY = 0;
constexpr uint64_t INT4_DTYPE_PARAM = 2;
constexpr uint32_t WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024
constexpr int32_t DB_BUFFER = 2;
constexpr int32_t EXTRA_GROUP_NUM = 2;
constexpr uint64_t K_ALIGN_SIZE = 64;
constexpr uint64_t N_ALIGN_SIZE = 64;

constexpr int64_t B64_BITS = 64;
constexpr int64_t B8_BITS = 8;
constexpr int64_t BLK_NUM_V100 = 32;
constexpr int64_t L0A_SIZE_V100 = 65536;
constexpr int64_t L0B_SIZE_V100 = 65536;
constexpr int64_t L0C_SIZE_V100 = 262144;
constexpr int64_t MTE2_MIN_LOAD_SIZE_V100 = 32768; // 实测16KB带宽较差，此处取32KB
constexpr int64_t MIN_CACHE_LINE_V100 = 128;
constexpr int64_t CACHE_LINE_V100 = 256;
constexpr int64_t GROUP_ALIGN_SIZE = 32;
constexpr int64_t NZ_GROUP_SIZE_32 = 32;
constexpr int64_t NZ_C0_SIZE = 32;
constexpr int64_t NZ_GROUP_SIZE_64 = 64;
constexpr int64_t MIN_SHAPE_SIZE = 1;
constexpr int64_t MAX_SHAPE_SIZE = 65535;
constexpr int64_t VALID_BIAS_SHAPE_SIZE = 1;

constexpr double FREQUENCY_v100 = 1.6;
constexpr double HBM_BANDWIDTH_V100 = 1.6;
constexpr double L2_BANDWIDTH_V100 = 5.4;

enum class QuantType : uint32_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4
};

enum class KernelTemplateType : uint32_t {
    BASIS = 0
};

enum class WeightFormat : uint32_t {
    ND = 0,
    FRACTAL_NZ = 1,
};

struct QuantBatchMatmulInfo {
    bool transA;
    bool transB;
    bool hasX1Scale;
    bool hasX2Scale;
    bool hasBias;
    bool hasAntiQuantOffset;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    bool weightNz;
    uint32_t libApiWorkSpaceSize;
    uint64_t groupSize;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t batchSize;
    uint64_t vecInnerAxisAlignUnit;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
    ge::DataType biasDtype;
    DtypeEnum templateDtype;
    QuantType antiQuantType;
    const char *opName;
    ge::Format bFormat = ge::FORMAT_ND;
};
}  // namespace matmul_v4

BEGIN_TILING_DATA_DEF(QuantBatchMatmulV4TilingData)
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint8_t, vecCoreParallel);
TILING_DATA_FIELD_DEF(uint8_t, reserve1);
TILING_DATA_FIELD_DEF(uint16_t, AL1Pingpong);
TILING_DATA_FIELD_DEF(uint16_t, BL1Pingpong);

TILING_DATA_FIELD_DEF(uint64_t, kAlign);
TILING_DATA_FIELD_DEF(uint64_t, nAlign);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF(uint64_t, mSize);

TILING_DATA_FIELD_DEF(uint64_t, nBubSize);
TILING_DATA_FIELD_DEF(uint64_t, kBubSize);
TILING_DATA_FIELD_DEF(uint64_t, mAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, nBL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kBL1Size);
TILING_DATA_FIELD_DEF(uint64_t, hasX1Scale);
TILING_DATA_FIELD_DEF(uint64_t, hasX2Scale);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV4, QuantBatchMatmulV4TilingData)
REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV4TilingDataOp, QuantBatchMatmulV4TilingData)

class QuantBatchMatmulV4TilingBase : public TilingBaseClass {
public:
    explicit QuantBatchMatmulV4TilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    explicit QuantBatchMatmulV4TilingBase(gert::TilingContext *context, QuantBatchMatmulV4TilingData *out)
        : TilingBaseClass(context)
    {
        Reset();
        tilingData_ = out;
        InitCompileInfo();
    }
    ~QuantBatchMatmulV4TilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus PostTiling() override;
    virtual bool CheckFinalTilingData()
    {
        return true;
    }

    void Reset();
    void InitCompileInfo();
    ge::graphStatus CheckContext() const;
    bool AnalyzeDtype();
    bool AnalyzeBiasDtype(const gert::CompileTimeTensorDesc *biasDesc);
    bool AnalyzeX1scaleDtype(const gert::CompileTimeTensorDesc *x1ScaleDesc);
    bool AnalyzeX2scaleDtype(const gert::CompileTimeTensorDesc *x2ScaleDesc);
    bool AnalyzeAntiQuantDtype(ge::DataType antiQuantScaleDtype,
                               const gert::CompileTimeTensorDesc *antiQuantOffsetDesc) const;
    bool AnalyzeYScaleOffsetShape(const gert::StorageShape *yScaleShape, const gert::StorageShape *yOffsetShape) const;
    bool AnalyzeAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                               const gert::StorageShape *antiQuantOffsetShape);
    bool AnalyzeTranspose();
    bool AnalyzeAttrs();
    bool AnalyzeX2InputDim(const gert::StorageShape *x2Shape);
    bool AnalyzeInputs();
    bool AnalyzeX2ScalePerGroupShape(const gert::StorageShape *x2ScaleShape);
    bool AnalyzeShapeSize(const gert::StorageShape *x1Shape, const gert::StorageShape *x2Shape);
    bool AnalyzeBiasShape(const gert::StorageShape *biasShape);
    bool AnalyzeX1ScaleShape(const gert::StorageShape *x1ScaleShape);
    bool AnalyzeX2ScaleShape(const gert::StorageShape *x2ScaleShape);
    bool AnalyzeQuantType();
    virtual bool SetQuantType(const gert::StorageShape *antiQuantScaleShape,
                              const gert::StorageShape *antiQuantOffsetShape) = 0;
    virtual bool CalcUBSize(uint64_t vecSingleN, uint64_t vecSingleK) const = 0;
    void PrintTilingData(bool debugLevel);
    virtual void PrintMatMulTiling() const;
    virtual bool GetTilingFromCache();

    uint32_t CalcAntiQuantTmpSize(uint64_t vecSingleN, uint64_t vecSingleK) const;
    void Convert2AscendCTiling(const CacheTilingData &tbeTiling, TCubeTiling &matmulTiling);
    MatrixTraverse GetIteratorOrder(const CacheTilingData &tbeTiling, int32_t singleCoreM, int32_t singleCoreN,
                                    int32_t singleCoreK) const;

    QuantBatchMatmulInfo inputParams_;
    uint64_t cubeBaseN_;
    int32_t templateId_ = -1;
    ge::Format aFormat;
    ge::Format bFormat;
    ge::Format cFormat;

    matmul_tiling::DataType mmInputDtype_;
    matmul_tiling::DataType mmOutputDtype_;
    matmul_tiling::DataType mmBiasDtype_;
    matmul_tiling::DataType mmScaleADtype_;
    matmul_tiling::DataType mmScaleBDtype_;
    uint32_t aivNum_;
    uint32_t aicNum_;
    std::unique_ptr<QuantBatchMatmulV4TilingData> tilingDataManager_;
    QuantBatchMatmulV4TilingData *tilingData_ = nullptr;
    std::unique_ptr<QuantBatchMatmulV4CompileInfo> compileInfoPtr_;
};

class QuantBatchMatmulV4RegBase : public QuantBatchMatmulV4TilingBase {
public:
    explicit QuantBatchMatmulV4RegBase(gert::TilingContext *context)
        : QuantBatchMatmulV4TilingBase(context)
    {
        tilingSolver_.Init();
    }
    ~QuantBatchMatmulV4RegBase() override = default;

protected:
    bool IsCapable() override;
    void SetBubTiling();
    void GetBubTilingA8W4(int64_t &nBubSize, int64_t &kBubSize) const;
    void GetBubTilingA8W4BySize(int64_t &nBubSize, int64_t &kBubSize,
        int64_t &kBl1Size, int64_t &nBl1Size) const;
    bool CustomCheck() const;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    void SetMatmulTiling();
    bool CalcUBSize(uint64_t vecSingleN, uint64_t vecSingleK) const override
    {
        (void) vecSingleN;
        (void) vecSingleK;
        return true;
    }

    bool SetQuantType(const gert::StorageShape *quantScaleShape, const gert::StorageShape *quantOffsetShape) override
    {
        (void) quantScaleShape;
        (void) quantOffsetShape;
        return true;
    }
    void UpdateL1Tiling(uint64_t minKL1AL1Size, uint64_t minKL1BL1Size,
                        uint64_t fullLoadAl1Size, uint64_t fullLoadBl1Size, uint64_t minKL1);
    uint64_t GetGroupNumBub(uint64_t kDimSzie) const;
    uint64_t GetBubSize(uint64_t bubN, uint64_t bubD, bool isWeightNz) const;
    void PrintCVTilingData(const bool debugLevel) const;
    ge::graphStatus PostTiling() override;

    QuantBatchMatmulV4BasicBlockTiling tilingSolver_;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_TILING_H
