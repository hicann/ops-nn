/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "op_host/tiling_templates_registry.h"
#include "../../../../common/op_host/op_tiling/tiling_type_mm.h"
#include "op_cache_tiling.h"

#include "../../../../weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_tool.h"
#include "../quant_batch_matmul_v4_compile_info.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_data_apt.h"

namespace optiling {
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
constexpr size_t VALID_X1_SCALE_DIM_NUM = 3UL;
constexpr size_t VALID_X2_SCALE_DIM_NUM = 3UL;
constexpr size_t VALID_WEIGHT_NZ_DIM_NUM = 4UL;
constexpr size_t VALID_BIAS_MIN_DIM = 1;
constexpr size_t VALID_BIAS_MAX_DIM = 2;
constexpr uint64_t VEC_INNER_AXIS_ALIGN_UINT = 128UL;
constexpr uint64_t MAX_SHAPE_DIM = 0x7fffffffUL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr int32_t BASIC_PRIORITY = 1;
constexpr uint64_t INT4_DTYPE_PARAM = 2;
constexpr uint32_t WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024
constexpr int32_t DB_BUFFER = 2;
constexpr int32_t EXTRA_GROUP_NUM = 2;
constexpr uint64_t K_ALIGN_SIZE = 32;
constexpr uint64_t K_ALIGN_SIZE_MX = 8;
constexpr uint64_t N_ALIGN_SIZE = 8;

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
constexpr int64_t VALID_BIAS_SHAPE_SIZE = 1;

constexpr double FREQUENCY_v100 = 1.6;
constexpr double HBM_BANDWIDTH_V100 = 1.6;
constexpr double L2_BANDWIDTH_V100 = 5.4;

enum class QuantType : uint32_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
    PER_TILE = 5,
    INT4_ASYMMETRICAL = 6
};

enum class KernelTemplateType : uint32_t { BASIS = 0, LUT_ASW = 1, LUT_AL1FULL = 2 };

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
    const char* opName;
    ge::Format bFormat = ge::FORMAT_ND;
};
} // namespace matmul_v4
using namespace matmul_v4;

class QuantBatchMatmulV4TilingBase : public TilingBaseClass {
public:
    explicit QuantBatchMatmulV4TilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }

    ~QuantBatchMatmulV4TilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual bool SetQuantType(const gert::StorageShape* antiQuantScaleShape,
                              const gert::StorageShape* antiQuantOffsetShape) = 0;
    virtual bool CalcUBSize(uint64_t vecSingleN, uint64_t vecSingleK) const = 0;
    virtual bool CheckCoreNum() const;
    uint64_t GetTilingKey() const override;

    ge::graphStatus SerializeTilingData(const void* data, size_t size, uint32_t usedCoreNum);
    ge::graphStatus CheckTilingDataCapacity(const void* data, size_t size) const;
    bool CheckA8W4Params() const;
    bool CustomCheck() const;

    matmul_v4::QuantBatchMatmulInfo inputParams_;
    uint32_t aivNum_;
    uint32_t aicNum_;
    std::unique_ptr<QuantBatchMatmulV4CompileInfo> compileInfoPtr_;

private:
    void Reset();
    void InitCompileInfo();
    ge::graphStatus CheckContext() const;
    ge::graphStatus CheckInputParams() const;
    bool AnalyzeDtype();
    bool AnalyzeBiasDtype(const gert::CompileTimeTensorDesc* biasDesc);
    bool AnalyzeX1scaleDtype(const gert::CompileTimeTensorDesc* x1ScaleDesc);
    bool AnalyzeX2scaleDtype(const gert::CompileTimeTensorDesc* x2ScaleDesc);
    bool AnalyzeYScaleOffsetShape(const gert::StorageShape* yScaleShape, const gert::StorageShape* yOffsetShape) const;
    bool AnalyzeTranspose();
    bool AnalyzeAttrs();
    bool AnalyzeX2InputDim(const gert::StorageShape* x2Shape);
    bool AnalyzeInputs();
    bool AnalyzeX2ScalePerGroupShape(const gert::StorageShape* x2ScaleShape);
    bool AnalyzeShapeSize(const gert::StorageShape* x1Shape, const gert::StorageShape* x2Shape);
    bool ValidateShapeDimensions();
    bool AnalyzeBiasShape(const gert::StorageShape* biasShape);
    bool AnalyzeX1ScaleShape(const gert::StorageShape* x1ScaleShape);
    bool AnalyzeX2ScaleShape(const gert::StorageShape* x2ScaleShape);
    bool AnalyzeQuantType();
};

} // namespace optiling
