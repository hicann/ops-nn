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
 * \file quant_batch_matmul_inplace_add_helper.h
 * \brief Common parser and validator for QuantBatchMatmulInplaceAdd basic-api tiling strategies.
 */
#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_HELPER_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_HELPER_H

#include <cstdint>
#include "matmul/common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "../quant_batch_matmul_inplace_add_host_utils.h"
#include "../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_data.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h"
#include "../../../quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {
using namespace QuantBatchMatmulInplaceAddTilingConstant;
using Ops::NN::FormatString;

template <typename BaseT>
class QuantBatchMatmulInplaceAddHelper : public BaseT {
public:
    explicit QuantBatchMatmulInplaceAddHelper(gert::TilingContext* context) : BaseT(context)
    {}
    ~QuantBatchMatmulInplaceAddHelper() override = default;

protected:
    using BaseT::BaseT;

    ge::graphStatus GetShapeAttrsInfo() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckDtype() const override;
    const char *GetDefaultOpName() const override;
    bool IsFp8Dtype(const ge::DataType dtype) const;
    bool IsHiFloat8Dtype(const ge::DataType dtype) const;
    bool IsMxQuant() const;
    bool IsHiFloat8TTQuant() const;
    bool CheckParamsForMxQuant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape) const;
    bool CheckParamsForPerTensorQuant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape) const;
    bool CheckShapeVaild(const gert::Shape& x1Shape, const gert::Shape& x2Shape) const;
    bool InitMatmulSize(const gert::Shape& x1Shape, const gert::Shape& x2Shape);
    bool ValidateQuantParams(const gert::Shape& x1ScaleShape, const gert::Shape& scaleShape);
    uint64_t GetBatchCoreCnt() const override;
    template <typename TilingData>
    void ResetInplaceTilingData(TilingData& tilingData);
    void CopyV3BasicApiTilingData(
        const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& src,
        QMMIA::QuantBatchMatmulInplaceAddTilingData& dst);
};

template <typename BaseT>
const char *QuantBatchMatmulInplaceAddHelper<BaseT>::GetDefaultOpName() const
{
    return "QuantBatchMatmulInplaceAdd";
}

template <typename BaseT>
ge::graphStatus QuantBatchMatmulInplaceAddHelper<BaseT>::GetShapeAttrsInfo()
{
    this->tilingDataSize_ = sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::InitMatmulSize(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape)
{
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (x1ShapeLen < X1_MINIMUM_DIMENSION_LENGTH || x2ShapeLen < X2_MINIMUM_DIMENSION_LENGTH) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            this->inputParams_.opName, "x1, x2", FormatString("%zuD, %zuD", x1ShapeLen, x2ShapeLen).c_str(),
            "the shape dims of x1 and x2 must be greater than or equal to 2");
        return false;
    }

    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    this->inputParams_.mSize = static_cast<uint64_t>(this->inputParams_.transA ? x1Inner : x1Outer);
    this->inputParams_.kSize = static_cast<uint64_t>(this->inputParams_.transA ? x1Outer : x1Inner);
    this->inputParams_.nSize = static_cast<uint64_t>(this->inputParams_.transB ? x2Outer : x2Inner);
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::ValidateQuantParams(
    const gert::Shape& x1ScaleShape, const gert::Shape& scaleShape)
{
    if (this->inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0) {
        return CheckParamsForMxQuant(x1ScaleShape, scaleShape);
    }
    return CheckParamsForPerTensorQuant(x1ScaleShape, scaleShape);
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::AnalyzeAttrs()
{
    auto attrs = this->context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "attrs", "null", "attrs can not be null"),
        return false);
    OP_CHECK_IF(
        attrs->GetAttrNum() < ATTR_INDEX_NUMBERS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "attrs num", std::to_string(attrs->GetAttrNum()).c_str(),
            FormatString("the num of attrs must be greater than or equal to %u", ATTR_INDEX_NUMBERS).c_str()),
        return false);
    const bool* transposeXPtr = attrs->template GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X1);
    OP_CHECK_IF(
        transposeXPtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "transposeX1", "null", "transposeX1 can not be null"),
        return false);
    const bool* transposeWeightPtr = attrs->template GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X2);
    OP_CHECK_IF(
        transposeWeightPtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "transposeX2", "null", "transposeX2 can not be null"),
        return false);
    const int64_t* groupSizePtr = attrs->template GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
    OP_CHECK_IF(
        groupSizePtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "groupSize", "null", "groupSize can not be null"),
        return false);
    this->inputParams_.groupSize = *groupSizePtr;
    this->inputParams_.transA = *transposeXPtr;
    this->inputParams_.transB = *transposeWeightPtr;
    if (this->inputParams_.groupSize != 0ULL) {
        this->inputParams_.groupSizeK = this->inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
        this->inputParams_.groupSizeN = (this->inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE;
        this->inputParams_.groupSizeM = (this->inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE;
    }
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::AnalyzeDtype()
{
    auto xDesc = this->context_->GetInputDesc(X1_INDEX);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x1", "null", "x1 can not be null"),
        return false);
    this->inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = this->context_->GetInputDesc(X2_INDEX);
    OP_CHECK_IF(
        wDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x2", "null", "x2 can not be null"),
        return false);
    this->inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = this->context_->GetInputDesc(X2_SCALE_INDEX);
    this->inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : this->inputParams_.scaleDtype;
    auto pertokenScaleDesc = this->context_->GetOptionalInputDesc(X1_SCALE_INDEX);
    this->inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : this->inputParams_.perTokenScaleDtype;
    auto outDesc = this->context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OP_CHECK_IF(
        outDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "output", "null", "output can not be null"),
        return false);
    this->inputParams_.cDtype = outDesc->GetDataType();
    return CheckDtype();
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::IsFp8Dtype(const ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::IsHiFloat8Dtype(const ge::DataType dtype) const
{
    return dtype == ge::DT_HIFLOAT8;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::IsMxQuant() const
{
    return IsFp8Dtype(this->inputParams_.aDtype) && IsFp8Dtype(this->inputParams_.bDtype) &&
           this->inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           this->inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0 && this->inputParams_.isMxPerGroup;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::IsHiFloat8TTQuant() const
{
    return IsHiFloat8Dtype(this->inputParams_.aDtype) && IsHiFloat8Dtype(this->inputParams_.bDtype) &&
           this->inputParams_.scaleDtype == ge::DT_FLOAT &&
           this->inputParams_.perTokenScaleDtype == ge::DT_FLOAT && this->inputParams_.isDoubleScale &&
           this->inputParams_.isPerTensor;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::CheckDtype() const
{
    OP_CHECK_IF(
        this->inputParams_.cDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            this->inputParams_.opName, "output",
            ge::TypeUtils::DataTypeToSerialString(this->inputParams_.cDtype).c_str(),
            "the dtype of output must be FLOAT"),
        return false);
    OP_CHECK_IF(
        this->inputParams_.transA != true || this->inputParams_.transB != false,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            this->inputParams_.opName, "transposeX1, transposeX2",
            FormatString("%s, %s", this->inputParams_.transA ? "true" : "false",
                this->inputParams_.transB ? "true" : "false").c_str(),
            "transposeX1 must be true and transposeX2 must be false"),
        return false);
    bool isFp8 = IsFp8Dtype(this->inputParams_.aDtype) && IsFp8Dtype(this->inputParams_.bDtype);
    bool isHiFloat8 = IsHiFloat8Dtype(this->inputParams_.aDtype) && IsHiFloat8Dtype(this->inputParams_.bDtype);
    if (isFp8) {
        OP_CHECK_IF(
            this->inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 ||
                this->inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                this->inputParams_.opName, "x1Scale, x2Scale",
                FormatString("%s, %s",
                    ge::TypeUtils::DataTypeToSerialString(this->inputParams_.perTokenScaleDtype).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(this->inputParams_.scaleDtype).c_str()).c_str(),
                "when the dtype of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2, the dtype of x1Scale and x2Scale must be FLOAT8_E8M0"),
            return false);
    } else if (isHiFloat8) {
        OP_CHECK_IF(
            this->inputParams_.scaleDtype != ge::DT_FLOAT || this->inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                this->inputParams_.opName, "x1Scale, x2Scale",
                FormatString("%s, %s",
                    ge::TypeUtils::DataTypeToSerialString(this->inputParams_.perTokenScaleDtype).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(this->inputParams_.scaleDtype).c_str()).c_str(),
                "when the dtype of x1 and x2 is HIFLOAT8, the dtype of x1Scale and x2Scale must be FLOAT"),
            return false);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            this->inputParams_.opName, "x1, x2",
            FormatString("%s, %s",
                ge::TypeUtils::DataTypeToSerialString(this->inputParams_.aDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(this->inputParams_.bDtype).c_str()).c_str(),
            "the dtypes of x1 and x2 must both be FLOAT8_E4M3FN/FLOAT8_E5M2 or both be HIFLOAT8");
        return false;
    }
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::CheckShapeVaild(
    const gert::Shape& x1Shape, const gert::Shape& x2Shape) const
{
    auto x1ShapeLength = x1Shape.GetDimNum();
    auto x2ShapeLength = x2Shape.GetDimNum();
    OP_CHECK_IF(
        x1ShapeLength != X1_MINIMUM_DIMENSION_LENGTH || x2ShapeLength != X2_MINIMUM_DIMENSION_LENGTH,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            this->inputParams_.opName, "x1, x2",
            FormatString("%zuD, %zuD", x1ShapeLength, x2ShapeLength).c_str(),
            "the shape dims of x1 and x2 must be 2"),
        return false);
    auto x2KDimValue =
        static_cast<uint64_t>(this->inputParams_.transB ? x2Shape.GetDim(1) : x2Shape.GetDim(0));
    auto x1KDimValue =
        static_cast<uint64_t>(this->inputParams_.transA ? x1Shape.GetDim(0) : x1Shape.GetDim(1));
    OP_CHECK_IF(
        x1KDimValue != x2KDimValue,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            this->inputParams_.opName, "x1K, x2K",
            FormatString("%lu, %lu", x1KDimValue, x2KDimValue).c_str(),
            "the K dimension of x1 must be equal to the K dimension of x2"),
        return false);
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::CheckParamsForMxQuant(
    const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape) const
{
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(
        x1ScaleDimNum != MX_X1_SCALE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            this->inputParams_.opName, "x1Scale", FormatString("%zuD", x1ScaleDimNum).c_str(),
            "when the quantization mode is mx, the shape dim of x1Scale must be 3"),
        return false);
    OP_CHECK_IF(
        x2ScaleDimNum != MX_X2_SCALE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            this->inputParams_.opName, "x2Scale", FormatString("%zuD", x2ScaleDimNum).c_str(),
            "when the quantization mode is mx, the shape dim of x2Scale must be 3"),
        return false);
    auto x2ScaleNDim =
        static_cast<uint64_t>(this->inputParams_.transB ? x2ScaleShape.GetDim(0) : x2ScaleShape.GetDim(1));
    auto x2ScaleKDim =
        static_cast<uint64_t>(this->inputParams_.transB ? x2ScaleShape.GetDim(1) : x2ScaleShape.GetDim(0));
    auto x1ScaleMDim =
        static_cast<uint64_t>(this->inputParams_.transA ? x1ScaleShape.GetDim(1) : x1ScaleShape.GetDim(0));
    auto x1ScaleKDim =
        static_cast<uint64_t>(this->inputParams_.transA ? x1ScaleShape.GetDim(0) : x1ScaleShape.GetDim(1));
    auto x1ScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(MX_X1_SCALE_DIM - 1));
    auto x2ScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(MX_X2_SCALE_DIM - 1));
    auto expectedKDimValue = ops::CeilDiv(this->inputParams_.kSize, MXFP_BASEK_FACTOR);
    OP_CHECK_IF(
        x2ScaleKDim != expectedKDimValue || x2ScaleNDim != this->inputParams_.nSize ||
            x2ScaleLastDim != MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            this->inputParams_.opName, "x2Scale",
            FormatString("[%lu, %lu, %lu]", x2ScaleKDim, x2ScaleNDim, x2ScaleLastDim).c_str(),
            FormatString("when the quantization mode is mx, the shape of x2Scale must be [%lu, %lu, 2]",
                expectedKDimValue, this->inputParams_.nSize).c_str()),
        return false);
    OP_CHECK_IF(
        x1ScaleMDim != this->inputParams_.mSize || x1ScaleKDim != expectedKDimValue ||
            x1ScaleLastDim != MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            this->inputParams_.opName, "x1Scale",
            FormatString("[%lu, %lu, %lu]", x1ScaleKDim, x1ScaleMDim, x1ScaleLastDim).c_str(),
            FormatString("when the quantization mode is mx, the shape of x1Scale must be [%lu, %lu, 2]",
                expectedKDimValue, this->inputParams_.mSize).c_str()),
        return false);
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::CheckParamsForPerTensorQuant(
    const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape) const
{
    OP_CHECK_IF(
        x1ScaleShape.GetDimNum() != 1 || x2ScaleShape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            this->inputParams_.opName, "x1Scale, x2Scale",
            FormatString("%zuD, %zuD", x1ScaleShape.GetDimNum(), x2ScaleShape.GetDimNum()).c_str(),
            "when the quantization mode is per-tensor, the shape dims of x1Scale and x2Scale must be 1"),
        return false);
    OP_CHECK_IF(
        x1ScaleShape.GetDim(0) != 1 || x2ScaleShape.GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            this->inputParams_.opName, "x1Scale, x2Scale",
            FormatString("[%ld], [%ld]", x1ScaleShape.GetDim(0), x2ScaleShape.GetDim(0)).c_str(),
            "when the quantization mode is per-tensor, the shape of x1Scale and x2Scale must be [1]"),
        return false);
    return true;
}

template <typename BaseT>
bool QuantBatchMatmulInplaceAddHelper<BaseT>::AnalyzeInputs()
{
    const gert::StorageShape* x1StorageShape = this->context_->GetInputShape(X1_INDEX);
    OP_CHECK_IF(
        x1StorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x1", "null", "x1 can not be null"),
        return false);
    const gert::StorageShape* x2StorageShape = this->context_->GetInputShape(X2_INDEX);
    OP_CHECK_IF(
        x2StorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(this->inputParams_.opName, "x2", "null", "x2 can not be null"),
        return false);
    auto x1Shape = x1StorageShape->GetOriginShape();
    auto x2Shape = x2StorageShape->GetOriginShape();
    auto scaleShape = this->GetScaleShape(X2_SCALE_INDEX);
    auto pertokenShape = this->GetPertokenShape(X1_SCALE_INDEX);
    OP_CHECK_IF(
        pertokenShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            this->inputParams_.opName, "x1Scale", "null", "x1Scale can not be null"),
        return false);
    auto& x1ScaleShape = pertokenShape->GetStorageShape();
    this->inputParams_.isPertoken = true;

    this->inputParams_.hasBias = 0;
    this->inputParams_.batchBias = 1;
    if (!InitMatmulSize(x1Shape, x2Shape)) {
        return false;
    }
    if (!this->AnalyzeGroupInfo(scaleShape, pertokenShape)) {
        return false;
    }
    this->inputParams_.batchA = this->GetBatchSize(x1Shape);
    this->inputParams_.batchB = this->GetBatchSize(x2Shape);
    this->AnalyzeBatchInfo(x1StorageShape->GetOriginShape(), x2StorageShape->GetOriginShape());
    OP_TILING_CHECK(!this->InferOutBatchDim(x1Shape, x2Shape),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        this->inputParams_.opName, "x1DimNum, x2DimNum",
                        FormatString("%zu, %zu", x1Shape.GetDimNum(), x2Shape.GetDimNum()).c_str(),
                        "the batch dimensions of x1 and x2 must be broadcastable"),
                    return false);
    if (!this->SetQuantMode(scaleShape, pertokenShape) || !ValidateQuantParams(x1ScaleShape, scaleShape) ||
        !CheckShapeVaild(x1Shape, x2Shape)) {
        return false;
    }
    OP_LOGD(
        this->inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %d, isPertoken: %d",
        this->inputParams_.batchA, this->inputParams_.batchB, this->inputParams_.batchC,
        static_cast<int>(this->inputParams_.isPerTensor), static_cast<int>(this->inputParams_.isPertoken));
    return true;
}

template <typename BaseT>
uint64_t QuantBatchMatmulInplaceAddHelper<BaseT>::GetBatchCoreCnt() const
{
    return this->inputParams_.batchC;
}

template <typename BaseT>
template <typename TilingData>
void QuantBatchMatmulInplaceAddHelper<BaseT>::ResetInplaceTilingData(TilingData& tilingData)
{
    if (!this->isTilingOut_) {
        tilingData = TilingData();
        OP_TILING_CHECK(
            memset_s(
                this->context_->GetRawTilingData()->GetData(), this->context_->GetRawTilingData()->GetCapacity(), 0,
                this->context_->GetRawTilingData()->GetCapacity()) != EOK,
            CUBE_INNER_ERR_REPORT(this->inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

template <typename BaseT>
void QuantBatchMatmulInplaceAddHelper<BaseT>::CopyV3BasicApiTilingData(
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& src,
    QMMIA::QuantBatchMatmulInplaceAddTilingData& dst)
{
    static_assert(
        sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData) ==
            sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData),
        "QuantBatchMatmulInplaceAdd basic-api tiling data must stay layout-compatible with QuantBatchMatmulV3.");
    OP_TILING_CHECK(
        memcpy_s(&dst, sizeof(dst), &src, sizeof(src)) != EOK,
        CUBE_INNER_ERR_REPORT(this->inputParams_.opName, "Fail to copy basic-api tiling data"), return);
}

} // namespace optiling

#endif
