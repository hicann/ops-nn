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
 * \file matmul_v3_tiling_advanced.cc
 * \brief
 */

#include "matmul_v3_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "./matmul_tiling_registry.h"
#include "./matmul_tiling_cfg.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_strategy.h"

namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

constexpr uint64_t ONE_BATCH_DIM = 1UL;
constexpr uint64_t TWO_BATCH_DIM = 2UL;
constexpr uint64_t THREE_BATCH_DIM = 3UL;
constexpr uint64_t FOUR_BATCH_DIM = 4UL;
constexpr size_t HF32_ATTR_NUM = 4UL;
constexpr size_t HF32_ATTR_INDEX = 3UL;
constexpr size_t OP_IMPL_MODE_ATTR_NUM = 4UL;
constexpr size_t OP_IMPL_MODE_ATTR_INDEX = 3UL;
constexpr size_t BIAS_IDX = 2UL;

inline void GetDtype(const gert::TilingContext &context, MatMulV3Args &args)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(BIAS_IDX)->GetDataType();
    }
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (strcmp(context.GetNodeType(), "MatMulV3") == 0) {
        if (context.GetAttrs()->GetAttrNum() >= OP_IMPL_MODE_ATTR_NUM) {
            args.isHf32 = *context.GetAttrs()->GetAttrPointer<int64_t>(OP_IMPL_MODE_ATTR_INDEX) == 0x40;
        }
    } else {
        if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
            args.isHf32 = *((context.GetAttrs())->GetAttrPointer<bool>(HF32_ATTR_INDEX));
        }
    }
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);
}

ge::graphStatus IsValidDtype(const MatMulV3Args &args)
{
    std::vector<ge::DataType> dtype = { args.aType, args.bType, args.cType };
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }
    const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
        // x1,              x2,             y,              bias
        { ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16 },
        { ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT },
        { ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT },
        { ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT },
        { ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16 } // david supports bias-bf16
    };
    for (auto &supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (args.hasBias) {
        OP_LOGE(args.opName,
            "Unsupported data type: x1[%s], x2[%s], y[%s], bias[%s], input dtype of x1 and x2 and output dtype "
            "must be same, only support[DT_FLOAT16, DT_FLOAT, DT_BF16], and bias dtype must be same to input type "
            "or equals DT_FLOAT when input dtype is DT_FLOAT16 | DT_BF16",
            Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(),
            Ops::Base::ToString(args.cType).c_str(), Ops::Base::ToString(args.biasType).c_str());
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(args.opName,
            "Unsupported data type: x1[%s], x2[%s], y[%s], input dtype of x1 and x2 and output dtype must be same, "
            "only support[DT_FLOAT16, DT_FLOAT, DT_BF16]",
            Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(),
            Ops::Base::ToString(args.cType).c_str());
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus GetInputDims(const gert::Shape& storageShape, const gert::Shape& oriShape, uint64_t dtypeSize,
                             ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    const size_t oriDimNum = oriShape.GetDimNum();
    dims[0] = oriShape[oriDimNum - TWO_BATCH_DIM];
    dims[1] = oriShape[oriDimNum - ONE_BATCH_DIM];
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        int64_t storageShape0 = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        int64_t storageShape1 = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
        if (ops::CeilAlign(dims[0], static_cast<int64_t>(BASIC_BLOCK_SIZE_16)) != storageShape0 ||
            ops::CeilAlign(dims[1], static_cast<int64_t>(BLOCK_BYTE_SIZE / dtypeSize)) != storageShape1) {
            OP_LOGE("MatMulV3", "NZ aligned oriShape (%ld, %ld) is not equal to storageShape (%ld, %ld))", dims[0],
                    dims[1], storageShape0, storageShape1);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

}

namespace optiling {
namespace matmul_v3_advanced {
ge::graphStatus MatMulV3Tiling::CheckSelfSlice(int64_t (&dims)[TWO_BATCH_DIM])
{
    auto selfShape = context_->GetInputShape(0)->GetOriginShape();
    if (args_.isATrans) {
        OP_LOGE(args_.opName, "non-contiguous slice does not support transA");
        return ge::GRAPH_FAILED;
    }
    dims[0] = selfShape[0] * selfShape[1]; // m = batch * sliceM
    dims[1] = selfShape[2];                    // k = viewShape[2]
    isSelfSlice_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::CheckInputTranspose(int64_t (&dims)[TWO_BATCH_DIM], int64_t idx)
{
    auto inputViewShape = context_->GetInputShape(idx)->GetOriginShape();
    const size_t oriDimNum = inputViewShape.GetDimNum();
    if (oriDimNum != THREE_BATCH_DIM) {
        OP_LOGE(args_.opName, "non-contiguous transpose viewShape dim is not 3");
        return ge::GRAPH_FAILED;
    }
    dims[0] = inputViewShape[oriDimNum - TWO_BATCH_DIM];
    dims[1] = inputViewShape[oriDimNum - ONE_BATCH_DIM];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), reinterpret_cast<void *>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    NpuArch npuArch = reinterpret_cast<const MatmulV3CompileInfo *>(tilingCfg.compileInfo)->npuArch;
    MMRegisterCfg registerCfg{ "MatMulV3", npuArch, strategy::GetMatMulV3Priorities(npuArch) };
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

bool MatMulV3Tiling::CheckIsNonContiguous(int64_t (&mkDims)[TWO_BATCH_DIM], int64_t (&knDims)[TWO_BATCH_DIM])
{
    auto selfShape = context_->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context_->GetInputShape(1)->GetOriginShape();
    auto selfStorageShape = context_->GetInputShape(0)->GetStorageShape();
    auto mat2StorageShape = context_->GetInputShape(1)->GetStorageShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t mat2DimNum = mat2Shape.GetDimNum();
    // transpose非连续校验，根据storageshape 1d和右矩阵维度3d判断
    bool isANonContiguous =
        context_->InputIsView(0) && (selfStorageShape.GetDimNum() == 1) && (selfDimNum == THREE_BATCH_DIM);
    bool isBTransposeNonContiguous =
        context_->InputIsView(1) && (mat2StorageShape.GetDimNum() == 1) && (mat2DimNum == THREE_BATCH_DIM);
    bool isASliceNonContiguous = isANonContiguous && mat2DimNum == 2;
    bool isATransposeNonContiguous = isANonContiguous && mat2DimNum == 3;
    // createView with TensorV2 & 3D *2D & storageShape 1d -> support  slice
    if (isASliceNonContiguous) {
        if (CheckSelfSlice(mkDims) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else if (isATransposeNonContiguous) { // only 3d support  transpose
        // transpose非连续校验，根据storageshape 1d 和右矩阵维度 3d 判断
        if (CheckInputTranspose(mkDims, 0) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else {
        if ((GetInputDims(selfStorageShape, selfShape, args_.aDtypeSize, args_.aFormat, mkDims) != ge::GRAPH_SUCCESS)) {
            OP_LOGE(args_.opName, "invalid input dim num for self");
            return false;
        }
    }

    if (isBTransposeNonContiguous) { // only 3d support  transpose
        if (CheckInputTranspose(knDims, 1) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else {
        if ((GetInputDims(mat2StorageShape, mat2Shape, args_.bDtypeSize, args_.bFormat, knDims) != ge::GRAPH_SUCCESS)) {
            OP_LOGE(args_.opName, "invalid input dim num for mat2");
            return false;
        }
    }
    return true;
}

ge::graphStatus MatMulV3Tiling::GetShape()
{
    // get transpose
    args_.isATrans = *((context_->GetAttrs())->GetAttrPointer<bool>(0));
    args_.isBTrans = *((context_->GetAttrs())->GetAttrPointer<bool>(1));

    // get (m, k, n)
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];

    // NZ异常校验
    if (args_.aFormat == ge::FORMAT_FRACTAL_NZ || args_.outFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(args_.opName, "Parameter format error: The 'self', and 'out' tensors not support NZ formats.");
        return ge::GRAPH_FAILED;
    }

    // 非连续校验
    if (!CheckIsNonContiguous(mkDims, knDims)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t kIdxA = args_.isATrans ? 0ULL : 1ULL;
    uint64_t kIdxB = args_.isBTrans ? 1ULL : 0ULL;
    int64_t k = mkDims[kIdxA];
    if (k != knDims[kIdxB]) {
      OP_LOGE(args_.opName, "unequal input kDim values: k_left[%ld], k_right[%ld]", k, knDims[kIdxB]);
      return ge::GRAPH_FAILED;
    }
    uint64_t mIdx = args_.isATrans ? 1ULL : 0ULL;
    uint64_t nIdx = args_.isBTrans ? 0ULL : 1ULL;
    int64_t m = mkDims[mIdx];
    int64_t n = knDims[nIdx];
    args_.mValue = static_cast<uint64_t>(m);
    args_.kValue = static_cast<uint64_t>(k);
    args_.nValue = static_cast<uint64_t>(n);
    if (args_.kValue == 0UL && args_.hasBias) {
        OP_LOGE(args_.opName, "Can not support inputShape has bias when kValue equals zero.");
        return ge::GRAPH_FAILED;
    }
    auto isValidDimValue = [](int64_t dim) -> bool {
        return (dim > 0) && (dim <= INT32_MAX);
    };
    auto isValidDimValueK = [](int64_t dim) -> bool {
        return (dim >= 0) && (dim <= INT32_MAX);
    };
    if (!isValidDimValue(args_.mValue) || !isValidDimValueK(args_.kValue) || !isValidDimValue(args_.nValue)) {
        OP_LOGE(args_.opName, "illegal value: m[%lu], k[%lu], n[%lu]", args_.mValue, args_.kValue, args_.nValue);
        return ge::GRAPH_FAILED;
    }
    // check output dim num
    const gert::Shape &cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE(args_.opName, "illegal value: output dim num (%zu)", cDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::BaseOpSpecificCheck()
{
    const bool isMatMulV3 = (strcmp(context_->GetNodeType(), "MatMulV3") == 0);
    const bool isBatchMatMulV3 = (strcmp(context_->GetNodeType(), "BatchMatMulV3") == 0);
    if (!isBatchMatMulV3 && !isMatMulV3) {
        // apply no additional checks for ops other than MMV3, BMMV3, for now
        return ge::GRAPH_SUCCESS;
    }

    // check input dim num equals to 2
    if (isMatMulV3) {
        auto isMatrix = [this](const gert::Shape& oriShape) {
            if (isSelfSlice_) {
                return oriShape.GetDimNum() == TWO_BATCH_DIM || oriShape.GetDimNum() == THREE_BATCH_DIM;
            } else {
                return oriShape.GetDimNum() == TWO_BATCH_DIM;
            }
        };
        if (!isMatrix(context_->GetInputShape(0)->GetOriginShape()) ||
            !isMatrix(context_->GetInputShape(1)->GetOriginShape())) {
            OP_LOGE(args_.opName, "invalid input dim num");
            return ge::GRAPH_FAILED;
        }
    }

    // check bias
    if (args_.hasBias) {
        const gert::Shape &biasShape = context_->GetInputShape(BIAS_IDX)->GetOriginShape();
        const gert::Shape &cShape = context_->GetOutputShape(0)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        const int64_t nOriValue = cShape[cShape.GetDimNum() - 1];
        if (biasValue != nOriValue) {
            OP_LOGE(args_.opName, "illegal value: bias[%ld], n[%ld]", biasValue, nOriValue);
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    return IsValidDtype(args_);
}

void MatMulV3Tiling::GetFormat()
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOutputDesc(0)->GetStorageFormat()));
    args_.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args_.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args_.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

ge::graphStatus MatMulV3Tiling::GetArgs()
{
    GetFormat();
    GetDtype(*context_, args_);
    if (GetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return BaseOpSpecificCheck();
}

ge::graphStatus MatMulV3Tiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("matmul", "get op name invalid context"),
        return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    // 区分MatMul和GemmV2，只有3个输入的为MatMul，并设置bias标志
    if (context_->GetInputDesc(idx) != nullptr && context_->GetInputDesc(idx + 1) == nullptr) {
        args_.hasBias = true;
    }
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(HF32_ATTR_INDEX - 1));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

}
}