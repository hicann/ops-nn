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
 * \file conv3d_backprop_input_v2_base_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/tbe_tiling_api.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "tiling_base/tiling_key.h"
#include "error_util.h"
#include "conv3d_backprop_input_v2_base_tiling.h"

using Ops::NN::Optiling::RecursiveSum;

namespace {
constexpr size_t OUTPUT_PADDING_DIM = 5;
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t F8_C0_BITS = 5;
constexpr uint32_t F16_C0_BITS = 4;
constexpr uint32_t F32_C0_BITS = 3;
constexpr uint32_t BIT8_DATA_SIZE = 1; // for hif8 and fp8
constexpr int64_t USER_WORKSIZE_LIMIT = static_cast<int64_t>(140 * 1024 * 1024);
constexpr uint64_t L1_BIAS_SIZE = static_cast<uint64_t>(4 * 1024);
constexpr int32_t DIM_FACTOR = 2;
constexpr int32_t BUFFER_NUM_L1 = 4;
constexpr float CORE_USED_THRESHOLD = 0.6;
constexpr float CORE_USED_D_THRESHOLD = 0.95;
constexpr uint64_t MAX_UINT16 = 65535;
const int32_t DIM_LOW = 1;
const int32_t PAD_DIM_LOW = 0;
const int32_t PAD_DIM_UP = 255;
const int32_t STRIDES_DIM_HW_UP = 63;
const int32_t STRIDES_DIM_DEPTH_UP = 255;
const int32_t GROUPS_LOW = 1;
const int32_t GROUPS_UP = 65535;
const int32_t FP32_FIXPIPE_BOUND_K_LIMIT = 528; // 理论值,输出fp32时,当 K >= 528 时才能不fixpipe bound

// 0: best base block; 1: threshold base block
constexpr uint32_t BASE_BLOCK_TYPE_BEST = 0;
constexpr uint32_t BASE_BLOCK_TYPE_THRESHOLD = 1;

const std::vector<Ops::NN::Conv::TilingBestBaseBlock> TILING_BEST_BASE_BLOCK_DEF{
    {128, 128, 256},
    {128, 128, 256},
};

const std::vector<Ops::NN::Conv::TilingBestBaseBlock> TILING_BEST_BASE_BLOCK_D1{
    {256, 128, 256},
    {256, 128, 128},
};

const std::map<platform_ascendc::SocVersion, std::vector<Ops::NN::Conv::TilingBestBaseBlock>> TILING_BEST_BASE{
    {platform_ascendc::SocVersion::ASCEND910_95, TILING_BEST_BASE_BLOCK_D1},
};
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

static inline bool CheckRange(int32_t value, int32_t valueLow, int32_t valueUp)
{
    if (value < valueLow || value > valueUp) {
        return false;
    }
    return true;
}

static inline bool CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize)
{
    int64_t l0aSize = static_cast<int64_t>(baseM) * static_cast<int64_t>(baseK) * static_cast<int64_t>(byteSize) *
                      static_cast<int64_t>(DB_ON);
    int64_t l0bSize = static_cast<int64_t>(baseN) * static_cast<int64_t>(baseK) * static_cast<int64_t>(byteSize) *
                      static_cast<int64_t>(DB_ON);

    return l0aSize <= L0_AB_SIZE && l0bSize <= L0_AB_SIZE;
}

void Conv3DBackpropInputV2TilingArch35::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    opName_ = nullptr;
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropInputV2TilingArch35::GetShapeFormatInfo()
{
    size_t aMatrixIndex = OUTPUT_BP_INDEX;
    size_t bMatrixIndex = FILTER_INDEX;

    if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2) {
        aMatrixIndex = FILTER_INDEX;
        bMatrixIndex = OUTPUT_BP_INDEX;
    }

    const auto out_backprop_desc = context_->GetInputDesc(aMatrixIndex);
    OP_TILING_CHECK(
        out_backprop_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "out_backprop_desc is null"), return false);
    runInfo_.outBackpropFormat = out_backprop_desc->GetStorageFormat();

    const auto filter_desc = context_->GetInputDesc(bMatrixIndex);
    OP_TILING_CHECK(filter_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "filter_desc is null"), return false);
    runInfo_.filterFormat = filter_desc->GetStorageFormat();

    const auto y_desc = context_->GetOutputDesc(Y_INDEX);
    OP_TILING_CHECK(y_desc == nullptr, CUBE_INNER_ERR_REPORT(opName_, "y_desc is null"), return false);
    runInfo_.yFormat = y_desc->GetStorageFormat();
    return true;
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(
        !GetShapeFormatInfo(), CUBE_INNER_ERR_REPORT(opName_, "fail to shape format info"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !AnalyzeDtype(), CUBE_INNER_ERR_REPORT(opName_, "fail to analyze context info"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropInputV2TilingArch35::IsCapable()
{
    if (context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->shortSocVersion !=
        platform_ascendc::SocVersion::ASCEND910_95) {
        return false;
    }
    return true;
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::DoOpTiling()
{
    auto compileInfoPtr = static_cast<const Conv3DBackpropV2CompileInfo*>(context_->GetCompileInfo());
    OP_TILING_CHECK(
        compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropInputV2", "compile_info is null"),
        return false);
    shortSocVersion_ = compileInfoPtr->shortSocVersion;
    if (TILING_BEST_BASE.find(shortSocVersion_) == TILING_BEST_BASE.end()) {
        OP_LOGE(context_, "soc version is invalid");
        return false;
    }

    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed");
        return ge::GRAPH_FAILED;
    }

    if (!GetTbeTiling(context_, tbeTiling_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }

    auto biasShape = context_->GetOptionalInputShape(BAIS_INDEX);
    hasBiasFlag_ = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    blockSize_ = BYTE_BLOCK / runInfo_.a_dtype_bytes;
    dtypeByte_ = runInfo_.a_dtype_bytes;
    dtypeByteL0c_ = runInfo_.c_dtype_bytes;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::DoLibApiTiling()
{
    SetDxTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DBackpropInputV2TilingArch35::GetTilingKey() const
{
    return RecursiveSum(loadB2Condition_, 0, groupConvMode_);
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 框架预留16M
    workspaces[0] = static_cast<size_t>(WORKSIZE);
    if (enableSplitDk_) {
        workspaces[0] += usrSpaceSizeForSplitDk_;
        OP_LOGD(opName_, "Enable split Dk, usrSpaceSize = %ld", usrSpaceSizeForSplitDk_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropInputV2TilingArch35::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_.GetDataSize());

    OP_TILING_CHECK(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    uint32_t dstStride = tilingData_.conv3DDxTiling.get_baseM() / blockSize_; // 为load3d的dstStride做截断保护
    OP_TILING_CHECK(
        dstStride > MAX_UINT16, CUBE_INNER_ERR_REPORT(opName_, "dstStride > MAX_UINT16"), return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.params.get_coreNum());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropInputV2TilingArch35::CheckDtypeFormatAttrs(
    size_t aMatrixesIndex, size_t bMatrixesIndex, bool hif8flag, bool fp8e4m3flag) const
{
    const auto out_backprop_desc = context_->GetInputDesc(aMatrixesIndex);
    const auto filter_desc = context_->GetInputDesc(bMatrixesIndex);
    const auto y_desc = context_->GetOutputDesc(Y_INDEX);
    bool isFormatNotDn = runInfo_.outBackpropFormat != ge::FORMAT_NCDHW || runInfo_.filterFormat != ge::FORMAT_NCDHW ||
                         runInfo_.yFormat != ge::FORMAT_NCDHW;

    OP_TILING_CHECK(
        hif8flag && fp8e4m3flag && isFormatNotDn,
        CUBE_INNER_ERR_REPORT(
            opName_,
            "the current output_backprop_dtype is [%s], filter_dtype is [%s], y_dtype is [%s], "
            "and format only support NCDHW, but actually get out_backprop_format is [%s], filter_format is [%s], "
            "y_format is [%s]",
            ge::TypeUtils::DataTypeToSerialString(out_backprop_desc->GetDataType()).c_str(),
            ge::TypeUtils::DataTypeToSerialString(filter_desc->GetDataType()).c_str(),
            ge::TypeUtils::DataTypeToSerialString(y_desc->GetDataType()).c_str(),
            ge::TypeUtils::FormatToSerialString(runInfo_.outBackpropFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(runInfo_.filterFormat).c_str(),
            ge::TypeUtils::FormatToSerialString(runInfo_.yFormat).c_str()),
        return false);

    return true;
}

bool Conv3DBackpropInputV2TilingArch35::AnalyzeDtype() const
{
    size_t inputSizeIndex = INPUT_SIZE_INDEX;
    size_t outputBackpropIndex = OUTPUT_BP_INDEX;
    size_t filterIndex = FILTER_INDEX;

    if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2) {
        outputBackpropIndex = FILTER_INDEX;
        filterIndex = OUTPUT_BP_INDEX;
    }
    OP_TILING_CHECK(
        context_->GetInputDesc(outputBackpropIndex) == nullptr || context_->GetInputDesc(filterIndex) == nullptr ||
            context_->GetOutputDesc(Y_INDEX) == nullptr || context_->GetInputDesc(inputSizeIndex) == nullptr,
        CUBE_INNER_ERR_REPORT(opName_, "failed to get out_backprop/filter/y/input_size tensor desc from context"),
        return false);

    ge::DataType outputBackpropDtype = context_->GetInputDesc(outputBackpropIndex)->GetDataType();
    ge::DataType filterDtype = context_->GetInputDesc(filterIndex)->GetDataType();
    ge::DataType inputSizeDtype = context_->GetInputDesc(inputSizeIndex)->GetDataType();
    ge::DataType yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    bool hif8flag =
        outputBackpropDtype == ge::DT_HIFLOAT8 && filterDtype == ge::DT_HIFLOAT8 && yDtype == ge::DT_HIFLOAT8;
    bool fp8e4m3flag = outputBackpropDtype == ge::DT_FLOAT8_E4M3FN && filterDtype == ge::DT_FLOAT8_E4M3FN &&
                       yDtype == ge::DT_FLOAT8_E4M3FN;
    bool bf16flag = outputBackpropDtype == ge::DT_BF16 && filterDtype == ge::DT_BF16 && yDtype == ge::DT_BF16;
    bool f16flag = outputBackpropDtype == ge::DT_FLOAT16 && filterDtype == ge::DT_FLOAT16 && yDtype == ge::DT_FLOAT16;
    bool f32flag = outputBackpropDtype == ge::DT_FLOAT && filterDtype == ge::DT_FLOAT && yDtype == ge::DT_FLOAT;
    OP_TILING_CHECK(
        !hif8flag && !fp8e4m3flag && !bf16flag && !f16flag && !f32flag,
        CUBE_INNER_ERR_REPORT(
            opName_,
            "the dtype of outputBackprop, filter, y only support "
            "DT_HIFLOAT8/DT_FLOAT8_E4M3FN/DT_BF16/DT_FLOAT16/DT_FLOAT, "
            "but actually get outputBackpropDtype is [%s], filterDtype is [%s], yDtype is [%s]",
            ge::TypeUtils::DataTypeToSerialString(outputBackpropDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(filterDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        opType_ == optiling::OpTypeV2::kConv3DTransposeV2 && inputSizeDtype != ge::DT_INT32 &&
            inputSizeDtype != ge::DT_INT64,
        CUBE_INNER_ERR_REPORT(
            opName_, "input_size dtype should be int32 or int64, but actually get inputSizeDtype is [%s]",
            ge::TypeUtils::DataTypeToSerialString(inputSizeDtype).c_str()),
        return false);
    if (!CheckDtypeFormatAttrs(outputBackpropIndex, filterIndex, hif8flag, fp8e4m3flag)) {
        return false;
    }
    return true;
}

int32_t Conv3DBackpropInputV2TilingArch35::GetDimFactor(
    const int64_t& value, const std::vector<int32_t>& factorLits) const
{
    int32_t dimFactor = 1;
    for (uint32_t i = 0; i < factorLits.size(); i++) {
        if (value % factorLits[i] == 0) {
            dimFactor = factorLits[i];
            break;
        }
    }
    return dimFactor;
}

void Conv3DBackpropInputV2TilingArch35::GetSubFactors(
    const std::vector<int32_t>& src, int32_t factor, int32_t base, std::vector<int32_t>& dst)
{
    for (size_t i = 0; i < src.size(); i++) {
        if ((factor % src[i] == 0) && (factor >= base * src[i])) {
            dst.push_back(src[i]);
        }
    }

    if (dst.size() == 0) {
        dst.push_back(1);
    }
}

bool Conv3DBackpropInputV2TilingArch35::GetCoreDimForMN(
    int32_t curCoreNum, TilingCoreDimDx& coreDim, const std::vector<int32_t>& coreFactors, int32_t remainFactor)
{
    int64_t maxM = static_cast<int64_t>(runInfo_.dedx_h) * static_cast<int64_t>(runInfo_.dedx_w);
    int64_t maxN = static_cast<int64_t>(runInfo_.dedx_cin1) * static_cast<int64_t>(blockSize_);

    // 剩余的在M, N方向如果能均匀分核且切块粒度不小于基本块，也结束
    std::vector<int32_t> mFactors = {};
    std::vector<int32_t> nFactors = {};

    Ops::NN::Conv::TilingBestBaseBlock baseBlock = TILING_BEST_BASE.at(shortSocVersion_).at(BASE_BLOCK_TYPE_THRESHOLD);

    GetSubFactors(coreFactors, maxM, baseBlock.baseM, mFactors);
    GetSubFactors(coreFactors, maxN, baseBlock.baseN, nFactors);
    for (size_t j = 0; j < mFactors.size(); j++) {
        for (size_t i = 0; i < nFactors.size(); i++) {
            if (nFactors[i] * mFactors[j] % remainFactor == 0) {
                coreDim.nDim = MathUtil::GetGcd(remainFactor, nFactors[i]);
                coreDim.mDim = remainFactor / coreDim.nDim;
                return true;
            }
        }
    }

    // m的粒度合适，执行M的非因子分核
    if (maxM >= (remainFactor * baseBlock.baseM)) {
        coreDim.mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, remainFactor));
        uint64_t usedCoreNum = static_cast<uint64_t>(coreDim.batchDim) * tbeTiling_.group_dim * coreDim.mDim *
                               tbeTiling_.k_dim * coreDim.nDim * coreDim.dDim;
        if (usedCoreNum >= static_cast<uint64_t>(curCoreNum)) {
            return true;
        }
    }

    return false;
}

bool Conv3DBackpropInputV2TilingArch35::GetCoreDim(int32_t curCoreNum, TilingCoreDimDx& coreDim)
{
    if (curCoreNum < static_cast<int32_t>(coreNum_ * CORE_USED_THRESHOLD)) {
        return false;
    }

    // 分核目标：
    // (1) 保证singleCoreM和singleCoreN足够大, 可以使能128/64/256的基本块tiling;
    // (2) 数据读取的方向连续且顺序访问，提高cache复用率
    // (3) 所有核因子中间变量以Factor后缀命名，公约数和除法只操作因子返回因子，避免除0

    std::vector<int32_t> coreFactors = {};
    MathUtil::GetFactors(coreFactors, curCoreNum, curCoreNum);
    std::sort(coreFactors.rbegin(), coreFactors.rend());

    // B和D方向的最大公因子乘积是核的倍数，直接均匀分核，结束
    int32_t dMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.dedx_d), coreFactors);
    int32_t bMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.batch_n), coreFactors);
    if ((dMaxFactor * bMaxFactor) % curCoreNum == 0) {
        coreDim.dDim = dMaxFactor;
        coreDim.batchDim = curCoreNum / dMaxFactor;
        return true;
    }

    // B和D分不完，找B*D方向的最大公因子，尽可能在B和D多分
    int32_t batchDepthMaxFactor = GetDimFactor(static_cast<int64_t>(dMaxFactor * bMaxFactor), coreFactors);
    int32_t remainFactor = curCoreNum / batchDepthMaxFactor;
    coreDim.dDim = MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
    coreDim.batchDim = batchDepthMaxFactor / coreDim.dDim;

    if (GetCoreDimForMN(curCoreNum, coreDim, coreFactors, remainFactor)) {
        return true;
    }

    // 当前核数无法分核，尝试[coreNum_ - 1, coreNum_ * 60%]
    return GetCoreDim(curCoreNum - 1, coreDim);
}

void Conv3DBackpropInputV2TilingArch35::SetTilingParamByDimInfo(
    TilingValueDavid& tilingParams, const TilingCoreDimDx& coreDim)
{
    tilingParams.coreNum = static_cast<uint64_t>(coreDim.batchDim) * tbeTiling_.group_dim * coreDim.mDim *
                           tbeTiling_.k_dim * coreDim.nDim * coreDim.dDim;
    tilingParams.batchDim = coreDim.batchDim;
    tilingParams.dDim = coreDim.dDim;
    tilingParams.groupDim = tbeTiling_.group_dim;
    tilingParams.mDim = coreDim.mDim;
    tilingParams.nDim = coreDim.nDim;
    tilingParams.kDim = tbeTiling_.k_dim;
    tilingParams.singleCoreBatch = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.batch_n), coreDim.batchDim);
    tilingParams.singleCoreGroup = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.real_g), tbeTiling_.group_dim);
    tilingParams.singleCoreM =
        static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.dedx_h, coreDim.mDim)) * runInfo_.dedx_w;
    tilingParams.singleCoreCout = runInfo_.dedy_cout_g;
    tilingParams.singleCoreCout1 = runInfo_.dedy_cout1_g;
    tilingParams.singleCoreCin1 = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), coreDim.nDim);
    tilingParams.singleCoreDin = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_d), coreDim.dDim);
    tilingParams.singleCoreHo = 1;

    // for enlarge > 1, need slice cin1 when mov_ub_to_l1
    if (runInfo_.outBackpropFormat == ge::FORMAT_NCDHW && runInfo_.enlarge == 1) {
        tilingParams.singleCoreCin = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin_g), coreDim.nDim);
    } else {
        tilingParams.singleCoreCin =
            Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), coreDim.nDim) * BLOCK_CUBE;
    }
}

bool Conv3DBackpropInputV2TilingArch35::CalCoreDimTiling(TilingValueDavid& tilingParams, const uint32_t coreNum)
{
    TilingCoreDimDx coreDim;
    if (!GetCoreDim(coreNum, coreDim)) {
        return false;
    }

    int64_t coreNumUsed = static_cast<int64_t>(coreDim.batchDim) * coreDim.dDim * coreDim.mDim * coreDim.nDim;
    if (coreNumUsed < coreNum_ * CORE_USED_THRESHOLD || coreNumUsed > coreNum_) {
        return false;
    }

    // 因M轴非因子切分可能导致实际使用的mDim小于原始mDim，此处需要修正
    coreDim.mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, coreDim.mDim));
    if (runInfo_.enlarge > 1) {
        coreDim.nDim = Ops::Base::CeilDiv(runInfo_.dedx_cin1_g, Ops::Base::CeilDiv(runInfo_.dedx_cin1_g, coreDim.nDim));
    }
    SetTilingParamByDimInfo(tilingParams, coreDim);

    TilingBestBaseBlock baseBlock = TILING_BEST_BASE.at(shortSocVersion_).at(BASE_BLOCK_TYPE_BEST);
    return CalBaseBlockTiling(tilingParams, baseBlock);
}

void Conv3DBackpropInputV2TilingArch35::UpdateBaseBlock(
    uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValueDavid& tilingParams,
    const TilingBestBaseBlock& baseBlock)
{
    // 调换基本块tiling的M和N方向，确保singleCoreM和singleCoreN方向够用
    if (tilingParams.singleCoreM > tilingParams.singleCoreCin) {
        baseM = baseBlock.baseN;
        baseN = baseBlock.baseM;
    }

    // 超限处理
    if (baseM > tilingParams.singleCoreM) {
        baseM = Ops::Base::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreM), BLOCK_CUBE) * BLOCK_CUBE;
    }
    if (baseN > tilingParams.singleCoreCin) {
        baseN = Ops::Base::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreCin), BLOCK_CUBE) * BLOCK_CUBE;
    }

    uint64_t singleC0BaseK = static_cast<uint64_t>(runInfo_.kernel_w) * runInfo_.kernel_h * blockSize_;
    if (baseK % singleC0BaseK == 0U) {
        baseK = baseBlock.baseK / dtypeByte_;
    } else if (singleC0BaseK < baseK) {
        baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
    } else if (singleC0BaseK > baseK) {
        baseK = runInfo_.kernel_w * blockSize_;
    }

    // kernel侧应该保证下述条件的功能正确，当前在tiling侧进行约束
    uint64_t singleDepthMaxK =
        static_cast<uint64_t>(tilingParams.singleCoreCout1) * blockSize_ * runInfo_.kernel_h * runInfo_.kernel_w;
    if (baseK > singleDepthMaxK) {
        baseK = singleDepthMaxK;
    } else if (singleDepthMaxK % baseK != 0U) {
        baseK = runInfo_.kernel_w * blockSize_;
    }
}

int32_t Conv3DBackpropInputV2TilingArch35::CalFmapH(const int32_t& mL1Size) const
{
    int32_t hiCal;
    if (mL1Size % runInfo_.dedx_w == 0 || runInfo_.dedx_w % mL1Size == 0) {
        hiCal = Ops::Base::CeilDiv(mL1Size, runInfo_.dedx_w);
    } else if (mL1Size > runInfo_.dedx_w) {
        hiCal = mL1Size / runInfo_.dedx_w + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    int32_t khDilation = (runInfo_.kernel_h - 1) * runInfo_.dilation_h + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1;
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

void Conv3DBackpropInputV2TilingArch35::UpdateBaseStep(
    uint32_t& stepKa, uint32_t& stepKb, TilingValueDavid& tilingParams)
{
    uint32_t hoCal = CalFmapH(tilingParams.baseM); // 此处默认stepM=1
    uint32_t cout1A1 = std::max(
        static_cast<uint64_t>(1),
        L1_SIZE / BUFFER_NUM_L1 /
            static_cast<uint64_t>(dtypeByte_ * hoCal * runInfo_.dedy_w * runInfo_.stride_w * blockSize_));
    uint32_t cout1B1 = Ops::Base::CeilDiv(
        L1_SIZE,
        static_cast<uint64_t>(dtypeByte_ * tilingParams.baseN * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_) *
            BUFFER_NUM_DB);
    if (cout1A1 >= tilingParams.singleCoreCout1) {
        cout1A1 = tilingParams.singleCoreCout1;
        tilingParams.al1Pbuffer = 1;
    }
    if (cout1B1 >= tilingParams.singleCoreCout1) {
        if (runInfo_.kernel_d == 1 || (runInfo_.kernel_d > 1 && cout1B1 >= tilingParams.singleCoreCout1 * DB_ON)) {
            tilingParams.bl1Pbuffer = 1;
            cout1B1 = tilingParams.singleCoreCout1;
        } else {
            cout1B1 = tilingParams.singleCoreCout1 / DB_ON;
        }
    } else if (cout1B1 == 1U) {
        tilingParams.bl1Pbuffer = 1;
    } else {
        cout1B1 = cout1B1 / DB_ON;
    }
    cout1B1 = std::max(static_cast<uint32_t>(1), cout1B1);
    if (cout1A1 > cout1B1) {
        while (cout1A1 % cout1B1 > 0U) {
            cout1B1--;
        }
    } else {
        while (cout1B1 % cout1A1 > 0U) {
            cout1A1--;
        }
    }
    stepKa = std::max(
        static_cast<uint64_t>(1),
        Ops::Base::CeilDiv(
            static_cast<uint64_t>(cout1A1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
            static_cast<uint64_t>(tilingParams.baseK)));
    stepKb = std::max(
        static_cast<uint64_t>(1),
        Ops::Base::CeilDiv(
            static_cast<uint64_t>(cout1B1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
            static_cast<uint64_t>(tilingParams.baseK)));
    if (stepKa > stepKb) {
        stepKa = Ops::Base::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = Ops::Base::FloorAlign(stepKb, stepKa);
    }
}

bool Conv3DBackpropInputV2TilingArch35::CheckBaseBlockTiling(TilingValueDavid& tilingParams)
{
    if (!CheckL0Size(tilingParams.baseM, tilingParams.baseN, tilingParams.baseK, dtypeByte_)) {
        OP_LOGD(opName_, "Check L0 size fail.");
        return false;
    }
    uint32_t stepParaCheck = (tilingParams.stepKa > tilingParams.stepKb) ? (tilingParams.stepKa % tilingParams.stepKb) :
                                                                           (tilingParams.stepKb % tilingParams.stepKa);
    if (stepParaCheck != 0U) {
        OP_LOGD(opName_, "Check stepK fail.");
        return false;
    }
    uint64_t biasNonFullLoadSize = tilingParams.baseN * dtypeByteL0c_;
    uint64_t biasFullLoadSize = tilingParams.singleCoreCin * dtypeByteL0c_;
    while (tilingParams.stepKa > 0 && tilingParams.stepKb > 0) {
        // 为保证A和B矩阵使能double buffer，两者加和要小于L1 size的一半
        uint64_t l1UsedSize = static_cast<uint64_t>(tilingParams.al1Pbuffer) * dtypeByte_ *
                                  CalFmapH(tilingParams.baseM) * runInfo_.dedy_w * runInfo_.stride_w *
                                  tilingParams.stepKa * tilingParams.baseK / (runInfo_.kernel_h * runInfo_.kernel_w) +
                              static_cast<uint64_t>(tilingParams.bl1Pbuffer) * dtypeByte_ * tilingParams.stepKb *
                                  tilingParams.baseN * tilingParams.baseK;
        if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2) {
            isBiasFullLoad = biasFullLoadSize <= L1_BIAS_SIZE && (biasFullLoadSize + l1UsedSize <= L1_SIZE);
            if (isBiasFullLoad || (biasNonFullLoadSize + l1UsedSize) <= L1_SIZE) {
                break;
            }
        }
        if (l1UsedSize <= L1_SIZE) {
            break;
        }
        if (tilingParams.stepKa > tilingParams.stepKb) {
            tilingParams.stepKa -= tilingParams.stepKb;
        } else if (tilingParams.stepKb > tilingParams.stepKa) {
            tilingParams.stepKb -= tilingParams.stepKa;
        } else {
            // tilingParams.stepKa必定等于tilingParams.stepKb
            if ((tilingParams.baseK / blockSize_) % (runInfo_.kernel_h * runInfo_.kernel_w) == 0) {
                // 第一个分支是baseK包含hkwk
                tilingParams.stepKa--;
                tilingParams.stepKb--;
            } else if (
                ((tilingParams.baseK / blockSize_) % runInfo_.kernel_w == 0) &&
                (static_cast<int32_t>(tilingParams.stepKa) > runInfo_.kernel_h)) {
                // 第二个分支是baseK包含wk，stepka大于kernel_h和stepkb大于kernel_h
                // 由于在UpdateBaseStep中限制，stepKa/stepKb必定是kernel_h的倍数
                tilingParams.stepKa -= runInfo_.kernel_h;
                tilingParams.stepKb -= runInfo_.kernel_h;
            } else {
                // 由于在UpdateBaseBlock中basek只能是包含hkwk或者包含wk两种情况
                // 最后一个是baseK包含wk，stepka小于等于kernel_h或stepkb小于等于kernel_h
                OP_LOGD(opName_, "baseK is invalid, use tbe tiling.");
                return false;
            }
        }
    }
    if (tilingParams.stepKa * tilingParams.stepKb == 0) {
        OP_LOGD(opName_, "stepK is 0.");
        return false;
    }
    OP_LOGD(opName_, "Use basic-block tiling.");
    return true;
}

bool Conv3DBackpropInputV2TilingArch35::CalBaseBlockTiling(
    TilingValueDavid& tilingParams, const TilingBestBaseBlock& baseBlock)
{
    // 默认开启double buffer
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = 1;
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_ON;

    // 默认采用最优基本块tiling
    uint32_t baseM = baseBlock.baseM;
    uint32_t baseK = baseBlock.baseK / dtypeByte_;
    uint32_t baseN = baseBlock.baseN;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    // 更新并设置基本块tiling
    UpdateBaseBlock(baseM, baseK, baseN, tilingParams, baseBlock);
    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
    tilingParams.baseD = 1U;
    tilingParams.baseBatch = 1;
    tilingParams.baseGroup = 1;
    UpdateBaseStep(stepKa, stepKb, tilingParams);

    tilingParams.stepKa = stepKa;
    tilingParams.stepKb = stepKb;
    tilingParams.stepM = 1;
    tilingParams.stepN = 1;
    tilingParams.stepBatch = 1;
    tilingParams.stepGroup = 1;
    tilingParams.iterateOrder = static_cast<uint32_t>(IterateOrder::ORDER_N);

    return CheckBaseBlockTiling(tilingParams);
}

void Conv3DBackpropInputV2TilingArch35::CalTbeBlockTiling(TilingValueDavid& tilingParams)
{
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = static_cast<uint32_t>(tbeTiling_.db_l0c);
    uint32_t baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    uint32_t baseK = tbeTiling_.k_l0 * blockSize_;
    uint32_t baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
    uint32_t tmpBaseKMax = std::max(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);
    uint32_t tmpBaseKMin = std::min(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);
    if (CheckL0Size(baseM, baseN, runInfo_.kernel_h * runInfo_.kernel_w * blockSize_, dtypeByte_)) {
        baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
    } else if (dtypeByte_ == FP32_DATA_SIZE) {
        baseK = blockSize_;
    } else if (CheckL0Size(baseM, baseN, tmpBaseKMax, dtypeByte_)) {
        baseK = tmpBaseKMax;
    } else if (CheckL0Size(baseM, baseN, tmpBaseKMin, dtypeByte_)) {
        baseK = tmpBaseKMin;
    } else {
        baseK = blockSize_;
    }
    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
    tilingParams.baseD = 1U;
    tilingParams.baseBatch = 1U;
    tilingParams.baseGroup = 1U;
    tilingParams.al1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_al1);
    tilingParams.bl1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_bl1);
    tbeTiling_.k_al1 = (tbeTiling_.k_al1 > static_cast<int32_t>(tilingParams.singleCoreCout1)) ?
                           static_cast<int32_t>(tilingParams.singleCoreCout1) :
                           tbeTiling_.k_al1;
    tbeTiling_.k_bl1 = (tbeTiling_.k_bl1 > static_cast<int32_t>(tilingParams.singleCoreCout1)) ?
                           static_cast<int32_t>(tilingParams.singleCoreCout1) :
                           tbeTiling_.k_bl1;
    tilingParams.stepKa = Ops::Base::CeilDiv(
        static_cast<uint64_t>(tbeTiling_.k_al1) * runInfo_.kernel_h * runInfo_.kernel_w,
        static_cast<uint64_t>(baseK / blockSize_));
    tilingParams.stepKb = Ops::Base::CeilDiv(
        static_cast<uint64_t>(tbeTiling_.k_bl1) * runInfo_.kernel_h * runInfo_.kernel_w,
        static_cast<uint64_t>(baseK / blockSize_));
    tilingParams.stepM = 1;
    tilingParams.stepN = 1;
    tilingParams.stepBatch = 1;
    tilingParams.stepGroup = 1;
    tilingParams.iterateOrder = static_cast<uint32_t>(IterateOrder::ORDER_N);
    if (hasBiasFlag_) {
        uint64_t biasFullLoadSize = runInfo_.dedx_cin * dtypeByteL0c_;
        uint64_t l1UsedSize = static_cast<uint64_t>(tilingParams.al1Pbuffer) * dtypeByte_ *
                                  CalFmapH(tilingParams.baseM) * runInfo_.dedy_w * runInfo_.stride_w *
                                  tilingParams.stepKa * tilingParams.baseK / (runInfo_.kernel_h * runInfo_.kernel_w) +
                              static_cast<uint64_t>(tilingParams.bl1Pbuffer) * dtypeByte_ * tilingParams.stepKb *
                                  tilingParams.baseN * tilingParams.baseK;
        isBiasFullLoad = biasFullLoadSize <= L1_BIAS_SIZE && (biasFullLoadSize + l1UsedSize <= L1_SIZE);
    }
}

void Conv3DBackpropInputV2TilingArch35::InitTilingValue(TilingValueDavid& tilingParams, const uint32_t coreNum)
{
    if (!CalCoreDimTiling(tilingParams, coreNum)) {
        OP_LOGD(opName_, "Use tbe cache tiling.");

        TilingCoreDimDx coreDim;
        coreDim.batchDim = tbeTiling_.batch_dim;
        coreDim.dDim = tbeTiling_.d_dim;
        coreDim.mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, tbeTiling_.m_dim));
        coreDim.nDim = tbeTiling_.n_dim;
        SetTilingParamByDimInfo(tilingParams, coreDim);
        CalTbeBlockTiling(tilingParams);
    }

    if ((context_->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_BF16 ||
         context_->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_FLOAT16) &&
        runInfo_.yFormat == ge::FORMAT_NCDHW) {
        uint64_t singleShapeK = tilingParams.singleCoreCout * runInfo_.kernel_h * runInfo_.kernel_w;
        // strideD > 1时单个Dk有效计算会减少，切Dk后fix_pipe写出次数变多会导致性能劣化
        // fixpipe bound的case切Dk性能会劣化
        enableSplitDk_ = (runInfo_.kernel_d > 1) && (runInfo_.enlarge == 1) &&
                         (tilingParams.baseK * tilingParams.stepKb >= singleShapeK) &&
                         (tilingParams.baseN * tilingParams.stepN >= tilingParams.singleCoreCin) &&
                         runInfo_.stride_d == 1 && runInfo_.dilation_d == 1 &&
                         (singleShapeK >= FP32_FIXPIPE_BOUND_K_LIMIT);
        uint64_t singleCoreUsrSpaceSize =
            Ops::Base::CeilAlign(tilingParams.singleCoreM, static_cast<uint64_t>(tilingParams.baseM)) *
            Ops::Base::CeilAlign(tilingParams.singleCoreCin, static_cast<uint64_t>(tilingParams.baseN)) *
            tilingParams.singleCoreDin * sizeof(float);
        usrSpaceSizeForSplitDk_ = tilingParams.coreNum * singleCoreUsrSpaceSize;
        enableSplitDk_ = enableSplitDk_ && (usrSpaceSizeForSplitDk_ <= USER_WORKSIZE_LIMIT);
        if (enableSplitDk_) {
            singleIterateDk_ = 1;
            tilingParams.iterateOrder = static_cast<uint32_t>(IterateOrder::ORDER_K); // 切Dk时走ORDER_K
        } else {
            singleIterateDk_ = runInfo_.kernel_d;
        }
    } else {
        enableSplitDk_ = false;
        singleIterateDk_ = runInfo_.kernel_d;
    }
}

void Conv3DBackpropInputV2TilingArch35::SetLoadB2Condition(const TilingValueDavid& tilingParams)
{
    if (runInfo_.filterFormat == ge::FORMAT_DHWCN) {
        loadB2Condition_ = 2; // 2表示load2b1时同时做转置的情况
        return;
    }

    if (runInfo_.kernel_d * runInfo_.kernel_h * runInfo_.kernel_w == 1) {
        loadB2Condition_ = 3; // 3表示DK*Hk*Wk = 1的情况
    } else {
        loadB2Condition_ = 1;
    }

    if (runInfo_.filterFormat == ge::FORMAT_NDHWC &&
        static_cast<uint64_t>(tilingParams.baseN) >=
            static_cast<uint64_t>(runInfo_.kernel_h) * static_cast<uint64_t>(runInfo_.kernel_w) &&
        loadB2Condition_ != 3) { // 3表示DK*Hk*Wk = 1的情况
        loadB2Condition_ = 2;    // 2表示load2b1时同时做转置的情况
    }

    if (groupConvMode_ == TILING_GROUP_MODE_ENLARGE || (dtypeByte_ == FP32_DATA_SIZE) ||
        (dtypeByte_ == BIT8_DATA_SIZE) ||
        (loadB2Condition_ == 1 && // kernel拆分后cout基本都大于cin
         tilingParams.baseN * tilingParams.stepN <=
             ((tilingParams.baseK * tilingParams.stepKb) / (runInfo_.kernel_h * runInfo_.kernel_w)))) {
        loadB2Condition_ = 2; // 2表示load2b1时同时做转置的情况
    }
}

void Conv3DBackpropInputV2TilingArch35::SetGroupConvMode(TConv3DInputV2TilingAdvance& dxt)
{
    if (dxt.get_enlarge() == 1) {
        groupConvMode_ = TILING_GROUP_MODE_ORIGIN;
    } else {
        groupConvMode_ = TILING_GROUP_MODE_ENLARGE;
    }
}

void Conv3DBackpropInputV2TilingArch35::SetTilingValue(
    TConv3DInputV2TilingAdvance& dxt, const TilingValueDavid& tilingParams)
{
    tilingData_.params.set_batchDim(tilingParams.batchDim);
    tilingData_.params.set_groupDim(tilingParams.groupDim);
    tilingData_.params.set_mDim(tilingParams.mDim);
    tilingData_.params.set_kDim(tilingParams.kDim);
    tilingData_.params.set_nDim(tilingParams.nDim);
    tilingData_.params.set_dDim(tilingParams.dDim);
    tilingData_.params.set_coreNum(tilingParams.coreNum);
    // singleCore
    dxt.set_singleCoreBatch(tilingParams.singleCoreBatch);
    dxt.set_singleCoreGroup(tilingParams.singleCoreGroup);
    dxt.set_singleCoreM(tilingParams.singleCoreM);
    dxt.set_singleCoreCout(tilingParams.singleCoreCout);
    dxt.set_singleCoreCin(tilingParams.singleCoreCin);
    dxt.set_singleCoreDin(tilingParams.singleCoreDin);

    dxt.set_baseM(tilingParams.baseM);
    dxt.set_baseK(tilingParams.baseK);
    dxt.set_baseN(tilingParams.baseN);

    dxt.set_stepM(tilingParams.stepM);
    dxt.set_stepN(tilingParams.stepN);

    dxt.set_stepKa(tilingParams.stepKa);
    dxt.set_stepKb(tilingParams.stepKb);

    dxt.set_al0Pbuffer(tilingParams.al0Pbuffer); // 默认开
    dxt.set_bl0Pbuffer(tilingParams.bl0Pbuffer); // 默认开
    dxt.set_cl0Pbuffer(tilingParams.cl0Pbuffer);
    dxt.set_al1Pbuffer(tilingParams.al1Pbuffer);
    dxt.set_bl1Pbuffer(tilingParams.bl1Pbuffer);

    dxt.set_iterateOrder(tilingParams.iterateOrder);
    tilingData_.conv3DDxKSTiling.set_kSCoutFullLoad(0);
    tilingData_.conv3DDxKSTiling.set_kSUseWorkSpace(0);

    a1DbFlag_ = tilingParams.al1Pbuffer == DB_ON;
    b1DbFlag_ = tilingParams.bl1Pbuffer == DB_ON;
    c0DbFlag_ = tilingParams.cl0Pbuffer == DB_ON;
    SetGroupConvMode(dxt);
    SetLoadB2Condition(tilingParams);
}

void Conv3DBackpropInputV2TilingArch35::SetBackpropPadInfo(TConv3DInputV2TilingAdvance& dxt)
{
    int64_t bpPadTail = runInfo_.dedx_d - (static_cast<int64_t>(runInfo_.dedy_d - 1) * runInfo_.stride_d + 1) +
                        (runInfo_.kernel_d - 1) * runInfo_.dilation_d - runInfo_.backprop_pad_h;
    if (bpPadTail < PAD_DIM_LOW || bpPadTail > PAD_DIM_UP) {
        dxt.set_backpropPadTail(runInfo_.backprop_pad_t);
    } else {
        dxt.set_backpropPadTail(static_cast<uint32_t>(bpPadTail));
    }
    OP_LOGD(opName_, "backprop tail pad: %ld, origin backprop_pad_t: %d", bpPadTail, runInfo_.backprop_pad_t);

    dxt.set_backpropPadUp(runInfo_.backprop_pad_u);
    int64_t bpPadDown = runInfo_.dedx_h - (static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1) +
                        (runInfo_.kernel_h - 1) * runInfo_.dilation_h - runInfo_.backprop_pad_u;
    if (bpPadDown < PAD_DIM_LOW || bpPadDown > PAD_DIM_UP) {
        dxt.set_backpropPadDown(runInfo_.backprop_pad_d);
    } else {
        dxt.set_backpropPadDown(static_cast<uint32_t>(bpPadDown));
    }
    OP_LOGD(opName_, "backprop down pad: %ld, origin backprop_pad_d: %d", bpPadDown, runInfo_.backprop_pad_d);

    dxt.set_backpropPadLeft(runInfo_.backprop_pad_l);
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    if (bpPadRight < PAD_DIM_LOW || bpPadRight > PAD_DIM_UP) {
        dxt.set_backpropPadRight(runInfo_.backprop_pad_r);
    } else {
        dxt.set_backpropPadRight(static_cast<uint32_t>(bpPadRight));
    }
    OP_LOGD(opName_, "backprop right pad: %ld, origin backprop_pad_r: %d", bpPadRight, runInfo_.backprop_pad_r);
}

void Conv3DBackpropInputV2TilingArch35::SetRunBaseShapeInfoTiling(TConv3DInputV2TilingAdvance& dxt)
{
    dxt.set_batch(runInfo_.batch_n);
    dxt.set_cin(runInfo_.dedx_cin);
    dxt.set_cout(runInfo_.dedy_cout);
    dxt.set_cinG(runInfo_.dedx_cin_g);
    dxt.set_coutG(runInfo_.dedy_cout_g);
    dxt.set_cin1(runInfo_.dedx_cin1);
    dxt.set_cout1(runInfo_.dedy_cout1);
    dxt.set_cin1G(runInfo_.dedx_cin1_g);
    dxt.set_cout1G(runInfo_.dedy_cout1_g);
    dxt.set_c0(blockSize_);

    if (dtypeByte_ == BIT8_DATA_SIZE) {
        dxt.set_c0Bits(F8_C0_BITS);
    } else if (dtypeByte_ == FP32_DATA_SIZE) {
        dxt.set_c0Bits(F32_C0_BITS);
    } else {
        dxt.set_c0Bits(F16_C0_BITS);
    }

    dxt.set_ho(runInfo_.dedy_h);
    dxt.set_wo(runInfo_.dedy_w);
    dxt.set_dout(runInfo_.dedy_d);
    dxt.set_di(runInfo_.dedx_d);
    dxt.set_hi(runInfo_.dedx_h);
    dxt.set_wi(runInfo_.dedx_w);
    dxt.set_hk(runInfo_.kernel_h);
    dxt.set_wk(runInfo_.kernel_w);
    dxt.set_dk(runInfo_.kernel_d);
}

void Conv3DBackpropInputV2TilingArch35::SetRunInfoTiling(TConv3DInputV2TilingAdvance& dxt)
{
    // shape
    SetRunBaseShapeInfoTiling(dxt);
    dxt.set_enlarge(runInfo_.enlarge);
    dxt.set_group(runInfo_.real_g);
    dxt.set_oriGroup(runInfo_.groups);
    dxt.set_strideH(runInfo_.stride_h);
    dxt.set_strideW(runInfo_.stride_w);
    dxt.set_strideD(runInfo_.stride_d);
    dxt.set_padFront(runInfo_.pad_h);
    dxt.set_padBack(runInfo_.pad_t);
    dxt.set_padUp(runInfo_.pad_u);
    dxt.set_padDown(runInfo_.pad_d);
    dxt.set_padLeft(runInfo_.pad_l);
    dxt.set_padRight(runInfo_.pad_r);
    SetBackpropPadInfo(dxt);

    dxt.set_dilationH(runInfo_.dilation_h);
    dxt.set_dilationW(runInfo_.dilation_w);
    dxt.set_dilationD(runInfo_.dilation_d);
    dxt.set_hf32Flag(runInfo_.hf32_flag);
    dxt.set_initOutputFlag(runInfo_.initOutputFlag);
    dxt.set_isBiasFullLoad(isBiasFullLoad);
    dxt.set_singleIterateDk(singleIterateDk_);
}

void Conv3DBackpropInputV2TilingArch35::SetDxTilingFromTbeTiling()
{
    TConv3DInputV2TilingAdvance& dxt = tilingData_.conv3DDxTiling;
    TilingValueDavid tilingParams;
    // key:
    // "N_Do_Co1_Ho_Wo_Di_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
    // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
    std::string key = std::to_string(runInfo_.batch_n) + "_" + std::to_string(runInfo_.dedy_d) + "_" +
                      std::to_string(runInfo_.dedy_cout1) + "_" + std::to_string(runInfo_.dedy_h) + "_" +
                      std::to_string(runInfo_.dedy_w) + "_" + std::to_string(runInfo_.dedx_d) + "_" +
                      std::to_string(runInfo_.dedx_cin1) + "_" + std::to_string(runInfo_.dedx_h) + "_" +
                      std::to_string(runInfo_.dedx_w) + "_" + std::to_string(runInfo_.kernel_d) + "_" +
                      std::to_string(runInfo_.kernel_h) + "_" + std::to_string(runInfo_.kernel_w) + "_" +
                      std::to_string(runInfo_.stride_d) + "_" + std::to_string(runInfo_.stride_h) + "_" +
                      std::to_string(runInfo_.stride_w) + "_" + std::to_string(runInfo_.pad_h) + "_" +
                      std::to_string(runInfo_.pad_t) + "_" + std::to_string(runInfo_.pad_u) + "_" +
                      std::to_string(runInfo_.pad_d) + "_" + std::to_string(runInfo_.pad_l) + "_" +
                      std::to_string(runInfo_.pad_r) + "_" + std::to_string(runInfo_.dilation_d) + "_" +
                      std::to_string(runInfo_.dilation_h) + "_" + std::to_string(runInfo_.dilation_w);
    coreNum_ = context_->GetCompileInfo<Conv3DBackpropV2CompileInfo>()->core_num;
    InitTilingValue(tilingParams, coreNum_);
    SetRunInfoTiling(dxt);
    SetTilingValue(dxt, tilingParams);
}

void Conv3DBackpropInputV2TilingArch35::PrintTilingData()
{
    TConv3DInputV2TilingAdvance& tiling = tilingData_.conv3DDxTiling;
    Conv3DBackpropInputV2ParamsAdvance& params = tilingData_.params;
    TConv3DInputV2KSTilingAdvance& ksTiling = tilingData_.conv3DDxKSTiling;
    std::stringstream ss;
    ss << "batchDim: " << params.get_batchDim() << " groupDim: " << params.get_groupDim()
       << " mDim: " << params.get_mDim() << " kDim: " << params.get_kDim() << " nDim: " << params.get_nDim()
       << " dDim: " << params.get_dDim() << " coreNum: " << params.get_coreNum()
       << " al0Pbuffer: " << static_cast<uint32_t>(tiling.get_al0Pbuffer())
       << " bl0Pbuffer: " << static_cast<uint32_t>(tiling.get_bl0Pbuffer())
       << " cl0Pbuffer: " << static_cast<uint32_t>(tiling.get_cl0Pbuffer())
       << " al1Pbuffer: " << static_cast<uint32_t>(tiling.get_al1Pbuffer())
       << " bl1Pbuffer: " << static_cast<uint32_t>(tiling.get_bl1Pbuffer())
       << " iterateOrder: " << static_cast<uint32_t>(tiling.get_iterateOrder())
       << " c0: " << static_cast<uint32_t>(tiling.get_c0()) << " c0Bits: " << static_cast<uint32_t>(tiling.get_c0Bits())
       << " enlarge: " << static_cast<uint32_t>(tiling.get_enlarge())
       << " hf32Flag: " << static_cast<uint32_t>(tiling.get_hf32Flag())
       << " initOutputFlag: " << static_cast<uint32_t>(tiling.get_initOutputFlag())
       << " isBiasFullLoad: " << static_cast<uint32_t>(tiling.get_isBiasFullLoad()) << " batch: " << tiling.get_batch()
       << " cin: " << tiling.get_cin() << " cout: " << tiling.get_cout() << " cinG: " << tiling.get_cinG()
       << " coutG: " << tiling.get_coutG() << " cout1: " << tiling.get_cout1() << " cin1: " << tiling.get_cin1()
       << " cout1G: " << tiling.get_cout1G() << " cin1G: " << tiling.get_cin1G() << " dout: " << tiling.get_dout()
       << " ho: " << tiling.get_ho() << " wo: " << tiling.get_wo() << " di: " << tiling.get_di()
       << " hi: " << tiling.get_hi() << " wi: " << tiling.get_wi() << " dk: " << tiling.get_dk()
       << " hk: " << tiling.get_hk() << " wk: " << tiling.get_wk() << " group: " << tiling.get_group()
       << " oriGroup: " << tiling.get_oriGroup() << " strideD: " << tiling.get_strideD()
       << " strideH: " << tiling.get_strideH() << " strideW: " << tiling.get_strideW()
       << " padFront: " << tiling.get_padFront() << " padBack: " << tiling.get_padBack()
       << " padUp: " << tiling.get_padUp() << " padDown: " << tiling.get_padDown()
       << " padLeft: " << tiling.get_padLeft() << " padRight: " << tiling.get_padRight()
       << " backpropPadTail: " << tiling.get_backpropPadTail() << " backpropPadUp: " << tiling.get_backpropPadUp()
       << " backpropPadDown: " << tiling.get_backpropPadDown() << " backpropPadLeft: " << tiling.get_backpropPadLeft()
       << " backpropPadRight: " << tiling.get_backpropPadRight() << " dilationD: " << tiling.get_dilationD()
       << " dilationH: " << tiling.get_dilationH() << " dilationW: " << tiling.get_dilationW()
       << " singleCoreGroup: " << tiling.get_singleCoreGroup() << " singleCoreCout: " << tiling.get_singleCoreCout()
       << " singleCoreCin: " << tiling.get_singleCoreCin() << " singleCoreDin: " << tiling.get_singleCoreDin()
       << " baseM: " << tiling.get_baseM() << " baseK: " << tiling.get_baseK() << " baseN: " << tiling.get_baseN()
       << " stepM: " << tiling.get_stepM() << " stepN: " << tiling.get_stepN() << " stepKa: " << tiling.get_stepKa()
       << " stepKb: " << tiling.get_stepKb() << " singleIterateDk: " << tiling.get_singleIterateDk()
       << " singleCoreBatch: " << tiling.get_singleCoreBatch() << " singleCoreM: " << tiling.get_singleCoreM()
       << " kSCoutFullLoad: " << ksTiling.get_kSCoutFullLoad() << " kSUseWorkSpace: " << ksTiling.get_kSUseWorkSpace();
    OP_LOGD(opName_, "api tiling: %s", ss.str().c_str());
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DBackpropInputV2TilingArch35, 102);

} // namespace Conv
} // namespace NN
} // namespace Ops
