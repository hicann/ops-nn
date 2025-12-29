/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "batch_matmul_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/op_api_def.h"
#include "matmul/mat_mul_v3/op_host/op_api/matmul.h"
#include "matmul/batch_mat_mul_v3/op_host/op_api/batch_matmul.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/batch_matmul_v2tov3.h"
#include "aclnn_kernels/transdata.h"
#include "cube_util.h"
#include "matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "matmul/common/op_host/math_util.h"

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;

using namespace Ops::NN;
using namespace op;

namespace {
static const uint64_t NUM_TWO = 2UL;
static const int64_t BLOCK_BYTE_SIZE = 32L;
static const uint64_t BLOCK_SIZE_256 = 256UL;
static const uint64_t BLOCK_CUBE = 16UL;
static const uint64_t KB_SIZE = 1024UL;
static const uint64_t UB_SIZE = 248UL * 1024UL;
static const uint64_t MIN_BATCH_NUM = 128UL;
static const int32_t PENULTIMATE_DIM = 2;
static const int32_t LAST_DIM = 1;

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape bmmEmptyShape = {(self->GetViewShape())[0], (self->GetViewShape())[1], (mat2->GetViewShape())[2]};
    auto output = executor->AllocTensor(bmmEmptyShape, self->GetDataType());
    if (output->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return output;
    }
    FVector<int64_t> fillShape = GetShape(output);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, self->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    OP_LOGD("set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor*>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
        input->GetStorageFormat() != op::Format::FORMAT_ND) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static aclnnStatus SetBatchMatMulOpSupportInfoV2(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, MmOpInfo& matmulOpInfo, int8_t cubeMathType)
{
    // 判断传入L0接口，用于计算的Dtype
    std::shared_ptr<SocMatMulRuleBase> socRule = SocMatMulRule::getInstance();
    CHECK_RET(socRule != nullptr, ACLNN_ERR_PARAM_INVALID);

    aclnnStatus status = socRule -> PromoteDtype(self, mat2, bias, out, cubeMathType, matmulOpInfo);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    // 1971场景 ACLNN中BMM全部走ND格式，1980场景进入函数路由
    if (CheckSocVersionIsSupportBf16()) {
        matmulOpInfo.support_info.output_format = Format::FORMAT_ND;
        matmulOpInfo.support_info.self_format = Format::FORMAT_ND;
        if (matmulOpInfo.ori_info.mat2_format == Format::FORMAT_FRACTAL_NZ) {
            matmulOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
        } else {
            matmulOpInfo.support_info.mat2_format = Format::FORMAT_ND;
        }
    } else {
        SetMmSupportFormat(self, mat2, matmulOpInfo);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CreateBatchMatmulOpInfo(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, MmOpInfo& matmulOpInfo, int8_t cubeMathType)
{
    matmulOpInfo.ori_info.self_dtype = self->GetDataType();
    matmulOpInfo.ori_info.self_format = GetPrimaryFormat(self->GetStorageFormat());
    matmulOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    matmulOpInfo.ori_info.mat2_format = GetPrimaryFormat(mat2->GetStorageFormat());
    matmulOpInfo.ori_info.output_dtype = out->GetDataType();
    matmulOpInfo.ori_info.output_format = GetPrimaryFormat(out->GetStorageFormat());

    matmulOpInfo.support_info = matmulOpInfo.ori_info;

    SetBatchMatMulOpSupportInfoV2(self, mat2, bias, out, matmulOpInfo, cubeMathType);
    bool inputFp32Flag = matmulOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
                         matmulOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    matmulOpInfo.opImplModeEnum =
        inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32)) ? 0x40 : 0x1;
    matmulOpInfo.enableHf32 =
        inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32));

    bool inputFp16Flag = matmulOpInfo.support_info.self_dtype == DataType::DT_FLOAT16 &&
                         matmulOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT16;
    bool inputBf16Flag = matmulOpInfo.support_info.self_dtype == DataType::DT_BF16 &&
                         matmulOpInfo.support_info.mat2_dtype == DataType::DT_BF16;
    // 在A2/A3平台下，如果输入数据类型为fp16或bf16，且进行高精度计算，则使能输出数据类型为fp32
    matmulOpInfo.enableFp16Bf16InFp32Out = (inputFp16Flag || inputBf16Flag) &&
                                           (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                                            GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) &&
                                           (cubeMathType == KEEP_DTYPE);

    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, cubeMathType=%d, enableFp16Bf16InFp32Out=%d", matmulOpInfo.opImplModeEnum, matmulOpInfo.enableHf32,
        cubeMathType, matmulOpInfo.enableFp16Bf16InFp32Out);

    GetMmInfo(matmulOpInfo);
    return ACLNN_SUCCESS;
}

static inline FVector<int64_t> GetBatchDim(const aclTensor* inputTensor)
{
    size_t tensorDim = inputTensor->GetViewShape().GetDimNum();
    // When dimension > 2 , the case pattern is [B, M, K]
    int64_t batchA3 = tensorDim > 2 ? inputTensor->GetViewShape().GetDim(tensorDim - 3) : 1;
    // When dimension > 3 , the case pattern is [B1, B2, M, K]
    int64_t batchA2 = tensorDim > 3 ? inputTensor->GetViewShape().GetDim(tensorDim - 4) : 1;
    // When dimension > 4 , the case pattern is [B1, B2, B3, M, K]
    int64_t batchA1 = tensorDim > 4 ? inputTensor->GetViewShape().GetDim(tensorDim - 5) : 1;
    // When dimension > 5 , the case pattern is [B1, B2, B3, B4, M, K]
    int64_t batchA0 = tensorDim > 5 ? inputTensor->GetViewShape().GetDim(tensorDim - 6) : 1;
    FVector<int64_t> batchDim = {batchA0, batchA1, batchA2, batchA3};

    return batchDim;
}

inline static bool IsBatchEqual(const FVector<int64_t>& batchDimForX1, const FVector<int64_t>& batchDimForX2)
{
    const size_t dimNumA = batchDimForX1.size();
    const size_t dimNumB = batchDimForX2.size();
    if (dimNumA != dimNumB) {
        return false;
    }
    for (size_t i = 0; i < dimNumA; i++) {
        if (batchDimForX1[i] != batchDimForX2[i]) {
            return false;
        }
    }
    return true;
}

inline static uint64_t GetBatchDimAll(const aclTensor* x)
{
    const FVector<int64_t> batchDims = GetBatchDim(x);
    int64_t result = 1L;
    for (int64_t d : batchDims) {
        result *= d;
    }
    return static_cast<uint64_t>(result);
};

static bool CheckAscendCScenario(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const MmOpInfo& mmOpInfo, const bool adjX1,
    const bool adjX2)
{
    if (mmOpInfo.support_info.self_format == ge::FORMAT_ND &&
        mmOpInfo.support_info.mat2_format != ge::FORMAT_ND) {
        OP_LOGI("Hit batch_mat_mul_v3 weightnz");
        return true;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return true;
    }
    if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) ||
        mmOpInfo.support_info.self_format != ge::FORMAT_ND || mmOpInfo.support_info.mat2_format != ge::FORMAT_ND) {
        OP_LOGI("Not batch_mat_mul_v3 case for unsupported SOC version or unsupported Format.");
        return false;
    }
    if ((x1->GetDataType() != DataType::DT_FLOAT16 && x1->GetDataType() != DataType::DT_BF16 &&
         x1->GetDataType() != DataType::DT_FLOAT) ||
        (x2->GetDataType() != DataType::DT_FLOAT16 && x2->GetDataType() != DataType::DT_BF16 &&
         x2->GetDataType() != DataType::DT_FLOAT)) {
        OP_LOGI("Not batch_mat_mul_v3 case due to unsupported dtype.");
        return false;
    }
    if (bias != nullptr) {
        OP_LOGI("batch_mat_mul_v3 case does not support bias yet");
        return false;
    }

    return l0op::BmmCheckHitV3Shape(x1, x2, bias, adjX1, adjX2, mmOpInfo.support_info.self_format,
                                    mmOpInfo.support_info.mat2_format, mmOpInfo.enableFp16Bf16InFp32Out);
}

const aclIntArray* GetOutputSize(
    const aclTensor* x1, const aclTensor* x2, const bool adjX1, const bool adjX2, aclOpExecutor* executor)
{
    constexpr size_t maxDim = 6;
    constexpr size_t minDim = 2;
    constexpr size_t maxBatchDim = 4;
    size_t x1DimSize = x1->GetViewShape().GetDimNum();
    size_t x2DimSize = x2->GetViewShape().GetDimNum();
    if ((x1DimSize < minDim || x1DimSize > maxDim) || (x2DimSize < minDim || x2DimSize > maxDim)) {
        OP_LOGE(
            ACLNN_ERR_INNER_NULLPTR,
            "Calculate BatchMatMul out unsuccessfully, one of input dim belows 2 or exceeds 6, which is %zu and %zu.",
            x1DimSize, x2DimSize);
        return nullptr;
    }
    size_t outDimSize = std::max(x1DimSize, x2DimSize);
    int64_t outM = adjX1 ? x1->GetViewShape().GetDim(x1DimSize - 1) : x1->GetViewShape().GetDim(x1DimSize - 2);
    int64_t outN = adjX2 ? x2->GetViewShape().GetDim(x2DimSize - 2) : x2->GetViewShape().GetDim(x2DimSize - 1);
    FVector<int64_t> batchDimForX1 = GetBatchDim(x1);
    FVector<int64_t> batchDimForX2 = GetBatchDim(x2);

    std::vector<int64_t> outShape;
    for (size_t i = maxDim - outDimSize; i < maxBatchDim; i++) { // outDimSize is 2~6, i is 0~3
        outShape.emplace_back(std::max(batchDimForX1[i], batchDimForX2[i]));
    }
    outShape.emplace_back(outM);
    outShape.emplace_back(outN);
    return executor->AllocIntArray(outShape.data(), outShape.size());
}

const aclTensor* TransBmm2Mm(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, bool enableHf32, bool adjX1, bool adjX2,
    const bool offsetX, aclOpExecutor* executor)
{
    OP_LOGI("Hit bmm2mm scenario.");
    auto x1Bmm2Mm = l0op::Reshape(x1, {-1, x1->GetViewShape().GetDim(x1->GetViewShape().GetDimNum() - 1)}, executor);
    auto x2Bmm2Mm = l0op::Reshape(x2, {-1, x2->GetViewShape().GetDim(x2->GetViewShape().GetDimNum() - 1)}, executor);
    const aclTensor* mmOut = l0op::MatMulV3Nd(x1Bmm2Mm, x2Bmm2Mm, bias, adjX1, adjX2, offsetX, enableHf32, executor);
    CHECK_RET(mmOut != nullptr, nullptr);
    auto outShapeIntArray = GetOutputSize(x1, x2, adjX1, adjX2, executor);
    CHECK_RET(outShapeIntArray != nullptr, nullptr);
    return l0op::Reshape(mmOut, outShapeIntArray, executor);
}

bool CheckSocIfBatchMatMulToMul910B(const aclTensor* self, const aclTensor* mat2, bool adjX1, bool adjX2)
{
    (void) adjX1;
    if (self->GetDataType() == DataType::DT_BF16 || mat2->GetDataType() == DataType::DT_BF16) {
        return checkBF16SizeValid(mat2, adjX2);
    }
    return true;
}

bool CheckShapeEqualToMul(const uint64_t& mDim, const uint64_t& nDim, const uint64_t& batchNum,
                          const uint64_t& dataSize, const uint64_t& alignNum) {
  // 判断batch数是否符合条件(大于等于128)
  if (batchNum < MIN_BATCH_NUM) {
    return false;
  }
  if (nDim > BLOCK_BYTE_SIZE / dataSize && nDim <= BLOCK_SIZE_256 / dataSize) {
    return false;
  }
  if (nDim == 1) {
    return false;
  }
  uint64_t alignM = CeilAlign(mDim, alignNum);
  uint64_t alignN = CeilAlign(nDim, alignNum);
  if ((alignM + alignN + alignM * alignN) * dataSize > UB_SIZE) {
    return false;
  }
  // 判断n是否256B对齐
  return nDim % (BLOCK_SIZE_256 / dataSize) != 0;
}

bool CheckSocIfBatchMatMulToMul91095(const aclTensor* self, const aclTensor* mat2, bool adjX1, bool adjX2)
{
    // now only basic api iterbatch temp not need convert to mul
    uint64_t aicoreNum = static_cast<uint64_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    uint64_t dtypeSize = static_cast<uint64_t>(op::TypeSize(self->GetDataType()));
    uint64_t c0 = static_cast<uint64_t>(BLOCK_BYTE_SIZE) / dtypeSize;
    constexpr uint64_t floatSize = 4UL;
    constexpr uint64_t pingPong = 2UL;
    constexpr uint64_t l0aSize = 64 * KB_SIZE;
    constexpr uint64_t l0bSize = 64 * KB_SIZE;
    constexpr uint64_t l0cSize = 256 * KB_SIZE;
    constexpr uint64_t l1Size = 512 * KB_SIZE;

    uint64_t mDim = adjX1 ? self->GetViewShape()[self->GetViewShape().GetDimNum() - 1] :
                           self->GetViewShape()[self->GetViewShape().GetDimNum() - NUM_TWO];
    uint64_t kDim = adjX1 ? self->GetViewShape()[self->GetViewShape().GetDimNum() - NUM_TWO] :
                           self->GetViewShape()[self->GetViewShape().GetDimNum() - 1];
    uint64_t nDim = adjX2 ? mat2->GetViewShape()[mat2->GetViewShape().GetDimNum() - NUM_TWO] :
                           mat2->GetViewShape()[mat2->GetViewShape().GetDimNum() - 1];
    uint64_t batchNum = GetBatchDimAll(self);
    bool batchEqual = IsBatchEqual(GetBatchDim(self), GetBatchDim(mat2));
    bool batchLargerThanAicNum = batchNum > aicoreNum;
    uint64_t alignMValue = CeilAlign(mDim, BLOCK_CUBE);
    uint64_t alignKaValue = adjX1 ? CeilAlign(kDim, BLOCK_CUBE) : CeilAlign(kDim, c0);
    uint64_t alignKbValue = adjX2 ? CeilAlign(kDim, c0) : CeilAlign(kDim, BLOCK_CUBE);
    uint64_t alignNValue = CeilAlign(nDim, BLOCK_CUBE);
    bool lessThanL0a = (alignMValue * alignKaValue * dtypeSize * pingPong <= l0aSize);
    bool lessThanL0b = (alignKbValue * alignNValue * dtypeSize * pingPong <= l0bSize);
    bool lessThanL0c = (alignMValue * alignNValue * floatSize * pingPong <= l0cSize);
    bool lessThanL1 = (alignMValue * alignKaValue + alignKbValue * alignNValue) * dtypeSize * pingPong <= l1Size;
    OP_LOGI("Checking If IterBatch Template in this socversion: %ld", static_cast<int64_t>(batchEqual &&
            batchLargerThanAicNum && lessThanL0a && lessThanL0b && lessThanL0c && lessThanL1));
    bool fitIterBatch = batchEqual && batchLargerThanAicNum && lessThanL0a && lessThanL0b && lessThanL0c && lessThanL1;
    bool fitBatchMatMulToMul = CheckShapeEqualToMul(mDim, nDim, batchNum, dtypeSize, c0);
    return !(fitIterBatch || fitBatchMatMulToMul);
}

using CheckSocIfBatchMatMulToMulFunc = bool (*)(const aclTensor* self, const aclTensor* mat2, bool adjX1, bool adjX2);
const static std::map<SocVersion, CheckSocIfBatchMatMulToMulFunc> CheckSocIfBatchMatMulToMulFuncMap = {
    {SocVersion::ASCEND910_95, CheckSocIfBatchMatMulToMul91095},
    {SocVersion::ASCEND910B, CheckSocIfBatchMatMulToMul910B},
    {SocVersion::ASCEND910_93, CheckSocIfBatchMatMulToMul910B},
};

const aclTensor* GetBatchMatmulOp(
    const aclTensor* selfTransdata, const aclTensor* mat2Transdata, const aclTensor* bias, const MmOpInfo& matmulOpInfo,
    bool adjX1, bool adjX2, const bool offsetX, aclOpExecutor* executor, bool isBaddbmm)
{
    auto bmmOpOut = selfTransdata;
    if (CheckAscendCScenario(selfTransdata, mat2Transdata, bias, matmulOpInfo, adjX1, adjX2)) {
        if (GetCurrentPlatformInfo().GetSocVersion() ==
                SocVersion::ASCEND910_95 && // 1.多维*2维(左非转置)2.多维*多维batch为1
            (GetBatchDimAll(mat2Transdata) <= 1 &&
             (!adjX1 || GetBatchDimAll(selfTransdata) <= 1))) { // 仅910_95路由该场景
            return TransBmm2Mm(
                selfTransdata, mat2Transdata, bias, matmulOpInfo.enableHf32, adjX1, adjX2, offsetX, executor);
        }
        OP_LOGI("Hit batch_mat_mul_v3 scenario.");
        if  ((matmulOpInfo.support_info.self_dtype == op::DataType::DT_FLOAT16 || matmulOpInfo.support_info.self_dtype == op::DataType::DT_BF16) && isBaddbmm) {
            OP_LOGI("Hit batch_mat_mul_v3 fp16/bf16 in - fp32 out scenario.");
            bmmOpOut = l0op::BatchMatMulV3NdFp16Bf162Fp32(
                selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.enableHf32, executor);
        } else {
            bmmOpOut = l0op::BatchMatMulV3Nd(
                selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.enableHf32, executor);
        }
        return bmmOpOut;
    }
    // 输入是FP16的场景
    if (matmulOpInfo.support_info.self_dtype == op::DataType::DT_FLOAT16) {
        if (matmulOpInfo.support_info.output_dtype == op::DataType::DT_FLOAT16) {
            // 输入是FP16, 输出是FP16的场景
            if (matmulOpInfo.support_info.self_format == op::Format::FORMAT_ND) {
                bmmOpOut = l0op::BatchMatMulNd(
                    selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.opImplModeEnum,
                    executor);
            } else {
                bmmOpOut = l0op::BatchMatMulNzFp162Fp16(
                    selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.opImplModeEnum,
                    executor);
            }
        } else {
            // 输入是FP16, 输出是FP32的场景
            if (matmulOpInfo.support_info.self_format == op::Format::FORMAT_ND) {
                bmmOpOut = l0op::BatchMatMulNdFp162Fp32(
                    selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.opImplModeEnum,
                    executor);
            } else {
                bmmOpOut = l0op::BatchMatMulNzFp162Fp32(
                    selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.opImplModeEnum,
                    executor);
            }
        }
    } else {
        // 输入是FP32/BF16,输出是FP32/BF16的场景
        bmmOpOut = l0op::BatchMatMulNd(
            selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX, matmulOpInfo.opImplModeEnum, executor);
    }
    return bmmOpOut;
}

bool CheckTransNonContiguousShapeSupport(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, bool adjX1)
{
    if (bias) {
        return false;
    }
    uint64_t aicoreNum = static_cast<uint64_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    uint64_t dtypeSize = static_cast<uint64_t>(op::TypeSize(self->GetDataType()));
    constexpr uint64_t floatSize = 4UL;
    constexpr uint64_t pingPong = 2UL;
    // 91095芯片参数
    constexpr uint64_t l0aSize = 64 * KB_SIZE;
    constexpr uint64_t l0bSize = 64 * KB_SIZE;
    constexpr uint64_t l0cSize = 256 * KB_SIZE;
    constexpr uint64_t l1Size = 512 * KB_SIZE;
    uint64_t mDim = adjX1 ? self->GetViewShape()[self->GetViewShape().GetDimNum() - 1] :
                            self->GetViewShape()[self->GetViewShape().GetDimNum() - NUM_TWO];
    uint64_t kDim = adjX1 ? self->GetViewShape()[self->GetViewShape().GetDimNum() - NUM_TWO] :
                            self->GetViewShape()[self->GetViewShape().GetDimNum() - 1];
    uint64_t nDim = mat2->GetViewShape()[mat2->GetViewShape().GetDimNum() - 1]; // 非连续场景viewshape一定是bkn格式
    uint64_t batchNum = GetBatchDimAll(self);
    bool batchEqual = IsBatchEqual(GetBatchDim(self), GetBatchDim(mat2));
    bool batchLargerThanAicNum = batchNum > aicoreNum;
    uint64_t alignMValue = CeilAlign(mDim, BLOCK_CUBE);
    uint64_t alignKaValue = CeilAlign(kDim, BLOCK_CUBE);
    uint64_t alignKbValue = CeilAlign(kDim, BLOCK_CUBE);
    uint64_t alignNValue = CeilAlign(nDim, BLOCK_CUBE);
    bool lessThanL0a = (alignMValue * alignKaValue * dtypeSize * pingPong <= l0aSize);
    bool lessThanL0b = (alignKbValue * alignNValue * dtypeSize * pingPong <= l0bSize);
    bool lessThanL0c = (alignMValue * alignNValue * floatSize * pingPong <= l0cSize);
    bool lessThanL1 = (alignMValue * alignKaValue + alignKbValue * alignNValue) * dtypeSize * pingPong <= l1Size;
    if (!batchEqual || !batchLargerThanAicNum) {
        return false;
    }
    bool l0CanLoadBatch = lessThanL0a && lessThanL0b && lessThanL0c && lessThanL1;
    constexpr static double defaultBalanceOfBatch = 0.8;
    uint64_t iterBatchL1 =
        FloorDiv(l1Size / pingPong, (alignMValue * alignKaValue + alignKbValue * alignNValue) * dtypeSize);
    if (!l0CanLoadBatch) {
        // if l0 can not load multi part of batch, cal to avoid formulate unbalance issue.
        double avgIterBatch = static_cast<double>(batchNum) / static_cast<double>(aicoreNum);
        double actualMaxIterBatch = static_cast<double>(
            CeilDiv(
                CeilDiv(batchNum, iterBatchL1),
                aicoreNum) *
            iterBatchL1);                                              // calculate one of the core load max batch
        double balanceRateOfBatch = avgIterBatch / actualMaxIterBatch; // calculate fb rate of batch
        if (balanceRateOfBatch < defaultBalanceOfBatch) {              // balance of batch lower than 0.8
            OP_LOGI("FormulteBalanceRate lower than 0.8, unable to enter in bmm iterbatch module");
            return false;
        }
    }
    return true;
}

bool CheckNonContiguousTranspose(
    const aclTensor* self, const aclTensor* mat2, bool& isNeedSwapInnerTwoDim, const aclTensor* bias, bool adjX1)
{
    if (!Ops::NN::IsTransposeNonContiguous(mat2, isNeedSwapInnerTwoDim)) {
        return false;
    }
    // 不支持NZ
    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        self->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("format NZ is not supported for transpose.");
        return false;
    }
    // 对shape校验，只支持多batch载入模板
    if (!CheckTransNonContiguousShapeSupport(self, mat2, bias, adjX1)) {
        OP_LOGI("shape is not supported for transpose");
        return false;
    }
    // transpose场景下，增加dtype判断，仅支持左右矩阵dtype相同
    if (self->GetDataType() != mat2->GetDataType()) {
        OP_LOGI("The data type of the self does not match the type of mat2 for transpose");
        return false;
    }
    return true;
}

bool CheckSocIfBatchMatMulToMulDefault(const aclTensor* self, const aclTensor* mat2, bool adjX1, bool adjX2)
{
    (void) self;
    (void) mat2;
    (void) adjX1;
    (void) adjX2;
    return false;
}

bool CheckSocIfBatchMatMulToMul(const aclTensor* self, const aclTensor* mat2, bool adjX1, bool adjX2)
{
    auto iter = (CheckSocIfBatchMatMulToMulFuncMap.find(GetCurrentPlatformInfo().GetSocVersion()) ==
                    CheckSocIfBatchMatMulToMulFuncMap.end()) ? CheckSocIfBatchMatMulToMulDefault :
                    CheckSocIfBatchMatMulToMulFuncMap.at(GetCurrentPlatformInfo().GetSocVersion());
    return iter(self, mat2, adjX1, adjX2);
}

static inline int64_t ProcessEqual1Cases(
    const aclTensor*& selfCast, const aclTensor*& mat2Cast, MmOpInfo& matmulOpInfo, const aclTensor*& bias, bool& adjX1,
    bool& adjX2, const aclTensor*& selfReshape, const aclTensor*& mat2Reshape, aclOpExecutor* executor, bool& ifKEqual1)
{
    ifKEqual1 = IfKEqual1(selfCast, matmulOpInfo, adjX1, bias) &&
                     CheckSocIfBatchMatMulToMul(selfCast, mat2Cast, adjX1, adjX2); // distincted by different soc
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes = IfKEqual1Mat2ToKN(selfCast, selfReshape, adjX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, -1);
        aclnnStatus kEqual1Mat2ToKNRes = IfKEqual1Mat2ToKN(mat2Cast, mat2Reshape, adjX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, -1);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes =
            IfMEqual1SelfToMK(selfCast, selfReshape, matmulOpInfo.support_info.self_format, adjX1, executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, -1);
        aclnnStatus nEqual1Mat2ToNKRes =
            IfNEqual1Mat2ToNK(mat2Cast, mat2Reshape, matmulOpInfo.support_info.mat2_format, adjX2, executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, -1);
    }
    return 0;
}

} // namespace

namespace Ops {
namespace NN {
const aclTensor* ExecBmmOpWithBiasV2(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType,
    aclOpExecutor* executor, bool isBaddbmm)
{
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(self, mat2, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    // reformat，全部转成ND
    auto reformatSelf = self;
    reformatSelf = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(reformatSelf != nullptr, nullptr);
    auto transposeSelf = Ops::NN::IsTransposeLastTwoDims(self);
    auto transposeMat2 = false;

    bool isNeedSwapInnerTwoDim = false; // 非连续场景下右矩阵转置属性
    bool isTransposeMat2Contiguous =
        CheckNonContiguousTranspose(self, mat2, isNeedSwapInnerTwoDim, bias, transposeSelf);

    const aclTensor* reformatMat2 = nullptr;
    if (mat2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ && !isTransposeMat2Contiguous) {
        OP_LOGI("mat2 StorageFormat not FORMAT_FRACTAL_NZ.");
        reformatMat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    } else {
        reformatMat2 = mat2;
    }
    auto reformatOut = l0op::ReFormat(out, op::Format::FORMAT_ND);
    CHECK_RET(reformatOut != nullptr, nullptr);

    auto contiguousSelf = reformatSelf;
    auto contiguousMat2 = reformatMat2;

    transposeSelf = Ops::NN::IsTransposeLastTwoDims(self);
    if (transposeSelf) {
        contiguousSelf = executor->CreateView(self, SwapLastTwoDimValue(self->GetViewShape()), self->GetViewOffset());
    } else {
        contiguousSelf = l0op::Contiguous(self, executor);
    }
    CHECK_RET(contiguousSelf != nullptr, nullptr);

    if (isTransposeMat2Contiguous) {
        if (isNeedSwapInnerTwoDim) {
            // Swap inner two axis (dim 1 & dim2) b1 b2 n k b1 b2 k n
            contiguousMat2 = executor->CreateView(
                mat2, SwapLastTwoDimValue(mat2->GetViewShape(), LAST_DIM, PENULTIMATE_DIM), mat2->GetViewOffset());
            op::Strides strides = mat2->GetViewStrides();
            const size_t size = strides.size();
            std::swap(strides[size - LAST_DIM], strides[size - NUM_TWO]);
            const_cast<aclTensor*>(contiguousMat2)->SetViewStrides(strides);
            transposeMat2 = true;
        }
    } else {
        // 原内轴转置
        transposeMat2 = Ops::NN::IsTransposeLastTwoDims(mat2);
        if (transposeMat2) {
            contiguousMat2 =
                executor->CreateView(mat2, SwapLastTwoDimValue(mat2->GetViewShape()), mat2->GetViewOffset());
        } else {
            contiguousMat2 = l0op::Contiguous(mat2, executor);
        }
        CHECK_RET(contiguousMat2 != nullptr, nullptr);
    }

    // weightnz storageshape刷新, 适配SwapLastTwoDimValue
    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        contiguousMat2->SetStorageShape(mat2->GetStorageShape());
    }

    auto batchMatmulOut = ExecBatchMatmulOpWithBiasAndAttrsV2(
        contiguousSelf, contiguousMat2, bias, reformatOut, transposeSelf, transposeMat2, cubeMathType,
        executor, isTransposeMat2Contiguous, isBaddbmm);

    CHECK_RET(batchMatmulOut != nullptr, nullptr);

    // reformat成原来out的数据格式
    auto reformatBmmOut = l0op::ReFormat(batchMatmulOut, out->GetViewFormat());
    CHECK_RET(reformatBmmOut != nullptr, nullptr);

    return reformatBmmOut;
}

const aclTensor* ExecBatchMatmulOpWithBiasAndAttrsV2(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, bool adjX1, bool adjX2,
    int8_t cubeMathType, aclOpExecutor* executor, bool isTransposeMat2Contiguous, bool isBaddbmm)
{
    MmOpInfo matmulOpInfo;
    CreateBatchMatmulOpInfo(self, mat2, bias, out, matmulOpInfo, cubeMathType);

    auto selfCast = l0op::Cast(self, matmulOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCast != nullptr, nullptr);

    auto mat2Cast = mat2;
    if (isTransposeMat2Contiguous) {
        // 刷新oriShape
        mat2Cast = executor->CreateView(
            mat2, mat2->GetViewShape(), mat2->GetStorageShape(), mat2->GetViewStrides(), mat2->GetViewOffset());
        mat2Cast = SetTensorToNDFormat(mat2Cast);
    } else {
        mat2Cast = l0op::Cast(mat2, matmulOpInfo.support_info.mat2_dtype, executor);
        CHECK_RET(mat2Cast != nullptr, nullptr);
    }

    // bias Contiguous cast
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        bool biasCastRes = ContiguousAndCastBias(bias, contiguousBias, matmulOpInfo.support_info.bias_dtype, executor);
        CHECK_RET(biasCastRes, nullptr);
    }

    // k,m,n=1特殊场景
    auto selfReshape = selfCast;
    auto mat2Reshape = mat2Cast;

    bool ifKEqual1 = false;
    if (!isTransposeMat2Contiguous) {
        CHECK_RET(
            ProcessEqual1Cases(
                selfCast, mat2Cast, matmulOpInfo, contiguousBias, adjX1, adjX2, selfReshape, mat2Reshape, executor,
                ifKEqual1) != -1, nullptr);
    }

    auto selfTransdata = l0op::TransData(selfReshape, matmulOpInfo.support_info.self_format, 0, executor);
    CHECK_RET(selfTransdata != nullptr, nullptr);
    auto mat2Transdata = mat2Reshape;
    if (!isTransposeMat2Contiguous) {
        mat2Transdata = l0op::TransData(mat2Reshape, matmulOpInfo.support_info.mat2_format, 0, executor);
        CHECK_RET(mat2Transdata != nullptr, nullptr);
    }

    const aclTensor* bmmOpOut = nullptr;
    if (isTransposeMat2Contiguous) {
        bmmOpOut = GetBatchMatmulOp(selfTransdata, mat2Transdata, contiguousBias, matmulOpInfo, adjX1, adjX2, 0, executor, isBaddbmm);
    } else if (ifKEqual1) {
        auto selfTransdataCast = l0op::Cast(selfTransdata, op::DataType::DT_FLOAT, executor);
        auto mat2TransdataCast = l0op::Cast(mat2Transdata, op::DataType::DT_FLOAT, executor);
        bmmOpOut = l0op::Mul(selfTransdataCast, mat2TransdataCast, executor);
    } else {
        bmmOpOut = GetBatchMatmulOp(selfTransdata, mat2Transdata, contiguousBias, matmulOpInfo, adjX1, adjX2, 0, executor, isBaddbmm);
    }

    CHECK_RET(bmmOpOut != nullptr, nullptr);

    auto transdataOut = l0op::TransData(bmmOpOut, matmulOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(transdataOut != nullptr, nullptr);

    return transdataOut;
}

const aclTensor* ExecBmmOpV2(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor, bool isBaddbmm)
{
    return ExecBmmOpWithBiasV2(self, mat2, nullptr, out, cubeMathType, executor, isBaddbmm);
}
}  // namespace Ops
}  // namespace NN
