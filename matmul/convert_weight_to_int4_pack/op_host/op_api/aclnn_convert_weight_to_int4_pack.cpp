/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <vector>
#include "runtime/mem.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_convert_weight_to_int4_pack.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
namespace {
const uint64_t INPUT_DIM_MIN_VALUE = 2;
const uint64_t INPUT_DIM_MAX_VALUE = 3;
const uint64_t GMM_WEIGHT_DIM = 3;
const uint64_t INT4_NUM_IN_BYTE = 2;
const int64_t INT4_NUM_IN_INT32 = 8;
const int64_t INT4_NUM_IN_32B = 64;
const int64_t C0_16 = 16;
const int64_t C0_32 = 32;
const int64_t CUBE_BLOCK_SIZE = 16;
const int64_t INT4_NZ_DIM_NUM = 4;
const int64_t INT4_NZ_DIM_NUM_GMM = 5;
const int64_t INT4_NZ_DIM_ID1 = 1;
const int64_t INT4_NZ_DIM_ID2 = 2;
const int64_t INT4_NZ_DIM_ID3 = 3;
const int64_t INT4_NZ_DIM_ID4 = 4;
const uint32_t FP32_TO_FP4_MASK = 0xFFC00000;
const std::unordered_map<uint32_t, uint32_t> FP32_BIT_TO_FP4_E2M1 = {
    {0x00000000, 0b0000}, // 0.0
    {0x3F000000, 0b0001}, // 0.5
    {0x3F800000, 0b0010}, // 1.0
    {0x3FC00000, 0b0011}, // 1.5
    {0x40000000, 0b0100}, // 2.0
    {0x40400000, 0b0101}, // 3.0
    {0x40800000, 0b0110}, // 4.0
    {0x40C00000, 0b0111}, // 6.0

    {0x80000000, 0b1000}, // -0.0
    {0xBF000000, 0b1001}, // -0.5
    {0xBF800000, 0b1010}, // -1.0
    {0xBFC00000, 0b1011}, // -1.5
    {0xC0000000, 0b1100}, // -2.0
    {0xC0400000, 0b1101}, // -3.0
    {0xC0800000, 0b1110}, // -4.0
    {0xC0C00000, 0b1111}, // -6.0
};

static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = { DataType::DT_INT32,
                                                                           DataType::DT_FLOAT };
static const std::initializer_list<DataType> WEIGHTINT4PACK_DTYPE_SUPPORT_LIST = {DataType::DT_INT4,
                                                                                  DataType::DT_INT32,
                                                                                  DataType::DT_FLOAT4_E2M1,
                                                                                  DataType::DT_FLOAT};

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return valueNum;
    }
    return (value + factor - 1) / factor;
}

static uint32_t ConvertFp32ToFp4e2m1(int32_t data)
{
    uint32_t fp32_bits = static_cast<uint32_t>(data) & FP32_TO_FP4_MASK;
    auto it = FP32_BIT_TO_FP4_E2M1.find(fp32_bits);
    if (it != FP32_BIT_TO_FP4_E2M1.end()) {
        return it->second;
    }
    return 0b0000;
}

static int32_t PackToINT8(int32_t a, int32_t b, DataType dtype) {
    if (dtype == DataType::DT_INT32) {
        // 将2个int4的数按照顺序交错排布存入到一个int8的数
        return static_cast<int32_t>(((static_cast<uint32_t>(b) & 0xF) << 4) | (static_cast<uint32_t>(a) & 0xF));
    }
    // 提取2个FP4数据并交错排布存入到一个int8的数
    return static_cast<int32_t>(((ConvertFp32ToFp4e2m1(b) & 0xF) << 4) | (ConvertFp32ToFp4e2m1(a) & 0xF));
}

static int64_t GetNdToNzC0Size(bool c032Flag)
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        if (c032Flag) {
            return static_cast<int64_t>(C0_32) / static_cast<int64_t>(INT4_NUM_IN_BYTE);
        } else {
            return static_cast<int64_t>(C0_16) / static_cast<int64_t>(INT4_NUM_IN_BYTE);
        }
    }

    return static_cast<int64_t>(INT4_NUM_IN_32B) / static_cast<int64_t>(INT4_NUM_IN_BYTE);
}

static void ConvertToB4Pack(const std::vector<int32_t>& weightData, std::vector<int8_t>& weightInt4PackData,
                            DataType dtype)
{
    size_t n = weightInt4PackData.size();
    for (size_t i = 0; i < n; ++i) {
        int32_t a = weightData[i * 2];
        int32_t b = weightData[i * 2 + 1];
        // 存入weightInt4PackData中，顺序交错排布
        weightInt4PackData[i] = PackToINT8(a, b, dtype);
    }
}

static bool IsFormatSupport(const aclTensor *input, Format format, const std::string &inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's format should be ND. Actual is [%s].", inputName.c_str(),
                op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool IsFormatSupportWeightPack(const aclTensor *input, const std::string &inputName)
{
    if (input != nullptr && input->GetStorageFormat() != Format::FORMAT_ND &&
        input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's format should be ND/NZ. Actual is [%s].", inputName.c_str(),
                op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool IsDimSupport(const aclTensor *input, std::vector<uint64_t> &dimRange, const std::string &inputName)
{
    if (input != nullptr &&
        (input->GetViewShape().GetDimNum() < dimRange[0] || input->GetViewShape().GetDimNum() > dimRange[1])) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's dim should be in range [%ld, %ld]. Actual is [%zu].", inputName.c_str(),
                dimRange[0], dimRange[1], input->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(weightInt4Pack, return false);
    return true;
}

static bool CheckContiguous(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    if (IsContiguous(weight) && IsContiguous(weightInt4Pack)) {
        return true;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the input weight and weightInt4Pack only support contiguous Tensor");
    return false;
}

static bool CheckDtypeVaild(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weightInt4Pack, WEIGHTINT4PACK_DTYPE_SUPPORT_LIST, return false);
    if ((weight->GetDataType() == DataType::DT_INT32 && weightInt4Pack->GetDataType() != DataType::DT_INT32 &&
         weightInt4Pack->GetDataType() != DataType::DT_INT4) ||
        (weight->GetDataType() == DataType::DT_FLOAT && weightInt4Pack->GetDataType() != DataType::DT_FLOAT &&
         weightInt4Pack->GetDataType() != DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Invalid dtype combination: weight dtype: %s, weightInt4Pack dtype: %s. Supported combination: "
                "weightInt4Pack dtype DT_INT32 or DT_INT4 with weight dtype DT_INT32. weightInt4Pack dtype DT_FLOAT or "
                "DT_FLOAT4_E2M1 with weight dtype DT_FLOAT",
                op::ToString(weight->GetDataType()).GetString(),
                op::ToString(weightInt4Pack->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormatVaild(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    CHECK_RET(IsFormatSupport(weight, Format::FORMAT_ND, "weight"), false);
    CHECK_RET(IsFormatSupportWeightPack(weightInt4Pack, "weightInt4Pack"), false);
    return true;
}

static bool CheckDimVaild(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    std::vector<uint64_t> dimRange = {INPUT_DIM_MIN_VALUE, INPUT_DIM_MAX_VALUE};
    CHECK_RET(IsDimSupport(weight, dimRange, "weight"), false);
    CHECK_RET(IsDimSupport(weightInt4Pack, dimRange, "weightInt4Pack"), false);
    return true;
}

static bool CheckStorageShapeVaild(const aclTensor *weightInt4Pack, int64_t weightInt4PackDimFirst,
                                   int64_t weightInt4PackDimLast)
{
    size_t weightInt4PackDimNum = weightInt4Pack->GetStorageShape().GetDimNum();
    if (weightInt4PackDimNum != INT4_NZ_DIM_NUM && weightInt4PackDimNum != INT4_NZ_DIM_NUM_GMM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "when weightInt4Pack's datatype is int4 and format is FORMAT_FRACTAL_NZ, "
                "weightInt4Pack's storage dim num should be [%ld] or [%ld], but actual is [%lu]",
                INT4_NZ_DIM_NUM, INT4_NZ_DIM_NUM_GMM, weightInt4PackDimNum);
        return false;
    }
    int64_t storageDim0 = weightInt4Pack->GetStorageShape().GetDim(weightInt4PackDimNum - INT4_NZ_DIM_ID4);
    int64_t storageDim1 = weightInt4Pack->GetStorageShape().GetDim(weightInt4PackDimNum - INT4_NZ_DIM_ID3);
    int64_t storageDim2 = weightInt4Pack->GetStorageShape().GetDim(weightInt4PackDimNum - INT4_NZ_DIM_ID2);
    int64_t storageDim3 = weightInt4Pack->GetStorageShape().GetDim(weightInt4PackDimNum - INT4_NZ_DIM_ID1);
    int64_t expDim1 = CeilDiv(weightInt4PackDimFirst, CUBE_BLOCK_SIZE);
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (weightInt4Pack->GetDataType() == DataType::DT_INT4) {
        int64_t lastDimSize = socVersion == SocVersion::ASCEND910_95 ? storageDim3 : INT4_NUM_IN_32B;
        int64_t expDim0 = CeilDiv(weightInt4PackDimLast, lastDimSize);
        if ((storageDim0 != expDim0) || (storageDim1 != expDim1) || (storageDim2 != CUBE_BLOCK_SIZE) ||
            (storageDim3 != lastDimSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when weightInt4Pack's datatype is int4 and format is FORMAT_FRACTAL_NZ, "
                    "weightInt4Pack's storage shape should be [%ld,%ld,%ld,%ld], "
                    "but actual is [%s]",
                    expDim0, expDim1, CUBE_BLOCK_SIZE, lastDimSize,
                    op::ToString(weightInt4Pack->GetStorageShape()).GetString());
            return false;
        }
    }

    if (weightInt4Pack->GetDataType() == DataType::DT_INT32) {
        int64_t lastDimSize = socVersion == SocVersion::ASCEND910_95 ? C0_16 / INT4_NUM_IN_INT32 : INT4_NUM_IN_INT32;
        int64_t expDim0 = CeilDiv(weightInt4PackDimLast, lastDimSize);
        if ((storageDim0 != expDim0) || (storageDim1 != expDim1) || (storageDim2 != CUBE_BLOCK_SIZE) ||
            (storageDim3 != lastDimSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when weightInt4Pack's datatype is int32 and format is FORMAT_FRACTAL_NZ, "
                    "weightInt4Pack's storage shape should be [%ld,%ld,%ld,%ld], "
                    "but actual is [%s]",
                    expDim0, expDim1, CUBE_BLOCK_SIZE, lastDimSize,
                    op::ToString(weightInt4Pack->GetStorageShape()).GetString());
            return false;
        }
    }

    return true;
}

static bool CheckShapeVaild(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    size_t weightDimNum = weight->GetViewShape().GetDimNum() - 1;
    size_t weightInt4PackDimNum = weightInt4Pack->GetViewShape().GetDimNum() - 1;
    int64_t weightDimFirst = weight->GetViewShape().GetDim(weightDimNum - 1);
    int64_t weightDimLast = weight->GetViewShape().GetDim(weightDimNum);
    int64_t weightInt4PackDimFirst = weightInt4Pack->GetViewShape().GetDim(weightInt4PackDimNum - 1);
    int64_t weightInt4PackDimLast = weightInt4Pack->GetViewShape().GetDim(weightInt4PackDimNum);

    if (weightInt4Pack->GetDataType() == DataType::DT_INT4 ||
        weightInt4Pack->GetDataType() == DataType::DT_FLOAT4_E2M1) {
        if (weightDimLast % INT4_NUM_IN_BYTE != 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weightInt4Pack's datatype is int4 or fp4, the last dim of weight should be even, Actual is %ld",
                weightDimLast);
            return false;
        }
        if (weightDimFirst != weightInt4PackDimFirst || weightDimLast != weightInt4PackDimLast) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when weightInt4Pack's datatype is int4, "
                    "weight's size should be the same as weightInt4Pack's size, "
                    "but [%ld, %ld] does not equal [%ld, %ld]",
                    weightDimFirst, weightDimLast, weightInt4PackDimFirst, weightInt4PackDimLast);
            return false;
        }
    }
    if (weightInt4Pack->GetDataType() == DataType::DT_INT32 || weightInt4Pack->GetDataType() == DataType::DT_FLOAT) {
        if (weightDimLast % INT4_NUM_IN_INT32 != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when weightInt4Pack's datatype is int32 or fp32, "
                    "the last dim of weight should be a multiple of 8, Actual is %ld",
                    weightDimLast);
            return false;
        }
        if (weightDimFirst != weightInt4PackDimFirst || (weightDimLast / INT4_NUM_IN_INT32) != weightInt4PackDimLast) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when weightInt4Pack's datatype is int32 or fp32, "
                    "weight's size first dim should be the same as weightInt4Pack's size first dim, "
                    "weight's size last dim should be the 8 multiple as weightInt4Pack's size last dim "
                    "but got weight [%ld, %ld], and weightInt4Pack [%ld, %ld]",
                    weightDimFirst, weightDimLast, weightInt4PackDimFirst, weightInt4PackDimLast);
            return false;
        }
    }

    if (weightInt4Pack->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        return CheckStorageShapeVaild(weightInt4Pack, weightInt4PackDimFirst, weightInt4PackDimLast);
    }
    return true;
}

static aclnnStatus ParamsCheck(const aclTensor *weight, const aclTensor *weightInt4Pack)
{
    CHECK_RET(CheckNotNull(weight, weightInt4Pack), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckContiguous(weight, weightInt4Pack), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeVaild(weight, weightInt4Pack), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatVaild(weight, weightInt4Pack), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimVaild(weight, weightInt4Pack), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeVaild(weight, weightInt4Pack), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool TransNdToNz(std::vector<int8_t>& weightArray, const Shape& weightShape, bool c032Flag)
{
    int64_t weightDimNum = weightShape.GetDimNum();
    int64_t k = weightShape.GetDim(weightDimNum - 2); // weightDimNum - 2为k轴索引
    int64_t n =
        weightShape.GetDim(weightDimNum - 1) / 2; // weightDimNum - 1为k轴索引，因为用int8_t表示int4，所以需要除以2
    int64_t g = 1;
    if (n == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The inner axis of weightInt4PackData should not be 0.");
        return false;
    }

    int64_t c0SizeInt8 = GetNdToNzC0Size(c032Flag);
    if (c0SizeInt8 == 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "The nz storage shape c0 size should not be 0.");
        return false;
    }
    // k为外轴，n为内轴，外轴按16对齐，内轴按32B对齐，(k, n) -> (n1,k1,16,c0SizeInt8)
    int64_t k1 = CeilDiv(k, CUBE_BLOCK_SIZE);
    int64_t n1 = CeilDiv(n, c0SizeInt8);
    int64_t weightNzSize = k1 * CUBE_BLOCK_SIZE * n1 * c0SizeInt8;
    if(weightDimNum == GMM_WEIGHT_DIM){
     g = weightShape.GetDim(0);
     weightNzSize = g * weightNzSize;   
    }
    std::vector<int8_t> weightNzArray(weightNzSize, 0);
    for (size_t idxG = 0; idxG < static_cast<size_t>(g); idxG++) {
     for (size_t idx = 0; idx < static_cast<size_t>(k * n); idx++) {
            size_t idxK = idx / n;
            size_t idxN = idx % n;
            size_t idxK0 = idxK % CUBE_BLOCK_SIZE;
            size_t idxK1 = idxK / CUBE_BLOCK_SIZE;
            size_t idxN0 = idxN % c0SizeInt8;
            size_t idxN1 = idxN / c0SizeInt8;
            weightNzArray
                [idxN1 * k1 * CUBE_BLOCK_SIZE * c0SizeInt8 + idxK1 * c0SizeInt8 * CUBE_BLOCK_SIZE + idxK0 * c0SizeInt8 +
                 idxN0 + idxG * k * n] = weightArray[idx + idxG * k * n];
     }
    }
    weightArray = weightNzArray;
    return true;
}

static void TransOriginalShape(aclTensor *weightInt4Pack)
{
    auto viewShape = weightInt4Pack->GetViewShape();
    weightInt4Pack->SetOriginalShape(viewShape);
    OP_LOGD("The correction of original shape for weightInt4PackNZ is completed.");
}

aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    (void)weight;
    (void)weightInt4Pack;
    (void)workspaceSize;
    (void)executor;
    auto checkRet = ParamsCheck(weight, weightInt4Pack);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    int64_t weightInt4PackDimNum = weightInt4Pack->GetStorageShape().GetDimNum();
    int64_t weightInt4PackDimLast = weightInt4Pack->GetStorageShape().GetDim(weightInt4PackDimNum - 1);
    // 若传入的weightInt4Pack的storageShape的最后一维为32，且为910_95,则为s8s4分支
    bool c032Flag =
        weightInt4PackDimLast == C0_32 && GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
    auto shapeSize = weight->GetViewShape().GetShapeSize();

    // 从device拷贝数据到host
    std::vector<int32_t> weightData(shapeSize, 0);
    auto ret = rtMemcpy(
        weightData.data(), weightData.size() * sizeof(weightData[0]), weight->GetTensor()->GetAddr(),
        weightData.size() * sizeof(weightData[0]), RT_MEMCPY_DEVICE_TO_HOST);
    OP_CHECK(
        ret == RT_ERROR_NONE, OP_LOGE(ret, "copy result from device to host failed."), return ACLNN_ERR_RUNTIME_ERROR);
    // 从原先的1个int32的数承载1个int4的数改变成用1个int8的数承载2个int4的数，size缩减为原来的一半
    std::vector<int8_t> weightInt4PackData(shapeSize / 2, 0);
    ConvertToB4Pack(weightData, weightInt4PackData, weight->GetDataType());
    auto weightShape = weight->GetViewShape();
    // 如果weightInt4Pack是FRACTAL_NZ，需要对ND数据重排为FRACTAL_NZ输出, original shape需要跟随view shape
    if (weightInt4Pack->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
     ret = TransNdToNz(weightInt4PackData, weightShape, c032Flag);
     CHECK_RET(ret, ACLNN_ERR_PARAM_INVALID);
     TransOriginalShape(weightInt4Pack);
    }

    // 从host拷贝数据到device
    ret = rtMemcpy(
        weightInt4Pack->GetTensor()->GetAddr(), weightInt4PackData.size() * sizeof(weightInt4PackData[0]),
        weightInt4PackData.data(), weightInt4PackData.size() * sizeof(weightInt4PackData[0]), RT_MEMCPY_HOST_TO_DEVICE);
    OP_CHECK(
        ret == RT_ERROR_NONE, OP_LOGE(ret, "copy result from host to device failed."), return ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    (void)workspace;
    (void)workspaceSize;
    (void)executor;
    (void)stream;
    return ACLNN_SUCCESS;
}
}
#ifdef __cplusplus
}
#endif