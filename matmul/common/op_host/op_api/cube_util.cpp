/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cube_util.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
using namespace op;

namespace Ops {
namespace NN {

// 910B额外支持BF16，其余只支持FP16 + FP32
static const std::initializer_list<DataType> V100_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> V200_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
namespace {
static const std::initializer_list<DataType> DAV_3510_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_HIFLOAT8};
static const std::initializer_list<DataType> DAV_3510_CONVBP_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E4M3FN};
} // namespace
// 根据dtype进行初步拦截，后续需要再和cubemathtype + 芯片再进行一次拦截
const std::initializer_list<DataType>& GetDtypeSupportListBySocVersion()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return DAV_3510_DTYPE_SUPPORT_LIST;
    }
    return (IsCubeSupportFp32()) ? V200_DTYPE_SUPPORT_LIST : V100_DTYPE_SUPPORT_LIST;
}

const std::initializer_list<DataType>& GetDtypeSupportListBySocVersion4ConvBackward(bool transposed)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return transposed ? DAV_3510_DTYPE_SUPPORT_LIST : DAV_3510_CONVBP_DTYPE_SUPPORT_LIST;
    }
    return (IsCubeSupportFp32()) ? V200_DTYPE_SUPPORT_LIST : V100_DTYPE_SUPPORT_LIST;
}

// 检查芯片是否支持输入的dtype allowFp32为True：芯片不允许输入为BF16   allowFp32为False：芯片不允许输入为BF16和FP32
static bool CheckSocSupportDtype(const op::DataType cubeTensorDtype, bool allowFp32)
{
    bool dtypeValid = allowFp32 ? (cubeTensorDtype == DataType::DT_BF16) :
                                  (cubeTensorDtype == DataType::DT_FLOAT || cubeTensorDtype == DataType::DT_BF16);
    // 如果芯片不支持FP32 + dtype为FP32 / BF16，报错
    OP_CHECK(
        !(dtypeValid && !IsCubeSupportFp32()),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The soc version does not support bf16 / fp32 for calculations, please change the setting of "
            "cubeMathType or the Dtype of input tensor."),
        return false);
    return true;
}

bool CheckUnSupportDtype(const aclTensor* input, const aclTensor* weight)
{
    if ((input->GetDataType() == op::DataType::DT_HIFLOAT8) != (weight->GetDataType() == op::DataType::DT_HIFLOAT8)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Hif8 not in aclnn framework support dtype list");
        return true;
    }
    return false;
}

// 路由cubeMathType4到cubeMathType0, 供不支持cubeMathType4的场景使用, 仅在DAV_2201场景下生效
// 该函数是考虑在部分算子中， 在DAV_2201架构下，cubeMathType = 4的场景并不被支持（具体可以参考资料），
// 但是用户可以提供该输入；出于易用性考虑，以及为了保持API的一致性，在DAV_2201场景下
// cubeMathType = 4的场景会被路由到cubeMathType = 0的场景
int8_t routeCubeMathType4ToCubeMathType0DAV_2201(int8_t cubeMathType)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 && cubeMathType == USE_FP32_ADD) {
        OP_LOGD("The cubeMathType is USE_FP32_ADD for which the API does not support temporarily, "
                "route to KEEP_DTYPE instead.");
        return KEEP_DTYPE;
    }
    return cubeMathType;
}

// 校验cubeMathType的值是否符合预期
bool CheckCubeMathType(const op::DataType cubeTensorDtype, int8_t cubeMathType)
{
    OP_LOGD("The cubeMathType is %d.", cubeMathType);
    switch (cubeMathType) {
        case KEEP_DTYPE:
            OP_LOGD("The cubeMathType is KEEP_DTYPE.");
            return CheckSocSupportDtype(cubeTensorDtype, false);
        case ALLOW_FP32_DOWN_PRECISION:
            // 注意：非910B场景，BF16报错，FP32支持  正常来说在校验cubemathtype前, dtype应该拦截掉910 + BF16场景
            OP_LOGD("The cubeMathType is ALLOW_FP32_DOWN_PRECISION.");
            return CheckSocSupportDtype(cubeTensorDtype, true);
        case USE_FP16:
            // 注意：非910B场景，BF16报错，FP32支持  正常来说在校验cubemathtype前, dtype应该拦截掉910 + BF16场景
            OP_LOGD("The cubeMathType is USE_FP16.");
            return CheckSocSupportDtype(cubeTensorDtype, true);
        case USE_HF32:
            OP_LOGD("The cubeMathType is USE_HF32.");
            return CheckSocSupportDtype(cubeTensorDtype, false);
        case USE_FP32_ADD:
            OP_LOGD("The cubeMathType is USE_FP32_ADD.");
            return CheckSocSupportDtype(cubeTensorDtype, false);
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The value of cubeMathType only support {0: KEEP_DTYPE, 1: "
                "ALLOW_FP32_DOWN_PRECISION, 2: USE_FP16, 3: USE_HF32, 4: USE_FP32_ADD}, but got %d",
                cubeMathType);
            return false;
    }
}

// 校验mm算子cubeMathType的值是否符合预期
bool CheckCubeMathTypeForMm(const op::DataType cubeTensorDtype, int8_t cubeMathType)
{
    if (cubeMathType == USE_FP16 && cubeTensorDtype == DataType::DT_BF16) {
        OP_LOGW("The cubeMathType is USE_FP16. For input BF16, it will not be enabled.");
    } else if (
        cubeMathType == USE_HF32 && (cubeTensorDtype == DataType::DT_BF16 || cubeTensorDtype == DataType::DT_FLOAT16)) {
        OP_LOGW("The cubeMathType is USE_HF32. For input FP16/BF16, it will not be enabled.");
    }

    if (cubeMathType == -1) {
        OP_LOGD("The inner cubeMathType is FP16FP32_KEEP_DTYPE.");
        return CheckSocSupportDtype(cubeTensorDtype, false);
    } else {
        return CheckCubeMathType(cubeTensorDtype, cubeMathType);
    }
}

bool CheckAddmmTensorShapeNeedBroadcast(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* self)
{
    uint64_t dimNum = mat1->GetViewShape().GetDimNum();
    uint64_t selfDimNum = self->GetViewShape().GetDimNum();
    if (dimNum != selfDimNum) {
        OP_LOGI("self's dimnum != matmul out's dimnum.");
        return true;
    }
    const op::Shape selfShape = self->GetViewShape();
    const op::Shape mat1Shape = mat1->GetViewShape();
    const op::Shape mat2Shape = mat2->GetViewShape();
    if (dimNum == 3UL) {
        OP_CHECK(selfShape[0] == mat1Shape[0] && selfShape[1] == mat1Shape[1] && selfShape[2] == mat2Shape[2],
            OP_LOGI("self shape not equal to matmul out shape."),
            return true);
    } else if (dimNum == 2UL) {
        OP_CHECK(selfShape[0] == mat1Shape[0] && selfShape[1] == mat2Shape[1],
            OP_LOGI("self shape not equal to matmul out shape."),
            return true);
    }
    return false;
}


bool CheckCubeMathTypeForAddMm(
    const aclTensor* mat1, const aclTensor* mat2, const aclTensor* self, const aclTensor* out, int8_t cubeMathType)
{
    (void)out;
    if (cubeMathType > USE_FP32_ADD) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The value of cubeMathType only support {0: KEEP_DTYPE, 1: "
            "ALLOW_FP32_DOWN_PRECISION, 2: USE_FP16, 3: USE_HF32, 4: USE_FP32_ADD}, but got %d",
            cubeMathType);
        return false;
    }

    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (cubeMathType != USE_FP32_ADD) {
        return true;
    }
    // 平台校验
    if (npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "current platform not support cubeMathType = 4: USE_FP32_ADD.");
        return false;
    }
    // A2平台上，当cubeMathType=USE_FP32_ADD时，当前不支持self与mmout broadcast
    if (npuArch == NpuArch::DAV_2201) {
        bool needBroadcast = CheckAddmmTensorShapeNeedBroadcast(mat1, mat2, self);
        OP_CHECK(!needBroadcast,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "when cubeMathType = 4:USE_FP32_ADD, do not support broadcast between self and mmout."),
            return false;
        );
    }
    return true;
}

// 根据promote type + cubemathtype的组合算出最终算子应用的dtype
DataType CalcPromoteTypeCubemathtype(const DataType cubeTensorPromoteType, int8_t cubeMathType)
{
    bool cubeSupportFp32Flag = IsCubeSupportFp32();
    // USE_FP16场景，如果promote type为bf16，提示不支持该选项
    if (cubeMathType == USE_FP16) {
        if (cubeTensorPromoteType == DataType::DT_BF16) {
            OP_LOGW("The cubeMathType can't be set to USE_FP16 when the dtype is BF16.");
        }
        return DataType::DT_FLOAT16;
    }

    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
            return DataType::DT_FLOAT16;
        case DataType::DT_FLOAT:
            return cubeSupportFp32Flag ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
        case DataType::DT_BF16:
            return cubeSupportFp32Flag ? DataType::DT_BF16 : DataType::DT_FLOAT16;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, but got %s",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_UNDEFINED;
    }
}

DataType CalcUseFp16PromoteType(const DataType cubeTensorPromoteType)
{
    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
        case DataType::DT_FLOAT:
            return DataType::DT_FLOAT16;
        case DataType::DT_BF16:
            OP_LOGW(
                "Cube cannot support dtype: %s when cubeMathType is USE_FP16.",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_FLOAT16;
        case DataType::DT_HIFLOAT8:
        case DataType::DT_FLOAT8_E4M3FN:
            OP_LOGW(
                "Cube cannot support dtype: %s when cubeMathType is USE_FP16.",
                op::ToString(cubeTensorPromoteType).GetString());
            return cubeTensorPromoteType;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, HIF8, FP8_E4M3FN, but got %s.",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_UNDEFINED;
    }
}

DataType CalcUseHf32PromoteType(const DataType cubeTensorPromoteType)
{
    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_HIFLOAT8:
        case DataType::DT_FLOAT8_E4M3FN:
            OP_LOGW(
                "Cube cannot support dtype: %s when cubeMathType is USE_HF32.",
                op::ToString(cubeTensorPromoteType).GetString());
            return cubeTensorPromoteType;
        case DataType::DT_FLOAT:
            return DataType::DT_FLOAT;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, HIF8, FP8_E4M3FN, but got %s.",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_UNDEFINED;
    }
}

DataType CalcAllowFp32DownPrecisionPromoteType(const DataType cubeTensorPromoteType)
{
    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
            return cubeTensorPromoteType;
        case DataType::DT_FLOAT:
            if (IsCubeSupportHf32()) {
                return DataType::DT_FLOAT;
            }
            OP_LOGD("Cube cannot support dtype: HF32 in this soc version when cubeMathType is "
                    "ALLOW_FP32_DOWN_PRECISION, using FP16 instead.");
            return DataType::DT_FLOAT16;
        case DataType::DT_HIFLOAT8:
        case DataType::DT_FLOAT8_E4M3FN:
            OP_LOGW(
                "Cube cannot support dtype: %s when cubeMathType is ALLOW_FP32_DOWN_PRECISION.",
                op::ToString(cubeTensorPromoteType).GetString());
            return cubeTensorPromoteType;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, HIF8, FP8_E4M3FN, but got %s.",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_UNDEFINED;
    }
}

DataType CalcKeepDtypePromoteType(const DataType cubeTensorPromoteType)
{
    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_FLOAT:
        case DataType::DT_HIFLOAT8:
        case DataType::DT_FLOAT8_E4M3FN:
            return cubeTensorPromoteType;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, HIF8, FP8_E4M3FN, but got %s.",
                op::ToString(cubeTensorPromoteType).GetString());
            return cubeTensorPromoteType;
    }
}

DataType CalcForceGrpAccForFp32PromoteType(const DataType cubeTensorPromoteType)
{
    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT:
            return cubeTensorPromoteType;
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Cube only support FP32, but got %s.",
                op::ToString(cubeTensorPromoteType).GetString());
            return cubeTensorPromoteType;
    }
}

// 根据promote type + cubemathtype的组合算出最终算子应用的dtype
DataType CalcPromoteTypeCubeMathTypeNew(const DataType cubeTensorPromoteType, int8_t cubeMathType)
{
    // USE_FP16场景，无论什么dtype + 芯片，均用FP16进行计算
    if (cubeMathType == USE_FP16) {
        return CalcUseFp16PromoteType(cubeTensorPromoteType);
    } else if (cubeMathType == USE_HF32) {
        return CalcUseHf32PromoteType(cubeTensorPromoteType);
    } else if (cubeMathType == ALLOW_FP32_DOWN_PRECISION) {
        return CalcAllowFp32DownPrecisionPromoteType(cubeTensorPromoteType);
    } else if (cubeMathType == KEEP_DTYPE) {
        return CalcKeepDtypePromoteType(cubeTensorPromoteType);
    } else if (cubeMathType == USE_FP32_ADD) {
        return CalcForceGrpAccForFp32PromoteType(cubeTensorPromoteType);
    }
    OP_LOGW("The cubeMathType: %d can't be matched.", static_cast<int32_t>(cubeMathType));
    return cubeTensorPromoteType;
}

// 根据promoteType + cubeMathType 判断是否要走HF32分支
bool NeedCubeGoHF32(const DataType cubeTensorPromoteType, int8_t cubeMathType)
{
    // USE_HF32场景，如果promoteType为BF16或FP16时，提示不支持该选项
    if (cubeMathType == USE_HF32) {
        if (cubeTensorPromoteType == DataType::DT_BF16) {
            OP_LOGW("The cubeMathType can't be set to USE_HF32 when the dtype is BF16.");
        }
        if (cubeTensorPromoteType == DataType::DT_FLOAT16) {
            OP_LOGW("The cubeMathType can't be set to USE_HF32 when the dtype is FP16.");
        }
    }

    // Ascend910B + promoteType为FP32 + ALLOW_DOWN / USE_HF32 才需要走HF32分支
    if (IsCubeSupportHf32() && (cubeTensorPromoteType == DataType::DT_FLOAT) &&
        (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_HF32)) {
        return true;
    }
    return false;
}

} // namespace NN
} // namespace Ops