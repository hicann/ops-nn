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
 * \file batch_norm_v3_tiling_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "batch_norm_v3_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_FULL_REDUCE = 200000;
constexpr int64_t TILINGKEY_RA_FULL_REDUCE = 400000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t SMALL_BUFFER_NUM = 8;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 6;
constexpr int64_t SMALL_BUFFER_NUM_T = 2;
constexpr int64_t LARGE_BUFFER_NUM = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;
constexpr int64_t NHWC_DIM_NUM = 4;
constexpr int64_t NDHWC_DIM_NUM = 5;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t RA_BINARY_ADD_THRESHOLD = 8;
constexpr int64_t CHANGE_TO_WELFORD_THRESHOLD = 64;
constexpr float DEFAULT_EPSILON = 1e-5;
constexpr float DEFAULT_MOMENTUM = 0.1;
constexpr int64_t RUNNING_PARAM_NUM = 2;
constexpr int64_t SAVE_PARAM_NUM = 2;
constexpr int64_t BIAS_INDEX = 2;
constexpr int64_t RUNNING_MEAN_INDEX = 3;
constexpr int64_t RUNNING_VAR_INDEX = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;

constexpr int64_t INPUT_NUM = 5;
constexpr int64_t OUTPUT_NUM = 5;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT};
} // namespace

namespace optiling {
ge::graphStatus BatchNormV3RegbaseTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const BatchNormV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"), return ge::GRAPH_FAILED);
    vl = compileInfoPtr->vectorLength / sizeof(float);
    blockSize = compileInfoPtr->blockSize;

    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.numBlocks = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormV3", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(0);
    epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const float* momentumPtr = attrs->GetFloat(1);
    momentum = (momentumPtr == nullptr) ? DEFAULT_MOMENTUM : *momentumPtr;
    const bool* isTrainingPtr = attrs->GetBool(2);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    if (!isTraining) {
        OP_LOGI(context_, "This node not support infer.");
        return ge::GRAPH_PARAM_INVALID;
    }

    // 获取输入shape
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    if (CheckShapeAllPositive(xStorageShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported shape info.");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(0);
    auto weightDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    dataType = xDesc->GetDataType();

    weightDataType = weightDesc->GetDataType();
    format = xDesc->GetFormat().GetStorageFormat();
    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCHW_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 4 with NCHW format."), return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0);
        a = xStorageShape.GetDim(DIM_1);
        r0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCDHW_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 5 with NCDHW format."), return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0);
        a = xStorageShape.GetDim(DIM_1);
        r0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_NHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NHWC_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 4 with NHWC format."), return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2);
        a = xStorageShape.GetDim(DIM_3);
        r0 = 0;
    } else if (format == FORMAT_NDHWC) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NDHWC_DIM_NUM,
            OP_LOGE(context_->GetNodeName(), "Dims should be 5 with NDHWC format."), return ge::GRAPH_FAILED);
        r1 = xStorageShape.GetDim(DIM_0) * xStorageShape.GetDim(DIM_1) * xStorageShape.GetDim(DIM_2) *
             xStorageShape.GetDim(DIM_3);
        a = xStorageShape.GetDim(DIM_4);
        r0 = 0;
    } else {
        OP_LOGE(context_->GetNodeName(), "Not supported format.");
        return ge::GRAPH_FAILED;
    }

    if (CheckInputValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported input info.");
        return ge::GRAPH_FAILED;
    }

    if (CheckOutputValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported output info.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckOneInputShape(int64_t idx)
{
    auto xShape = context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    OP_CHECK_IF(
        xStorageShape.GetDimNum() != 1, OP_LOGE(context_->GetNodeName(), "Dims of input %ld should be 1.", idx),
        return ge::GRAPH_FAILED);
    int64_t actualA = xStorageShape.GetDim(DIM_0);
    OP_CHECK_IF(
        a != actualA,
        OP_LOGE(context_->GetNodeName(), "Shape of input %ld expect %ld, but actual %ld.", idx, a, actualA),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckOneOutputShape(int64_t idx)
{
    auto outputShape = context_->GetOutputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputStorageShape = outputShape->GetStorageShape();
    OP_CHECK_IF(
        outputStorageShape.GetDimNum() != 1, OP_LOGE(context_->GetNodeName(), "Dims of output %ld should be 1.", idx),
        return ge::GRAPH_FAILED);
    int64_t actualA = outputStorageShape.GetDim(DIM_0);
    OP_CHECK_IF(
        a != actualA,
        OP_LOGE(context_->GetNodeName(), "Shape of output %ld expect %ld, but actual %ld.", idx, a, actualA),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckInputValid()
{
    for (int64_t i = 1; i < INPUT_NUM; i++) {
        if (CheckOneInputShape(i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(
        std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), dataType) == DTYPE_LIST.end(),
        OP_LOGE(context_->GetNodeName(), "Unsupported dtype %s for input 0.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str()), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        // 支持weight和x的混合数据类型
        (weightDataType != dataType) && (weightDataType != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "Dtype of input weight expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(weightDataType).c_str()),
        return ge::GRAPH_FAILED);

    auto biasDesc = context_->GetInputDesc(BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    ge::DataType biasDataType = biasDesc->GetDataType();
    OP_CHECK_IF(
        (biasDataType != weightDataType),
        OP_LOGE(context_->GetNodeName(), "Dtype of input bias expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(weightDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(biasDataType).c_str()), return ge::GRAPH_FAILED);

    auto runningMeanDesc = context_->GetInputDesc(RUNNING_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanDesc);
    ge::DataType runningMeanDataType = runningMeanDesc->GetDataType();
    OP_CHECK_IF(
        // 支持runningMean和x的混合数据类型
        (runningMeanDataType != dataType) && (runningMeanDataType != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "Dtype of runningMean expect %s, but actual %s.",
            ge::TypeUtils::DataTypeToSerialString(dataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(runningMeanDataType).c_str()), return ge::GRAPH_FAILED);
    
    auto runningVarDesc = context_->GetInputDesc(RUNNING_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningVarDesc);
    ge::DataType runningVarDataType = runningVarDesc->GetDataType();
    OP_CHECK_IF(
        runningMeanDataType != runningVarDataType,
        OP_LOGE(context_->GetNodeName(), "Dtype of runningMean is %s, not equal with runningVar's Dtype %s.",
            ge::TypeUtils::DataTypeToSerialString(runningMeanDataType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(runningVarDataType).c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckOutputValid()
{
    if (CheckOutputDtypeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported output datatype info.");
        return ge::GRAPH_FAILED;
    }

    if (CheckOutputShapeValid() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported output shape info.");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckOutputDtypeValid()
{
    // Step1：校验y的类型与x是否一致
    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDataType = yDesc->GetDataType();
    OP_CHECK_IF(
        dataType != yDataType,
        OP_LOGE(context_->GetNodeName(), "Output y dtype %s is not same as input x dtype %s.",
                ge::TypeUtils::DataTypeToSerialString(yDataType).c_str(),
                ge::TypeUtils::DataTypeToSerialString(dataType).c_str()),
        return ge::GRAPH_FAILED);
    
    // Step2：校验输出runningMean/Var参数的类型与输入runningMean是否一致
    auto runningMeanDesc = context_->GetInputDesc(RUNNING_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, runningMeanDesc);
    ge::DataType runningMeanDataType = runningMeanDesc->GetDataType();

    for (int64_t i = 1; i < (OUTPUT_NUM - SAVE_PARAM_NUM); i++) {
        auto xDesc = context_->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
        ge::DataType subDataType = xDesc->GetDataType();
        OP_CHECK_IF(
            // 校验输出runningMean/Var参数类型是否与输入一致
            subDataType != runningMeanDataType,
            OP_LOGE(
                context_->GetNodeName(), "Dtype of output %ld expect %s, but actual %s.", i,
                ge::TypeUtils::DataTypeToSerialString(runningMeanDataType).c_str(),
                ge::TypeUtils::DataTypeToSerialString(subDataType).c_str()),
            return ge::GRAPH_FAILED);
    }

    // Step3：校验输出saveMean/saveRstd参数类型是否是float32
    for (int64_t i = DIM_3; i < (DIM_3 + SAVE_PARAM_NUM); i++) {
        auto xDesc = context_->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
        ge::DataType subDataType = xDesc->GetDataType();
        OP_CHECK_IF(
            subDataType != ge::DT_FLOAT,
            OP_LOGE(
                context_->GetNodeName(), "Dtype of output %ld expect %s, but actual %s.", i,
                ge::TypeUtils::DataTypeToSerialString(ge::DT_FLOAT).c_str(),
                ge::TypeUtils::DataTypeToSerialString(subDataType).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckOutputShapeValid()
{
    // Step1：校验输出y的shape、format与x是否一致
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xStorageShape = xShape->GetStorageShape();
    int64_t xShapeSize = xStorageShape.GetDimNum();

    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = yShape->GetStorageShape();
    int64_t yShapeSize = yStorageShape.GetDimNum();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::Format yFormat = yDesc->GetFormat().GetStorageFormat();

    OP_CHECK_IF(
        (xShapeSize != yShapeSize),
        OP_LOGE(
            context_->GetNodeName(), "Input X dim size [%ld] is not equal to Output Y dim size [%ld]",
            xShapeSize, yShapeSize), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (format != yFormat),
        OP_LOGE(
            context_->GetNodeName(), "Input X format [%s] does not match Output Y format [%s]",
            ge::TypeUtils::FormatToAscendString(format).GetString(), ge::TypeUtils::FormatToAscendString(yFormat).GetString()),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < xShapeSize; i++) {
        OP_CHECK_IF((xStorageShape.GetDim(i) != yStorageShape.GetDim(i)),
            OP_LOGE(
                context_->GetNodeName(),
                "Output Y dim [%ld] is [%ld] and Input X dim [%ld] is [%ld] should be same", i,
                yStorageShape.GetDim(i), i, xStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    OP_LOGI(context_->GetNodeName(), "CheckXYShapeValid success.");
    
    // Step2：校验其他输出的shape
    for (int64_t i = 1; i < OUTPUT_NUM; i++) {
        if (CheckOneOutputShape(i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGI(context_->GetNodeName(), "CheckOutputShapeValid success.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input expect be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RegbaseTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

class BatchNormV3FullReduceTilingBase : public BatchNormV3RegbaseTilingBase {
public:
    explicit BatchNormV3FullReduceTilingBase(gert::TilingContext* context) : BatchNormV3RegbaseTilingBase(context)
    {}
    ~BatchNormV3FullReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormV3RegbaseTilingBase::Reset(context);
        blockNum = 0;
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override
    {
        if (format != FORMAT_NCHW && format != FORMAT_NCDHW) {
            return false;
        }
        int64_t elemSize = FP32_BYTE;
        int64_t weightElemSize = FP32_BYTE;
        if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
            elemSize = FP16_BYTE;
        }
        if (weightDataType == ge::DT_FLOAT16 || weightDataType == ge::DT_BF16) {
            weightElemSize = FP16_BYTE;
        }
        int64_t r1r0 = r0 * r1;
        binaryAddQuotient = vl;
        while (binaryAddQuotient < r1r0) {
            binaryAddQuotient *= BINARY_ADD_COEF;
        }
        binaryAddQuotient /= BINARY_ADD_COEF;
        int64_t quotientVcaddNum = binaryAddQuotient / vl;
        int64_t quotientVcaddSizeAlign = ((quotientVcaddNum * FP32_BYTE + blockSize - 1) / blockSize) * blockSize;
        if (static_cast<uint64_t>(quotientVcaddSizeAlign) >= aicoreParams_.ubSize) {
            return false;
        }
        // reserve 8 block for 8 A tensor alignment
        int64_t ubCanUseSize =
            ((((aicoreParams_.ubSize - quotientVcaddSizeAlign) / DOUBLE_BUFFER) / blockSize) * blockSize);
        if (SMALL_BUFFER_NUM * blockSize >= ubCanUseSize) {
            return false;
        }
        ubCanUseSize -= SMALL_BUFFER_NUM * blockSize;
        int64_t r1r0Align = (((r1r0 * elemSize + blockSize - 1) / blockSize) * blockSize) / elemSize;
        // two AR tensor, two A tensor, six fp32 A tensor
        int64_t ubSizePerA =
            LARGE_BUFFER_NUM * r1r0Align * elemSize + SMALL_BUFFER_NUM_T * weightElemSize + SMALL_BUFFER_NUM_FP32 * FP32_BYTE;
        int64_t aFactor = ubCanUseSize / ubSizePerA;
        if (aFactor >= 1) {
            batchNormV3TilingData.set_aFactor(aFactor);
            batchNormV3TilingData.set_binaryAddQuotient(binaryAddQuotient);
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    int64_t blockNum;
    int64_t binaryAddQuotient;
    BatchNormV3FullReduceRegbaseTilingData batchNormV3TilingData;
};

ge::graphStatus BatchNormV3FullReduceTilingBase::DoOpTiling()
{
    // dim
    batchNormV3TilingData.set_r1(r1);
    batchNormV3TilingData.set_a(a);
    batchNormV3TilingData.set_r0(r0);
    int64_t rDim = r1 * r0;
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < rDim) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormV3TilingData.set_powerOfTwoForR(powerOfTwoForR);

    // attr
    batchNormV3TilingData.set_epsilon(epsilon);
    batchNormV3TilingData.set_momentum(momentum);

    // core num
    int64_t blockFactor = (a + aicoreParams_.numBlocks - 1) / aicoreParams_.numBlocks;
    blockNum = (a + blockFactor - 1) / blockFactor;
    batchNormV3TilingData.set_aBlockFactor(blockFactor);
    batchNormV3TilingData.set_blockNum(blockNum);

    // vf loop count
    int64_t r1r0LoopCount = ((r1 * r0) + vl - 1) / vl;
    batchNormV3TilingData.set_r1r0LoopCount(r1r0LoopCount);

    // binary add k
    int64_t vcaddNum = binaryAddQuotient / vl; // 2的幂次方的数据要做二分
    if (vcaddNum <= vl) {
        batchNormV3TilingData.set_binaryAddK(0);
        batchNormV3TilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vl; // vl为一块，要累加的块，当前肯定是2的幂次方
        int64_t binaryAddK = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < binaryAddNum) {
            binaryAddK++;
            curBinaryAddNum *= BINARY_ADD_COEF;
        }
        batchNormV3TilingData.set_binaryAddK(binaryAddK);
        batchNormV3TilingData.set_binaryAddLastNum(vl);
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormV3FullReduceTilingBase::GetTilingKey() const
{
    return TILINGKEY_FULL_REDUCE;
}

ge::graphStatus BatchNormV3FullReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(blockNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    batchNormV3TilingData.SaveToBuffer(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(batchNormV3TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

class BatchNormV3RAFullReduceTilingBase : public BatchNormV3RegbaseTilingBase {
public:
    explicit BatchNormV3RAFullReduceTilingBase(gert::TilingContext* context) : BatchNormV3RegbaseTilingBase(context)
    {}
    ~BatchNormV3RAFullReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormV3RegbaseTilingBase::Reset(context);
        blockNum = 0;
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override
    {
        if (format != FORMAT_NHWC && format != FORMAT_NDHWC) {
            // NCHW和NCDHW场景中，如果r0是1，也放在RA模版处理
            if (r0 != 1) {
                return false;
            }
        }
        int64_t elemSize = FP32_BYTE;
        if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
            elemSize = FP16_BYTE;
        }
        int64_t weightElemSize = FP32_BYTE;
        if (weightDataType == ge::DT_FLOAT16 || weightDataType == ge::DT_BF16) {
            weightElemSize = FP16_BYTE;
        }

        int64_t ubCanUseSize = (((aicoreParams_.ubSize / DOUBLE_BUFFER) / blockSize) * blockSize);
        int64_t ubSizePerA =
            (LARGE_BUFFER_NUM * r1 + 1) * elemSize + SMALL_BUFFER_NUM_T * weightElemSize + SMALL_BUFFER_NUM_FP32 * FP32_BYTE;
        if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
            ubSizePerA = LARGE_BUFFER_NUM * r1 * elemSize + (r1 + 1) * FP32_BYTE + SMALL_BUFFER_NUM_T * elemSize +
                         SMALL_BUFFER_NUM_FP32 * FP32_BYTE;
        }
        int64_t aFactor = ubCanUseSize / ubSizePerA;
        int64_t aFactorAlign = (((aFactor * elemSize) / blockSize) * blockSize) / elemSize;
        if (aFactorAlign >= 1) {
            batchNormV3TilingData.set_aFactor(aFactorAlign);
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus BinaryAddTiling();
    bool IsNeedChangeToWelford(int64_t elemSize);

private:
    int64_t blockNum;
    int64_t binaryAddQuotient;
    BatchNormV3RAFullReduceTilingData batchNormV3TilingData;
};

bool BatchNormV3RAFullReduceTilingBase::IsNeedChangeToWelford(int64_t elemSize)
{
    int64_t blockFactor = batchNormV3TilingData.get_aBlockFactor();
    int64_t ubFactor = batchNormV3TilingData.get_aFactor();
    int64_t blockFactorSize = blockFactor * elemSize;
    // 核内last轴大于64B时，如果ub内可以放下全部last轴或者计算带宽可以用满，那么无需切到welford模版
    return ((blockFactorSize >= CHANGE_TO_WELFORD_THRESHOLD) && (ubFactor < std::min(vl, blockFactor)));
}

ge::graphStatus BatchNormV3RAFullReduceTilingBase::BinaryAddTiling()
{
    int64_t binaryQuotient = RA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < r1) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    batchNormV3TilingData.set_binaryAddQuotient(binaryQuotient);
    int64_t binaryAddNum = binaryQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        batchNormV3TilingData.set_binaryAddK(binaryAddK);
        batchNormV3TilingData.set_binaryAddLast(0);
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        batchNormV3TilingData.set_binaryAddK(binaryAddK - 1);
        batchNormV3TilingData.set_binaryAddLast(1);
    } else {
        OP_LOGE(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3RAFullReduceTilingBase::DoOpTiling()
{
    // dim
    batchNormV3TilingData.set_r1(r1);
    batchNormV3TilingData.set_a(a);
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r1) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormV3TilingData.set_powerOfTwoForR(powerOfTwoForR);

    // attr
    batchNormV3TilingData.set_epsilon(epsilon);
    batchNormV3TilingData.set_momentum(momentum);

    // core num
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    int64_t theLeastAPerCore = blockSize / elemSize;
    int64_t blockFactor = (a + aicoreParams_.numBlocks - 1) / aicoreParams_.numBlocks;
    if (blockFactor < theLeastAPerCore) {
        blockFactor = theLeastAPerCore;
    }
    blockNum = (a + blockFactor - 1) / blockFactor;
    batchNormV3TilingData.set_aBlockFactor(blockFactor);
    batchNormV3TilingData.set_blockNum(blockNum);

    if (r1 <= RA_BINARY_ADD_THRESHOLD) {
        return ge::GRAPH_SUCCESS;
    }

    if (IsNeedChangeToWelford(elemSize)) {
        OP_LOGW(context_->GetNodeName(), "Change to welford tiling.");
        return ge::GRAPH_PARAM_INVALID;
    }

    return BinaryAddTiling();
}

uint64_t BatchNormV3RAFullReduceTilingBase::GetTilingKey() const
{
    return TILINGKEY_RA_FULL_REDUCE;
}

ge::graphStatus BatchNormV3RAFullReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(blockNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        batchNormV3TilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            batchNormV3TilingData.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    batchNormV3TilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(batchNormV3TilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3RAFullReduceTilingBase, 10000);
REGISTER_TILING_TEMPLATE("BatchNormV3", BatchNormV3FullReduceTilingBase, 20000);
} // namespace optiling