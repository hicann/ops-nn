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
 * \file add_example_tiling.cpp
 * \brief Add算子的tiling(分块)策略实现
 * 
 * 本文件提供了tiling逻辑，将大型计算任务划分为较小的块，以便在AI Core上高效并行执行。
 * Tiling对于最大化NPU利用率和性能至关重要。
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "examples/add_example/op_kernel/add_example_tiling_data.h"
#include "examples/add_example/op_kernel/add_example_tiling_key.h"

namespace optiling {

// 矢量系统缓冲区大小为0
constexpr uint32_t WS_SYS_SIZE = 0U;
// 输入shape的维度限制为4
constexpr int32_t DIMS_LIMIT = 4;

// 维度索引常量定义
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;

// 缓冲区数量常量
constexpr int64_t BUFFER_NUM = 6;
// 类型大小常量(字节)
constexpr int64_t TYPE_SIZE = 4;
// 标量形状常量{1}
static const gert::Shape g_vec_1_shape = {1};

// Add编译信息结构
struct AddExampleCompileInfo {
};

/*!
 * \brief 确保标量shape转换为{1}以保持一致性
 * 
 * 对于标量输入，需要将其转换为1维张量以确保处理的一致性。
 * 
 * \param in_shape 要检查的输入shape
 * \return 如果输入是标量则返回{1}，否则返回原始shape的引用
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

/*!
 * \brief 获取平台信息，包括UB(统一缓冲区)大小和AI Core数量
 * 
 * 该函数查询平台以获取硬件资源信息，这对于确定最优tiling策略至关重要。
 * 
 * \param context 指向tiling上下文的指针
 * \param ubSize 输出参数：UB大小(字节)
 * \param coreNum 输出参数：可用的AI Core数量
 * \return 如果成功获取平台信息则返回ge::GRAPH_SUCCESS
 */
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取平台信息指针
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    
    // 创建AscendC平台对象
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    
    // 获取AI Core数量
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    
    // 获取UB(统一缓冲区)大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 从输入输出张量获取shape和属性信息
 * 
 * 该函数检索并验证张量shape和数据类型。它确保：
 * - 所有张量都有4维shape
 * - 数据类型是支持的(FLOAT或INT32)
 * - 计算总元素数量
 * 
 * \param context 指向tiling上下文的指针
 * \param totalIdx 输出参数：要处理的元素总数
 * \param dataType 输出参数：输入张量的数据类型
 * \return 如果成功检索和验证shape信息则返回ge::GRAPH_SUCCESS
 */
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入x的shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    
    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    
    // 获取输入y的shape信息
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());
    
    // 获取输出z的shape信息
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());

    // shape校验：确保所有张量都是4维
    OP_CHECK_IF(
        inputShapeX.GetDimNum() != DIMS_LIMIT || inputShapeY.GetDimNum() != DIMS_LIMIT ||
            outShapeZ.GetDimNum() != DIMS_LIMIT,
        OP_LOGE(
            context, "AddExample: inputx,inputy,outputz shape dim = %zu, %zu, %zu, should be equal 4",
            inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
        return ge::GRAPH_FAILED);

    // 获取shape的维度值
    totalIdx = inputShapeX.GetShapeSize();
    
    // 数据类型校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 获取算子执行所需的工作空间大小
 * 
 * 工作空间是计算期间使用的额外内存。对于Add算子，
 * 不需要工作空间，因此大小设置为0。
 * 
 * \param context 指向tiling上下文的指针
 * \return
 */
ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    // 获取工作空间大小数组
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief Add算子的主tiling函数
 * 
 * 该函数计算最优的tiling策略：
 * 1. 检索平台信息(UB大小、AI Core数量)
 * 2. 检索shape和属性信息
 * 3. 计算Core tiling：将工作划分到多个AI Core以实现并行
 * 4. 计算UB tiling：根据可用的UB内存在每个Core内划分工作
 * 5. 根据数据类型设置tiling key(FLOAT和INT32有不同的tiling策略)
 * 
 * Tiling数据存储在AddExampleTilingData结构中，将被传递给kernel。
 * 
 * \param context 指向tiling上下文的指针
 * \return 如果成功计算tiling策略则返回ge::GRAPH_SUCCESS
 */
static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    
    // 2、获取shape、属性信息
    int64_t totalIdx;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), 
        return ge::GRAPH_FAILED);
    
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    AddExampleTilingData* tiling = context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    
    // 初始化tiling数据为0
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AddExampleTilingData), 0, sizeof(AddExampleTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), 
        return ge::GRAPH_FAILED);
    
    // 优先做核切分，尽量用更多的核并行计算
    // 计算每个AI Core处理的元素数量
    tiling->totalNum = totalIdx;
    tiling->blockFactor = Ops::Base::CeilDiv(totalIdx, coreNum);
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
    
    // 计算UB切分，考虑最大同时使用的UB tensor数量，计算单个UB tensor可用大小
    // 当前场景：2个输入和1个输出，考虑使用双缓冲，共需要6块UB tensor
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->ubFactor = Ops::Base::FloorAlign(Ops::Base::FloorDiv((ubCanUse / TYPE_SIZE), BUFFER_NUM), ubBlockSize);

    // 设置使用的AI Core数量
    context->SetBlockDim(usedCoreNum);
    
    // 根据数据类型设置tiling key
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        // 浮点数类型使用mode 0的tiling key
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_INT32) {
        // 32位整数类型使用mode 1的tiling key
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

/*!
 * \brief 解析Add算子的tiling信息
 * 
 * 该函数在模型加载期间被调用，以解析任何静态tiling信息。
 * 对于Add算子，不需要静态tiling，因此直接返回成功。
 * 
 * \param context 指向tiling解析上下文的指针
 * \return ge::GRAPH_SUCCESS
 */
static ge::graphStatus TilingParseForAddExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
// 将tiling函数、tiling解析函数和编译信息注册到系统中
IMPL_OP_OPTILING(AddExample).Tiling(AddExampleTilingFunc).TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);
} // namespace optiling
