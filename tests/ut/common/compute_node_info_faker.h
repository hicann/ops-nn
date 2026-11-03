/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_TESTS_UT_COMMON_COMPUTE_NODE_INFO_FAKER_H_
#define OPS_NN_TESTS_UT_COMMON_COMPUTE_NODE_INFO_FAKER_H_

#include <cstdint>
#include <vector>

#include "graph/types.h"
#include "base/context_builder/op_kernel_run_context_builder.h"

namespace gert {
/**
 * 通过公开的 OpKernelContextBuilder 构造 ComputeNodeInfo(含 RuntimeAttrs 属性区),
 * 供 fallback 类 UT 使用: 属性由库按 AppendAttr 顺序写入, 测试不再手写 RuntimeAttrs
 * 的私有内存布局, 也不重定义库符号 gert::RuntimeAttrs::GetPointerByIndex(ODR 违规,
 * 依赖 --allow-multiple-definition 才能链接)。
 * 未 AppendAttr 的属性 index, 运行期 GetBool/GetInt 等按真实库语义返回 nullptr,
 * 可用于覆盖"属性缺失走默认值/报错"的分支。
 */
class ComputeNodeInfoFaker {
public:
    ComputeNodeInfoFaker() = default;
    ~ComputeNodeInfoFaker() = default;
    ComputeNodeInfoFaker(const ComputeNodeInfoFaker&) = delete;
    ComputeNodeInfoFaker& operator=(const ComputeNodeInfoFaker&) = delete;

    ComputeNodeInfoFaker& NodeType(const char* const nodeType)
    {
        builder_.OpType(nodeType).OpName(nodeType);
        return *this;
    }

    // 每个 IR 输入/输出的实例化个数(可选输入未实例化时为 0), 顺序与 IR 原型一致
    ComputeNodeInfoFaker& IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                        const std::vector<uint32_t>& outputInstanceNum)
    {
        inputInstanceNum_ = inputInstanceNum;
        builder_.IOInstanceNum(inputInstanceNum, outputInstanceNum);
        return *this;
    }

    // 设置 IR 原型第 irIndex 个输入的 TensorDesc(该输入须已实例化, 内部换算为实例下标);
    // 须在 IrInstanceNum 之后调用
    ComputeNodeInfoFaker& InputTd(const size_t irIndex, const ge::DataType dtype, const ge::Format originFormat,
                                  const ge::Format storageFormat)
    {
        size_t instanceIndex = 0;
        for (size_t i = 0; i < irIndex && i < inputInstanceNum_.size(); ++i) {
            instanceIndex += inputInstanceNum_[i];
        }
        builder_.InputTensorDesc(instanceIndex, dtype, originFormat, storageFormat);
        return *this;
    }

    // 设置第 index 个实例输出的 TensorDesc
    ComputeNodeInfoFaker& OutputTd(const size_t index, const ge::DataType dtype, const ge::Format originFormat,
                                   const ge::Format storageFormat)
    {
        builder_.OutputTensorDesc(index, dtype, originFormat, storageFormat);
        return *this;
    }

    // 追加 bool 属性, 顺序即属性 index
    ComputeNodeInfoFaker& AppendAttr(const bool attr)
    {
        builder_.AppendAttr(attr);
        return *this;
    }

    // 追加 int64 属性, 顺序即属性 index
    ComputeNodeInfoFaker& AppendAttr(const int64_t attr)
    {
        builder_.AppendAttr(attr);
        return *this;
    }

    // 构建 ComputeNodeInfo 并返回其地址; 资源由本对象持有, 生命周期须覆盖使用方
    const void* Build()
    {
        holder_ = builder_.Build();
        return holder_.GetContext()->GetContext()->compute_node_info;
    }

private:
    OpKernelContextBuilder builder_;
    ContextHolder<KernelContext> holder_;
    std::vector<uint32_t> inputInstanceNum_;
};
} // namespace gert
#endif // OPS_NN_TESTS_UT_COMMON_COMPUTE_NODE_INFO_FAKER_H_
