/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_DEV_TESTS_UT_COMMON_OP_IMPL_REGISTRY_H
#define OPS_NN_DEV_TESTS_UT_COMMON_OP_IMPL_REGISTRY_H

#include "base/registry/op_impl_space_registry_v2.h"

// gert::OpImplRegistry::GetInstance().GetOpImpl("WeightQuantBatchMatmulV2")->infer_shape
// 适配GE接口变化，对测试用例代码尽可能屏蔽差异

namespace Ops {
namespace NN {
class OpImplRegistry {
public:
    ~OpImplRegistry() = default;

    static OpImplRegistry& GetInstance()
    {
        static OpImplRegistry instance;
        return instance;
    }

    const gert::OpImplKernelRegistry::OpImplFunctions* GetOpImpl(const std::string& opType)
    {
        auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        if (registry == nullptr) {
            return nullptr;
        }
        return registry->GetOpImpl(opType.c_str());
    }

private:
    OpImplRegistry() = default;
};
} // namespace NN
} // namespace Ops

namespace gert {
    using OpImplRegistry = Ops::NN::OpImplRegistry;
}

#endif  // OPS_NN_DEV_TESTS_UT_COMMON_OP_IMPL_REGISTRY_H