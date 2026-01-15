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
 * \file lock.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_LOCK_H_
#define CANN_OPS_BUILT_IN_LOCK_H_

#include <shared_mutex>

namespace Ops {
namespace NN {
namespace optiling {
class RWLock {
public:
    RWLock() = default;
    ~RWLock() = default;
    RWLock(const RWLock &) = delete;
    RWLock(RWLock &&) = delete;
    RWLock &operator=(const RWLock &) = delete;
    RWLock &operator=(RWLock &&) = delete;

    void rdlock();

    void wrlock();

    void unlock();

private:
    std::shared_mutex _mtx;
};
} // namespace optiling
} // namespace NN
} // namespace Ops
#endif // CANN_OPS_BUILT_IN_LOCK_H_