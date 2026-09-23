/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IN_TRAINING_UPDATE_V2_EMPTY_H
#define IN_TRAINING_UPDATE_V2_EMPTY_H

namespace INTrainingUpdateV2Ops {

class INTrainingUpdateV2Empty {
public:
    __aicore__ inline void Process() const {}
};

} // namespace INTrainingUpdateV2Ops

#endif // IN_TRAINING_UPDATE_V2_EMPTY_H
