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
 * \file test_rms_norm_common_proto.h
 * \brief
 */

#ifndef TEST_PROTO_RMS_NORM_COMMON_H
#define TEST_PROTO_RMS_NORM_COMMON_H

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"

class RmsNormAtb : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormAtb Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormAtb Proto Test TearDown" << std::endl;
    }
};

#endif
