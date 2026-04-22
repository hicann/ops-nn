/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "gtest/gtest.h"
#include "cube_utils/cube_fp16_t.h"

using namespace std;
namespace ops
{

class fp16_t_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "fp16_t_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "fp16_t_test TearDown" << std::endl;
    }
};

static void CheckVal(fp16_t a)
{
    std::cout << static_cast<float>(a);
}

TEST_F(fp16_t_test, fp16_t_test_1)
{
    fp16_t val = 8.1f;
    fp16_t val2 = 9.5f;
    fp16_t midVal = (uint16_t)35500;
    fp16_t largeVal = 65500.0f;
    int16_t int16Val = val.ToInt16();
    int16_t midInt16Val = midVal.ToInt16();
    int16_t largeInt16Val = largeVal.ToInt16();
    uint16_t uint16Val = val.ToUInt16();
    std::cout << midInt16Val << largeInt16Val << uint16Val << endl;
    fp16_t int16ToFp16 = (uint16_t)int16Val;
    CheckVal(int16ToFp16);
    fp16_t largeInt16ToFp16 = (uint16_t)largeInt16Val;
    CheckVal(largeInt16ToFp16);
    fp16_t uint16ToFp16 = uint16Val;
    CheckVal(uint16ToFp16);
    fp16_t resVal = val;
    resVal = largeVal;
}
}  // namespace ops
