# Copyright (c) 2025 Huawei Technologies Co., Ltd.

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(matmul_add "ascend910B1" "" )
endif()
