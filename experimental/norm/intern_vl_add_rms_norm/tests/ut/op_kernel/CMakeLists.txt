# Copyright (c) 2026 Huawei Technologies Co., Ltd.

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(intern_vl_add_rms_norm "ascend910B1" "" )
endif()
