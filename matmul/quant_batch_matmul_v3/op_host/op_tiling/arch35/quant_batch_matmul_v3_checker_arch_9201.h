/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_checker_arch_9201.h
 * \brief
 */
#pragma once
#include "quant_batch_matmul_v3_checker.h"

namespace optiling {

class QuantBatchMatmulV3CheckerArch9201 : public QuantBatchMatmulV3Checker {
public:
    QuantBatchMatmulV3CheckerArch9201(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams)
        : QuantBatchMatmulV3Checker(context, inputParams)
    {}

    ~QuantBatchMatmulV3CheckerArch9201() override = default;

protected:
    const std::vector<ge::DataType>& GetLegalInputDtypes() const override;
    const std::vector<ge::DataType>& GetLegalOutputDtypes() const override;
    const char* GetLegalInputDtypesDesc() const override;
    const char* GetLegalOutputDtypesDesc() const override;
    bool IsFp4Input() const override;
    bool IsHif4Input() const override;
    bool CheckExtraABDtypePairs() const override;
    bool CheckHif4ScaleDtypeWithPertoken() const override;
    bool CheckExtraOutputDtype() const override;
    bool CheckExtraBiasConstraint(const gert::Shape& scaleShape,
                                  const gert::StorageShape* pertokenShape) const override;
    bool CheckExtraShapeValid(const gert::Shape& scaleShape, const gert::StorageShape* pertokenShape,
                              const std::vector<int64_t>& dimValueOfMKN) const override;

private:
    bool Hif4PertokenScaleShapeCheck(const gert::StorageShape* pertokenShape) const;
    bool Hif4ScaleShapeCheck(const gert::Shape& scaleShape) const;
    bool CheckInputValidInHif4PerGroupMode(const gert::Shape& scaleShape, const gert::StorageShape* pertokenShape,
                                           const std::vector<int64_t>& dimValueOfMKN) const;
};
} // namespace optiling
