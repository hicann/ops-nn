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
 * \file qbmm_streamk_tiling_960.h
 * \brief StreamK tiling implementation for Ascend960DT/PR(DAV_9201/9202) QuantBatchMatmulV3,
 *        960DT/960PR(9201/9202)的StreamK量化场景统一走Blaze（复用QbmmMxStreamKBasicApiKernel），
 *        非StreamK场景由ASW MX BasicAPI模板分流（9201 Blaze，9202 CMCT）。
 *        tiling data复用QuantBatchMatmulV3StreamKBasicAPITilingData
 *        （与3510 Blaze StreamK共用布局），按SocVersion注册调度。
 */
#ifndef QBMM_STREAMK_TILING_960_H
#define QBMM_STREAMK_TILING_960_H

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"

namespace optiling {

class QBMMV3StreamKTiling960 : public QuantBatchMatmulV3TilingBase {
public:
    explicit QBMMV3StreamKTiling960(gert::TilingContext* context);
    QBMMV3StreamKTiling960(gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData* out);
    ~QBMMV3StreamKTiling960() override = default;

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData，mc2使用的直接接口
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    bool IsCapable() override;
    bool CheckDtype() const override;

private:
    bool IsPertensorStreamKInput() const;
    bool IsPostDequantBiasInput() const;

private:
    bool CheckShape(const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN) const override;
    void Reset();

    //  streamk 收益是否够好
    bool IsProfilingBetter();
    bool CalcBasicBlockV2();
    void CalL1Tiling();
    uint64_t GetBatchMode() const;
    uint64_t GetKernelType() const;
    uint64_t GetApiLevel(NpuArch npuArch) const;

    void SetTilingData();

    DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData& tilingData_;
};

} // namespace optiling

#endif // QBMM_STREAMK_TILING_960_H
