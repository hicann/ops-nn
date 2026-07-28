# HingeLoss

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：计算逐元素 Hinge Loss 分类损失，用于 SVM 等间隔分类训练场景，并与 HingeLossGrad 配套完成反向传播。

- 计算公式：

对于每个元素 i，模型预测值为 predict_i、标签为 target_i，输出损失为：

$$
loss_i = max(0, 1 - target_i * predict_i)
$$


## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| predict | 输入 | 模型预测值，对应公式中的 `predict`。 | FLOAT、FLOAT16、BF16 | ND | 1 至 8 维；与 target、loss 一致。 |
| target | 输入 | 标签输入，对应公式中的 `target`；通常取值为 1 或 -1。 | 与 predict 保持一致 | ND | 1 至 8 维；必须与 predict 完全一致。 |
| loss | 输出 | 逐元素 Hinge Loss 结果，对应公式中的 `loss`。 | 与 predict 保持一致 | ND | 与 predict 一致。 |

## 约束说明

- predict 和 target 的 rank、各维度大小必须完全一致；首版不支持广播。
- 目标值通常使用 1 或 -1；算子不额外校验 target 数值。
- 输出为逐元素结果，不执行 mean 或 sum reduction。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| ACLNN API | [test_aclnn_hinge_loss.cpp](examples/test_aclnn_hinge_loss.cpp) | 通过 `aclnnHingeLossGetWorkspaceSize` 与 `aclnnHingeLoss` 执行。接口说明见 [aclnnHingeLoss.md](docs/aclnnHingeLoss.md)。 |
| GE 图模式 | - | 当前实验算子通过 ACLNN 接口调用。 |
| PyTorch API | - | 当前未提供 PyTorch Extension。 |

## 本地编译运行 UT

环境准备请参考项目环境部署文档。Atlas A2 系列产品使用 `ascend910b`；其他可选 SoC 为 `ascend910_93` 和 `ascend950`。

```bash
# Host UT：注册、shape/type 推导和 tiling
bash build.sh -u --experimental --ophost --ops=hinge_loss --soc=ascend910b

# Kernel UT：逐元素公式、边界值和 golden 对比
bash build.sh -u --experimental --opkernel --ops=hinge_loss --soc=ascend910b
```

当前未提供独立的 ACLNN API UT；ACLNN 调用通过示例用例验证。

构建、安装并执行 eager 示例：

```bash
cd /home/ma-user/work/ops-nn
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 构建 experimental 下的自定义算子包
bash build.sh --pkg --ops=hinge_loss --soc=ascend910b -j16 --experimental

# 安装默认 vendor_name=custom 的自定义算子包
./build_out/cann-ops-nn-custom_linux-aarch64.run

# 运行 ACLNN eager 示例
bash build.sh --experimental --run_example hinge_loss eager cust \
    --example_name=hinge_loss --vendor_name=custom --soc=ascend910b
```

执行前需确保自定义包已安装到 `ASCEND_HOME_PATH/opp/vendors`，且 `vendor_name` 与打包时保持一致。若打包时显式指定了 `--vendor_name`，需同时替换 run 包名称和 eager 命令中的 `--vendor_name`。

## 本地自测 UT 覆盖率

安装 `lcov` 后，在上述命令末尾增加 `--cov`：

```bash
bash build.sh -u --experimental --ophost --ops=hinge_loss --soc=ascend910b --cov
bash build.sh -u --experimental --opkernel --ops=hinge_loss --soc=ascend910b --cov
```

## 参考资源

- Kernel 实现：`op_kernel/hinge_loss.h`
- Host tiling 实现：`op_host/hinge_loss_tiling.cpp`
- Kernel UT golden 数据：`tests/ut/op_kernel/hinge_loss_data/gen_data.py`
