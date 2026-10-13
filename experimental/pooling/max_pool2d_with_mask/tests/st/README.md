# MaxPool2dWithMask 自测

精度和性能均使用任务附件全部用例：A2 为 153 条，310P 为 FLOAT32/FLOAT16 共 103 条。
精度测试通过公开 ACLNN 接口执行，默认逐字节比较 out，以及 indices 起始的
`4 * out.size` 个有效字节（连续 INT32 索引），不比较容器剩余尾部。
原始 Golden 文件保持不变；以下比较仅覆盖接口定义的有效输出范围。
indices 仍预填非零值；形状、数据类型、容器容量和非连续输出存储间隙继续检查。
有效范围内任一差异均使程序返回非零状态。

添加 `--indices-check full` 可复现原始 Golden 的完整容器比较（包括零尾）；
默认值为 `--indices-check valid`。报告及每条通过记录都标明 `indices_check`，
valid 模式通过不表示尾部符合原始 Golden。

## 准备

按算子 [README](../../README.md) 构建安装，加载 CANN 环境并设置 `ASCEND_CUSTOM_OPP_PATH`。
Python 依赖为 `numpy` 和 `ml_dtypes`，测试程序无需编译。
在仓库根目录执行以下命令，下载并解压[任务原始附件](https://www.hiascend.com/p/resource/202609/33d5365752d34991b3d90ef52b3fb35a.zip)：

```bash
python3 -m pip install numpy ml_dtypes
cd experimental/pooling/max_pool2d_with_mask/tests/st
mkdir -p results
curl --fail --location \
  https://www.hiascend.com/p/resource/202609/33d5365752d34991b3d90ef52b3fb35a.zip \
  --output results/task_package.zip
python3 -m zipfile -e results/task_package.zip results/task_package
export CASES="$(find "$PWD/results/task_package" -name max_pool2d_with_mask_cases.json -type f -print -quit)"
export GOLDEN="$(find "$PWD/results/task_package" -name max_pool2d_with_mask_golden_func.py -type f -print -quit)"
export LIB="$CUSTOM_OPP_ROOT/op_api/lib/libcust_opapi.so"
export TYPES=float,float16,bfloat16
```

离线环境可将任务原始附件放到 `results/task_package.zip`，从解压命令继续执行。
310P 使用 `export TYPES=float,float16`。以下命令默认使用逻辑设备 0，可通过 `--device` 指定。

## 精度

```bash
python3 run_aclnn_cases.py --cases "$CASES" --golden "$GOLDEN" \
  --library "$LIB" --types "$TYPES" --report results/full.json
```

额外回归用例：

| 文件 | 覆盖内容 |
| --- | --- |
| generalization_cases.json | 非连续布局、未对齐切片、矩形窗口、空 stride、ceilMode 和多 tile。 |
| generalization_extended_cases.json | 44 条通道数、输入尺寸、核间输出划分及 BF16 联合分块索引写回用例。 |
| wide_input_cases.json | 12 条宽输入用例，310P 执行其中 8 条。 |
| large_window_cases.json | A2 的 27 条大窗口、窗口分块、输出缓冲复用及不足一个向量重复的尾部用例。 |

替换 `--cases` 并添加 `--layout both`，即可检查非连续输入与输出。例如：

```bash
python3 run_aclnn_cases.py --cases generalization_extended_cases.json \
  --golden "$GOLDEN" --library "$LIB" --types "$TYPES" \
  --layout both --report results/generalization.json
```

`--pattern` 支持 `normal`、`ascending`、`ties`、`zeros`、`adjacent`、`subnormal`、
`positive_inf`、`half_bits` 和 `bf16_bits`。后两者分别配合 `--types float16` 和
`--types bfloat16`，循环生成 16 位编码并排除 NaN/-Inf，检查合法位模式。
测试报告按用例集合、数据模式和布局分别列出实际验证范围。

## 性能

使用 CANN 官方 `msopprof` 采集 Kernel Task Duration，不统计 Golden 生成和主机开销。
对照时保持输入、设备、频率和计时范围一致，并保留逐例数据。

```bash
msopprof --application="python3 -u run_aclnn_cases.py --cases $CASES \
  --golden $GOLDEN --library $LIB --types $TYPES" \
  --output=results/profile-custom-1 --aic-metrics=BasicInfo \
  --warm-up=1 --launch-count=1000
```

A2 基线为 CANN 9.1.0 内置 ACLNN 实现。另开已加载 CANN、未设置自定义算子路径的终端，
使用同一程序，将 `--library` 改为 `$ASCEND_HOME_PATH/lib64/libopapi.so`。
INT32 前缀格式一致的路径可按默认 valid 模式校验；历史位掩码路径不能用此比较规则。
仅计时且未比较输出时使用 `--no-verify`，记录为 `RAN`，不作为精度通过记录。
基线与自定义实现交替采集，分别保存每轮结果；从官方导出文件读取对应 kernel 的
Task Duration 和 aiv 当前频率。仅统计目标频率下的样本，A2 为 1800 MHz，310P 为 1080 MHz。

单例验证添加 `--match '^F036_'`，默认 `--repeat 1`。

A2 如遇官方 msopprof 注入组件的设备编号问题，使用官方 msopcom
`a0eb0c6262110140ad16645f5514994da07ab1c7` 或包含该修复的版本，在目标系统构建对应组件。
