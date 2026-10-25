# 实验结果与证据索引

[← 首页](../README.md) · [完整复现](../docs/operations.md) · [详细A1–A5教程](../README.md#学习与实现导航)

![结果总览](../assets-figures/results-overview.png)

| 作业 | 证据入口 | 读数与限制 |
|---|---|---|
| A1 | [配置](a1/config.json)、[metrics](a1/metrics.jsonl)、[状态](a1/status.json) | 5000步，val1.41168；固定随机窗口，未遍历完整验证集 |
| A2 | [全部21项](a2)、[数值验证](a2/flash-numerics.json)、[Triton dh64前后向](a2/a2-triton-d64-fwd-bwd.json) | CPU双rank不是多GPU；共享设备测时 |
| A3 | [9点测量](a3/measurements.json)、[拟合与边界](a3/fit.json) | 实际训练FLOPs按6ND估算、整步取整；不是B200 API |
| A4 | [各组配置/逐步指标/best](a4)、[small消融](a4/a4-small-analysis.json) | 同轮优先比较，扩容raw/rules差异很小 |
| A5 | [各候选与选择](a5)、[派生汇总](summary.json) | val选模，test各1319题一次；不是多seed结论 |

## A4为什么从失败对照开始

旧轮raw/clean loss约5.4831/9.17865。筛选后独立数据只有2.25M token，却重复训练65.536M token；同时Wiki代理硬筛改变分布。优化轮先匹配token做ranked/random消融，再扩容到100M+独立token，raw/rules降到4.14461/4.13960。旧轮验证协议与本轮不同，不能据跨轮数值归因所有改善；详见[A4](../docs/a4.md)。

## A5如何得到最终表格

| 方法 | 最佳配置 / step | validation | test |
|---|---|---:|---:|
| SFT | lr5e-6 / 1350 | 91/256（35.55%） | 330/1319（25.02%） |
| DPO | lr1e-7、beta0.1 / 125 | 83/256（32.42%） | 357/1319（27.07%） |
| GRPO | P4G8、KL0.01 / 200 | 119/256（46.48%） | 476/1319（36.09%） |

所有选模只使用完整256题validation。测试单次生成、seed42、max_new_tokens512；没有按测试分数挑候选。DPO/GRPO分别从同一选中的SFT分叉，预算与数据反馈机制不同，不能当公平算法排名。

`a5/optimized-*-scores.json`发布固定测试顺序索引、严格/格式/答案分数和长度诊断，省略题目与模型生成文本。`initial-*-scores.json`用于历史配方对比；`selected-*.json`与各组best/complete文件区分最新恢复状态和最佳导出点。

[summary.json](summary.json)通过[scripts/plot_results.py](../scripts/plot_results.py)计算配对bootstrap。SFT→DPO增益的95%区间跨0，SFT→GRPO为正；区间只包含固定生成结果下的题目重采样，不涵盖训练种子、生成种子和模型选择不确定性。

## 来源与处理

2026-10-25重新通过SSH下载最终配置、指标与来源摘要，并逐文件验证SHA256。对应训练源commit、服务器相对路径、原始hash与发布hash记录在[server-verification](../provenance/server-verification.json)。发布JSON移除PID、规范化格式；原始与发布hash因此可不同。

未上传权重、网页/数学数据正文、模型生成文本、本地测试、上游fixtures、探针调试日志和重复进度快照。图表内存栏为不同测量口径中的PyTorch allocated近似值；详细记录见[memory](memory.json)，不能把它与NVML或reserved混同。
