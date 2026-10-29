# 本次整理的复现检查

日期：2026-10-29。检查在独立的 GitHub 发布工作副本中执行，原始项目与完整整合目录的策略代码保持不变。

## 代码与文件

- 61 个主线 Python 策略文件与完整档案中的最新版逐字节一致。
- 主线代码、引用用历史代码和审计脚本通过 Python 语法检查。
- Markdown 中的相对文件链接均指向仓库内已有文件。
- 拟提交的代码与文档未检出高置信度凭据模式；历史执行凭据不放入主仓库。

## 实际回放

使用本机已有的 Anaconda Python 3.13.5 环境，从发布工作副本运行：

```bash
python scripts/run_latest_baseline.py --output-dir ../checks/p16_replay
```

程序从仓库中的两块 replay 分片恢复面板，完成固定选择读取、test 回放、结果和图表生成。生成的关键指标与原 P16 结果包比较，绝对差异均小于 `1e-6`；本次没有重新选择参数。

| 字段 | 本次回放值 |
|---|---:|
| 记录收益期数 | 172 |
| folds | 23 |
| 历史净累计收益，6bps | 16.623338% |
| 历史 gross 累计收益 | 21.061839% |
| 已记录 72h 端点口径 Sharpe | 2.426798 |
| 已记录端点口径 MaxDD | -1.787740% |

这证明发布副本能够复现旧研究快照，未修复 [架构审查](../reviews/20260908_architecture_review/REVIEW_ZH.md) 中记录的连续账本与评价口径问题。它不等于新一轮独立样本验证。

机器可读记录：[回放指标](validation/p16_replay_metrics.json)、[与旧结果的差异](validation/replay_comparison.json)。

## 完整档案

完整目录压缩为 8 个 Release 分片，总计约 10.77 GB。核验程序依次读取所有压缩分片，验证分片 SHA-256，并解读 tar 内的每个原始文件：

- 46,765 个来源普通文件的 SHA-256 全部匹配。
- 65 个保留或补齐的符号链接全部匹配。
- 导览与核验资料一并归档；完整 tar 包含 46,791 个普通文件。

核验记录：[complete_archive_verification.json](validation/complete_archive_verification.json)。下载与恢复步骤见 [完整档案说明](FULL_ARCHIVE.md)。
