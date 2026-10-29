# 完整档案、数据与恢复

仓库首页展示一条代表性的资产级 residual 研究主线。完整整合目录以分片压缩包放在 [research-archive-20261029 Release](https://github.com/SONGTAIKUN/binance_alphatrading/releases/tag/research-archive-20261029)。

## 保存内容

| 目录 | 内容 |
|---|---|
| `binance_alphatrading_clean/` | 最新本地研究代码、长历史输入与 P18–P29 全部实验产物 |
| `crypto-stat-arb/` | 早期 P1–P17、完整 residual panel、原始 Binance 衍生品缓存和中间实验 |
| `pair-trading/` | 早期 AlphaGPT 配对路线、20 币 15m OHLCV、输出、配置与历史环境 |
| `supplements/pair-trading-main/` | 完整 quant_pairs 源码、测试、q10.7 文档与 runs |
| `execution/P16_CS72H_LIVE_TRADING_SAFE_V2_20260721/` | SAFE_V2 独立执行交付包 |
| `reviews/` | 架构审查、源码审计和已有结果诊断 |
| `integration/` | 来源映射、逐文件 SHA-256、链接与完整性核验 |

完整原始文件索引包含 **46,765 个来源普通文件，19,574,786,698 字节**。额外生成的导览和核验资料也随整合目录归档。压缩包不排除历史环境或 Git 目录；恢复后各来源仍分别保存，同名历史版本不相互覆盖。

顶层数据入口与补齐的依赖使用目录内相对链接。历史 `.venv` 里依赖本机安装的解释器链接原样保留；运行研究时按 README 创建新的环境。

## 下载与恢复

把 Release 的所有 `alphagpt-complete-20261029.tar.gz.part-*`、`SHA256SUMS.txt` 和 `ARCHIVE_MANIFEST.json` 下载到同一个目录，然后执行：

```bash
shasum -a 256 -c SHA256SUMS.txt
cat alphagpt-complete-20261029.tar.gz.part-* | tar -xz

cd alphagpt_binance_complete_20261029
python3 integration/verify_bundle.py
```

分片按三位数字编号排序拼接，恢复为一个标准 gzip + tar 归档。`ARCHIVE_MANIFEST.json` 记录每个分片的准确大小与 SHA-256；仓库内 [完整来源文件索引](../archive/COMPLETE_FILE_INDEX.csv) 可直接查询某项数据或实验的位置。

## 仓库与完整档案的边界

- 主仓库：最新主线源码、配置、P16 replay 分片及固定选择、长历史阶段对比、后期研究报告和架构审查。
- 完整档案：所有原始行情与衍生品缓存、长历史特征面板、大型 pickle、全部中间试验、配对与执行分支以及各版 Git 历史。
- 原始本地目录：保持不变，不因上传而移动、删除或覆盖。

Release 是研究资料归档；其中的执行脚本随历史现场保存，解压与核验过程不会启动交易程序。
