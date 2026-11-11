# 实验记录

GR00T 与 π0.5 rebuttal 实验材料，打包日期：2026-11-12。

保留 `rebuttal_gr00t_pi05_20261111` 目录中除顶层 `inventory/` 外的全部文件，包括模型资源、历史证据、新实验记录、脚本、报告及原有复算 ZIP。共 **11,556 个文件**，原始总大小 **20,859,473,803 字节（约 19.43 GiB）**。结果解释、统计分母、失败记录与复现限制见原始报告，材料公开不等同于新增实验验收。

## 下载

[完整 Release 下载页](https://github.com/115239/RASbench/releases/tag/experiments-261112)

以下是同一个 tar.gz 的 10 个分段，必须全部下载后按编号合并。各分段不能单独解压。

| 分段 | 大小（字节） | 下载 |
| --- | ---: | --- |
| RASbench-experiments.tar.gz.part-001 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-001) |
| RASbench-experiments.tar.gz.part-002 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-002) |
| RASbench-experiments.tar.gz.part-003 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-003) |
| RASbench-experiments.tar.gz.part-004 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-004) |
| RASbench-experiments.tar.gz.part-005 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-005) |
| RASbench-experiments.tar.gz.part-006 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-006) |
| RASbench-experiments.tar.gz.part-007 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-007) |
| RASbench-experiments.tar.gz.part-008 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-008) |
| RASbench-experiments.tar.gz.part-009 | 1990000000 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-009) |
| RASbench-experiments.tar.gz.part-010 | 1094842697 | [下载](https://github.com/115239/RASbench/releases/download/experiments-261112/RASbench-experiments.tar.gz.part-010) |

## 校验与还原

将全部 10 个分段及 `SHA256SUMS.txt` 放在同一目录，然后运行：

```bash
shasum -a 256 -c SHA256SUMS.txt
mkdir -p rebuttal_gr00t_pi05_20261111
cat RASbench-experiments.tar.gz.part-* | tar -xz -C rebuttal_gr00t_pi05_20261111
```

Linux 也可用 `sha256sum -c SHA256SUMS.txt`。还原后的文件内容、相对路径与源目录一致；请预留至少约 40 GiB 可用空间供分段和还原文件使用。

- [FILE_MANIFEST.csv](FILE_MANIFEST.csv)：全部原文件路径、字节数与 SHA-256。
- [SHA256SUMS.txt](SHA256SUMS.txt)：各分段 SHA-256。
- [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)：发布总数、体积、分段及排除范围。

## 报告入口

解压后可从 `实验核查与rebuttal证据.md`、`S1_选型证据.md`、`商超_对照实验结果.md`、`复现实验说明.md` 和 `rebuttal_表述边界.md` 开始阅读。

## 排除与访问范围

顶层 `inventory/` 及其全部内容未包含在本次发布中；其余内容未主动删减。本页下载链接指向公开 GitHub，会显示源仓库账号，不是匿名评审下载链接。
