# 比赛开源

## 开源整理材料

[下载发布页](https://github.com/115239/RASbench/releases/tag/competition-261111) · [下载源码 ZIP](https://github.com/115239/RASbench/releases/download/competition-261111/RASbench-competition-source.zip)

包含 2026-11-09 整理目录中的 `candidate/`、`tools/` 和顶层说明，共 5,384 个文件、90,577,705 字节。已排除 `internal_audit/` 及其全部内容。该整理包是开源候选材料，不代表完整复现实验已经验收；缺口及许可证边界以包内说明为准。

[逐文件清单](FILE_MANIFEST.csv) · [ZIP SHA-256](SHA256SUMS.txt)

## 基线镜像

镜像由阿里云容器镜像仓库提供，允许匿名拉取，无需 `docker login`。2026-11-11 已验证匿名读取镜像 manifest 成功，摘要与服务器现有镜像一致；未执行全量镜像下载。镜像本体未上传 GitHub。

完整镜像地址：

```text
ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner:260915_baseline
```

拉取命令：

```bash
docker pull ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner:260915_baseline
```

为固定复现版本，也可以使用摘要拉取：

```bash
docker pull ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner@sha256:cded24562310ba9aa6fae5052f720f77e844aac6cfc6f93d0fcca47fec3e01ad
```

镜像内项目目录为 `/opt/safety_embodiment/`。如只需提取文件，可创建一个不启动的容器，再复制目录：

```bash
docker create --name rasbench-baseline-export ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner@sha256:cded24562310ba9aa6fae5052f720f77e844aac6cfc6f93d0fcca47fec3e01ad /bin/true
docker cp rasbench-baseline-export:/opt/safety_embodiment ./safety_embodiment
docker rm rasbench-baseline-export
```

上面命令仅提取目录，不是实验运行命令；实验依赖和运行方式见提取目录内的 README。容器名称如已被占用，请改用其他名称。

## 访问与匿名性

匿名拉取表示无需账号登录，不表示隐藏镜像发布者身份。GitHub 和镜像仓库地址均包含发布账号或组织线索；以上地址不是匿名评审下载链接。
