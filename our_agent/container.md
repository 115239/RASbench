# Competition baseline container

The competition baseline image supports pulls without `docker login`.

```bash
docker pull ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner:260915_baseline
```

For a fixed image version:

```bash
docker pull ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner@sha256:cded24562310ba9aa6fae5052f720f77e844aac6cfc6f93d0fcca47fec3e01ad
```

Project files are under `/opt/safety_embodiment/`. Extract them without starting the container:

```bash
docker create --name rasbench-export ccic-dmp-eval-images-registry.cn-hangzhou.cr.aliyuncs.com/cieval/embodied-competition-runner@sha256:cded24562310ba9aa6fae5052f720f77e844aac6cfc6f93d0fcca47fec3e01ad /bin/true
docker cp rasbench-export:/opt/safety_embodiment ./safety_embodiment
docker rm rasbench-export
```

This image is the competition baseline. Its digest is not a verified binding to the paper's final experiment version.
