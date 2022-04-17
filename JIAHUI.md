# 说明

- 本项目来自Tensorboard（在upstream），现在的分支jiahui追踪heiwang1997的fork（origin）

## 编译与测试

```shell
ca
bazel build tensorboard:tensorboard
./bazel-bin/tensorboard/tensorboard --logdir demo_logs/logs
```

安装：
```shell
bazel run //tensorboard/pip_package:extract_pip_package
```
但是好像运行起来有问题的样子，无法打包成whl文件，所以目前是把`tb`的TENSORBOARD_DIR直接设成`./bazel-bin/tensorboard/tensorboard`

## 测试Hparams

```shell
cd demo_logs
python write_test_log.py
```

# Changelog

1. 增强了`hparams`插件，在HPARAMS选项卡默认只打上不同名称的勾，另外支持为每个实验输入Comment
2. 增强了`hparams`插件，支持操作Done，支持显示服务器名称以及最新更新时间

