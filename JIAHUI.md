# 说明

- 本项目来自Tensorboard（在upstream），现在的分支jiahui追踪heiwang1997的fork（origin）

## 编译与测试

```shell
ca
bazel build tensorboard:tensorboard
./bazel-bin/tensorboard/tensorboard --logdir demo_logs
```

安装：
```shell
bazel run //tensorboard/pip_package:extract_pip_package
```
但是好像运行起来有问题的样子，无法打包成whl文件，所以目前是把`tb`的TENSORBOARD_DIR直接设成`./bazel-bin/tensorboard/tensorboard`

## 测试Hparams

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

writer = SummaryWriter('demo_logs/v1')
hparams_dict = {"pa": 5.0, "pb": 1.0, "pc": True, "pd": 7.0}
hparams_metrics = {"ma": 1.0, "mb": 2.0}
exp, ssi, sei = hparams(hparams_dict, hparams_metrics)
writer.file_writer.add_summary(exp)
writer.file_writer.add_summary(ssi)
writer.file_writer.add_summary(sei)

writer.add_scalar("ma", 2.0)
writer.close()
```

# Changelog

1. 增强了`hparams`插件，在HPARAMS选项卡默认只打上不同名称的勾，另外支持为每个实验输入Comment

