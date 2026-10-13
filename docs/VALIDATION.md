# v0.1 验证记录

日期：2026-10-13。范围：本地 CPU 数学参考代码；没有启动环境实验或远端训练。

## 实测环境

- Windows x64，Python 3.12.14。
- PyTorch 2.6.0+cpu，NumPy 2.2.6。
- editable package 构建与安装成功。
- GitHub Actions 本次未配置或执行；其他 Python/PyTorch、GPU/CUDA 组合没有据此验收。

## 执行结果

```text
python -m unittest discover -s tests -v
Ran 30 tests in 0.091s
OK

python -m examples.synthetic_trace
all_gradients_finite: true
example_horizon: [2]
example_returns: [[11.791, 10.9, 0.0]]
example_valid_mask: [[true, true, false]]
optimizer_steps: 0
```

上述耗时是这一次本地测试记录，不是性能基准或运行时间承诺。合成演示也核对了各模型损失、辅助监督和梯度连接；未训练网络的 loss 与风险概率不作为算法效果证据。

| 测试文件 | 数量 | 独立核对内容 |
|---|---:|---|
| test_distributions.py | 3 | KL/JS 手算值、分组归约、ST 解析梯度 |
| test_rssm_losses.py | 9 | Real/Dream 时间索引、reset、teacher/start 断梯度、padding |
| test_reliability.py | 9 | 历史因果性、前缀最大误差标签、未知标签屏蔽、辅助梯度隔离 |
| test_reliability_precision.py | 4 | 极端 fp16 logits 与奖励的溢出回归 |
| test_returns.py | 5 | 连续前缀、H=0、bootstrap、终止/折扣边界、拒绝尾部无影响 |

独立复查发现并修复了两类低精度问题：有限 fp16 logits 在分布差异计算中溢出；奖励先相减再归一化时溢出。相关操作在运算前提升到 float32，新增回归测试通过。这不代表完整模型已完成混合精度训练验证。

## 仍需实验验证

当前证据只能说明这些数学接口与边界测试通过，不能证明 HelixWM 提升了环境回报、样本效率、校准质量或泛化。没有生成学习曲线、环境视频、训练 checkpoint 或多 seed 结果。

下一阶段应先讨论冻结模型的诊断设计，再决定如何适配作者维护的 DreamerV3。完整训练循环、checkpoint/replay/RNG 恢复和正式验收仍属于后续工作。
