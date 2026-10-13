# HelixWM

**Real-Dream Double-Helix World Model**

HelixWM 是从真实观测与模型想象的双路径关系出发构建的世界模型研究构想。本仓库保存第一版设计思路、数学公式、流程图、PyTorch 参考代码和正确性测试。

> 当前阶段：v0.1 原理与可运行的数学参考实现。尚无 HelixWM 正式实验结果；本文的性能改进均为研究假设。代码使用小型向量输入，不是完整 DreamerV3 或 Walker 训练程序。

阅读顺序：[本页核心公式](#2-时间索引与双路径公式) → [详细设计与流程图](docs/DESIGN.md) → [合成公式演示](examples/synthetic_trace.py) → [验证记录](docs/VALIDATION.md)。

## 1. 研究目标与最小核心

真实观测形成 Real Strand；模型从同一起点生成 Dream Strand。两条路径通过两种不同机制发生联系：

- **Cross correction**：用同动作真实序列的观测后验监督多步开环预测，更新世界模型。
- **Reliability calibration**：结合已经发生的预测误差与模型分歧，估计候选想象的风险，决定哪些连续片段用于策略学习。

第一版围绕“预测如何生成、预测如何校正、预测如何用于学习”组织。暂不扩展机器人应用；context residual 和主动采样保留为后续可选模块。

### 首版建模选择

原始构想允许独立的 Real/Dream 动力学、共享主干双分支或共享动力学。首版建议采用 **共享 RSSM**：两条路径共享转移参数，Real 每步吸收新观测，Dream 递推时不使用未来观测。

这是明确的初步设计选择，不是对独立双模型的否定，也不是已验证的最优结构。

## 2. 时间索引与双路径公式

执行动作 $a_t$ 后，收到 $o_{t+1}$、奖励 $r_{t+1}$ 和继续标记 $c_{t+1}$。令模型状态 $b_t=(h_t,z_t)$，其中 $h$ 为确定性记忆、$z$ 为随机潜变量。

### Real Strand：转移后吸收观测

$$
h^R_{t+1}=f_\theta(h^R_t,z^R_t,a_t),\qquad
q^R_{t+1}=q_\phi(\cdot\mid h^R_{t+1},E_\phi(o_{t+1})).
$$

$$
z^R_{t+1}\sim q^R_{t+1},\qquad
b^R_{t+1}=(h^R_{t+1},z^R_{t+1}).
$$

这里的 Real 状态是观测条件下的潜在估计，不是环境物理真值。

### Dream Strand：从同一起点开环展开

$$
\hat b_{t,0}=\operatorname{sg}(b^R_t),\qquad
\hat h_{t,k+1}=f_\theta(\hat h_{t,k},\hat z_{t,k},\tilde a_{t,k}),
$$

$$
p^D_{t,k+1}=p_\theta(\cdot\mid\hat h_{t,k+1}),\qquad
\hat z_{t,k+1}\sim p^D_{t,k+1}.
$$

$\operatorname{sg}$ 表示停止梯度。校正时使用实际记录动作 $\tilde a_{t,k}=a_{t+k}$；策略学习时使用 actor 产生的候选动作。这两种动作来源不能混用监督标签。

预测头输出奖励 $\hat r_{t,k+1}$ 与继续概率 $\hat c_{t,k+1}$。模型不可信不等同于环境终止。

## 3. 模型训练与 Cross correction

令 $p_i^1=p_\theta(\cdot\mid h_i^R)$ 为看到当前观测之前的一步先验。基础世界模型损失包括观测、奖励和继续标记的预测，以及先验/后验 KL：

$$
\ell_{\mathrm{pred},i}
=-\log p_\beta(o_i\mid b_i^R)
-\log p_\beta(r_i\mid b_i^R)
-\log p_\beta(c_i\mid b_i^R),
$$

$$
\mathcal L_{\mathrm{base}}
=\mathbb E_i\left[
\ell_{\mathrm{pred},i}
+\lambda_d D_{\mathrm{KL}}(\operatorname{sg}[q_i^R]\Vert p_i^1)
+\lambda_r D_{\mathrm{KL}}(q_i^R\Vert\operatorname{sg}[p_i^1])
\right].
$$

在相同起点、相同记录动作下，添加多步监督：

$$
\mathcal L_{\mathrm{cc}}
=
\frac{
\sum_t\sum_{k=2}^{K}m_{t,k}w_k
\mathbb E_{\mathrm{roll}}\left[
D_{\mathrm{KL}}(\operatorname{sg}[q^R_{t+k}]\Vert p^D_{t,k})
\right]}
{\sum_t\sum_{k=2}^{K}m_{t,k}w_k+\epsilon},
$$

$$
\mathcal L_{\mathrm{WM}}
=\mathcal L_{\mathrm{base}}
+\lambda_{\mathrm{cc}}\mathcal L_{\mathrm{cc}}.
$$

- $m$ 屏蔽 padding、跨 episode/reset 和无效配对。
- 共享版本的一步校正已包含在基础 KL 中，因此新增项从两步开始。
- 起点和未来 posterior 教师目标停止梯度；校正梯度沿 Dream 路径更新动力学参数。
- 该目标是对随机预测路径的条件 KL 取期望，不等于精确多步边缘分布 KL。
- 多步一致性不自动保证物理预测更准确或环境回报更高。

## 4. 真实误差、分歧与风险

对分组 categorical latent，记录一步预测与随后观测后验之间的差异：

$$
d_i^z=\frac1J\sum_{j=1}^J\operatorname{JS}(q^R_{i,j},p^{1,\mathrm{saved}}_{i,j}),
\qquad
d_i^r=\frac{|r_i-\hat r_i^{1,\mathrm{saved}}|}{s_r+\epsilon}.
$$

误差历史 $D_t=(d_{t-L+1},\ldots,d_t)$ 只能包含时刻 $t$ 已获得的信息。episode 开始时截短历史或使用有效掩码，不跨 reset。预测必须在真实结果到来之前保存，并与模型版本关联。

代码参数 `reward_scale` 接收严格正的有效分母；若采用上式的 epsilon 稳定项，传入 `s_r + epsilon`。

多个独立初始化、bootstrap 监督的预测成员提供分歧代理：

$$
u_{t,k}
=\frac1{MJ}\sum_{m,j}
D_{\mathrm{KL}}(P^{(m)}_{t,k,j}\Vert\bar P_{t,k,j}),
\qquad
\bar P_{t,k,j}=\frac1M\sum_m P^{(m)}_{t,k,j}.
$$

分歧不直接等于错误概率；所有成员也可能共同出错。

首版风险目标沿用原理稿中的累计奖励预测错误事件：

$$
Y_{t,k}=
\mathbf1\left[
\max_{1\le j\le k}
\frac{|r_{t+j}-\hat r_{t,j}|}{s_r+\epsilon}
>\varepsilon_{\mathrm{task}}
\right].
$$

$$
\hat p_{t,k}
=g_\omega\left(
\operatorname{sg}[b_t^R],
\tilde a_{t,0:k-1},
\operatorname{sg}[u_{t,1:k}],
\operatorname{sg}[D_t],k
\right).
$$

用有有效未来标签的样本优化 BCE。风险监督使用同动作的开环奖励预测；不能把未来的一步误差历史直接替代该多步标签。未知或跨 reset 的标签不填成零，而是屏蔽。

未来真实观测和奖励只作训练标签，不能输入当前风险估计器。BCE 训练不自动意味着有限模型已经校准；反事实动作与分布变化仍需验证。

## 5. 可信前缀与价值计算

选择从起点连续可信的前缀：

$$
H_t=\max\{h\in\{0,\ldots,H_{\max}\}:
\hat p_{t,j}\le\delta,\ \forall\,1\le j\le h\}.
$$

在各自截断位置保留价值 bootstrap，再向前反推：

$$
G_{t,H_t}=V_{\bar\nu}(\hat b_{t,H_t}),
$$

$$
G_{t,i}=
\hat r_{t,i+1}
+\gamma\hat c_{t,i+1}
\left[(1-\lambda)V_{\bar\nu}(\hat b_{t,i+1})+\lambda G_{t,i+1}\right],
\quad i<H_t.
$$

只在有效前缀内构造 actor/critic 目标。不能从完整尾部先计算回报再只对损失乘 mask，否则被拒绝的后缀仍会污染前缀目标。

当 $H=0$ 时，该起点不生成想象策略更新样本；整批没有有效样本时跳过更新。

后续 actor 的首版参考形式采用固定门控样本上的 score-function surrogate。采样动作、状态、门控权重及优势均按所需路径停止梯度；不能声称它是整个自适应停止目标的无偏梯度。完整 actor/critic 优化器目前尚未声明实现。

## 6. 总体流程

~~~mermaid
flowchart TD
    A["真实交互：观测、动作、奖励"] --> B["Real Strand：观测后验"]
    B --> C["同起点 + 记录动作：Dream 多步预测"]
    B --> D["停止梯度的未来后验目标"]
    C --> E["Cross correction"]
    D --> E
    E --> F["更新世界模型"]
    C --> G["预测与实际结果比较"]
    A --> G
    G --> H["已发生误差历史"]
    B --> I["候选 actor 动作：Dream 想象"]
    I --> J["模型分歧 + 风险预测"]
    H --> J
    J --> K["连续可信前缀"]
    K --> L["截断价值 bootstrap"]
    L --> M["Actor / Critic 学习"]
    M --> A
    F --> C
    F --> I
~~~

该图表达学习周期。训练标签所需的未来数据必须已经收集；正在做真实动作决策时，不能读取尚未发生的结果。

## 7. 公式与代码对应关系

以下文件为已实现的数学核心；正式环境、replay、完整 actor/critic 和训练恢复尚未接入。

| 模块 | 代码 | 职责 |
|---|---|---|
| 分布运算 | [distributions.py](helixwm/distributions.py) | grouped KL/JS、straight-through 采样 |
| 双路径模型 | [rssm.py](helixwm/rssm.py) | observe_step、imagine_step、共享状态表示 |
| 模型损失 | [losses.py](helixwm/losses.py) | 基础损失、多步校正、梯度隔离 |
| 可靠性 | [reliability.py](helixwm/reliability.py) | 历史真实误差、bootstrap ensemble、因果风险头和标签 |
| 价值目标 | [returns.py](helixwm/returns.py) | 连续前缀、lambda-return、bootstrap |
| 公式演示 | [synthetic_trace.py](examples/synthetic_trace.py) | 合成输入、反传验证、手算回报核对，优化器更新次数为零 |
| 正确性测试 | [tests](tests/) | 因果时序、梯度、掩码、回报边界、半精度回归 |

参考实现的观测/奖励头使用固定单位方差 Gaussian 损失，continuation 使用 Bernoulli 损失；没有照搬 DreamerV3 的 distributional heads、free bits、归一化及其他训练细节。所有网络尺寸和示例阈值仅用于数学验证。

### 本地运行

在独立 Python 3.12 环境、仓库根目录运行：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.synthetic_trace
```

CPU 安装方式见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。本仓库只验证 CPU 数学核心，不据此宣称 GPU/CUDA 或 RTX 5090 训练兼容性。包声明的版本范围不代表每个组合都已验证。

运行脚本只生成合成张量并检查梯度，不连接 SSH、不创建实验、不更新模型参数。示例输出中 loss/风险概率来自未训练模型，不是性能结果。

主要验证内容：

1. Dream 接口不读取未来观测。
2. 校正损失不沿教师 posterior 或起点输入回传，但动力学获得有限梯度。
3. 跨 reset 的无效配对不影响前一回合损失。
4. 风险序列 [0.1, 0.8, 0.1]、阈值 0.5 时，可信长度为 1；全拒绝时为 0。
5. 两步奖励均为 1、折扣 0.99、lambda=1、末端价值 10 时，起点目标为 11.791，而不是错误清除末端价值后的 1.99。

## 8. 环境与范围

首个候选环境为 **DeepMind Control Suite / Walker Walk，像素观测、连续动作**。

此前 EAWM、DRAGO、DrQ-v2 等实验仅用于选择环境和提供工程经验；其分数、模型权重或验收记录均不作为 HelixWM 的结果。暂不涉及 Franka。

第一版先完成基本模型、公式和正确性测试，再讨论实验方案。Context residual、主动信息采集、独立双模型比较不挤入当前最小核心。

## 9. 当前状态与后续运行要求

| 阶段 | 状态 |
|---|---|
| 基本设计与公式 | 第一版草案，待持续讨论 |
| 参考实现 | v0.1 数学核心、合成演示和测试已实现 |
| 本地测试 | 30 项 unittest 与合成演示通过；见验证记录 |
| CI | 本次未配置或执行 GitHub Actions |
| Walker 正式训练 | 尚未开展 |
| 多 seed 与论文级比较 | 尚未开展 |

后续进入实验时遵守：

- 记录源码版本、依赖、配置、硬件、随机种子及完整启动命令，区分环境帧与控制步/action repeat。
- 基线比较保留各作者原算法、原预算和原评估协议；不把 HelixWM 的修改包装成作者原版复现。
- 经环境兼容、smoke、checkpoint 重载和冻结零更新评估门槛后，再开始正式训练。
- 远端运行使用 tmux；按用户要求每六小时检查一次，只在实质变化、完成、失败或需要处理时通知。此条是后续运行要求，不代表已创建巡检或已启动训练。
- 不擅自中断训练、修改正在运行的源码/预算、切换 seed、重复启动或释放实例。
- Checkpoint 明确列出模型、优化器及其他保存状态；若缺少 replay、RNG 或环境状态，不宣称可精确恢复完整训练。
- 结果按用户保留的验收规范归档到 D 盘：模型、原始数据、日志、源码/配置、元数据、学习曲线及 SHA256；每个实际实验阶段保存一个视频和一张学习曲线。
- 区分 smoke、单 seed 完成和多 seed 论文级证据；不以合成示例代替环境实验。
- GitHub 保存代码与文档；大型模型和实验数据保存在实验归档中。SSH 密码、私钥及其他凭据不进入仓库。

## 10. 来源与研究定位

研究构想来自用户提供的 HelixWM_Research_Summary_CN.txt；本 README 对应“先原理、再公式和流程、随后实验”的第一版整理。

基础概念来源：

- [DreamerV3](https://arxiv.org/abs/2301.04104)：世界模型与想象策略学习。
- [PlaNet / Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html)：RSSM 与多步 latent overshooting。
- [Perceiving the Knowledge Boundary](https://ojs.aaai.org/index.php/AAAI/article/view/39576)：不确定性驱动的探索与想象相关工作。
- [dm_control](https://github.com/google-deepmind/dm_control)：候选控制环境实现。

已有组件的组合和重新命名不能单独证明新颖性。核心待验证问题是：历史真实预测误差是否能在模型分歧之外提供额外信息，以及联合校正与可信前缀能否改善模型学习和控制表现。
