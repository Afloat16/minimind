# MiniMind 独立修复审阅导航

这里按复核目标列出 #927–#946 的 20 项独立修复。每个 PR 均直接基于上游 `cc9629e5f464ef46ef9411bfac916364e0a6dff1`；可在该基线上单独审阅、验证和合并。表内改动规模来自 GitHub 当前 PR 的差异统计，验证结果对应 [提交清单](SUBMISSIONS.md) 中的确切 head。

建议先查看下方六项训练目标与恢复正确性修复：它们分别有明确的数值、更新次数或保存/恢复结果作为复核标准。其余两组覆盖模型与资源使用、接口及工具链；各项复现入口集中在 [README](README.md#commands-for-all-20-contributions)。

## 训练目标与恢复正确性

| PR / 差异 | 改动规模 | 已记录的行为证据 | 结果 |
|---|---|---|---|
| [#928 · 半精度因果损失以 FP32 归约](https://github.com/jingyaogong/minimind/pull/928/files) | 1 文件，+2/−1 | 半精度溢出样例从 inf 恢复为有限值，并与 FP32 参考一致。 | [原始结果](results/model-dataset-results.json) |
| [#931 · 蒸馏路由正则与 alpha 解耦](https://github.com/jingyaogong/minimind/pull/931/files) | 1 文件，+4/−5 | alpha 为 0、0.5、1 时，路由正则都产生预期的独立更新。 | [原始结果](results/training-results.json) |
| [#935 · 在梯度累积完成后保存 checkpoint](https://github.com/jingyaogong/minimind/pull/935/files) | 7 文件，+23/−9 | 避免保存未提交梯度的中间状态；实际训练循环的恢复结果与连续训练一致。 | [原始结果](results/training-results.json) |
| [#943 · 移除 PPO Critic 的重复 RMSNorm](https://github.com/jingyaogong/minimind/pull/943/files) | 1 文件，+1/−1 | 实际小模型中，对 backbone 加 value head 的最大偏差从 0.277123 变为 0。 | [原始结果](results/logs/ppo-single-critic-norm-fixed.log) |
| [#944 · PPO 达到 KL 阈值后停止更新](https://github.com/jingyaogong/minimind/pull/944/files) | 1 文件，+6/−2 | 首次拒绝从 4 次更新变为 0 次；跨 rollout 残余梯度刷新也通过验证。 | [原始结果](results/logs/ppo-stop-at-kl-limit-fixed.log) |
| [#946 · 学习 Agent 每轮生成的结束符](https://github.com/jingyaogong/minimind/pull/946/files) | 1 文件，+1/−7 | 两轮工具交互中参与监督的生成 EOS 从 0/2 变为 2/2，后续轮次动作保留。 | [原始结果](results/logs/agent-train-turn-endings-fixed.log) |

## 模型、数据与资源使用

| PR / 差异 | 改动规模 | 已记录的行为证据 | 结果 |
|---|---|---|---|
| [#927 · 单 token 缓存解码启用 SDPA](https://github.com/jingyaogong/minimind/pull/927/files) | 1 文件，+5/−1 | 让缓存解码使用 SDPA；输出与梯度对照通过。CPU 微基准未显示提速，未测 GPU。 | [原始结果](results/model-dataset-results.json) |
| [#929 · MoE 路由概率使用 FP32](https://github.com/jingyaogong/minimind/pull/929/files) | 1 文件，+2/−2 | 避免低精度概率下溢或舍入到 1 导致的路由梯度丢失。 | [原始结果](results/model-dataset-results.json) |
| [#930 · 解析 Agent 历史中的字符串工具调用](https://github.com/jingyaogong/minimind/pull/930/files) | 1 文件，+2/−0 | JSON 字符串形式与结构化工具历史生成相同模板，原始记录保持完整。 | [原始结果](results/model-dataset-results.json) |
| [#932 · 蒸馏教师使用配置的混合精度](https://github.com/jingyaogong/minimind/pull/932/files) | 1 文件，+4/−4 | 教师实际使用 bf16；概率与 KL 以 FP32 计算，极值损失与梯度有限。 | [原始结果](results/training-results.json) |
| [#933 · CPU 中转 checkpoint，及时释放权重副本](https://github.com/jingyaogong/minimind/pull/933/files) | 6 文件，+8/−6 | 修复 6 个加载入口，验证权重逐位一致及设备转移前的临时张量释放。 | [原始结果](results/training-results.json) |
| [#934 · DPO 提前释放参考模型 logits](https://github.com/jingyaogong/minimind/pull/934/files) | 1 文件，+3/−2 | 策略模型 forward 前释放完整词表 logits，两步参数更新与原实现完全一致。 | [原始结果](results/training-results.json) |

## 接口、工具调用、导出与 rollout

| PR / 差异 | 改动规模 | 已记录的行为证据 | 结果 |
|---|---|---|---|
| [#936 · 修正 Web 工具对话历史](https://github.com/jingyaogong/minimind/pull/936/files) | 1 文件，+11/−4 | 模型历史不再混入 HTML 卡片或重复工具调用，并按完整用户轮次保留历史。 | [原始结果](results/serving-results.json) |
| [#937 · 让 Web 输出长度预算生效](https://github.com/jingyaogong/minimind/pull/937/files) | 1 文件，+1/−2 | 原生生成请求 2 token 时，输出从原先 5 个变为 2 个。 | [原始结果](results/serving-results.json) |
| [#938 · 工具评测遵守 token 和轮数预算](https://github.com/jingyaogong/minimind/pull/938/files) | 1 文件，+8/−2 | API 使用 CLI token 预算；达到工具轮数上限后终止重复调用。 | [原始结果](results/serving-results.json) |
| [#939 · API 正确标记长度截断](https://github.com/jingyaogong/minimind/pull/939/files) | 1 文件，+18/−3 | 流式与非流式区分 length、stop 和 tool_calls；10 个 HTTP 情形通过。 | [原始结果](results/serving-results.json) |
| [#940 · 导出前严格校验完整基模权重](https://github.com/jingyaogong/minimind/pull/940/files) | 1 文件，+2/−2 | 拒绝缺层、缺投影和多余键；完整导出及零增量 LoRA 合并数值正确。 | [原始结果](results/serving-results.json) |
| [#941 · 避免工具参数重复 JSON 编码](https://github.com/jingyaogong/minimind/pull/941/files) | 1 文件，+4/−1 | 真实 SDK 解码后得到参数对象；字符串和对象形式在流式、非流式中一致。 | [原始结果](results/serving-results.json) |
| [#942 · 支持 OpenAI 文本分段输入](https://github.com/jingyaogong/minimind/pull/942/files) | 1 文件，+15/−1 | 保留文本数组中的用户提问；不支持的内容类型明确返回 422，11 个 HTTP 请求通过。 | [原始结果](results/serving-results.json) |
| [#945 · 校验 SGLang rollout 概率完整性](https://github.com/jingyaogong/minimind/pull/945/files) | 1 文件，+9/−4 | 拒绝 token/logprob 不匹配、非有限概率和错误 batch 大小；合法响应保持完整。 | [原始结果](results/logs/sglang-rollout-integrity-fixed.log) |

## 复核时需要区分的证据范围

- **保存与恢复（#935）：**运行了预训练、SFT、LoRA、DPO 和蒸馏的实际训练循环并比较恢复结果；GRPO 和 Agent 部分检查的是实际保存条件。
- **资源使用（#933、#934）：**已验证 CPU 加载、临时张量生命周期及数值一致性；没有测量 GPU 峰值显存。
- **SDPA（#927）：**输出、梯度及分派检查通过；CPU 微基准没有显示提速，没有 GPU 性能数据。
- **接口依赖：**#930、#937、#939、#941、#942、#946 另用仓库锁定的 Pydantic 2.11.5 / Jinja2 3.1.2 完成了六组前后对照，详见 [锁定依赖验证](FOLLOWUP.md)。
- **SGLang（#945）：**HTTP 响应夹具验证了合法输出与无效数据拒绝；另对官方 v0.5.23 固定提交完成 [正常输出、EOS 和空完成的协议核查](SGLANG_PROTOCOL.md)。未运行真实 SGLang 服务。
- **组合兼容性：**20 项在记录顺序下自动 cherry-pick 成功，零冲突；18 个修改的 Python 文件通过 AST 和 diff 检查。[组合记录](results/combined-20-compatibility.json) 描述的是补丁及静态兼容性。

## 状态与复现

[当前状态快照](SUBMISSIONS.md) 包含所有 PR 链接、确切 head 和查询时间；[机器可读快照](results/pr-status.json) 同时记录评审、检查、评论数量及维护者编辑开关。当前没有上游自动检查记录；本目录中的通过结果来自注明环境和范围的验证运行。

按 [环境准备与单项复现](README.md#prepare-an-explicit-baseline-and-candidate) 选择对应候选分支，并在原始基线运行同一用例观察原问题。所有生产分支保持原有的小范围改动，检查脚本和审阅资料集中在此验证分支。
