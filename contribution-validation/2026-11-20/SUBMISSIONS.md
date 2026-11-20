# MiniMind：20 项 PR 跟进状态

核查时间：**2026/11/20 11:51:14（北京时间）**。

**20/20 项均已创建上游 PR，对应 #927–#946；20 个 Open，0 个 Draft，本批 0 个 merged。** 每个 PR 的作者、来源分支、标题、完整正文和实际提交 SHA 均与准备好的提交清单逐项核对；没有遗漏或重复。

GitHub 当前返回 **20/20 项 `mergeable=true`、`mergeable_state=clean`**。这表示当前代码可自动合并；最终审查和接受由上游维护者完成。

本次查询到的讨论/行内评论为 0 条，commit status 为 0 项，check run 为 0 项，PR workflow run 为 0 项。零项 CI 记录不表示自动 CI 已通过；下列验证证据来自记录过的针对性本地运行。

[复现命令与原始验证](README.md) · [本轮跟进与锁定依赖验证](FOLLOWUP.md) · [去重记录](DEDUPLICATION.md) · [机器可读清单](submissions.json) · [实时状态快照](results/pr-status.json)

| # | 上游 PR | 改进 | 状态 | 已验证提交 |
|---:|---|---|---|---|
| 1 | [#927](https://github.com/jingyaogong/minimind/pull/927) | 单 token 缓存解码启用 SDPA | Open / clean | [5b10cb0](https://github.com/Afloat16/minimind/commit/5b10cb055a9e744786b8b635362f310e9144e2d2) |
| 2 | [#928](https://github.com/jingyaogong/minimind/pull/928) | 半精度因果损失以 FP32 归约 | Open / clean | [8d4be33](https://github.com/Afloat16/minimind/commit/8d4be333c4e418b879b39aa84931a929cceb5e7b) |
| 3 | [#929](https://github.com/jingyaogong/minimind/pull/929) | MoE 路由概率使用 FP32 | Open / clean | [b3e2564](https://github.com/Afloat16/minimind/commit/b3e2564b66187bc1b7c048451a0a0c57b5f688d6) |
| 4 | [#930](https://github.com/jingyaogong/minimind/pull/930) | 解析 Agent 历史中的字符串工具调用 | Open / clean | [6093668](https://github.com/Afloat16/minimind/commit/6093668f948045535d2ccb3b4711de40093b8eec) |
| 5 | [#931](https://github.com/jingyaogong/minimind/pull/931) | 蒸馏路由正则与 alpha 解耦 | Open / clean | [c00eb1b](https://github.com/Afloat16/minimind/commit/c00eb1bd32c34c41859f1879642dc64746a560a2) |
| 6 | [#932](https://github.com/jingyaogong/minimind/pull/932) | 蒸馏教师使用配置的混合精度 | Open / clean | [c39d880](https://github.com/Afloat16/minimind/commit/c39d8802e48dce97cdcd1551f304b33d3c293979) |
| 7 | [#933](https://github.com/jingyaogong/minimind/pull/933) | CPU 中转 checkpoint，及时释放权重副本 | Open / clean | [afc2a22](https://github.com/Afloat16/minimind/commit/afc2a224cbb689929c530ed21ebbb0edf9527a7a) |
| 8 | [#934](https://github.com/jingyaogong/minimind/pull/934) | DPO 提前释放参考模型 logits | Open / clean | [5ed7373](https://github.com/Afloat16/minimind/commit/5ed7373aebe40eed74a6e812c835e69fd1e05545) |
| 9 | [#935](https://github.com/jingyaogong/minimind/pull/935) | 在梯度累积完成后保存 checkpoint | Open / clean | [1f3d49a](https://github.com/Afloat16/minimind/commit/1f3d49a1c2eeedb294e3c0018c5cdc1c5b51f559) |
| 10 | [#936](https://github.com/jingyaogong/minimind/pull/936) | 修正 Web 工具对话历史 | Open / clean | [6f5f3ce](https://github.com/Afloat16/minimind/commit/6f5f3cec12152391737a2ba7a2fb52e5de5f255c) |
| 11 | [#937](https://github.com/jingyaogong/minimind/pull/937) | 让 Web 输出长度预算生效 | Open / clean | [c7f395c](https://github.com/Afloat16/minimind/commit/c7f395c5e71b77b035641c66cc76b1f01b9a25c6) |
| 12 | [#938](https://github.com/jingyaogong/minimind/pull/938) | 工具评测遵守 token 和轮数预算 | Open / clean | [3fcad5b](https://github.com/Afloat16/minimind/commit/3fcad5be245721b76254f2efcab8edd4a2ee0afc) |
| 13 | [#939](https://github.com/jingyaogong/minimind/pull/939) | API 正确标记长度截断 | Open / clean | [df771bf](https://github.com/Afloat16/minimind/commit/df771bfee76f08c5d1e0d264d47c5388205897bb) |
| 14 | [#940](https://github.com/jingyaogong/minimind/pull/940) | 导出前严格校验完整基模权重 | Open / clean | [7908578](https://github.com/Afloat16/minimind/commit/79085785a9d00b108f4051613e525c8980b99775) |
| 15 | [#941](https://github.com/jingyaogong/minimind/pull/941) | 避免工具参数重复 JSON 编码 | Open / clean | [2327805](https://github.com/Afloat16/minimind/commit/2327805326f7964e61abe66236c98a51374eb9e8) |
| 16 | [#942](https://github.com/jingyaogong/minimind/pull/942) | 支持 OpenAI 文本分段输入 | Open / clean | [c38f380](https://github.com/Afloat16/minimind/commit/c38f380669795044ebc757e11f0257d3599b11bb) |
| 17 | [#943](https://github.com/jingyaogong/minimind/pull/943) | 移除 PPO Critic 的重复 RMSNorm | Open / clean | [5240635](https://github.com/Afloat16/minimind/commit/5240635fce0f0653ee4e2f8711857c90835a6b83) |
| 18 | [#944](https://github.com/jingyaogong/minimind/pull/944) | PPO 达到 KL 阈值后停止更新 | Open / clean | [0fc3dcc](https://github.com/Afloat16/minimind/commit/0fc3dcce93f421e409606ff570cad242d4c466df) |
| 19 | [#945](https://github.com/jingyaogong/minimind/pull/945) | 校验 SGLang rollout 概率完整性 | Open / clean | [b18b55a](https://github.com/Afloat16/minimind/commit/b18b55af63d5e63bccc711cbca72a2e0dcbc06bf) |
| 20 | [#946](https://github.com/jingyaogong/minimind/pull/946) | 学习 Agent 每轮生成的结束符 | Open / clean | [433cd5a](https://github.com/Afloat16/minimind/commit/433cd5a94517a2b657827bbf7b1113340fd8d833) |

## 验证与计数范围

这 20 项的生产代码没有因本次状态核查而变化。原有 20 组前后对照、数值及梯度验证继续对应各 PR 当前 head；当前上游 master 仍为 `cc9629e5f464ef46ef9411bfac916364e0a6dff1`。

此前按清单顺序组合全部 20 项：20/20 自动 cherry-pick 成功，零冲突，零手工解决；18 个修改的 Python 文件通过 AST 与累计 diff 检查。这是当前基线下的组合兼容性结果，运行时证据仍按各独立分支记录。

此前 #905–#924 的 20 项全部排除出本批；其中 5 个历史 merged 单独计数。只有本批上游 PR 实际 merged 后才增加本批合并数。

状态快照是上述核查时间的记录。访问对应 PR 可查看后续维护者操作。
