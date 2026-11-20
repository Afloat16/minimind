# SGLang rollout 协议核查：PR #945

## 结论与范围

针对 [#945](https://github.com/jingyaogong/minimind/pull/945) 的确切 head `b18b55af63d5e63bccc711cbca72a2e0dcbc06bf`，核查了 MiniMind 当前原生 `/generate` 请求与 SGLang 官方 **v0.5.23** 的对应实现。官方 tag 指向不可变提交 **`297e54fd105e974cdf401e5b4d20253a22c16f90`**，见 [tag 对象](https://api.github.com/repos/sgl-project/sglang/git/tags/7df5dd09bc2f0e7e1509a1f7c2a40dff00ad8a01)。以下源码链接均固定到该提交。

在核查的普通非 streaming 生成、EOS 停止及普通非推测解码零预算路径中，没有发现需要放宽 #945 完整性校验的兼容性问题。此结论来自源码路径核查，未运行真实 SGLang 服务，也未覆盖其他版本、所有服务配置或推测解码组合。生产补丁保持不变。

## MiniMind 实际请求与解析

[适配器代码](https://github.com/Afloat16/minimind/blob/b18b55af63d5e63bccc711cbca72a2e0dcbc06bf/trainer/rollout_engine.py) 发送批量 `input_ids`，设置 `return_logprob=true`、生成预算、温度和停止 token；请求未开启 streaming，也未请求 top logprobs 或指定 token 的分数。

解析时直接读取返回的原始 token IDs，并从 `meta_info.output_token_logprobs` 提取分数。因此，显示文本是否包含 EOS 不参与 token 数量计算。每个生成 token 应对应一个有效的行为策略 logprob；缺失时补零或多余时任意截断会改变后续 PPO/GRPO 的重要性比例。

## 普通生成与 EOS 的对应关系

| 待确认的问题 | 官方实现 | 对 #945 的含义 |
|---|---|---|
| 默认非 streaming 响应是否包含 IDs | [tokenizer manager](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/tokenizer_manager.py#L2665-L2670) 在完成响应中包含顶层 `output_ids` | 当前请求无需额外开关即可读取原始 IDs |
| EOS 是否被原始序列去掉 | [停止序列定义](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_batch.py#L1508-L1513) 保留触发停止的 token | 显示文本省略 EOS 不等于原始 token 缺失 |
| 文本解码是否改变 IDs 或分数 | [detokenizer](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/detokenizer_manager.py#L467-L491) 继续传递原始 IDs 和 logprobs | MiniMind 没有从文本重新 tokenize，文本显示不影响对齐 |
| 正长度输出的两组数据是否按同一前缀裁剪 | [发送 IDs](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/output_streamer.py#L489-L492) 使用停止前缀；[发送分数](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/output_streamer.py#L612-L624) 使用对应长度 | 普通正长度响应符合 token/logprob 数量约束 |

## 普通零预算路径

零预算需要同时检查分数的生成端与发送端。发送端的最小切片上限本身不会产生分数，不能据此推断实际响应必定有一条额外 logprob。

对当前请求参数及统一的普通非推测解码零预算 batch，相关路径如下：

1. [请求判定](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_batch.py#L1500-L1506) 将其归为 prefill-only。
2. [worker](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/tp_worker.py#L742-L762) 跳过普通采样，只尝试专用分数计算；[输出分数字段](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/logits_processor.py#L239-L243) 默认为空。
3. 没有 top/token-ID 分数请求时，[专用计算](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/logprob_processor.py#L967-L975) 直接返回。
4. [结果收集](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/logprob_result_processor.py#L319-L335) 仅在存在输出分数时追加。对空列表取至多一项仍为空。

这一路径与 `output_ids=[]`、`output_token_logprobs=[]` 相容，#945 接受这样的空完成，并保持零完成 mask。以上是源码推导的字段关系，没有把构造的 JSON 示例计作服务器实测结果。

## 已执行验证与审阅入口

- [适配器验证日志](results/logs/sglang-rollout-integrity-fixed.log) 记录了合法 token/概率、空完成，以及缺失、多余、非有限概率和错误响应 batch 的处理。
- [原问题日志](results/logs/sglang-rollout-integrity-baseline.log) 与 [回归脚本](root_checks.py) 可对照阅读；复现命令为 `python root_checks.py "$MINIMIND_CANDIDATE" sglang_integrity`，环境和 checkout 准备见 [README](README.md)。
- 本次新增的是固定官方版本的协议核查资料，没有新增服务集成测试，也没有更改 #945 的代码或既有测试结果。
