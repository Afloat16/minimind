# Prior-work audit

The audit covered the 176 visible upstream pull requests through #925, including open, merged, and closed-unmerged work. Python and repository-rule patches were inspected, with extra pages for large pull requests. Related model, PPO, rollout, and agent commit history was also checked. The latest-PR collection and upstream master were checked again before handoff; no newer PR was present and master remained cc9629e5.

The index in `prior-pr-index.json` records the audited PR set. The selected changes had no equivalent patch in that inspected set. This records the evidence checked; it does not claim access to unpublished work.

## Excluded prior Afloat16 contributions

Every prior contribution below was excluded regardless of whether it had been merged.

| PR | Title | Status |
|---|---|---|
| [#905](https://github.com/jingyaogong/minimind/pull/905) | [fix] keep synchronous inference off the API event loop | merged |
| [#906](https://github.com/jingyaogong/minimind/pull/906) | [fix] validate invalid generation requests before inference | closed, unmerged |
| [#907](https://github.com/jingyaogong/minimind/pull/907) | [fix] accept boolean padding masks in manual attention | merged |
| [#908](https://github.com/jingyaogong/minimind/pull/908) | [fix] resolve the conversion tokenizer relative to the script | closed, unmerged |
| [#909](https://github.com/jingyaogong/minimind/pull/909) | [fix] keep empty supervised distillation batches finite | merged |
| [#910](https://github.com/jingyaogong/minimind/pull/910) | [fix] retain fp32 precision for DPO sequence log probabilities | closed, unmerged |
| [#911](https://github.com/jingyaogong/minimind/pull/911) | [fix] keep generated return sequences grouped by prompt | closed, unmerged |
| [#912](https://github.com/jingyaogong/minimind/pull/912) | [fix] ignore sampling temperature for greedy generation | merged |
| [#913](https://github.com/jingyaogong/minimind/pull/913) | [perf] project only the final token logits during generation | closed, unmerged |
| [#914](https://github.com/jingyaogong/minimind/pull/914) | [fix] stop generation on any configured EOS token | closed, unmerged |
| [#915](https://github.com/jingyaogong/minimind/pull/915) | [fix] honor stopping criteria in native token generation | closed, unmerged |
| [#916](https://github.com/jingyaogong/minimind/pull/916) | [fix] bound top-k sampling by the model vocabulary size | closed, unmerged |
| [#917](https://github.com/jingyaogong/minimind/pull/917) | [fix] reject LoRA checkpoints missing adapter matrices | closed, unmerged |
| [#918](https://github.com/jingyaogong/minimind/pull/918) | [fix] initialize LoRA adapters in their target layer dtype | closed, unmerged |
| [#919](https://github.com/jingyaogong/minimind/pull/919) | [fix] accumulate LoRA deltas before rounding merged weights | merged |
| [#920](https://github.com/jingyaogong/minimind/pull/920) | [fix] avoid recursively adding LoRA adapters to adapters | closed, unmerged |
| [#921](https://github.com/jingyaogong/minimind/pull/921) | [fix] normalize MoE auxiliary load over all routed assignments | closed, unmerged |
| [#922](https://github.com/jingyaogong/minimind/pull/922) | [fix] preserve native attention and token configuration in Qwen exports | closed, unmerged |
| [#923](https://github.com/jingyaogong/minimind/pull/923) | [fix] normalize residual gradient accumulation by its actual length | closed, unmerged |
| [#924](https://github.com/jingyaogong/minimind/pull/924) | [fix] honor the timezone selected for current-time tools | closed, unmerged |

## Candidates removed during the audit

- Token-embedding resize: the baseline already passed the compatibility checks with the tested Transformers version; no production submission was made.
- DDP `no_sync` accumulation: an equivalent pretraining change was found in #803; not submitted.
- Absolute checkpoint paths: related fixes were already present in #791 and #756; not submitted.

## Nearby work distinguished

- #869 covers SSE metadata and error handling, not output-limit finish reasons, double-encoded tool arguments, or text-content arrays.
- #882 covers malformed training tool records, not API argument encoding.
- #917 checks LoRA adapter matrices; the new conversion change checks the complete base-model checkpoint.
- #870 handles reverse MoE weight unpacking; the new conversion check does not change that mapping.

## Repository instructions

The audited tree contained no CONTRIBUTING file, AGENTS file, pull-request template, or stated concurrent-open-PR cap. The README welcomes pull requests.
