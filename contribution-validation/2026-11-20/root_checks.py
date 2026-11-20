"""Run offline regressions against the repository passed as the first argument."""
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

torch.set_num_threads(1)


def config(vocab_size=64):
    return MiniMindConfig(hidden_size=16, num_hidden_layers=1, vocab_size=vocab_size,
                          num_attention_heads=2, num_key_value_heads=1,
                          intermediate_size=32, max_position_embeddings=1024)


def critic_norm():
    from trainer.train_ppo import CriticModel
    torch.manual_seed(8)
    model = CriticModel(config()).eval()
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.2, 2.4, 16))
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = model.value_head(model.model(ids)[0]).squeeze(-1)
    actual = model(ids)
    error = (actual - expected).abs().max().item()
    print(f"critic maximum error versus the backbone value head: {error:.9g}")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.square().mean().backward()
    assert model.value_head.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.model.norm.weight.grad).all()


class Batch(SimpleNamespace):
    def to(self, device):
        return self


def ppo_early_stop():
    import trainer.train_ppo as ppo
    from trainer.rollout_engine import RolloutResult
    torch.manual_seed(12)
    ppo.lm_config = config()
    ppo.actor_model = MiniMindForCausalLM(ppo.lm_config)
    ppo.critic_model = ppo.CriticModel(ppo.lm_config)
    ref = copy.deepcopy(ppo.actor_model).eval().requires_grad_(False)
    ids = torch.tensor([[3, 4, 5, 6, 7, 2]] * 4)
    with torch.no_grad():
        old = ref(ids).logits[:, :-1].log_softmax(-1).gather(2, ids[:, 1:, None]).squeeze(-1)[:, 2:]
    # Simulate a stale policy whose old log probabilities exceed the KL threshold.
    result = RolloutResult(ids, ids[:, 3:], old - 1000, ["answer"] * 4,
                           torch.tensor([3] * 4), torch.ones((4, 3), dtype=torch.long))
    engine = SimpleNamespace(rollout=lambda **kwargs: result, update_policy=lambda model: None)
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def __call__(self, prompts, **kwargs):
            return Batch(input_ids=ids[:, :3], attention_mask=torch.ones((4, 3), dtype=torch.long))
    ppo.tokenizer = Tokenizer()
    ppo.args = SimpleNamespace(device="cpu", max_seq_len=32, max_gen_len=3,
        debug_mode=False, gamma=1.0, lam=0.95, mini_batch_size=1,
        ppo_update_iters=2, debug_log_ratio=False, early_stop_kl=0.25,
        clip_epsilon=0.2, kl_coef=0.02, cliprange_value=0.2, vf_coef=0.5,
        accumulation_steps=1, grad_clip=1.0, save_interval=100, epochs=1)
    ppo.autocast_ctx = nullcontext()
    ppo.calculate_rewards = lambda prompts, responses, model: torch.zeros(4)
    ppo.is_main_process = lambda: False
    ppo.actor_optimizer = torch.optim.AdamW(ppo.actor_model.parameters(), lr=0.01, weight_decay=0.1)
    ppo.critic_optimizer = torch.optim.AdamW(ppo.critic_model.parameters(), lr=0.01, weight_decay=0.1)
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ppo.actor_optimizer, 8)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ppo.critic_optimizer, 8)
    before = {k: v.clone() for k, v in ppo.actor_model.state_dict().items()}
    with (patch.object(ppo.actor_model, "forward", wraps=ppo.actor_model.forward) as forward,
          patch.object(ppo.actor_optimizer, "step", wraps=ppo.actor_optimizer.step) as step):
        ppo.ppo_train_epoch(0, [{"prompt": ["prompt"] * 4}], 1, engine, ref,
                            actor_scheduler, critic_scheduler, None)
        print(f"PPO forwards after KL violation: {forward.call_count}; optimizer updates: {step.call_count}")
        assert forward.call_count == 1
        assert step.call_count == 0
    for key, value in before.items():
        torch.testing.assert_close(ppo.actor_model.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ppo.actor_model.parameters())
    result.per_token_logps = old
    ppo.args.early_stop_kl = 1e9
    with patch.object(ppo.actor_optimizer, "step", wraps=ppo.actor_optimizer.step) as step:
        ppo.ppo_train_epoch(0, [{"prompt": ["prompt"] * 4}], 1, engine, ref,
                            actor_scheduler, critic_scheduler, None)
        assert step.call_count == 8, "the following normal rollout must still update"
    assert all(torch.isfinite(p).all() for p in ppo.actor_model.parameters())
    with torch.no_grad():
        old = ppo.actor_model(ids).logits[:, :-1].log_softmax(-1).gather(2, ids[:, 1:, None]).squeeze(-1)[:, 2:]
    mixed_old = old.clone()
    mixed_old[1:] -= 1000
    mixed_result = RolloutResult(ids, ids[:, 3:], mixed_old, ["answer"] * 4,
                                 torch.tensor([3] * 4), torch.ones((4, 3), dtype=torch.long))
    rejected_result = RolloutResult(ids, ids[:, 3:], old - 1000, ["answer"] * 4,
                                    torch.tensor([3] * 4), torch.ones((4, 3), dtype=torch.long))
    rollouts = iter([mixed_result, rejected_result])
    pending_engine = SimpleNamespace(rollout=lambda **kwargs: next(rollouts), update_policy=lambda model: None)
    ppo.args.accumulation_steps, ppo.args.early_stop_kl = 4, 0.25

    # The preceding eight scheduler steps reached zero LR. Use positive LR to
    # prove that the pending gradients cause a real update before the next
    # rollout in this same epoch is rejected without another optimizer step.
    for optimizer in (ppo.actor_optimizer, ppo.critic_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = 0.01
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ppo.actor_optimizer, 8)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ppo.critic_optimizer, 8)
    models = {"actor": ppo.actor_model, "critic": ppo.critic_model}
    before_pending = {name: {k: v.clone() for k, v in model.state_dict().items()}
                      for name, model in models.items()}
    after_updates, gradient_sums = {}, {}

    def checked_step(name, optimizer):
        original_step = optimizer.step
        def update(*args, **kwargs):
            assert optimizer.param_groups[0]["lr"] > 0
            gradients = [p.grad for p in models[name].parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            gradient_sums[name] = sum(g.abs().sum().item() for g in gradients)
            assert gradient_sums[name] > 0, f"{name} lost its pending gradients"
            output = original_step(*args, **kwargs)
            after_updates[name] = {k: v.clone() for k, v in models[name].state_dict().items()}
            return output
        return update

    with (patch.object(torch, "randperm", side_effect=lambda n, **kwargs: torch.arange(n)),
          patch.object(ppo.actor_model, "forward", wraps=ppo.actor_model.forward) as forward,
          patch.object(ppo.actor_optimizer, "step", side_effect=checked_step("actor", ppo.actor_optimizer)) as actor_step,
          patch.object(ppo.critic_optimizer, "step", side_effect=checked_step("critic", ppo.critic_optimizer)) as critic_step):
        ppo.ppo_train_epoch(0, [{"prompt": ["prompt"] * 4}] * 2, 2, pending_engine, ref,
                            actor_scheduler, critic_scheduler, None)
        assert forward.call_count == 3, forward.call_count
        assert actor_step.call_count == critic_step.call_count == 1
    for name, model in models.items():
        assert any(not torch.equal(value, after_updates[name][key])
                   for key, value in before_pending[name].items()), f"{name} parameters did not update"
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, after_updates[name][key], rtol=0, atol=0)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"Same-epoch residual flush followed by rejection: 3 forwards, 1 actor/critic update; pending gradient sums {gradient_sums}")
    print("Normal rollout after rejection and positive-LR pending-gradient flush both passed")


def sglang_integrity():
    from trainer.rollout_engine import SGLangRolloutEngine
    engine = object.__new__(SGLangRolloutEngine)
    engine.base_url, engine.timeout = "http://fixture.invalid", 2
    engine.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
        decode=lambda ids, **kwargs: " ".join(map(str, ids)))
    def run(results):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: results)
        engine.http = SimpleNamespace(post=lambda *args, **kwargs: response)
        return engine.rollout(torch.tensor([[3, 4]]), torch.ones((1, 2), dtype=torch.long), 1, 4)
    good = {"output_ids": [5, 2], "meta_info": {"output_token_logprobs": [[-0.5, 5, None], [-0.7, 2, None]]}}
    result = run(good)
    torch.testing.assert_close(result.per_token_logps, torch.tensor([[-0.5, -0.7]]))
    cases = {
        "missing logprob": {"output_ids": [5, 2], "meta_info": {"output_token_logprobs": [[-0.7, 2, None]]}},
        "extra logprob": {"output_ids": [5], "meta_info": {"output_token_logprobs": [-0.5, -0.7]}},
        "nan logprob": {"output_ids": [5], "meta_info": {"output_token_logprobs": [float("nan")]}},
        "empty response batch": [],
        "extra response": [good, good],
    }
    rejected = []
    for name, payload in cases.items():
        try:
            run(payload)
        except RuntimeError:
            rejected.append(name)
        except Exception as exc:
            print(f"{name}: unexpected {type(exc).__name__}: {exc}")
    print(f"Rejected invalid rollout fixtures: {len(rejected)}/{len(cases)} ({', '.join(rejected)})")
    assert len(rejected) == len(cases)
    empty = run({"output_ids": [], "meta_info": {"output_token_logprobs": []}})
    assert empty.completion_mask.sum() == 0


def agent_eos():
    import trainer.train_agent as agent
    from trainer.rollout_engine import RolloutResult
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"), local_files_only=True)
    texts = ['<tool_call>\n{"name":"calculate_math","arguments":{"expression":"1+1"}}\n</tool_call>', 'The result is 2.']
    class Engine:
        def __init__(self): self.turn = 0
        def rollout(self, prompt_ids, attention_mask, **kwargs):
            text = texts[self.turn]
            self.turn += 1
            completion = torch.tensor([tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]])
            return RolloutResult(torch.cat([prompt_ids, completion], 1), completion,
                torch.full(completion.shape, -1.0), [text], torch.tensor([prompt_ids.shape[1]]), torch.ones_like(completion))
    sample = agent.rollout_single(Engine(), tokenizer, [{"role": "user", "content": "Calculate 1+1."}],
                                  agent.TOOLS[:1], thinking_ratio=0, device="cpu")
    _, _, prompt, response, mask, old_logps, _, _ = sample
    generated_ends = [i for i, (token, logp) in enumerate(zip(response, old_logps))
                      if token == tokenizer.eos_token_id and logp == -1.0]
    print(f"Generated EOS tokens supervised: {sum(mask[i] for i in generated_ends)}/{len(generated_ends)}")
    assert len(generated_ends) == 2
    assert all(mask[i] == 1 for i in generated_ends)
    assert any(v == 0 for v in mask[generated_ends[0] + 1:generated_ends[1]])
    # The real training loop must keep the second generated turn after the first EOS.
    source = (ROOT / "trainer/train_agent.py").read_text(encoding="utf-8")
    start = source.index("        completion_mask = full_response_masks[:, 1:]")
    end = source.index("        if args.debug_mode", start)
    namespace = {"full_response_masks": torch.tensor([[0] * len(prompt) + mask], dtype=torch.float32),
                 "input_ids": torch.tensor([prompt + response]), "tokenizer": tokenizer,
                 "torch": torch, "args": SimpleNamespace(device="cpu")}
    exec("\n".join(line[8:] for line in source[start:end].splitlines()), namespace)
    learned_mask = namespace["completion_mask"][0]
    assert learned_mask[len(prompt) + generated_ends[1] - 1] == 1
    assert learned_mask.sum() == sum(mask)
    torch.manual_seed(42)
    model = MiniMindForCausalLM(config(len(tokenizer)))
    ids = torch.tensor([prompt + response])
    logps = model(ids).logits[:, :-1].log_softmax(-1).gather(2, ids[:, 1:, None]).squeeze(-1)
    loss = -(logps * learned_mask).sum() / learned_mask.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad[tokenizer.eos_token_id].abs().sum() > 0


if __name__ == "__main__":
    globals()[sys.argv[2]]()
    print("PASS", sys.argv[2])
