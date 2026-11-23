import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward
from trainer import train_grpo
from trainer.rollout_engine import RolloutResult, compute_per_token_logps


def config(top_k=2, use_moe=True):
    return MiniMindConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=128,
        use_moe=use_moe, num_experts=4, num_experts_per_tok=top_k,
        router_aux_loss_coef=0.1, flash_attn=False,
    )


class FixedCompletions:
    """Keep the generated text fixed while changing only padding in a rollout."""

    def __init__(self, model, tokenizer, extra_padding):
        self.model, self.tokenizer, self.extra_padding = model, tokenizer, extra_padding

    def rollout(self, prompt_ids, attention_mask, num_generations, **kwargs):
        assert prompt_ids.size(0) == 1 and num_generations == 2
        texts = ['第一次测试。', '这是第二次完整测试。']
        completions = [self.tokenizer(text, add_special_tokens=False).input_ids
                       + [self.tokenizer.eos_token_id] for text in texts]
        width = max(map(len, completions)) + self.extra_padding
        completion_ids = torch.tensor([
            ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in completions
        ])
        completion_mask = torch.tensor([
            [1] * len(ids) + [0] * (width - len(ids)) for ids in completions
        ])
        output_ids = torch.cat([prompt_ids.repeat_interleave(2, dim=0), completion_ids], dim=1)
        full_mask = torch.cat([attention_mask.repeat_interleave(2, dim=0), completion_mask], dim=1)
        with torch.no_grad():
            old_logps = compute_per_token_logps(self.model, output_ids, width, attention_mask=full_mask)
        return RolloutResult(output_ids, completion_ids, old_logps, texts,
                             torch.full((2,), prompt_ids.size(1)), completion_mask)

    def update_policy(self, model):
        pass


class LengthReward:
    def get_score(self, messages, answer):
        return len(answer) / 10


class MoEAuxPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / 'model', local_files_only=True,
        )

    def make_model(self, top_k=2, use_moe=True):
        torch.manual_seed(87)
        return MiniMindForCausalLM(config(top_k, use_moe)).train()

    def test_padding_preserves_valid_logits_auxiliary_loss_and_router_gradients(self):
        ids = self.tokenizer('请解释为什么天空是蓝色的。', return_tensors='pt').input_ids
        for top_k in (1, 2):
            for side in ('left', 'right'):
                with self.subTest(top_k=top_k, side=side):
                    model = self.make_model(top_k)
                    baseline = model(ids, attention_mask=torch.ones_like(ids))
                    router = model.model.layers[0].mlp.gate.weight
                    expected_grad = torch.autograd.grad(baseline.aux_loss, router)[0]
                    padding = torch.zeros((1, 48), dtype=torch.long)
                    parts = [padding, ids] if side == 'left' else [ids, padding]
                    masks = [padding, torch.ones_like(ids)] if side == 'left' else [torch.ones_like(ids), padding]
                    padded = model(torch.cat(parts, dim=1), attention_mask=torch.cat(masks, dim=1))
                    actual_logits = padded.logits[:, -ids.size(1):] if side == 'left' else padded.logits[:, :ids.size(1)]
                    torch.testing.assert_close(actual_logits, baseline.logits, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(padded.aux_loss, baseline.aux_loss, rtol=1e-5, atol=1e-7)
                    actual_grad = torch.autograd.grad(padded.aux_loss, router)[0]
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)

    def test_interior_mask_and_unequal_lengths_match_only_the_valid_router_inputs(self):
        torch.manual_seed(19)
        x = torch.randn(2, 7, 32)
        mask = torch.tensor([[1, 0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1, 1]], dtype=torch.bool)
        module = MOEFeedForward(config()).train()
        module(x[mask].unsqueeze(0))
        expected = module.aux_loss.detach().clone()
        module(x, attention_mask=mask)
        torch.testing.assert_close(module.aux_loss, expected)

    def test_all_valid_mask_keeps_existing_auxiliary_scale(self):
        ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
        for top_k in (1, 2, 4):
            with self.subTest(top_k=top_k):
                model = self.make_model(top_k)
                unmasked = model(ids)
                masked = model(ids, attention_mask=torch.ones_like(ids))
                torch.testing.assert_close(masked.aux_loss, unmasked.aux_loss, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(masked.logits, unmasked.logits, rtol=0, atol=0)

    def test_all_padding_has_zero_differentiable_auxiliary_loss(self):
        model = self.make_model()
        ids = torch.zeros((2, 8), dtype=torch.long)
        result = model(ids, attention_mask=torch.zeros_like(ids))
        self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(result.aux_loss.item(), 0)
        result.aux_loss.backward()
        gradient = model.model.layers[0].mlp.gate.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(gradient.count_nonzero().item(), 0)

    def test_cached_forward_masks_only_current_router_tokens(self):
        model = self.make_model()
        prefix = torch.tensor([[10, 20, 30, 40]])
        with torch.no_grad():
            past = model(prefix, use_cache=True).past_key_values
        current = torch.tensor([[50, 60]])
        valid = model(current, past_key_values=past, attention_mask=torch.ones((1, 6), dtype=torch.long))
        padded = model(torch.tensor([[50, 60, 0]]), past_key_values=past,
                       attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]]))
        torch.testing.assert_close(padded.logits[:, :2], valid.logits)
        torch.testing.assert_close(padded.aux_loss, valid.aux_loss)

    def test_dense_model_still_accepts_padding_masks(self):
        model = self.make_model(use_moe=False)
        result = model(torch.tensor([[10, 20, 0]]), attention_mask=torch.tensor([[1, 1, 0]]))
        self.assertEqual(result.aux_loss.item(), 0)
        self.assertTrue(torch.isfinite(result.logits).all())

    def test_masked_auxiliary_gradients_are_finite_under_bfloat16_autocast(self):
        model = self.make_model()
        ids = torch.tensor([[10, 20, 30, 0], [40, 50, 0, 0]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = model(ids, attention_mask=ids.ne(0))
        self.assertTrue(torch.isfinite(result.aux_loss))
        result.aux_loss.backward()
        gradient = model.model.layers[0].mlp.gate.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.count_nonzero().item(), 0)

    def test_grpo_optimizer_update_does_not_depend_on_unused_padding_width(self):
        original = self.make_model()
        states = []
        prompt = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '请给出测试结果。'}],
            tokenize=False, add_generation_prompt=True,
        )
        args = SimpleNamespace(
            device='cpu', max_seq_len=32, num_generations=2, max_gen_len=16,
            debug_mode=False, loss_type='grpo', epsilon=0.2, beta=0.1,
            accumulation_steps=1, grad_clip=0, log_interval=1, save_interval=100,
            epochs=1,
        )
        for extra_padding in (0, 24):
            model = copy.deepcopy(original)
            ref_model = copy.deepcopy(original).eval().requires_grad_(False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            engine = FixedCompletions(model, self.tokenizer, extra_padding)
            with patch.multiple(train_grpo, create=True, args=args, model=model,
                                tokenizer=self.tokenizer, lm_config=model.config,
                                autocast_ctx=nullcontext(), optimizer=optimizer,
                                scheduler=scheduler, Logger=lambda _: None,
                                is_main_process=lambda: False):
                train_grpo.grpo_train_epoch(0, [{'prompt': [prompt]}], 1,
                                            engine, ref_model, LengthReward())
            states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        self.assertFalse(torch.equal(states[0]['model.layers.0.mlp.gate.weight'],
                                     original.state_dict()['model.layers.0.mlp.gate.weight']))
        for name in states[0]:
            with self.subTest(parameter=name):
                torch.testing.assert_close(states[0][name], states[1][name], rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
