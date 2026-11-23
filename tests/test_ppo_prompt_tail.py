import copy
import json
import random
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer import train_ppo
from trainer.rollout_engine import TorchRolloutEngine


class RecordingRollout(TorchRolloutEngine):
    def rollout(self, prompt_ids, attention_mask, **kwargs):
        self.prompt_ids = prompt_ids.detach().clone()
        self.attention_mask = attention_mask.detach().clone()
        return super().rollout(prompt_ids=prompt_ids, attention_mask=attention_mask, **kwargs)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class LengthReward:
    def get_score(self, messages, answer):
        return len(answer) / 10


class PPOPromptTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / 'model', local_files_only=True,
        )

    def prompt(self, long=False):
        messages = []
        if long:
            messages.extend([
                {'role': 'user', 'content': '请阅读背景资料。' + '航空航天技术需要逐步进行可靠的工程验证。' * 70},
                {'role': 'assistant', 'content': '我已经阅读了背景资料。'},
            ])
        messages.extend([
            {'role': 'user', 'content': '现在请概括离子推进的优点。'},
            {'role': 'assistant', 'content': '离子推进的效率较高。'},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.jsonl'
            path.write_text(json.dumps({'conversations': messages}, ensure_ascii=False) + '\n')
            random.seed(0)
            dataset = RLAIFDataset(str(path), self.tokenizer, thinking_ratio=0)
            return dataset[0]['prompt']

    def train_one_batch(self, prompts, budget):
        torch.manual_seed(31)
        config = MiniMindConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=64,
            max_position_embeddings=128, flash_attn=False,
        )
        actor = MiniMindForCausalLM(config)
        reference = copy.deepcopy(actor).eval().requires_grad_(False)
        critic = train_ppo.CriticModel(config)
        critic.load_state_dict(actor.state_dict(), strict=False)
        initial_actor = actor.model.layers[0].self_attn.q_proj.weight.detach().clone()
        initial_critic = critic.value_head.weight.detach().clone()
        actor_optimizer, critic_optimizer = CountingSGD(actor.parameters()), CountingSGD(critic.parameters())
        actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1)
        critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1)
        engine = RecordingRollout(actor, self.tokenizer, device='cpu')
        args = SimpleNamespace(
            device='cpu', max_seq_len=budget, max_gen_len=4, debug_mode=False,
            debug_log_ratio=False, gamma=1.0, lam=0.95, mini_batch_size=len(prompts),
            ppo_update_iters=1, early_stop_kl=100, clip_epsilon=0.2,
            kl_coef=0.02, cliprange_value=0.2, vf_coef=0.5,
            accumulation_steps=1, grad_clip=1.0, save_interval=100, epochs=1,
        )
        with patch.multiple(train_ppo, create=True, args=args, tokenizer=self.tokenizer,
                            actor_model=actor, critic_model=critic, lm_config=config,
                            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                            autocast_ctx=nullcontext(), Logger=lambda _: None,
                            is_main_process=lambda: False):
            train_ppo.ppo_train_epoch(0, [{'prompt': prompts}], 1, engine, reference,
                                      actor_scheduler, critic_scheduler, LengthReward())
        self.assertEqual((actor_optimizer.steps, critic_optimizer.steps), (1, 1))
        self.assertFalse(torch.equal(initial_actor, actor.model.layers[0].self_attn.q_proj.weight))
        self.assertFalse(torch.equal(initial_critic, critic.value_head.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in actor.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in critic.parameters()))
        return engine

    def test_long_dataset_prompt_keeps_latest_query_and_assistant_header(self):
        prompt = self.prompt(long=True)
        all_ids = self.tokenizer(prompt).input_ids
        self.assertGreater(len(all_ids), 768)
        engine = self.train_one_batch([prompt], 64)
        self.assertEqual(engine.prompt_ids[0].tolist(), all_ids[-64:])
        tail = self.tokenizer.decode(engine.prompt_ids[0])
        self.assertIn('现在请概括离子推进的优点。', tail)
        self.assertTrue(tail.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertEqual(self.tokenizer.truncation_side, 'right')

    def test_mixed_lengths_keep_left_padding_aligned_after_tail_truncation(self):
        long_prompt, short_prompt = self.prompt(long=True), self.prompt()
        engine = self.train_one_batch([long_prompt, short_prompt], 64)
        long_ids, short_ids = self.tokenizer(long_prompt).input_ids, self.tokenizer(short_prompt).input_ids
        self.assertEqual(engine.prompt_ids[0].tolist(), long_ids[-64:])
        self.assertEqual(engine.attention_mask[0].tolist(), [1] * 64)
        self.assertEqual(engine.prompt_ids[1].tolist(), [self.tokenizer.pad_token_id] * (64 - len(short_ids)) + short_ids)
        self.assertEqual(engine.attention_mask[1].tolist(), [0] * (64 - len(short_ids)) + [1] * len(short_ids))

    def test_short_prompts_keep_existing_token_ids_and_masks(self):
        prompts = [self.prompt(), self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '你好'}], tokenize=False, add_generation_prompt=True,
        )]
        previous = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                  truncation=True, max_length=64, padding_side='left')
        engine = self.train_one_batch(prompts, 64)
        torch.testing.assert_close(engine.prompt_ids, previous.input_ids, rtol=0, atol=0)
        torch.testing.assert_close(engine.attention_mask, previous.attention_mask, rtol=0, atol=0)

    def test_prompt_exactly_at_budget_is_unchanged(self):
        prompt = self.prompt()
        ids = self.tokenizer(prompt).input_ids
        engine = self.train_one_batch([prompt], len(ids))
        self.assertEqual(engine.prompt_ids[0].tolist(), ids)
        self.assertEqual(engine.attention_mask[0].tolist(), [1] * len(ids))


if __name__ == '__main__':
    unittest.main()
