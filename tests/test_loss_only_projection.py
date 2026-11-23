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

from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_lora import apply_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer import train_full_sft, train_lora, train_pretrain


def make_model(**kwargs):
    torch.manual_seed(72)
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=128,
        vocab_size=kwargs.pop('vocab_size', 128), flash_attn=False, **kwargs,
    )
    return MiniMindForCausalLM(config).train()


class LossOnlyProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / 'model', local_files_only=True,
        )

    def compare_training(self, model, input_ids, labels, autocast=False, **forward_kwargs):
        candidate = copy.deepcopy(model)
        expected_optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        actual_optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.002)
        losses = []
        for current, optimizer, return_logits in (
            (model, expected_optimizer, True), (candidate, actual_optimizer, False),
        ):
            context = torch.autocast('cpu', dtype=torch.bfloat16) if autocast else nullcontext()
            with context:
                result = current(input_ids, labels=labels, return_logits=return_logits, **forward_kwargs)
                loss = result.loss + result.aux_loss
            if not return_logits:
                self.assertIsNone(result.logits)
            losses.append(loss.detach())
            loss.backward()
        tolerance = {'rtol': 0.02, 'atol': 0.001} if autocast else {'rtol': 1e-5, 'atol': 1e-7}
        if next(model.parameters()).dtype == torch.float64:
            tolerance = {'rtol': 1e-10, 'atol': 1e-12}
        torch.testing.assert_close(losses[0], losses[1], **tolerance)
        expected_parameters = dict(model.named_parameters())
        actual_parameters = dict(candidate.named_parameters())
        self.assertEqual(expected_parameters.keys(), actual_parameters.keys())
        for name, expected in expected_parameters.items():
            with self.subTest(parameter=name):
                actual = actual_parameters[name]
                self.assertIsNotNone(expected.grad)
                self.assertIsNotNone(actual.grad)
                self.assertTrue(torch.isfinite(actual.grad).all())
                torch.testing.assert_close(actual.grad, expected.grad, **tolerance)
        expected_optimizer.step()
        actual_optimizer.step()
        for name, expected in expected_parameters.items():
            with self.subTest(updated_parameter=name):
                torch.testing.assert_close(actual_parameters[name], expected, **tolerance)

    def test_masked_dense_and_moe_loss_gradients_and_updates_match_full_projection(self):
        ids = torch.tensor([[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]])
        labels = ids.clone()
        labels[0, :3] = -100
        labels[0, 5] = -100
        labels[1] = -100
        for use_moe in (False, True):
            for tied in (False, True):
                with self.subTest(use_moe=use_moe, tied=tied):
                    self.compare_training(make_model(use_moe=use_moe, tie_word_embeddings=tied,
                                                     num_experts_per_tok=2), ids, labels)

    def test_float64_precision_with_dense_and_sparse_supervision(self):
        ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        model = make_model().double()
        result = model(ids, labels=ids, return_logits=False)
        self.assertEqual(result.loss.dtype, torch.float64)
        self.compare_training(model, ids, ids)
        labels = ids.clone()
        labels[:, :2] = -100
        self.compare_training(make_model().double(), ids, labels)

    def test_selected_logit_windows_keep_existing_causal_loss_semantics(self):
        ids = torch.tensor([[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]])
        for selection, indices in ((3, slice(-3, None)), (torch.tensor([1, 3, 5]), [1, 3, 5])):
            with self.subTest(selection=selection):
                labels = ids[:, indices].clone()
                labels[1, 1] = -100
                self.compare_training(make_model(), ids, labels, logits_to_keep=selection)

    def test_bfloat16_autocast_preserves_loss_gradients_and_updates(self):
        ids = torch.tensor([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])
        labels = ids.clone()
        labels[:, :2] = -100
        labels[1, -2:] = -100
        self.compare_training(make_model(), ids, labels, autocast=True)

    def test_default_logits_and_no_label_hidden_state_contract(self):
        model = make_model().eval()
        ids = torch.tensor([[10, 20, 30]])
        expected = model(ids)
        explicit = model(ids, return_logits=True)
        hidden_only = model(ids, return_logits=False)
        self.assertEqual(expected.logits.shape, (1, 3, 128))
        self.assertIsNone(expected.loss)
        torch.testing.assert_close(explicit.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(hidden_only.hidden_states, expected.hidden_states, rtol=0, atol=0)
        self.assertIsNone(hidden_only.loss)
        self.assertIsNone(hidden_only.logits)
        tail = model(ids, logits_to_keep=1)
        torch.testing.assert_close(tail.logits, expected.logits[:, -1:], rtol=1e-5, atol=1e-7)

    def test_empty_shifted_targets_keep_existing_mean_loss_and_zero_gradients(self):
        for ids, labels in (
            (torch.tensor([[10, 20, 30]]), torch.full((1, 3), -100)),
            (torch.tensor([[10]]), torch.tensor([[10]])),
        ):
            with self.subTest(sequence_length=ids.size(1)):
                baseline = make_model()
                candidate = copy.deepcopy(baseline)
                expected = baseline(ids, labels=labels)
                actual = candidate(ids, labels=labels, return_logits=False)
                # Empty-mean CE is already NaN; this optimization does not change that policy.
                self.assertTrue(torch.isnan(expected.loss))
                self.assertTrue(torch.isnan(actual.loss))
                self.assertIsNone(actual.logits)
                expected.loss.backward()
                actual.loss.backward()
                for left, right in zip(baseline.parameters(), candidate.parameters()):
                    self.assertIsNotNone(right.grad)
                    self.assertTrue(torch.isfinite(right.grad).all())
                    torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)

    def test_torch_compile_autograd_handles_changing_supervision_masks(self):
        eager = make_model()
        compiled_model = copy.deepcopy(eager)
        compiled = torch.compile(compiled_model, backend='aot_eager')
        ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        for ignored in (1, 3, 2):
            with self.subTest(ignored_prefix=ignored):
                labels = ids.clone()
                labels[:, :ignored] = -100
                eager.zero_grad(set_to_none=True)
                compiled.zero_grad(set_to_none=True)
                expected = eager(ids, labels=labels, return_logits=False)
                actual = compiled(ids, labels=labels, return_logits=False)
                self.assertIsNone(actual.logits)
                torch.testing.assert_close(actual.loss, expected.loss)
                expected.loss.backward()
                actual.loss.backward()
                for left, right in zip(eager.parameters(), compiled_model.parameters()):
                    torch.testing.assert_close(right.grad, left.grad, rtol=1e-5, atol=1e-7)

    def test_three_real_trainers_preserve_updates_on_tokenized_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            pretrain_path = Path(directory) / 'pretrain.jsonl'
            sft_path = Path(directory) / 'sft.jsonl'
            pretrain_path.write_text('\n'.join(json.dumps({'text': text}, ensure_ascii=False)
                                               for text in ('天空为什么是蓝色的？', '光在大气中发生散射。')) + '\n')
            conversations = [
                [{'role': 'user', 'content': '什么是散射？'},
                 {'role': 'assistant', 'content': '光与微粒相互作用后改变传播方向。'}],
                [{'role': 'user', 'content': '为什么天空是蓝色的？'},
                 {'role': 'assistant', 'content': '较短波长的光更容易被大气散射。'}],
            ]
            sft_path.write_text('\n'.join(json.dumps({'conversations': value}, ensure_ascii=False)
                                         for value in conversations) + '\n')
            random.seed(0)
            datasets = {
                train_pretrain: PretrainDataset(str(pretrain_path), self.tokenizer, max_length=64),
                train_full_sft: SFTDataset(str(sft_path), self.tokenizer, max_length=64),
            }
            datasets[train_lora] = datasets[train_full_sft]
            for trainer, dataset in datasets.items():
                with self.subTest(trainer=trainer.__name__):
                    batches = [(ids.unsqueeze(0), labels.unsqueeze(0)) for ids, labels in dataset]
                    self.assertTrue(all((labels != -100).any() for _, labels in batches))
                    states, metrics = [], []
                    initial = None
                    for force_full_logits in (True, False):
                        model = make_model(vocab_size=len(self.tokenizer))
                        if trainer is train_lora:
                            apply_lora(model, rank=4)
                            for name, parameter in model.named_parameters():
                                parameter.requires_grad_('lora' in name)
                        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
                        optimizer = torch.optim.AdamW(trainable, lr=0.002)
                        args = SimpleNamespace(device='cpu', epochs=1, learning_rate=0.002, warmup_ratio=0,
                                               accumulation_steps=2, grad_clip=1.0, log_interval=1, save_interval=100)
                        logs = []
                        original_forward = model.forward

                        def full_forward(*args, **kwargs):
                            kwargs['return_logits'] = True
                            return original_forward(*args, **kwargs)

                        forward_context = patch.object(model, 'forward', full_forward) if force_full_logits else nullcontext()
                        with forward_context, patch.multiple(
                            trainer, create=True, args=args, model=model, optimizer=optimizer,
                            scaler=torch.amp.GradScaler('cpu', enabled=False), autocast_ctx=nullcontext(),
                            lm_config=model.config, Logger=lambda _: None, is_main_process=lambda: False,
                        ):
                            extra = (trainable,) if trainer is train_lora else ()
                            trainer.train_epoch(0, batches, len(batches), *extra,
                                                wandb=SimpleNamespace(log=logs.append))
                        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
                        metrics.append(logs)
                    self.assertTrue(any(not torch.equal(states[1][key], initial[key]) for key in initial))
                    for key in states[0]:
                        torch.testing.assert_close(states[1][key], states[0][key], rtol=1e-5, atol=1e-7)
                    for expected, actual in zip(metrics[0], metrics[1]):
                        self.assertAlmostEqual(expected['loss'], actual['loss'], places=5)


if __name__ == '__main__':
    unittest.main()
