"""Offline regressions for independent MiniMind changes; --repo selects a worktree."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--case', required=True)
args = parser.parse_args()
sys.path.insert(0, args.repo)
import torch
from model.model_minimind import Attention, MiniMindConfig, MiniMindForCausalLM, MOEFeedForward, precompute_freqs_cis

torch.set_num_threads(1)

def config(**kwargs):
    values = dict(hidden_size=32, num_hidden_layers=1, vocab_size=128,
                  num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
                  max_position_embeddings=64, dropout=0.0)
    values.update(kwargs)
    return MiniMindConfig(**values)

class SingleTokenSDPA(unittest.TestCase):
    def compare(self, dtype, masked, causal=True):
        torch.manual_seed(123)
        efficient = Attention(config()).to(dtype).eval()
        manual = copy.deepcopy(efficient)
        manual.flash = False
        efficient.is_causal = manual.is_causal = causal
        hidden = torch.randn(2, 11, 32).to(dtype)
        cos, sin = precompute_freqs_cis(8, 11)
        mask = torch.tensor([[0, 0] + [1] * 9, [1] * 11]) if masked else None
        with torch.no_grad():
            _, cache = manual(hidden[:, :-1], (cos[:-1], sin[:-1]), use_cache=True,
                              attention_mask=mask[:, :-1] if masked else None)
            expected, _ = manual(hidden[:, -1:], (cos[-1:], sin[-1:]), past_key_value=cache,
                                  attention_mask=mask)
            real_sdpa = torch.nn.functional.scaled_dot_product_attention
            with mock.patch('torch.nn.functional.scaled_dot_product_attention', wraps=real_sdpa) as sdpa:
                actual, present = efficient(hidden[:, -1:], (cos[-1:], sin[-1:]),
                                            past_key_value=cache, attention_mask=mask, use_cache=True)
            self.assertEqual(sdpa.call_count, 1, 'cached one-token decoding must dispatch to SDPA')
            self.assertFalse(sdpa.call_args.kwargs['is_causal'])
            torch.testing.assert_close(actual, expected, atol=3e-3 if dtype == torch.bfloat16 else 1e-6,
                                       rtol=2e-2 if dtype == torch.bfloat16 else 1e-5)
            self.assertEqual(present[0].shape[1], 11)
            self.assertTrue(torch.isfinite(actual).all())

    def test_cached_fp32(self):
        self.compare(torch.float32, False)

    def test_cached_padding(self):
        self.compare(torch.float32, True)

    def test_cached_bfloat16(self):
        self.compare(torch.bfloat16, True)

    def test_noncausal(self):
        self.compare(torch.float32, False, False)

    def test_training_gradients(self):
        torch.manual_seed(77)
        efficient = Attention(config()).train()
        manual = copy.deepcopy(efficient)
        manual.flash = False
        x = torch.randn(2, 1, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        cache = (torch.randn(2, 7, 2, 8), torch.randn(2, 7, 2, 8))
        cos, sin = precompute_freqs_cis(8, 8)
        pos = (cos[-1:], sin[-1:])
        efficient(x, pos, past_key_value=cache)[0].square().sum().backward()
        manual(y, pos, past_key_value=cache)[0].square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
        for a, b in zip(efficient.parameters(), manual.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

class EmbeddingResize(unittest.TestCase):
    def check_resize(self, tied):
        torch.manual_seed(9)
        model = MiniMindForCausalLM(config(tie_word_embeddings=tied)).eval()
        original_input = model.model.embed_tokens.weight.detach().clone()
        original_output = model.lm_head.weight.detach().clone()
        model.resize_token_embeddings(136, mean_resizing=False)
        self.assertEqual(model.config.vocab_size, 136)
        self.assertEqual(model.get_input_embeddings().weight.shape, (136, 32))
        self.assertEqual(model.get_output_embeddings().weight.shape, (136, 32))
        torch.testing.assert_close(model.model.embed_tokens.weight[:128], original_input, rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head.weight[:128], original_output, rtol=0, atol=0)
        self.assertEqual(model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr(), tied)
        inputs = torch.tensor([[1, 135, 3]])
        with torch.no_grad():
            expected = model(inputs).logits
        self.assertEqual(expected.shape[-1], 136)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = MiniMindForCausalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                actual = loaded(inputs).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        model.resize_token_embeddings(120, mean_resizing=False)
        self.assertEqual(model.lm_head.weight.shape[0], 120)
        torch.testing.assert_close(model.model.embed_tokens.weight, original_input[:120], rtol=0, atol=0)

    def test_tied_resize_and_reload(self):
        self.check_resize(True)

    def test_untied_resize_and_reload(self):
        self.check_resize(False)

class CausalLossFP32(unittest.TestCase):
    def test_half_checkpoint_loss_is_finite(self):
        torch.manual_seed(3)
        model = MiniMindForCausalLM(config(hidden_size=8, num_attention_heads=2,
                                         num_key_value_heads=1, intermediate_size=16)).half().eval()
        inputs = torch.randint(0, 128, (1024, 16))
        result = model(inputs, labels=inputs)
        reference = torch.nn.functional.cross_entropy(result.logits[:, :-1].float().reshape(-1, 128),
                                                       inputs[:, 1:].reshape(-1))
        self.assertTrue(torch.isfinite(result.loss), f'native half loss was {result.loss.item()}')
        self.assertEqual(result.loss.dtype, torch.float32)
        torch.testing.assert_close(result.loss, reference, rtol=0, atol=0)
        result.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ignored_labels_and_fp32_parity(self):
        model = MiniMindForCausalLM(config()).eval()
        inputs = torch.tensor([[1, 4, 6, 0]])
        labels = torch.tensor([[-100, 4, 6, -100]])
        output = model(inputs, labels=labels)
        expected = torch.nn.functional.cross_entropy(output.logits[:, :2].reshape(-1, 128),
                                                      torch.tensor([4, 6]))
        torch.testing.assert_close(output.loss, expected)

    def test_float64_loss_is_not_downcast(self):
        model = MiniMindForCausalLM(config()).double().eval()
        inputs = torch.tensor([[1, 4, 6, 0]])
        output = model(inputs, labels=inputs)
        self.assertEqual(output.loss.dtype, torch.float64)

class RouterSoftmaxFP32(unittest.TestCase):
    def test_scaled_float16_router_gradient_survives_underflow(self):
        router = MOEFeedForward(config(hidden_size=2, intermediate_size=4, use_moe=True,
                                      num_experts=4, num_experts_per_tok=1, router_aux_loss_coef=0.0)).train()
        with torch.no_grad():
            router.gate.weight.zero_()
            router.gate.weight[0].fill_(9)
            for expert in router.experts:
                for parameter in expert.parameters():
                    parameter.fill_(0.25)
        with torch.autocast('cpu', dtype=torch.float16):
            output = router(torch.ones(1, 1, 2))
        (output.sum() * 32768).backward()
        self.assertTrue(torch.isfinite(router.gate.weight.grad).all())
        self.assertGreater(router.gate.weight.grad.abs().sum().item(), 0,
                           'fp16 softmax rounded routing probabilities to one/zero and killed the task gradient')

    def test_bfloat16_top1_winner_retains_gradient(self):
        router = MOEFeedForward(config(hidden_size=2, intermediate_size=4, use_moe=True,
                                      num_experts=4, num_experts_per_tok=1, router_aux_loss_coef=0.0)).train()
        with torch.no_grad():
            router.gate.weight.zero_()
            router.gate.weight[0].fill_(4.5)
            for expert in router.experts:
                for parameter in expert.parameters():
                    parameter.fill_(0.25)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = router(torch.ones(1, 1, 2))
        output.sum().backward()
        self.assertGreater(router.gate.weight.grad[0].abs().sum().item(), 0)

    def test_fp32_forward_and_backward_finite(self):
        router = MOEFeedForward(config(use_moe=True, num_experts_per_tok=2)).train()
        x = torch.randn(2, 4, 32, requires_grad=True)
        output = router(x)
        (output.square().mean() + router.aux_loss).backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())

class AgentHistoryJSON(unittest.TestCase):
    def test_serialized_calls_match_structured_history(self):
        from transformers import AutoTokenizer
        from dataset.lm_dataset import AgentRLDataset
        tokenizer = AutoTokenizer.from_pretrained(str(Path(args.repo) / 'model'), local_files_only=True)
        call = [{'id': 'call_1', 'type': 'function', 'function': {
            'name': 'calculate_math', 'arguments': {'expression': '2+3'}}}]
        tools = [{'type': 'function', 'function': {'name': 'calculate_math', 'description': 'Evaluate arithmetic',
                 'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}}}}]
        conversations = [
            {'role': 'system', 'content': 'Use tools.', 'tools': json.dumps(tools)},
            {'role': 'user', 'content': 'What is 2+3?'},
            {'role': 'assistant', 'content': '', 'tool_calls': json.dumps(call)},
            {'role': 'tool', 'content': '5'},
            {'role': 'user', 'content': 'Explain the result.'},
            {'role': 'assistant', 'content': 'The answer is five.'},
        ]
        before = copy.deepcopy(conversations)
        dataset = AgentRLDataset.__new__(AgentRLDataset)
        messages, loaded_tools = dataset.parse_conversations(conversations)
        self.assertIsInstance(messages[2]['tool_calls'], list)
        self.assertEqual(conversations, before, 'parsing must not mutate the source record')
        structured = copy.deepcopy(conversations)
        structured[2]['tool_calls'] = call
        expected_messages, _ = dataset.parse_conversations(structured)
        render = lambda data: tokenizer.apply_chat_template(data, tools=loaded_tools,
                                                            tokenize=False, add_generation_prompt=True)
        self.assertEqual(render(messages), render(expected_messages))
        self.assertIn('"name": "calculate_math"', render(messages))

cases = {
    'decode-single-token-sdpa': SingleTokenSDPA,
    'hf-token-embedding-resize': EmbeddingResize,
    'causal-loss-fp32-reduction': CausalLossFP32,
    'moe-router-softmax-fp32': RouterSoftmaxFP32,
    'agent-history-tool-call-json': AgentHistoryJSON,
}
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(cases[args.case])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
