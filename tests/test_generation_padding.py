import unittest

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class RepetitionPaddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(65)
        self.model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=16, num_hidden_layers=1, vocab_size=8,
            num_attention_heads=2, num_key_value_heads=1,
            intermediate_size=32, max_position_embeddings=32,
            flash_attn=False,
        )).eval()

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(
            input_ids, attention_mask=attention_mask, do_sample=False,
            max_new_tokens=3, eos_token_id=None, repetition_penalty=2.0,
            **kwargs,
        )

    def test_left_padding_does_not_change_repetition_penalties(self):
        prompt = torch.tensor([[3, 4]])
        padded = torch.tensor([[0, 0, 3, 4]])
        for use_cache in (False, True):
            expected = self.generate(prompt, use_cache=use_cache)[:, 2:]
            # Zero is a real output token here, not an EOS or excluded token.
            self.assertEqual(expected[0, 0].item(), 0)
            for dtype in (torch.long, torch.bool):
                with self.subTest(use_cache=use_cache, mask_dtype=dtype):
                    mask = torch.tensor([[0, 0, 1, 1]], dtype=dtype)
                    original_mask = mask.clone()
                    actual = self.generate(padded, mask, use_cache=use_cache)[:, 4:]
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(mask, original_mask)

    def test_valid_zero_tokens_still_receive_the_penalty(self):
        prompt = torch.tensor([[0, 3, 4]])
        mask = torch.ones_like(prompt)
        with torch.no_grad():
            scores = self.model(prompt, attention_mask=mask).logits[0, -1].clone()
        seen = torch.tensor([0, 3, 4])
        scores[seen] = torch.where(scores[seen] > 0, scores[seen] / 2, scores[seen] * 2)
        expected = scores.argmax().item()
        for attention_mask in (None, mask):
            with self.subTest(has_mask=attention_mask is not None):
                actual = self.generate(prompt, attention_mask)
                self.assertEqual(actual[0, 3].item(), expected)

    def test_each_batch_row_uses_its_own_prompt_mask(self):
        batch = torch.tensor([[0, 0, 3, 4], [3, 4, 3, 4]])
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        actual = self.generate(batch, mask)[:, 4:]
        for index, prompt in enumerate((batch[:1, 2:], batch[1:])):
            with self.subTest(row=index):
                expected = self.generate(prompt)[:, prompt.shape[1]:]
                torch.testing.assert_close(actual[index:index + 1], expected)


if __name__ == "__main__":
    unittest.main()
