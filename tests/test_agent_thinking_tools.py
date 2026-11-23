import json
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from trainer.rollout_engine import RolloutResult
from trainer.train_agent import TOOLS, parse_tool_calls, rollout_single


def math_call(expression):
    payload = {"name": "calculate_math", "arguments": {"expression": expression}}
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


class ScriptedRolloutEngine:
    def __init__(self, tokenizer, completions):
        self.tokenizer = tokenizer
        self.completions = iter(completions)

    def rollout(self, prompt_ids, **kwargs):
        text = next(self.completions)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        completion_ids = prompt_ids.new_tensor([ids + [self.tokenizer.eos_token_id]])
        return RolloutResult(
            torch.cat([prompt_ids, completion_ids], dim=1),
            completion_ids,
            torch.zeros_like(completion_ids, dtype=torch.float32),
            [text],
            prompt_ids.new_tensor([prompt_ids.size(1)]),
            torch.ones_like(completion_ids),
        )


class AgentThinkingToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")

    def rollout_messages(self, completion):
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        engine = ScriptedRolloutEngine(self.tokenizer, [completion, "</think>\nThe answer is 4."])
        rollout_single(engine, self.tokenizer, messages, [TOOLS[0]], max_turns=2,
                       thinking_ratio=1, device="cpu")
        return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]

    def test_prefilled_thinking_executes_only_the_final_tool_call(self):
        completion = f"I could try {math_call('999 * 999')} but that is unrelated.\n</think>\n{math_call('2 + 2')}"
        self.assertEqual(self.rollout_messages(completion), [{"result": "4"}])

    def test_incomplete_thinking_does_not_execute_a_tool(self):
        self.assertEqual(self.rollout_messages(f"I am considering {math_call('999 * 999')}"), [])

    def test_explicit_thinking_prefix_ignores_reasoning_calls(self):
        text = f"<think>Consider {math_call('999 * 999')}</think>{math_call('2 + 2')}"
        self.assertEqual(parse_tool_calls(text), [{"name": "calculate_math", "arguments": {"expression": "2 + 2"}}])

    def test_thinking_delimiter_inside_arguments_is_preserved(self):
        call = {"name": "translate_text", "arguments": {"text": "Write </think> here", "target_language": "chinese"}}
        self.assertEqual(parse_tool_calls(f"<tool_call>{json.dumps(call)}</tool_call>"), [call])


if __name__ == "__main__":
    unittest.main()
