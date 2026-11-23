import asyncio
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer
from starlette.requests import ClientDisconnect

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import serve_openai_api as api


class PausingMiniMind(MiniMindForCausalLM):
    def __init__(self, token_id):
        super().__init__(MiniMindConfig(hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1,
                                        intermediate_size=32, max_position_embeddings=2048))
        self.token_id = token_id
        self.forward_calls = 0
        self.release = threading.Event()
        self.finished = threading.Event()

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        self.forward_calls += 1
        result.logits.fill_(-100)
        result.logits[..., self.token_id] = 100
        if self.forward_calls == 2:
            self.release.wait(5)
        return result

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.finished.set()


class StreamCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "model")
        cls.space_id, = cls.tokenizer(" ", add_special_tokens=False)["input_ids"]

    def setUp(self):
        self.model = PausingMiniMind(self.space_id).eval()
        self.globals = patch.multiple(api, model=self.model, tokenizer=self.tokenizer, device="cpu", create=True)
        self.globals.start()

    def tearDown(self):
        self.model.release.set()
        self.model.finished.wait(5)
        self.globals.stop()

    def test_closing_generator_stops_native_generation(self):
        stream = api.generate_stream_response([{"role": "user", "content": "hello"}], 0, 1, 64)
        self.assertIn("choices", json.loads(next(stream)))
        stream.close()
        self.model.release.set()
        self.assertTrue(self.model.finished.wait(5))
        self.assertLessEqual(self.model.forward_calls, 2)

    def test_asgi_send_failure_stops_native_generation(self):
        response = api.chat_completions(api.ChatRequest(model="minimind", messages=[{"role": "user", "content": "hello"}],
                                                        stream=True, temperature=0, max_tokens=64))

        async def fail_on_first_body():
            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    raise OSError("client disconnected")

            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}}
            with self.assertRaises(ClientDisconnect):
                await response(scope, receive, send)

        asyncio.run(fail_on_first_body())
        self.model.release.set()
        self.assertTrue(self.model.finished.wait(5))
        self.assertLessEqual(self.model.forward_calls, 2)

    def test_asgi_disconnect_stops_native_generation(self):
        response = api.chat_completions(api.ChatRequest(model="minimind", messages=[{"role": "user", "content": "hello"}],
                                                        stream=True, temperature=0, max_tokens=64))

        async def disconnect_after_first_body():
            first_body = asyncio.Event()

            async def receive():
                await first_body.wait()
                asyncio.get_running_loop().call_later(0.05, self.model.release.set)
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    first_body.set()

            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}
            await asyncio.wait_for(response(scope, receive, send), timeout=5)

        asyncio.run(disconnect_after_first_body())
        self.model.release.set()
        self.assertTrue(self.model.finished.wait(5))
        self.assertLessEqual(self.model.forward_calls, 2)

    def test_successful_generation_keeps_all_tokens(self):
        self.model.release.set()
        chunks = list(api.generate_stream_response([{"role": "user", "content": "hello"}], 0, 1, 3))
        self.assertTrue(self.model.finished.wait(5))
        self.assertEqual(self.model.forward_calls, 3)
        self.assertEqual(json.loads(chunks[-1])["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
