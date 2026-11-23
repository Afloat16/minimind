import ast
import contextlib
import io
import json
import os
import random
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import convert_model
from trainer.train_agent import TOOLS
from trainer.trainer_utils import safe_math_eval, setup_seed


REPO = Path(__file__).resolve().parents[1]


class State(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


class ScriptedModel:
    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = iter(turns)
        self.calls = 0
        self.delivered_chunks = 0

    def generate(self, input_ids, streamer, **kwargs):
        self.calls += 1
        streamer.put(input_ids)
        for item in next(self.turns):
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item()
                continue
            streamer.put(torch.tensor(self.tokenizer(item, add_special_tokens=False).input_ids))
            self.delivered_chunks += 1
        streamer.end()


class WebGenerationErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep Streamlit's page setup out of unit tests; execute the actual
        # generation, model-loader and main function bodies with widget fixtures.
        syntax = ast.parse((REPO / "scripts/web_demo.py").read_text())
        names = {"stream_model_response", "execute_tool", "main"}
        cls.code = compile(ast.Module(body=[node for node in syntax.body
                                           if isinstance(node, ast.FunctionDef) and node.name in names],
                                      type_ignores=[]), str(REPO / "scripts/web_demo.py"), "exec")
        cls.directory = tempfile.TemporaryDirectory(prefix="minimind-web-loader-")
        directory = Path(cls.directory.name)
        config = MiniMindConfig(hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=4, num_key_value_heads=2)
        native = MiniMindForCausalLM(config)
        checkpoint, exported = directory / "tiny.pth", directory / "exported"
        torch.save(native.state_dict(), checkpoint)
        cwd = Path.cwd()
        try:
            os.chdir(REPO / "scripts")
            with mock.patch.object(convert_model, "lm_config", config, create=True), contextlib.redirect_stdout(io.StringIO()):
                convert_model.convert_torch2transformers_minimind(str(checkpoint), str(exported), dtype=torch.float32)
        finally:
            os.chdir(cwd)
        loader = next(node for node in syntax.body if isinstance(node, ast.FunctionDef)
                      and node.name == "load_model_tokenizer")
        namespace = {"st": SimpleNamespace(cache_resource=lambda function: function),
                     "AutoModelForCausalLM": AutoModelForCausalLM, "AutoTokenizer": AutoTokenizer,
                     "torch": torch, "device": "cpu"}
        exec(compile(ast.Module(body=[loader], type_ignores=[]), str(REPO / "scripts/web_demo.py"), "exec"), namespace)
        cls.native, cls.tokenizer = namespace["load_model_tokenizer"](str(exported))
        assert type(cls.native).__name__ == "MiniMindForCausalLM"
        assert cls.native.config.max_position_embeddings == 32768

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_ui(self, model, prompt="你好，请简短回答。", tools=False,
               expected_workers=1, interrupt=None, release=None):
        state = State(history_chat_num=0, max_new_tokens=16, temperature=0.8,
                      selected_tools=["calculate_math"] if tools else [], enable_thinking=False)
        rendered, streamers, workers, worker_errors, consumer_errors = [], [], [], [], []
        created = threading.Condition()
        completed = threading.Event()

        def render(text, **kwargs):
            rendered.append(text)
            if interrupt is not None and text:
                raise interrupt

        st = SimpleNamespace(session_state=state, markdown=lambda *args, **kwargs: None,
                             chat_input=lambda **kwargs: prompt,
                             empty=lambda: SimpleNamespace(markdown=render))

        class RecordingStreamer(TextIteratorStreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                streamers.append(self)

        class RecordingThread(threading.Thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.completed = threading.Event()
                with created:
                    workers.append(self)
                    created.notify_all()

            def run(self):
                try:
                    super().run()
                except BaseException as error:
                    worker_errors.append(error)
                finally:
                    self.completed.set()

        namespace = dict(st=st, load_model_tokenizer=lambda path: (model, self.tokenizer),
                         model_path="cached-local-export", get_text=lambda key: key,
                         random=random, setup_seed=setup_seed, TOOLS=TOOLS, device="cpu",
                         TextIteratorStreamer=RecordingStreamer, Thread=RecordingThread,
                         Event=threading.Event, safe_math_eval=safe_math_eval,
                         process_assistant_content=lambda text, **kwargs: text,
                         re=re, json=json)
        exec(self.code, namespace)

        def run_main():
            try:
                namespace["main"]()
            except BaseException as error:
                consumer_errors.append(error)
            finally:
                completed.set()

        consumer = threading.Thread(target=run_main, daemon=True)
        consumer.start()
        try:
            with created:
                self.assertTrue(created.wait_for(lambda: len(workers) >= expected_workers, timeout=10))
            if release is not None:
                self.assertTrue(completed.wait(2), "render interruption should return to its caller")
                release.set()
            self.assertTrue(workers[-1].completed.wait(10), "generation worker must finish")
            for worker in workers:
                worker.join(2)
            stalled = not completed.wait(2)
            return SimpleNamespace(stalled=stalled, state=state, rendered=rendered,
                                   consumer_errors=consumer_errors, worker_errors=worker_errors)
        finally:
            # A baseline failure must not leave the test process blocked.
            for streamer in streamers:
                TextIteratorStreamer.on_finalized_text(streamer, "", stream_end=True)
            if release is not None:
                release.set()
            consumer.join(2)
            for worker in workers:
                worker.join(2)
            self.assertFalse(consumer.is_alive())
            self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_native_export_loader_error_reaches_consumer(self):
        paragraph = "这是一段需要总结的文档。请保留重要的事实和结论。\n"
        size = len(self.tokenizer(paragraph, add_special_tokens=False).input_ids)
        prompt = paragraph * (33000 // size + 1)
        result = self.run_ui(self.native, prompt)
        self.assertFalse(result.stalled, "worker exited but the UI is still waiting for the stream")
        self.assertEqual(len(result.consumer_errors), 1)
        self.assertIsInstance(result.consumer_errors[0], RuntimeError)
        self.assertIn("32768", str(result.consumer_errors[0]))
        self.assertEqual(result.worker_errors, [])

    def test_partial_output_keeps_original_exception(self):
        error = ValueError("generation failed after partial output")
        result = self.run_ui(ScriptedModel(self.tokenizer, [["partial answer ", error]]))
        self.assertFalse(result.stalled)
        self.assertEqual(result.consumer_errors, [error])
        self.assertIs(result.consumer_errors[0], error)
        self.assertIn("partial answer ", result.rendered)
        self.assertEqual(result.worker_errors, [])

    def test_tool_followup_error_reaches_consumer(self):
        call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>'
        error = RuntimeError("tool followup generation failed")
        model = ScriptedModel(self.tokenizer, [[call], ["partial followup ", error]])
        result = self.run_ui(model, tools=True, expected_workers=2)
        self.assertFalse(result.stalled)
        self.assertEqual(result.consumer_errors, [error])
        self.assertEqual(model.calls, 2)
        observations = [json.loads(m["content"]) for m in result.state.chat_messages if m["role"] == "tool"]
        self.assertEqual(observations, [{"result": 4}])
        self.assertEqual(result.worker_errors, [])

    def test_native_short_generation_is_unchanged(self):
        selection = iter([self.tokenizer("好", add_special_tokens=False).input_ids[0], self.tokenizer.eos_token_id])

        def select_one(probs, num_samples=1, **kwargs):
            return torch.full((probs.size(0), num_samples), next(selection), dtype=torch.long, device=probs.device)

        with mock.patch("torch.multinomial", side_effect=select_one):
            result = self.run_ui(self.native)
        self.assertFalse(result.stalled)
        self.assertEqual(result.consumer_errors, [])
        self.assertEqual(result.state.messages[-1], {"role": "assistant", "content": "好"})

    def test_complete_tool_followup_is_unchanged(self):
        call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>'
        model = ScriptedModel(self.tokenizer, [[call], ["答案是4。"]])
        result = self.run_ui(model, tools=True, expected_workers=2)
        self.assertFalse(result.stalled)
        self.assertEqual(result.consumer_errors, [])
        self.assertEqual(model.calls, 2)
        self.assertTrue(result.state.messages[-1]["content"].endswith("答案是4。"))

    def test_interrupted_consumer_releases_its_producer(self):
        release = threading.Event()
        error = RuntimeError("render was interrupted")
        model = ScriptedModel(self.tokenizer, [["started ", lambda: release.wait(10), "unused output "]])
        result = self.run_ui(model, interrupt=error, release=release)
        self.assertEqual(result.consumer_errors, [error])
        self.assertEqual(model.delivered_chunks, 1, "closed consumer must stop its producer at the next callback")
        self.assertEqual(result.worker_errors, [])


if __name__ == "__main__":
    unittest.main()
