"""Behavioral regressions for separate MiniMind serving branches.

Usage: python serving_regressions.py REPO CASE
Cases: history, budget, tool_budget, finish, conversion, arguments, text_parts
The web tests execute actual main() with deterministic UI/model fixtures.
The API and converter tests use the real repository tokenizer and PyTorch.
"""
import ast
import copy
import importlib
import io
import json
import os
from pathlib import Path
import queue
import random
import re
import sys
import tempfile
from contextlib import redirect_stdout
from threading import Event, Thread
from types import SimpleNamespace, MethodType
from unittest.mock import patch

REPO = Path(sys.argv[1]).resolve()
CASE = sys.argv[2]
sys.path.insert(0, str(REPO))


def load_functions(relative, namespace, wanted=None):
    source = REPO / relative
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and (wanted is None or n.name in wanted)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class State(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class FakeUI:
    def __init__(self):
        self.session_state = State(messages=[], chat_messages=[], history_chat_num=2,
                                   max_new_tokens=2, temperature=0.9, selected_tools=['calculate_math'])
        self.prompt = 'What is 2+3?'
        self.display = []

    def chat_input(self, **kwargs):
        return self.prompt

    def markdown(self, text, **kwargs):
        self.display.append(text)

    def empty(self):
        return self


class Batch(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, device):
        return self


class RecordingTokenizer:
    pad_token_id, eos_token_id = 0, 2

    def __init__(self):
        self.transcripts = []

    def apply_chat_template(self, messages, **kwargs):
        self.transcripts.append(copy.deepcopy(messages))
        return 'prompt'

    def __call__(self, text, **kwargs):
        return Batch(input_ids=SimpleNamespace(shape=(1, 2)), attention_mask=SimpleNamespace(shape=(1, 2)))


class FixtureStreamer:
    def __init__(self, *args, **kwargs):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(timeout=3)
        if item is None:
            raise StopIteration
        return item


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.kwargs = []

    def generate(self, **kwargs):
        self.kwargs.append(kwargs)
        streamer = kwargs['streamer']
        streamer.queue.put(next(self.outputs))
        streamer.queue.put(None)


TOOL_TEXT = '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+3"}}</tool_call>'


def web_namespace(outputs, model=None, tokenizer=None, streamer=FixtureStreamer):
    ui = FakeUI()
    model = model or ScriptedModel(outputs)
    tokenizer = tokenizer or RecordingTokenizer()
    ns = {'st': ui, 'load_model_tokenizer': lambda path: (model, tokenizer), 'model_path': 'fixture',
          'device': 'cpu', 'random': random, 're': re, 'json': json, 'Thread': Thread,
          'TextIteratorStreamer': streamer, 'setup_seed': lambda seed: None,
          'process_assistant_content': lambda text, **kwargs: text,
          'get_text': lambda key: key, 'execute_tool': lambda name, args: {'result': 5},
          'TOOLS': [{'type': 'function', 'function': {'name': 'calculate_math'}}]}
    load_functions('scripts/web_demo.py', ns, {'main', 'trim_chat_history'})
    return ns, ui, model, tokenizer


def check_history():
    ns, ui, model, tokenizer = web_namespace([TOOL_TEXT, 'Result 5.', 'Follow-up answer.'])
    ns['main']()
    assert ui.session_state.chat_messages[-1] == {'role': 'assistant', 'content': 'Result 5.'}, ui.session_state.chat_messages[-1]
    assert '<div' in ui.session_state.messages[-1]['content'], 'UI must still show tool results'
    ui.prompt = 'What was the expression?'
    ns['main']()
    transcript = tokenizer.transcripts[-1]
    assert [m['role'] for m in transcript] == ['user', 'assistant', 'tool', 'assistant', 'user'], transcript
    assert transcript[0]['content'] == 'What is 2+3?'
    assert not any('<div' in m['content'] for m in transcript)
    ui.session_state.history_chat_num = 0
    model.outputs = iter(['Fresh answer.'])
    ui.prompt = 'New question'
    ns['main']()
    assert tokenizer.transcripts[-1] == [{'role':'user', 'content':'New question'}]
    # Standard alternating user/assistant messages retain the existing message-count setting.
    ns, ui, model, tokenizer = web_namespace(['Next'])
    ui.session_state.chat_messages = [
        {'role': 'system', 'content': 'Old system'}, {'role': 'user', 'content': 'Old user'},
        {'role': 'assistant', 'content': 'Old answer'}, {'role': 'user', 'content': 'Recent user'},
        {'role': 'assistant', 'content': 'Recent answer'}]
    ui.session_state.selected_tools = []
    ns['main']()
    transcript = tokenizer.transcripts[-1]
    assert [m['role'] for m in transcript] == ['system', 'user', 'assistant', 'user'], transcript
    assert transcript[1]['content'] == 'Recent user'
    print('PASS: tool transcript stays plain, complete prior tool turn retained, zero/plain history preserved')


def check_budget():
    import torch
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from transformers import AutoTokenizer, TextIteratorStreamer
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'model', local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                vocab_size=len(tokenizer), max_position_embeddings=256)).eval()
    generated = []
    completed = Event()
    counter = [0]
    text_token = tokenizer.encode('hello', add_special_tokens=False)[0]
    def forward(self, input_ids, *args, **kwargs):
        counter[0] += 1
        # Deterministic logits isolate the actual native generate() budget handling.
        token = tokenizer.eos_token_id if counter[0] % 5 == 0 else text_token
        logits = torch.full((1, input_ids.shape[1], len(tokenizer)), -float('inf'))
        logits[..., token] = 0
        return SimpleNamespace(logits=logits, past_key_values=None)
    model.forward = MethodType(forward, model)
    original_generate = model.generate
    def recording_generate(**kwargs):
        result = original_generate(**kwargs)
        generated.append(result.shape[1] - kwargs['input_ids'].shape[1])
        completed.set()
        return result
    model.generate = recording_generate
    ns, ui, _, _ = web_namespace([], model=model, tokenizer=tokenizer, streamer=TextIteratorStreamer)
    ui.session_state.selected_tools = []
    ns['main']()
    assert completed.wait(3), 'generation worker did not complete'
    assert generated == [2], f'slider requested 2 tokens, native generate emitted {generated}'
    # Both initial response and post-tool response use the same output budget.
    ns, ui, model, tokenizer = web_namespace([TOOL_TEXT, 'Done'])
    ns['main']()
    assert len(model.kwargs) == 2
    assert [kw.get('max_new_tokens') for kw in model.kwargs] == [2, 2], model.kwargs
    assert all('max_length' not in kw for kw in model.kwargs)
    print('PASS: actual native generate stops at slider budget; post-tool generation uses identical budget')


def check_tool_budget():
    module = importlib.import_module('scripts.eval_toolcall')
    args = SimpleNamespace(api_model='fixture', temperature=0.8, top_p=0.9,
                           max_new_tokens=7, max_tool_rounds=2, backend='api', stream=0)
    seen = []
    final = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Done', tool_calls=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: seen.append(kw) or final)))
    with redirect_stdout(io.StringIO()):
        module.chat_api(client, [{'role':'user','content':'Hello'}], [], args, stream=False)
    assert seen[0]['max_tokens'] == 7, seen[0]
    call = {'id':'call_1','function':{'name':'calculate_math','arguments':'{"expression":"2+3"}'}}
    counts = {'generations':0, 'tools':0}
    def endless_api(*unused, **kwargs):
        counts['generations'] += 1
        assert counts['generations'] <= 4, 'tool evaluation did not stop at its round budget'
        return '', [call]
    def execute(*unused, **kwargs):
        counts['tools'] += 1
        return {'result':5}
    with patch.object(module, 'chat_api', endless_api), patch.object(module, 'execute_tool', execute), redirect_stdout(io.StringIO()) as output:
        module.run_case('Question', [], args, client=client)
    assert counts == {'generations':3, 'tools':2}, counts
    assert '最大工具调用轮数 2' in output.getvalue()
    args.max_tool_rounds = 0
    counts.update(generations=0, tools=0)
    with patch.object(module, 'chat_api', endless_api), patch.object(module, 'execute_tool', execute), redirect_stdout(io.StringIO()):
        module.run_case('Question', [], args, client=client)
    assert counts == {'generations':1, 'tools':0}, counts
    args.backend, args.max_tool_rounds = 'local', 1
    counts.update(generations=0, tools=0)
    def endless_local(*unused, **kwargs):
        counts['generations'] += 1
        assert counts['generations'] <= 3
        return TOOL_TEXT
    with patch.object(module, 'generate', endless_local), patch.object(module, 'execute_tool', execute), redirect_stdout(io.StringIO()):
        module.run_case('Question', [], args, model=object(), tokenizer=object())
    assert counts == {'generations':2, 'tools':1}, counts
    print('PASS: API token budget honored; repeated local/API tool requests terminate; zero budget executes no tools')


def check_finish():
    import torch
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer
    module = importlib.import_module('scripts.serve_openai_api')
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'model', local_files_only=True)
    text_ids = tokenizer.encode('hello world', add_special_tokens=False)
    plain = [text_ids[0], text_ids[-1]]
    class TokensModel:
        def __init__(self, tokens):
            self.tokens = tokens
        def generate(self, input_ids, streamer=None, **kwargs):
            tokens = self.tokens[:kwargs['max_new_tokens']]
            if streamer is not None:
                streamer.put(input_ids.cpu())
                for token in tokens:
                    streamer.put(torch.tensor([token]))
                streamer.end()
            return torch.cat([input_ids, input_ids.new_tensor([tokens])], dim=1)
    module.tokenizer, module.device = tokenizer, 'cpu'
    client = TestClient(module.app)
    base = {'model':'minimind','messages':[{'role':'user','content':'Hello'}]}
    cases = [(plain, 2, 'length'), ([plain[0], tokenizer.eos_token_id], 2, 'stop'),
             ([tokenizer.eos_token_id], 3, 'stop'), (plain, 3, 'stop')]
    for tokens, budget, expected in cases:
        module.model = TokensModel(tokens)
        for stream in (False, True):
            response = client.post('/v1/chat/completions', json={**base, 'stream':stream, 'max_tokens':budget})
            assert response.status_code == 200, response.text
            if stream:
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
                actual = frames[-1]['choices'][0]['finish_reason']
            else:
                actual = response.json()['choices'][0]['finish_reason']
            assert actual == expected, (tokens, budget, stream, actual, expected)
    tool_ids = tokenizer.encode(TOOL_TEXT, add_special_tokens=False) + [tokenizer.eos_token_id]
    module.model = TokensModel(tool_ids)
    for stream in (False, True):
        response = client.post('/v1/chat/completions', json={**base, 'stream':stream, 'max_tokens':len(tool_ids)+1})
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')] if stream else [response.json()]
        assert frames[-1]['choices'][0]['finish_reason'] == 'tool_calls', frames
    print('PASS: 10 real HTTP cases distinguish truncated/EOS/early-return/tool-call finishes in both response modes')


def check_conversion():
    import torch
    module = importlib.import_module('scripts.convert_model')
    from model.model_lora import apply_lora, save_lora
    module.lm_config = module.MiniMindConfig(hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=6400,
        max_position_embeddings=128)
    model = module.MiniMindForCausalLM(module.lm_config)
    original = copy.deepcopy(model.state_dict())
    old_cwd = os.getcwd()
    os.chdir(REPO / 'scripts')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = tmp/'complete.pth'
            torch.save(original, good)
            apply_lora(model, rank=4)
            lora = tmp/'adapter.pth'
            save_lora(model, str(lora))
            missing_key = 'model.layers.0.self_attn.v_proj.weight'
            # Omitting a complete layer reproduces selecting a deeper export config
            # than the checkpoint: strict=False silently retains a random layer.
            for name, bad_state in [('missing_layer', {k:v for k,v in original.items() if not k.startswith('model.layers.0.')}),
                                    ('missing', {k:v for k,v in original.items() if k != missing_key}),
                                    ('unexpected', {**original, 'model.typo.weight':torch.ones(1)})]:
                bad = tmp/(name+'.pth')
                torch.save(bad_state, bad)
                for kind in ('native', 'merge'):
                    destination = tmp/(name+'-'+kind)
                    rejected = False
                    try:
                        with redirect_stdout(io.StringIO()):
                            if kind == 'native':
                                module.convert_torch2transformers_minimind(str(bad), str(destination), dtype=torch.float32)
                            else:
                                module.convert_merge_base_lora(str(bad), str(lora), str(destination), lora_rank=4)
                    except RuntimeError as exc:
                        assert 'state_dict' in str(exc)
                        rejected = True
                    assert rejected, f'{kind} silently exported {name} base checkpoint'
                    assert not destination.exists(), destination
            with redirect_stdout(io.StringIO()):
                module.convert_torch2transformers_minimind(str(good), str(tmp/'native'), dtype=torch.float32)
                module.convert_merge_base_lora(str(good), str(lora), str(tmp/'merged.pth'), lora_rank=4)
            saved = torch.load(tmp/'native'/'pytorch_model.bin', weights_only=True)
            assert saved.keys() == original.keys()
            assert all(torch.equal(saved[k], v) for k,v in original.items())
            merged = torch.load(tmp/'merged.pth', weights_only=True)
            assert all(torch.equal(merged[k], v.half()) for k,v in original.items())
    finally:
        os.chdir(old_cwd)
    print('PASS: both exporters reject missing/unexpected base weights before writing; valid exports retain all values')




def check_arguments():
    import torch
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer
    from openai import OpenAI
    module = importlib.import_module('scripts.serve_openai_api')
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'model', local_files_only=True)
    class TokensModel:
        def __init__(self, text):
            self.tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        def generate(self, input_ids, streamer=None, **kwargs):
            if streamer is not None:
                streamer.put(input_ids.cpu())
                for token in self.tokens:
                    streamer.put(torch.tensor([token]))
                streamer.end()
            return torch.cat([input_ids, input_ids.new_tensor([self.tokens])], dim=1)
    module.tokenizer, module.device = tokenizer, 'cpu'
    http = TestClient(module.app)
    sdk = OpenAI(api_key='local-fixture', base_url='http://testserver/v1', http_client=http, max_retries=0)
    request = {'model':'minimind','messages':[{'role':'user','content':'Call the tool'}]}
    expected = {'text':'中文 "quoted" \\ path', 'options':{'count':0, 'enabled':True}}
    for raw in (expected, json.dumps(expected, ensure_ascii=False)):
        text = '<tool_call>'+json.dumps({'name':'text_length','arguments':raw}, ensure_ascii=False)+'</tool_call>'
        module.model = TokensModel(text)
        result = sdk.chat.completions.create(**request, stream=False, max_tokens=256)
        arguments = result.choices[0].message.tool_calls[0].function.arguments
        assert json.loads(arguments) == expected, (raw, arguments)
        response = http.post('/v1/chat/completions', json={**request, 'stream':True, 'max_tokens':256})
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        calls = [call for frame in frames for call in frame['choices'][0]['delta'].get('tool_calls', [])]
        assert len(calls) == 1
        assert json.loads(calls[0]['function']['arguments']) == expected, calls
        assert frames[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    text = '<tool_call>{"name":"clock"}</tool_call>'
    assert json.loads(module.parse_response(text)[2][0]['function']['arguments']) == {}
    bad = '<tool_call>{"name":"clock","arguments":"not JSON"}</tool_call>'
    content, _, calls = module.parse_response(bad)
    assert calls is None and content == bad
    sdk.close()
    print('PASS: object and JSON-string arguments both decode once to the same dict through real SDK/HTTP and streaming; missing/invalid argument cases checked')


def check_text_parts():
    import torch
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer
    module = importlib.import_module('scripts.serve_openai_api')
    tokenizer = AutoTokenizer.from_pretrained(REPO / 'model', local_files_only=True)
    class RecordingModel:
        def __init__(self):
            self.inputs = []
        def generate(self, input_ids, streamer=None, **kwargs):
            self.inputs.append(input_ids.clone())
            tokens = tokenizer.encode('OK', add_special_tokens=False) + [tokenizer.eos_token_id]
            if streamer is not None:
                streamer.put(input_ids.cpu())
                for token in tokens:
                    streamer.put(torch.tensor([token]))
                streamer.end()
            return torch.cat([input_ids, input_ids.new_tensor([tokens])], dim=1)
    module.tokenizer, module.device, module.model = tokenizer, 'cpu', RecordingModel()
    client = TestClient(module.app)
    parts = [{'type':'text', 'text':'你好 '}, {'type':'text', 'text':'world\n<literal>'}]
    plain = [{'role':'user', 'content':'你好 world\n<literal>'}]
    compound = [{'role':'user', 'content':parts}]
    for stream in (False, True):
        request = {'model':'minimind', 'stream':stream, 'max_tokens':16}
        response = client.post('/v1/chat/completions', json={**request, 'messages':plain})
        assert response.status_code == 200, response.text
        expected = module.model.inputs[-1]
        response = client.post('/v1/chat/completions', json={**request, 'messages':compound})
        assert response.status_code == 200, response.text
        assert torch.equal(module.model.inputs[-1], expected), tokenizer.decode(module.model.inputs[-1][0])
    original = [
        {'role':'system', 'content':'A plain system instruction'},
        {'role':'assistant', 'content':None, 'tool_calls':[{'id':'call_a','type':'function',
            'function':{'name':'clock','arguments':'{}'}}]},
        {'role':'tool', 'tool_call_id':'call_a', 'content':[{'type':'text','text':'12:'},{'type':'text','text':'30'}]},
        {'role':'user','content':parts}]
    unchanged = copy.deepcopy(original)
    normalized = module.ChatRequest(model='minimind', messages=original).messages
    assert original == unchanged, 'request normalization must not mutate caller messages'
    assert normalized[0]['content'] == unchanged[0]['content']
    assert normalized[1]['content'] is None and normalized[1]['tool_calls'] == unchanged[1]['tool_calls']
    assert normalized[2] == {'role':'tool','tool_call_id':'call_a','content':'12:30'}
    assert normalized[3]['content'] == '你好 world\n<literal>'
    expected = tokenizer(tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True), return_tensors='pt')['input_ids']
    response = client.post('/v1/chat/completions', json={'model':'minimind','messages':original,'stream':False,'max_tokens':16})
    assert response.status_code == 200 and torch.equal(module.model.inputs[-1], expected)
    bad_parts = [{'type':'image_url','image_url':{'url':'https://example.invalid/image.png'}},
                 {'type':'input_audio','input_audio':{'data':'AA==','format':'wav'}},
                 {'type':'text','text':123}]
    for part in bad_parts:
        for stream in (False, True):
            calls_before = len(module.model.inputs)
            response = client.post('/v1/chat/completions', json={'model':'minimind', 'messages':[
                {'role':'user','content':[{'type':'text','text':'Keep this input'},part]}], 'stream':stream})
            assert response.status_code == 422, response.text
            assert len(module.model.inputs) == calls_before
    print('PASS: 11 HTTP requests preserve text-part token IDs and tool history, reject unsupported parts before inference, and retain plain strings/assistant None')


{'history': check_history, 'budget': check_budget, 'tool_budget': check_tool_budget,
 'finish': check_finish, 'conversion': check_conversion,
 'arguments': check_arguments, 'text_parts':check_text_parts}[CASE]()
