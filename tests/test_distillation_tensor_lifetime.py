import copy
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer import train_distillation as trainer


class DistillationTensorLifetimeTests(unittest.TestCase):
    def test_teacher_logits_are_released_before_the_next_student_forward(self):
        torch.manual_seed(0)
        config = MiniMindConfig(hidden_size=16, num_hidden_layers=1, vocab_size=32,
                                num_attention_heads=2, num_key_value_heads=1,
                                max_position_embeddings=16)
        student = MiniMindForCausalLM(config)
        teacher_config = copy.deepcopy(config)
        teacher_config.vocab_size = 64
        teacher = MiniMindForCausalLM(teacher_config)
        logits_refs = []
        retained_at_forward = []

        def remember_teacher_logits(module, inputs, output):
            logits = output.logits
            while logits._base is not None:
                logits = logits._base
            logits_refs.append(weakref.ref(logits))

        def observe_student_forward(module, inputs):
            retained_at_forward.append(sum(ref() is not None for ref in logits_refs))

        teacher.register_forward_hook(remember_teacher_logits)
        student.register_forward_pre_hook(observe_student_forward)
        initial = {name: value.clone() for name, value in student.state_dict().items()}
        inputs = torch.tensor([[1, 4, 7, 9, 2]])
        labels = inputs.clone()
        labels[:, :2] = -100
        args = SimpleNamespace(device='cpu', epochs=1, learning_rate=0.01,
                               warmup_ratio=0.0, accumulation_steps=1,
                               grad_clip=1.0, log_interval=100, save_interval=100)
        optimizer = torch.optim.SGD(student.parameters(), lr=args.learning_rate)
        with patch.multiple(trainer, create=True, model=student, args=args,
                            optimizer=optimizer, autocast_ctx=nullcontext(),
                            scaler=torch.amp.GradScaler('cpu', enabled=False)), \
                patch.object(trainer, 'is_main_process', return_value=False), \
                patch.object(trainer, 'Logger'):
            trainer.train_epoch(0, [(inputs, labels)] * 3, 3, teacher, config,
                                alpha=0.5)

        self.assertEqual(retained_at_forward, [0, 0, 0])
        self.assertTrue(all(ref() is None for ref in logits_refs))
        self.assertTrue(any(not torch.equal(value, initial[name])
                            for name, value in student.state_dict().items()))
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in student.state_dict().values()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
