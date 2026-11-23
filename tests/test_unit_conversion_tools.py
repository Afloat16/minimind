import unittest

from scripts.eval_toolcall import execute_tool as evaluate_tool
from trainer.train_agent import execute_tool as train_tool


class UnitConversionToolTests(unittest.TestCase):
    def test_training_and_evaluation_use_the_requested_units(self):
        cases = [
            (0, 'celsius', 'fahrenheit', 32),
            (100, 'celsius', 'fahrenheit', 212),
            (32, 'fahrenheit', 'celsius', 0),
            (212, 'fahrenheit', 'celsius', 100),
            (-40, 'celsius', 'fahrenheit', -40),
            (10, 'miles', 'km', 16.0934),
            (2, 'kg', 'pounds', 4.40924),
            (10, 'feet', 'meters', 3.048),
            (12, 'meters', 'meters', 12),
            (10, 'KM', 'MILES', 6.21371),
        ]
        for execute, precision in [(train_tool, 4), (evaluate_tool, 2)]:
            for value, source, target, expected in cases:
                with self.subTest(executor=execute.__module__, source=source,
                                  target=target, value=value):
                    result = execute('unit_converter', {
                        'value': value, 'from_unit': source, 'to_unit': target,
                    })
                    self.assertEqual(result['result'], round(expected, precision))

    def test_unsupported_units_are_errors_instead_of_plausible_results(self):
        for execute in [train_tool, evaluate_tool]:
            for source, target in [('kg', 'miles'), ('unknown', 'unknown')]:
                with self.subTest(executor=execute.__module__, source=source, target=target):
                    result = execute('unit_converter', {
                        'value': 10, 'from_unit': source, 'to_unit': target,
                    })
                    self.assertIn('error', result)
                    self.assertNotIn('result', result)


if __name__ == '__main__':
    unittest.main()
