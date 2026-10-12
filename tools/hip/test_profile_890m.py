import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).with_name('profile_890m.py')
spec = importlib.util.spec_from_file_location('p890m', P)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Profile890mTests(unittest.TestCase):
    def test_stage_breakdown_and_decode(self):
        data = p.parse_log('''noise
strata prefill timing: 4096 tokens, GPU timeline 11000 ms, wall 15000 ms, host staging 200 ms: GDN 3000 (27.3%) MoE 5000 (45.5%)
strata decode timing: 5 windows, avg T 4.00, 2.80 tokens/window, 42.00 ms/window = verify 25.00 (GPU-reach wait 1.00)
''')
        self.assertEqual(data['summary']['prefill_stage_ms_medians']['MoE'], 5000)
        self.assertAlmostEqual(data['summary']['prefill_tps_median'], 4096/15)
        self.assertEqual(data['decode'][0]['tokens_per_window'], 2.8)

    def test_empty_log_is_not_invented_data(self):
        self.assertIsNone(p.parse_log('other output')['summary']['prefill_tps_median'])


if __name__ == '__main__':
    unittest.main()
