import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

P = Path(__file__).with_name('pack_dense_890m.py')
spec = importlib.util.spec_from_file_location('p890mw', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class LayoutTests(unittest.TestCase):
    def test_lossless_roundtrip_and_padding(self):
        class Tensor:
            def __init__(self, name, offset, n):
                self.name, self.offset, self.n = name, offset, n
                self.shape, self.type_name = [n], 'F16'
            def expected_bytes(self):
                return self.n
        g = SimpleNamespace(data_start=0, tensors=[Tensor('blk.0.attn_q.weight', 13, 17),
                  Tensor('blk.0.ffn_gate_exps.weight', 40, 20), Tensor('blk.1.ssm_alpha.weight', 100, 7)])
        entries = mod.make_plan(g)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]['packed_offset'], 256)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'orig', Path(d)/'pack'
            a.write_bytes(bytes(range(128)))
            mod.write_sidecar(a, b, entries)
            self.assertTrue(mod.verify_sidecar(a, b, entries))
            self.assertEqual(b.read_bytes()[0:17], a.read_bytes()[13:30])
            raw = bytearray(b.read_bytes())
            raw[256] ^= 1
            b.write_bytes(raw)
            self.assertFalse(mod.verify_sidecar(a, b, entries))


if __name__ == '__main__':
    unittest.main()
