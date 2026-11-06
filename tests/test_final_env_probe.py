"""CPU evidence-contract tests; never instantiate DMC or a model."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helixwm_final.env_probe import (GAPS, binary64, compare_outputs, fixed_inputs,
                                     validate_row, write_json)
from helixwm_matched.protocol import array_sha256, gap_segments


class EnvironmentProbeTests(unittest.TestCase):
    def row(self, index=0):
        actions, mask = fixed_inputs()
        action = actions[index]
        row = {'action_index': index, 'input_observation_timestamp': index,
               'observation_timestamp': index+1, 'input_available': bool(mask[index]),
               'availability': bool(mask[index+1]), 'DMC_calls': 2, 'episode_DMC_calls': 2*(index+1),
               'done': index == 499, 'terminated': False, 'truncated': index == 499,
               'executed_action_sha256': array_sha256(action), 'reward': binary64(0.75),
               'visible_RGB_sha256': array_sha256(np.zeros((64,64,3), np.uint8)),
               'subcalls': [{'native_call': 2*index+k, 'last': index == 499 and k == 2,
                             'executed_action': action.tolist(), 'reward': value}
                            for k, value in [(1, .25), (2, .5)]]}
        return row, action, mask

    def test_fixed_inputs_have_all_gaps_and_exact_seed(self):
        actions, mask = fixed_inputs()
        self.assertEqual(actions.dtype, np.float32)
        self.assertEqual(actions.shape, (500, 6))
        self.assertTrue(np.array_equal(actions, np.random.default_rng(944000).uniform(-1,1,(500,6)).astype(np.float32)))
        self.assertEqual(gap_segments(mask), list(GAPS))
        self.assertEqual(int((~mask).sum()), 186)
        self.assertTrue(mask[0] and mask[-1])

    def test_binary64_rejects_nonfinite_and_preserves_signbit(self):
        for bad in [float('nan'), float('inf'), np.float32(.5), 1]:
            with self.assertRaises(ValueError): binary64(bad)
        self.assertNotEqual(binary64(-0.0), binary64(0.0))

    def test_valid_initial_missing_and_last_rows(self):
        for index in [0, 9, 10, 49, 279, 499]:
            validate_row(*self.row(index)[:1], index, *self.row(index)[1:])

    def test_one_ulp_reward_change_rejected(self):
        row, action, mask = self.row()
        row['reward'] = binary64(math.nextafter(.75, 1))
        with self.assertRaisesRegex(ValueError, 'exact two'): validate_row(row, 0, action, mask)

    def test_counts_action_timestamp_and_hidden_rgb_rejected(self):
        for key, value in [('DMC_calls', 1), ('executed_action_sha256', 'changed'),
                           ('input_observation_timestamp', 1), ('terminated', True)]:
            row, action, mask = self.row()
            row[key] = value
            with self.assertRaises(ValueError): validate_row(row, 0, action, mask)
        row, action, mask = self.row(9)
        row['visible_RGB_sha256'] = array_sha256(np.ones((64,64,3),np.uint8))
        with self.assertRaisesRegex(ValueError, 'nonzero'): validate_row(row, 9, action, mask)

    def prepare_pair(self, root):
        lock = {'wrapper_sha': 'original', 'version': 1}
        result = {'seed': 44000, 'actions_seed': 944000, 'action_dtype': '<f4',
                  'actions_sha256': 'A', 'mask_sha256': 'B', 'mask': [True], 'gap_segments': [],
                  'action_decisions': 500, 'DMC_calls': 1000, 'trace_sha256': 'C',
                  'lock_sha256': 'D', 'training_updates': 0, 'model_imports': [], 'old_checkpoints_read': False}
        paths = [Path(root)/x for x in ['old','final']]
        for pid, path in enumerate(paths, 1):
            path.mkdir()
            (path/'trace.jsonl').write_bytes(b'{"reward_hex":"0x1.0p+0"}\n'*501)
            write_json(path/'environment_lock.json', lock)
            write_json(path/'worker_result.json', {**result, 'pid': pid})
        return *paths, lock

    def test_exact_pair_and_single_byte_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            old, final, lock = self.prepare_pair(root)
            self.assertTrue(compare_outputs(old, final, lock)['exact_trace_bytes'])
            p = final/'trace.jsonl'
            p.write_bytes(p.read_bytes().replace(b'0x1.0',b'0x1.1',1))
            with self.assertRaisesRegex(ValueError, 'row 0'): compare_outputs(old, final, lock)

    def test_runtime_drift_and_same_pid_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            old, final, lock = self.prepare_pair(root)
            with self.assertRaisesRegex(ValueError, 'runtime environment'): compare_outputs(old, final, {**lock, 'version': 2})
            p = final/'worker_result.json'
            value=json.loads(p.read_bytes());value['pid']=1;p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'independent'): compare_outputs(old, final, lock)


if __name__ == '__main__':
    unittest.main()
