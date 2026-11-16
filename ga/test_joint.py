"""Checks for paired controls, actual interventions and strict block budgets."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from joint_protocol import analyze_state, audit_path, make_pool


def case_fixture():
    current = list(range(100, 132))
    historical = list(current)
    for i in (0, 4, 5):
        historical[i] += 500
    probabilities = [.6, .7, .8, .99, .55, .54] + [.1] * 26
    state = {'id': 'fixture', 'positions': [0, 1, 2, 3], 'masked_positions': list(range(28)),
        'current_ids': current, 'historical_ids': historical, 'confidence': probabilities,
        'topk_ids': [[current[i], 200 + i] for i in range(32)],
        'input_ids': [999] * 28 + current[28:], 'mask': [0] * 28 + [1] * 4,
        'block_start': 32, 'source_forward': 2, 'round': 2}
    case = {'question_id': 'fixture', 'setting': 'T8', 'snapshot': state, 'actions': [
        {'name': 'Current', 'positions': [0, 1, 2, 3], 'tokens': current[:4]},
        {'name': 'Delay1', 'positions': [1, 2, 3], 'tokens': current[1:4], 'deferred_position': 0},
        {'name': 'Swap1', 'positions': [1, 2, 3, 4], 'tokens': current[1:5], 'replacement_position': 4}]}
    return case


def outcomes(pool):
    return [{'action_id': a['id'], 'reward': 0., 'grading': 'ok', 'forward_calls': 100} for a in pool['actions']]


class JointTests(unittest.TestCase):
    def test_reporting_separates_workpoints_with_identical_saved_state_ids(self):
        from joint import report
        first, second = case_fixture(), case_fixture()
        second['setting'] = 'T4'
        second['snapshot']['positions'] = list(range(8))
        current = second['snapshot']['current_ids']
        second['actions'] = [
            {'name': 'Current', 'positions': list(range(8)), 'tokens': current[:8]},
            {'name': 'Delay1', 'positions': list(range(1, 8)), 'tokens': current[1:8], 'deferred_position': 0},
            {'name': 'Swap1', 'positions': list(range(1, 9)), 'tokens': current[1:9], 'replacement_position': 8}]
        cases, rows = [first, second], []
        for case in cases:
            case['pool'] = make_pool(case)
            for row in outcomes(case['pool']):
                rows.append(dict(row, state_id='fixture', setting=case['setting'],
                    restore_forwards=2, inference_seconds=.1, validation={'passed': True}))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'cases.json').write_text(json.dumps(cases))
            (out / 'manifest.json').write_text(json.dumps({'unique_actions': len(rows)}))
            (out / 'run-gpu4.json').write_text(json.dumps(rows))
            with redirect_stdout(io.StringIO()):
                report(out)
            summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['completed_states'], 2)
        self.assertEqual(summary['settings']['T8']['states'], 1)
        self.assertEqual(summary['settings']['T4']['states'], 1)

    def test_original_controls_and_fixed_values_are_preserved(self):
        case = case_fixture()
        original = deepcopy(case)
        pool = make_pool(case)
        self.assertEqual(case, original)
        self.assertEqual(pool['anchors'], [3])
        for action in pool['actions']:
            self.assertIn(3, action['positions'])
            self.assertEqual(action['tokens'][action['positions'].index(3)], 103)
            self.assertTrue(set(action['positions']) <= set(case['snapshot']['masked_positions']))
        self.assertEqual(pool['position_plans']['P2'], [2, 3])
        self.assertEqual(pool['position_plans']['P4'], [2, 3, 4, 5])

    def test_factorial_cells_keep_all_controls_and_deduplicate_inactive_values(self):
        pool = make_pool(case_fixture())
        self.assertEqual(len(pool['matrix']), 20)
        self.assertEqual(pool['matrix']['P1/V2'], pool['matrix']['P1/V0'])
        self.assertEqual(pool['matrix']['P0/V3'], pool['current_id'])
        self.assertEqual(pool['history_mode'], 'adjacent_pair')
        self.assertEqual(pool['history_positions'], [4, 5])
        self.assertEqual(sum(len(a['cells']) for a in pool['actions']), 20)
        self.assertTrue(any(a['group'] == 'G3' for a in pool['actions']))

    def test_insufficient_mutable_positions_do_not_unfreeze_anchors(self):
        case = case_fixture()
        case['snapshot']['confidence'][1:4] = [.99] * 3
        pool = make_pool(case)
        self.assertEqual(pool['movable'], [0])
        self.assertNotIn('P2', pool['position_plans'])
        self.assertNotIn('P4', pool['position_plans'])
        self.assertEqual(len(pool['matrix']), 12)

    def test_mask_is_a_state_operation_and_eos_remains_a_legal_candidate(self):
        case = case_fixture()
        case['snapshot']['topk_ids'][2][1] = 999
        case['snapshot']['topk_ids'][0][1] = 777
        pool = make_pool(case, eos_ids=[777], vocab_size=1000)
        self.assertEqual(pool['value_plans']['V1'][2], 102)
        self.assertTrue(pool['rejected_values'])
        self.assertTrue(any(a['contains_eos'] for a in pool['actions']))
        self.assertTrue(all(999 not in a['tokens'] for a in pool['actions']))

    def test_joint_exclusive_success_requires_all_single_factor_actions_to_fail(self):
        case = case_fixture()
        pool = case['pool'] = make_pool(case)
        rows = outcomes(pool)
        joint_id = pool['matrix']['P2/V1']
        next(r for r in rows if r['action_id'] == joint_id)['reward'] = 1.
        summary = analyze_state(case, rows)
        self.assertTrue(summary['joint_exclusive'])
        self.assertTrue(any(i['paired_rescue'] for i in summary['interactions']))
        self.assertTrue(all(len(d['singles_ids']) == len(d['joint_ids']) for d in summary['matched']))
        pure = next(a['id'] for a in pool['actions'] if a['group'] == 'G2')
        next(r for r in rows if r['action_id'] == pure)['reward'] = 1.
        self.assertFalse(analyze_state(case, rows)['joint_exclusive'])

    def test_new_position_enabled_value_is_distinguished_from_shared_coordinate_changes(self):
        case = case_fixture()
        case['pool'] = make_pool(case)
        summary = analyze_state(case, outcomes(case['pool']))
        interaction = next(x for x in summary['interactions'] if x['position_plan'] == 'P3' and x['value_plan'] == 'V2')
        self.assertEqual(interaction['changed_new_positions'], [4])
        self.assertTrue(interaction['only_newly_enabled'])

    def test_grading_timeout_is_not_a_joint_failure_observation(self):
        case = case_fixture()
        case['pool'] = make_pool(case)
        rows = outcomes(case['pool'])
        rows[-1]['grading'] = 'timeout'
        self.assertFalse(analyze_state(case, rows)['eligible'])

    def test_audit_allows_source_alleles_but_rejects_later_overrides_and_extra_budget(self):
        case = case_fixture()
        pool = case['pool'] = make_pool(case)
        action_id = pool['matrix']['P2/V1']
        action = next(a for a in pool['actions'] if a['id'] == action_id)
        state = case['snapshot']
        ids, mask, events = list(state['input_ids']), list(state['mask']), []
        for round_id in range(2, 9):
            positions = action['positions'] if round_id == 2 else [i for i, flag in enumerate(mask) if flag == 0]
            if round_id not in (2, 8):
                positions = positions[:4]
            tokens = action['tokens'] if round_id == 2 else [state['current_ids'][i] for i in positions]
            after = list(mask)
            for i in positions:
                after[i] = 1
            events.append({'forward': round_id, 'block_start': 32, 'round': round_id,
                'phase': 'denoise', 'full_generation_block': True, 'input_ids': list(ids),
                'mask_before': list(mask), 'mask_after': after, 'positions': positions, 'tokens': tokens,
                'current_ids': state['current_ids'], 'topk_ids': state['topk_ids']})
            for i, value in zip(positions, tokens):
                ids[i] = value
            mask = after
        result = {'restore_forwards': 2, 'future_forwards': 6, 'forward_calls': 8,
                  'forward_types': {'restore': 2, 'future': 6}}
        self.assertTrue(audit_path(result, events, case, action)['passed'])
        bad = deepcopy(events)
        bad[1]['tokens'][0] = 888
        with self.assertRaises(AssertionError):
            audit_path(result, bad, case, action)
        with self.assertRaises(AssertionError):
            audit_path(dict(result, forward_calls=513, future_forwards=511), events, case, action)


if __name__ == '__main__':
    unittest.main()
