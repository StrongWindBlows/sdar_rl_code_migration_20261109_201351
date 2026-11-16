"""Small, causal commit rules for the fixed-B32 threshold study."""
import numpy as np


def threshold(rule, block, step, generated, limit, table):
    kind = rule['kind']
    fallback = clipped = False
    if kind == 'fixed':
        value = rule['threshold']
    elif kind == 'step':
        progress = min((step-1)/(rule['span']-1), 1.)
        value = rule['start']*(1-progress) + rule['end']*progress
    elif kind == 'cad':
        progress = min(max(generated, 0)/limit, 1.)
        value = rule['start']*(1-progress) + rule['end']*progress
    elif kind == 'osdt':
        profile = table.get(rule['mode'], [])
        available = block < len(profile)
        if available and rule['mode'] == 'step_block':
            available = step <= len(profile[block]) and profile[block][step-1] is not None
        if not available:
            value, fallback = rule['fallback'], True
        else:
            value = profile[block] if rule['mode'] == 'block' else profile[block][step-1]
            raw = min(value, rule['cap'])*(1-rule['slack'])
            value = float(np.clip(raw, rule['lower'], rule['upper']))
            clipped = value != raw
    else:
        raise ValueError(kind)
    return float(value), fallback, clipped


def factor_count(confidence, factor):
    """Largest k satisfying (k+1)*(1-p[k]) < factor; force progress."""
    p = np.sort(np.asarray(confidence, dtype=float))[::-1]
    eligible = (np.arange(2, len(p)+2)*(1-p) < factor)
    return max(1, int(eligible.sum())), not eligible.any()


def calibrate(events, prompt_length, output_length):
    blocks = {}
    first = prompt_length//32*32
    for e in events:
        if e['phase'] != 'denoise':
            continue
        values = [e['confidence'][i] for i in e['positions']
                  if prompt_length <= e['block_start']+i < prompt_length+output_length]
        blocks.setdefault((e['block_start']-first)//32, []).append(values)
    profile, steps = [], []
    for b in range(max(blocks)+1):
        groups = blocks[b]
        values = [v for group in groups for v in group]
        profile.append(float(np.quantile(values, .25)))
        steps.append([float(np.quantile(group, .25)) if group else None for group in groups])
    return {'block': profile, 'step_block': steps}
