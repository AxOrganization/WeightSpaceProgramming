"""Where a thinking trace starts to repeat itself (token level): used only to describe the starting model.

A position is a "repeat position" when the n tokens starting there already occurred within the previous `window`
tokens. A loop starts at the first position from which repeat positions make up at least `density` of the next `span`
tokens (so one restated passage is not a loop). The period is the distance back to the earlier occurrence.
"""


def repeat_flags(ids, n=8, window=3000):
    last, flags, back = {}, [False] * len(ids), [0] * len(ids)
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        j = last.get(g)
        if j is not None and i - j <= window:
            flags[i], back[i] = True, i - j
        last[g] = i
    return flags, back


def first_loop(ids, n=8, window=3000, span=600, density=0.6, min_pos=50):
    """-> {"onset", "period"} or None; onset = index of the first token of the first repeated block."""
    if len(ids) < span + n:
        return None
    flags, back = repeat_flags(ids, n, window)
    pre = [0]
    for f in flags:
        pre.append(pre[-1] + f)
    for i in range(min_pos, len(ids) - span):
        if flags[i] and pre[i + span] - pre[i] >= density * span:
            cnt = {}
            for j in range(i, i + span):
                if flags[j]:
                    cnt[back[j]] = cnt.get(back[j], 0) + 1
            return {"onset": i, "period": max(cnt, key=cnt.get)}
    return None
