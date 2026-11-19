"""Token positions used to fit and check the late-thinking gate."""


def trace_sites(kind, prompt_length, total_length, close_position, lo, hi, immediate_close=False):
    if kind not in ("think", "neg", "negall"):
        raise ValueError(f"Unknown trace kind {kind}")
    if not 1 <= prompt_length <= total_length or not prompt_length <= close_position <= total_length:
        raise ValueError("Invalid prompt, sequence or closing-token position")
    if not 0 <= lo < hi:
        raise ValueError("Require 0 <= lo < hi")
    if kind in ("neg", "negall"):
        return {"early": range(prompt_length - 1, prompt_length), "late": range(0),
                "ans": range(prompt_length, total_length)}
    if immediate_close or close_position == prompt_length:
        return {"early": range(0), "late": range(0), "ans": range(prompt_length, total_length)}
    return {"early": range(prompt_length, min(close_position, prompt_length + lo)),
            "late": range(prompt_length + hi, max(close_position, prompt_length + hi)),
            "ans": range(close_position, total_length)}
