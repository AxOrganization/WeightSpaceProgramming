"""The working of a truth-chain puzzle of the toolkit ("Question: A tells the truth. B says A lies. ... Does E tell the
truth?"), written from the puzzle text: the chain one person at a time (what the person says, whether that is so,
hence whether the person tells the truth), then the asked person. Ours."""
import re


def close(think, visible, boxed):
    return f"{think.rstrip()}\n</think>\n\n{visible.strip()}\n\n\\boxed{{{boxed}}}"


def state(v):
    return "tells the truth" if v else "lies"


def write(problem, gold):
    m = re.fullmatch(r"Question: (.+) Does (\w+) tell the truth\?", problem.strip())
    if not m:
        return None
    asked = m.group(2)
    sents = [s.strip() for s in m.group(1).split(".") if s.strip()]
    m0 = re.fullmatch(r"(\w+) (tells the truth|lies)", sents[0])
    if not m0:
        return None
    truth = {m0.group(1): m0.group(2) == "tells the truth"}
    t = [f"The user wants to know whether {asked} tells the truth.", "",
         "1.  **The chain, one person at a time:**",
         f"    *   {m0.group(1)} {m0.group(2)}. (given)"]
    order = [m0.group(1)]
    for s in sents[1:]:
        ms = re.fullmatch(r"(\w+) says (\w+) (tells the truth|lies)", s)
        if not ms or ms.group(2) not in truth:
            return None
        p, about, claim = ms.group(1), ms.group(2), ms.group(3) == "tells the truth"
        right = truth[about] == claim
        truth[p] = right
        order.append(p)
        t.append(f"    *   {p} says {about} {state(claim)}. {about} {state(truth[about])}, so what {p} says is "
                 f"{'true' if right else 'false'}: {p} {state(right)}.")
    if asked not in truth:
        return None
    ans = "Yes" if truth[asked] else "No"
    t += ["", f"2.  **Result:** {asked} {state(truth[asked])}, so the answer is {ans}."]
    vis = "Following the chain: " + ", ".join(f"{p} {state(truth[p])}" for p in order) + f". So the answer is {ans}."
    return close("\n".join(t), vis, ans), ans
