# Report: wrap-up to answer mode (see METHOD.md)

**Failure targeted.** Answers cut at the token limit with the thinking still open, on the start's model (mostly
maths and ten-option multiple choice). The start's own measurements on the `train` split are in
`$WSP_WORK/wuam/lengths.json` (cut share by kind, thinking length of right answers, how many long runs end right).

**Derivation (all in the run, practice data only).**
- The start's own answers on `train` (2 each) and `dev` (2 each).
- Wrap-up unit of `0009538688-0032` ("wrap-up", credited; `derive/wrapup14.py` is its published code): a ridge read
  of the thinking depth from the MLP input of layer 10, and one rarely used MLP unit rewritten so that past `theta`
  (the 75th percentile of the thinking length of the start's right `train` answers) it pushes toward the model's own
  end-of-thinking state; doses 2, 3 and 4 (1 = the measured mid-thinking to end-of-thinking gap at a read-out of
  22,000 tokens).
- Control: the same unit writing a random direction of the same size (`W_RAND`, dose 3).

**Choice.** The dose with the largest pooled dev gain among those that lose on no task type by more than
max(2 SE, 1 point) and beat the control; if none beats the control, the best passing dose; if none passes, the dose
with the largest pooled gain (`pick_rule.txt` names the rule that decided). Then confirmed once on holdout (chosen on
nothing there).

**Tables.** Generated below: every candidate and the control on dev, per task type, with the paired gain and its
standard error; the holdout table of the chosen edit.

**What did not work / caveats.** Four-option multiple choice moved -5 to +6 across our practice splits, partly
noise (its random control moved -3 on one split). A start that already closes late answers leaves the unit little
to do; the report then says so with its dev and holdout tables.
