# data/aux_if.jsonl: self-written instruction prompts

553 prompts in the toolkit's item format, task "ifeval": `wsp.grade` grades them with the lm-eval instruction
checkers (strict, prompt level), as it grades the toolkit's `ifsynth` practice set. A script built them from
practice data (`practice:gsm8k_train` topics) and self-written templates.

Uses in this method (`derive/run_items.py` turns them into case runs):
- `split = fit` (280 prompts): base traces whose decision positions are hard negatives of the gate, and whose prompt
  and answer positions are easy negatives (`hesitation_gate.py --neg-runs ... auxfit`);
- `split = test` (273 prompts, never in any fit): the held-out instruction check. Base and edit at the same seeds:
  the share of answers whose thinking never closes, and accuracy (`compare.py`, rule I of METHOD.md).
Fit and test share no topic and no wording of the main instruction (see Construction), so the test split measures
the gate on instruction prompts it has never seen.

## Fields
- `id` "aux_<type code>_<n>", `source` "aux_if", `task` "ifeval", `answer` null, `problem` (the user message).
- `instruction_id_list`, `kwargs`: lm-eval registry ids and arguments, as the registry draws them.
- `split`: fit | test. `type`: the main instruction id ("plain": none). `group`: rule | hardneg | plain.
- `wording`: wording id of the main instruction: `<id>/w0` is the registry's own description; `<id>/p1`, ... are
  self-written paraphrases with identical kwargs. `wordings`: one id per instruction.
- `genre`, `topic`, `topic_kind` (thing | setting), `placement` (constraint after the request: end; before: start).

## Counts
| type | group | fit | test | with a second instruction |
|---|---|---|---|---|
| punctuation:no_comma | rule | 50 | 49 | 26 |
| detectable_format:number_highlighted_sections | rule | 50 | 49 | 27 |
| change_case:english_lowercase | rule | 50 | 50 | 36 |
| change_case:english_capital | rule | 48 | 49 | 36 |
| detectable_format:json_format | hard negative | 9 | 8 | 0 |
| startend:quotation | hard negative | 8 | 7 | 0 |
| startend:end_checker | hard negative | 8 | 8 | 0 |
| detectable_format:number_bullet_lists | hard negative | 9 | 8 | 0 |
| detectable_format:title | hard negative | 9 | 8 | 0 |
| detectable_content:postscript | hard negative | 9 | 8 | 0 |
| plain | plain | 30 | 29 | 0 |

Genres: advice 44, description 47, dialogue 50, email 47, how-to 44, letter 42, poem 46, product blurb 46, review 51, speech 42, story 47, summary request 47.

## Construction
- Topics: concrete nouns mined from the 7469 practice:gsm8k_train questions. Things are plural
  nouns right after a number; settings are nouns after "at/to/in/... the". Self-written stop lists remove units,
  times, people, containers, abstract words and rooms of a home; a noun is kept if at least 3 questions use it.
  Result: 253 things and 51 settings. No numbers are taken over.
- Requests: 12 self-written genres, each with one form for a thing and one for a setting. A summary request
  quotes the narrative sentences of a gsm8k_train question that names the topic, without its question sentences.
- Constraints: ids and kwargs from the lm-eval registry, drawn the way `wsp.data.make_ifsynth` draws them
  (seed 0). Rule types: no_comma, number_highlighted_sections, english_lowercase, english_capital;
  about 30% of them carry one second instruction that the registry does not
  mark as conflicting (title, postscript, end phrase, bullets or quotation). Hard negatives: json_format,
  quotation, end_checker, number_bullet_lists, title, postscript, one instruction each. Plain: no instruction.
  The constraint follows the request in about 75% of the prompts and precedes
  it otherwise.
- Split by wording and by topic. Things and settings are each halved by a seeded shuffle, so fit and test
  share no topic. Rule types use the registry wording and two paraphrases in fit, and two other paraphrases in
  test; the other types use the registry wording and one paraphrase in fit, and two other paraphrases in test.
  The registry wording never appears in test.

## Decontamination
Every prompt passed the exclusion bars below, set after the templates, types and counts were fixed; none was
rejected. Nothing in this file is taken from evaluation text, and the method never reads any.
- The constraint sentences are removed from each prompt first; the remaining request text is compared.
- A prompt is rejected if its char 5-gram TF-IDF cosine is >= 0.5 with any benchmark prompt. Each
  benchmark prompt is compared whole, and again with its constraint-like sentences removed. The TF-IDF
  vocabulary is built from the candidates and these benchmark prompts, only to measure similarity; nothing from
  the benchmark is in this file.
- A prompt is rejected if it shares any word 8-gram with a benchmark prompt (lowercased alphanumeric
  words). This check covers the request text and the self-written constraint sentences. Strings supplied by
  the registry (end phrases, postscript markers) are masked first: like the registry wording, they come from
  the checker library.
- A prompt is rejected if its fingerprint (`wsp.data.fingerprint`) is in the toolkit's
  data/bench_fingerprints.txt.gz.
- Candidates: 649; rejected: none. The first survivors of each type and split, up to the target counts, are kept:
  553 prompts.
- Kept prompts: max cosine at most 0.283 (median 0.140). Longest
  shared word n-gram: 6 words for the request text, 5 words for the
  self-written constraint sentences.
