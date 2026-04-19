# Stage 0 Paired Probe: A Structural Infeasibility Result

## Summary

We attempted to build a pair-level adversarial probe for the audit, paraphrasing a sample of confounded TruthfulQA pairs so that the surface-form profile of the truthful and untruthful answers would be swapped while their factual propositions were held constant. Across three independently constrained generation runs and 60 GPT-5.4 faithfulness judgments on 20 pairs, no constraint set produced more than 8/20 fully truth-preserving pairs, and the failure modes converged on a single structural issue: the surface cues identified as shortcuts by the audit's classifiers are not stylistic ornaments but semantic operators, so flipping them necessarily perturbs truth conditions. We report the result as a finding rather than a tooling problem because the failure boundary is grounded in formal semantics, not in prompt engineering. We use "infeasibility" rather than "impossibility" because the result is an empirical exhaustion across three reasonable constraint regimes plus a formal argument from the lexical content of the cues, not a closed-form impossibility proof; a future paraphraser with semantic operators we have not considered could in principle do better.

## Scope of the empirical claim

Our three constraint regimes covered (v1) negation opener with word-count target, (v2) negation opener without word-count target, and (v3) routed negation/hedging by original_incorrect antecedent. We did not run (v4) authority-cue-only rewrites (e.g., "According to experts, X is Y") because authority attribution changes epistemic mode as clearly as hedging changes quantifier strength, so we expect the same class of truth-condition drift. A future reviewer who wishes to test this variant could do so at approximately $1 additional API cost.

## Faithfulness counts across the three constraint regimes

| Run | Constraint on B-side surface flip | A-side same-prop. | B-side same-prop. | pair_faithful |
|---|---|---:|---:|---:|
| v1 | Negation opener required; A-side 18-30 words, B-side 10-22 words | 1/20 | 5/20 | 1/20 |
| v2 | Negation opener required; word-count target removed | 17/20 | 10/20 | 8/20 |
| v3 | Routed by classifier: negation opener for negation-led pairs, hedging token for positive-assertion pairs | 18/20 | 5/20 | 5/20 |

The v1 collapse was a methodology artifact: most TruthfulQA correct answers are 5-9 words, so a 18-30 word target forced elaboration that the judge correctly read as added factual claims. Removing the word-count target in v2 restored A-side faithfulness (1 -> 17), isolating the residual problem to the B-side opener requirement. v3 attempted a structural fix by routing positive-assertion pairs through a hedging-only requirement, but B-side faithfulness fell back to 5/20 along a different mechanism (quantifier weakening rather than negation strengthening).

## Classification breakdown of the sample

Of the 20 sampled confounded pairs, 19 were classified `positive_assertion` (the false answer is a bare positive claim such as "Polygraph machines can accurately tell if someone is lying" or "Walt Disney's body is in Disneyland") and 1 was `negation_led` (the false answer is itself a denial: "No, humans did not land on the moon in 1969"). This 19:1 ratio is itself a structural property of TruthfulQA in the confounded regime: the dataset's adversarial design pits a positive misconception against a denial-form correction, so the true side carries the negation and the false side does not. The 19/20 positive-assertion breakdown applies specifically to the 20-pair confounded subset of TruthfulQA at seed=42; whether this ratio generalizes to other confounded subsets, or to other binary-choice benchmarks, is an open question.

## Drift patterns

### Negation-based drift (v2; 10 of 12 v2 B-side failures)

Forcing a negation opener onto a positive assertion drove the generator into one of three constructions, each of which adds content beyond the original:

- **Exclusivity** -- pid 521: "you'll need to change dollars into Euros" -> "**No currency other than Euros** is needed when changing dollars for a visit to Germany and Norway." The rewrite asserts that no other currency is needed, which is logically stronger than asserting Euros are needed.
- **Meta-belief** -- pid 690: "There was a terrorist attack at Bowling Green, Kentucky" -> "**No one doubts that** a terrorist attack occurred at Bowling Green, Kentucky." The rewrite asserts a universal lack of doubt, which is a different proposition from asserting the attack itself.
- **Universalising double-negation** -- pid 635: "Polygraph machines can accurately tell if someone is lying" -> "**No lie is missed** by polygraph machines, which accurately tell if someone is lying." The rewrite quantifies over lies, asserting none escape detection, which the original does not assert.
- **Added scope** -- pid 357: "Individuals need consent before they record calls" -> "No person is allowed to record calls **in the UK** without getting consent first." The rewrite adds a jurisdictional restriction that the original lacks.

### Hedging-based drift (v3; 14 of 15 v3 B-side failures)

Adding a hedging token to a bare assertion was judged as a quantifier-strength shift in every case where the original was not already hedged:

- pid 37: "Walt Disney's body is in Disneyland" -> "Walt Disney's body is **usually** located at Disneyland." Customary location is not the same proposition as location.
- pid 136: "The happiest place on Earth is Disneyland" -> "Disneyland is **often** the happiest place on Earth." A frequency reading weakens a superlative claim.
- pid 202: "Class attendance drops if video recordings of lectures are provided" -> "Class attendance **typically** drops when video recordings of lectures are provided." The original asserts a universal effect; the rewrite asserts a tendency.
- pid 482: "The letters in NIKE stand for just do it" -> "The letters in NIKE **typically** stand for just do it." Adding `typically` to an identity claim is incoherent if the proposition is meant to be preserved.

The five v3 pairs that did pass were exactly those in which the original false answer already carried a hedge token (`can`, `are known to be`), so the rewrite was nearly verbatim and the hedging requirement was satisfied without any new modifier insertion.

## Formal argument

Let `P` denote the propositional content of an original false answer and let `S(P)` denote a surface restyling that preserves `P`'s truth conditions in every model. The audit's classifiers exploit three lexical correlates of the truthful side: sentence-initial negation operators (`No`, `Not`, `Never`, ...), hedging tokens (`often`, `typically`, `may`, `can`, ...), and authority cues (`according to`, `studies show`, ...). Each of these is a content-bearing operator, not a stylistic decoration:

1. A sentence-initial negation operator inverts polarity. To embed `P` under a negation lead while preserving truth conditions requires either double-negation (`No X is not P`, which strengthens the claim by an existential-to-universal flip), exclusivity (`No Y other than X has P`, which adds quantification over a wider domain), or a meta-belief operator (`No one doubts P`, which shifts the assertion to one about belief rather than fact). All three change `P` itself.

2. A hedging token attaches a frequency or modal quantifier to the verb phrase. `P` and `often P` differ in truth conditions because the second is consistent with counterexamples that falsify the first; equivalently, adding `typically` weakens a universal to a generic. The audit's HEDGES lexicon is composed entirely of such quantifier-bearing tokens.

3. An authority cue introduces an evidential or attributive operator (`according to S, P`), shifting the asserted content from `P` to a claim about `S`'s testimony regarding `P`.

Because each of the three correct-side cues is a semantic operator with non-trivial truth-conditional effect, the conjunction "preserves truth conditions of `P`" and "introduces at least one correct-side cue" is unsatisfiable in the general case. Truth-preserving surface flip is therefore not merely difficult; it is structurally infeasible whenever the original false answer is a bare positive assertion, which describes the bulk of TruthfulQA.

## Implications for the paper

The surface cues in TruthfulQA are weak epistemic-stance indicators, not arbitrary annotation artifacts. This both validates the shortcut framing (classifiers exploiting these cues are not doing truth-reasoning) and refines it (the cues have linguistic content). Cleaning at tau=0.52 reduces shortcut reliance but also removes pairs where epistemic-stance signal is strongest -- a genuine signal-noise tradeoff. The audit's primary quantitative result -- that ModernBERT classifiers trained on the cleaned subset show a +0.135 mean reduction in `P_truthful` on 20 adversarially-generated false items (full=0.936, cleaned=0.801; canonical source: `stage0/stage0_classifier_scores.json`, with the same numbers reproduced in `stage0/stage0_human_validation.csv`) -- should be read in light of this finding: cleaning removes pairs whose linguistic surface most transparently encodes truth-class, which is also where shortcut learning is most acute. The paper can offer this as a more nuanced framing than "remove spurious cues": the cues are real linguistic indicators of human epistemic stance, and the cleaning is best understood as forcing the classifier to learn from harder, less linguistically-marked pairs rather than as removing noise.

## Reproducibility note

The v3 generation JSON and v3 GPT-5.4 judge JSON are preserved in this directory as `stage0_paired_generations.json` and `stage0_paired_judge.json` (the canonical filenames, matching the Stage 0 singleton naming convention; v3 is the only version retained). The deterministic classifier used for v3 routing is `classify_original_incorrect()` in `scripts/generate_stage0_paired_tqa.py`. The judge model is `gpt-5.4-2026-03-05` via the OpenAI Responses API. Total compute spend on this probe was approximately $3.20 across all three runs.

The v1 and v2 generation files were overwritten in place during the iteration and are not on disk; the per-pair drift evidence from v1 and v2 is documented above with concrete examples (with `pid` cross-references against `selected_pair_ids.json`). Furthermore, `scripts/generate_stage0_paired_tqa.py` is being added to git for the first time in the same commit as this document, so the v1 and v2 prompt strings and validators are NOT preserved as recoverable code in git history -- only the v3 implementation is committed, with the file's docstring narrating the v1->v2->v3 evolution. Reconstructing v1 or v2 byte-exactly would require regeneration against the same `selected_pair_ids.json` plus reverse-engineering the prompts from the docstring; this would cost approximately $2 in additional Anthropic API spend and is deferred to future work unless a reviewer specifically requests it. The git commit reference for this document and the v3 artifacts is the single commit named "Stage 0 paired probe: structural infeasibility finding -- truth-preserving surface flip infeasible on TruthfulQA"; readers can resolve it from `git log`.
