# Label sensitivity

Analyze 4B and 8B separately and keep trace-separated and problem-separated scores distinct. Use five BCE seeds, validation scores, fixed nuisance covariates and the banked correction tables. Training remains on stored labels.

S uses stored labels. A applies the binary-answer and letter-answer corrections per trace; newly correct traces leave the failure classes. B applies A and removes all problems with binary or letter-coded gold answers. B changes the selected population. C applies A and removes flagged unparseable traces. The flags cover capability gaps and pits but not near misses, so C is incomplete for near misses.

Report incorrect-trace counts, class shares, per-seed detection AUROC with mean and sample standard deviation, class miss rates at confidence 0.5, the pit-minus-near-miss difference, and the residualized pit average marginal effect. Miss-rate intervals use 1000 resamples of problems with seed 12345. Logistic intervals use 1000 resamples with seed 20260627. The default effect threshold is 0.08. The inferential gate requires a pit log-odds interval excluding zero, marginal effect at least 0.08, and positive differences in every populated comparison cell.

S must reproduce the banked stored-label outputs at their reported precision before A, B or C is interpreted. A correction set must be nonempty and fully matched to score IDs. A mismatch stops interpretation rather than changing the random seed or selected rows to obtain agreement.

If the paired pit-minus-near-miss interval includes zero in A or B at either scale, report the corrected miss-rate result with that qualification. If any class share changes by more than 0.05 in A, report the A counts in the taxonomy table. If the problem-split marginal-effect interval changes its position relative to 0.08 (above, crossing or below), report that corrected status. Otherwise retain the stored-label summary and disclose the largest measured change. The interval-based classification and the inferential gate are distinct decisions.

This file describes the public computation. The drivers hash these bytes into their existing metadata fields. It makes no claim to establish when the protocol was first specified.
