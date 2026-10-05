# Failure shares by problem type

Analyze the 4B and 8B broad collections separately. S uses stored incorrect-trace labels. A removes traces marked correct by the binary-answer or letter-answer corrections. Keep all defined types, including small groups.

For DeepMath, type is the second component of the topic field after splitting on ` -> `. For MATH, use the subject encoded in the problem identifier, prefixed by `MATH:`. Missing or unrecognized identifiers are errors.

Report each type's trace count, problem count and failure-class proportions. Obtain 95 percent percentile intervals from 1000 resamples of problems within each type, using seed 12345. Compute uncorrected Cramer's V on the trace-level class-by-type table. For its null, permute type labels across problems while keeping each problem's traces together: 1000 draws, seed 20260930, p-value `(1 + count(null >= observed)) / 1001`.

V below 0.15 at both scales under S supports a weak-association description, qualified by which type-specific intervals exclude the pooled share. V at least 0.15 at either scale calls for reporting the type-specific table without a similar-across-types description. Report the permutation p-value but do not use it as this decision threshold. This analysis does not estimate detection AUROC or paired accuracy.

The driver hashes this public protocol into its existing metadata fields. This file specifies the computation, not a historical timing claim.
