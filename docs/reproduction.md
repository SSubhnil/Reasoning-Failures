# Reproduction guide

`analysis/` contains table reducers and figure programs. `scripts/` contains probe training, feature extraction, scoring and inference commands. `sst/` contains their implementations. `tests/` contains synthetic tests. The package paths used by saved probe and corpus objects remain `sst.*`.

The study uses Qwen3-4B and Qwen3-8B reasoning traces. Monte Carlo continuation estimates separate incorrect traces into capability gaps, pits, and near misses. Confidence probes predict held-out correctness. Their miss rates differ across these classes. The tested individual localization probes do not establish an advantage over the length rule. The five-probe last-token ensemble is a separate estimator and clears that rule in the reported 4B comparison.

Code, data keys and output files label the near-miss class `reaches` (`pit_vs_reaches`, `n_reaches` and similar); it is the class the paper calls near miss. In the transcoder classifier outputs, `z_presence_union` is the transcoder classifier and `h_presence_single` the hidden-state classifier.

Run the tests with `python -m pytest -q tests`.

## Data

The data files are in the Hugging Face dataset `ssubhnil/Reasoning-Failures`. The repository mirrors the `runs/` and `data/` layout the commands read. Download it with the command in the README and link `hf_data/runs` and `hf_data/data` into the repository root as `runs` and `data`. The README command downloads the inputs of the table commands; without its `--include` and `--exclude` options it downloads everything. The scripts that read hidden-state features take them from `runs/hf_pull/probe_corpus_16k/qwen3-*/_p1_packed` and `runs/value_probe/ask_a_818/_packed`; download those directories from the dataset (omit the matching `--exclude` options) or regenerate the trace features with `python scripts/precompute_p1_features.py --shard-dir runs/hf_pull/probe_corpus_16k/qwen3-4b/_p1_packed --student-model models/student/Qwen3-4B` (GPU; `--shard-start`/`--shard-end` split the work across jobs). The models load from `models/student/Qwen3-4B` and `models/student/Qwen3-8B`; download each with `hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir models/student/Qwen3-4B` and `hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/student/Qwen3-8B`. These commands pin the model versions used in the paper; the transcoder versions are pinned in the classifier section. Transcoder activation stores are not distributed; run the extraction first (see the classifier section). The `.pt` files are pickles: load them only from the dataset repository above.

| Input | Consumers |
|---|---|
| `runs/resp/locus_{4b,8b}_k16/campaign/gold_pits.json` | Taxonomy and class-dependent confidence analyses |
| `runs/mc_gold/locus_{4b,8b}_k16/gold_pits.json` | Label sensitivity and type shares; same selected populations as the corresponding locus labels |
| `runs/conf_probe{,_psplit}/blindspot_scores/{4b,8b}_{bce,brier}_s{0..4}.parquet` | Detection and miss-rate tables, separately for each split |
| `runs/conf_probe/confound_lookup.parquet`, `runs/conf_probe/conceal/steps_{4b,8b}.parquet` | Nuisance controls and answer position |
| `runs/conf_probe/grouped_split_reeval.csv` | Stored trace-split calibration reference |
| `runs/mc_conf_probe/mc_vs_bce_{4b,8b}_perseed.parquet` | Per-trace rollout-supervised and terminal-label probe scores; the terminal-label scores are from the trace-split confidence probe (supporting-table row "rollout-supervised − terminal-label probe, held out") |
| `runs/conf_probe_psplit/mc_vs_bce_{4b,8b}_perseed.parquet` | The same with terminal-label scores from the problem-split confidence probe (pit versus near miss scores in the claims table and the miss-rate figure, panel C) |
| `runs/resp/bipp_locus_4b/` labels, validation problems, predictions and `score_excl818.json` | Paired localization and pointer figure |
| `data/paper_tables/probe_layer_localization.csv` | Per-layer distance-score table of the localization figure's right panel |
| `data/paper_tables/mc_regimes_full.csv`, `mc_pit_full_4b.csv`, `mc_pit_full_8b.csv` | Taxonomy figure |
| `runs/mc_gold/campaign_final/{gold_pits.json,mc_ground_truth.pt}` | 4B example continuation curves and dense-label pit dip |
| `runs/hf_pull/mc_rollouts_locus_k16/qwen3-8b/campaign/`, `runs/resp/bipp_locus_8b/gold_pits_by_md5.pt`, value-probe readouts | 8B localization comparison |
| `runs/failure_type_classifier/gd_{4b,8b}_{s,d}_seed{0..5}.json` and `.oof.npz` | Transcoder failure-type classifier table (stored per-partition scores; `s` = stored labels, `d` = defect-exposed problems removed) |
| `runs/failure_type_classifier/inputs/{4b,8b}/`: `problem_regimes.json`, `trace_ids.json`, `questions.json`, `trace_logprobs.json`, `trace_texts.json.gz`, `defects_removed.json` | Transcoder extraction and classifier fit |
| `runs/paper/labels/`, `runs/paper/correction_flags/{4b,8b}/unverified_flags.parquet` | Separate correction flags with unchanged trace/problem keys |
| `data/problem_lookup.parquet` | Problem type (DeepMath-103K level-two topic) for the type shares, and problem difficulty for the transcoder classifier fit |
| `runs/layer_selection/qwen3-{4b,8b}/{depth_curve.csv,count_curve.csv,decision.json}` | Read-site selection figure and the read-site band row of the supporting table |
| `runs/value_probe/ask_a_818_pertrace.parquet`, `runs/value_probe/ask_a_818_leakage_provenance.parquet` | Label-swap rows of the supporting and paired tables. Stored output of `scripts/score_gold_set_localizers.py`; its feature inputs are in `runs/value_probe/ask_a_818/_packed`, but the control-arm checkpoints it also reads (`runs/localizer/qwen3-4b`) are not distributed |
| `runs/resp/campaign_final/gold_pits_by_md5.pt` | Gold-collection failure steps keyed by trace MD5 (5,380 traces, 818 with a failure step) |
| `data/paper_tables/mc_ksweep.csv` | Refinement-depth agreement sub-row of the supporting table (0.754 at K = 16, 0.638 at K = 8): Hit@±2 of the located step against the K = 32 run, on the 818 pit traces of the gold collection (stored table; the underlying rollouts and the generating script are not retained). The `hit_at2_vs_independent` and `regime_flips` columns are not used by the paper |
| `data/paper_tables/mc_density.csv` | Rollout-search Hit@±2 and evaluations per trace in the supporting table (stored table; no generating script is included) |

The table lists the inputs the commands expect, at the paths the commands read.

## Tables and figures

Run modules from the repository root. The primary confidence analyses use problem-separated scores. Trace-separated scores are a separate comparison. Prepare output directories and the answer-position cache:

```sh
mkdir -p outputs/conceal figures
cp runs/conf_probe/conceal/steps_*.parquet outputs/conceal/
export SST_CONCEAL_SCORES=runs/conf_probe_psplit/blindspot_scores
export SST_CONCEAL_OUT=outputs/conceal
export SST_DECONF_SCORES=runs/conf_probe_psplit/blindspot_scores
```

| Table or figure | Command |
|---|---|
| Taxonomy threshold sensitivity, both scales | `python -m analysis.taxonomy_robustness > outputs/taxonomy.txt` |
| Taxonomy figure | `python -m analysis.make_pit_anatomy_figure` |
| Detection metrics across retraining seeds | `python -m scripts.aggregate_conf_probe --root runs/conf_probe_psplit --out outputs/detection.csv` |
| Detection with nuisance controls | `python -m scripts.deconfound_conf_probe --out outputs/detection_controls.json` |
| Calibration table (problem split) | `SST_ECE_SCORES=runs/conf_probe_psplit/blindspot_scores SST_ECE_REEVAL=0 SST_ECE_OUT=outputs/calibration_problem.json python -m analysis.calibration_by_loss` |
| Trace-split calibration comparison | `SST_ECE_SCORES=runs/conf_probe/blindspot_scores SST_ECE_OUT=outputs/calibration_trace.json python -m analysis.calibration_by_loss` |
| Value-matched miss rates | `python -m analysis.miss_rate_by_initial_recovery --loss bce` |
| Residualized miss-rate decision | `python -m analysis.residualized_miss_rate_effect --n-boot 1000 --skip-brier` |
| Miss rates by failure class | `python -m analysis.miss_rate_by_failure_class --loss bce` |
| Supervision comparison, problem split | `SST_CALIB_PERSEED_DIR=runs/conf_probe_psplit SST_CALIB_OUT=outputs/calibration_matched.json python -m analysis.rollout_vs_outcome_ordering` |
| Miss-rate and supervision figure | `SST_FIG_T1=outputs/conceal/t1_value_control.json SST_FIG_T3=outputs/conceal/t3_per_class_cer.json SST_FIG_CALIB=outputs/calibration_matched.json python -m analysis.make_miss_rate_figure` |
| Stored pit-dip control, 4B | `python -m analysis.confidence_dip_at_failure_step --model 4b --out t6_pit_dip_4b.json` |
| Stored pit-dip control, 8B | `python -m analysis.confidence_dip_at_failure_step --model 8b --out t6_pit_dip_8b.json` |
| Pointer versus length-rule paired comparison | `python -m analysis.paired_localization --pred-a runs/resp/bipp_locus_4b/score_lasttok_predictions.csv --vs-null --labels runs/resp/bipp_locus_4b/gold_pits_by_md5.pt --val-problems runs/resp/bipp_locus_4b/val_problems.json --n-boot 2000 --seed 0 > outputs/localization.txt` |
| Pointer and per-layer distance-score figure | `python -m analysis.make_wall_figure` (reads `outputs/detection_controls.json` from the detection-controls command) |
| 8B value-probe comparison | `python -m analysis.value_probe_vs_length_rule --readout runs/value_probe/qwen3-8b/readout.json --arm hidden --n-boot 2000 --boot-seed 0 --out outputs/localization_8b.json` |
| Transcoder failure-type classifier table | `python -m analysis.failure_type_classifier_table --scores-dir runs/failure_type_classifier --out outputs/failure_type_classifier.json` (add `--recompute` to recompute AUROCs and intervals from the out-of-fold scores) |
| Example continuation curves | `python -m analysis.make_reachability_figure` |
| Read-site selection figure | `python -m analysis.make_layers_figure` |
| Crossing rule versus argmin of step success | `python -m analysis.crossing_vs_argmin` |
| Uniform floor and length rule on the 818 gold-collection pits (pointer figure reference lines) | `python -m analysis.length_rule_baseline --labels runs/resp/campaign_final/gold_pits_by_md5.pt` |
| Broad-fitted length rule on the 818 gold-collection pits (pointer figure reference line) | `python -m analysis.length_rule_baseline --labels runs/resp/campaign_final/gold_pits_by_md5.pt --fit-labels runs/resp/bipp_locus_4b/gold_pits_by_md5.pt` |
| Label sensitivity, primary split | `python -m analysis.label_defect_sensitivity --split problem --out outputs/label_sensitivity_problem.json` |
| Failure shares by problem type | `python -m analysis.failure_shares_by_problem_type --out outputs/type_shares.json` |

The value-matched, residualized and per-class miss-rate commands print nothing; they write `t1_value_control.json`, `t2_residualized_logistic.json` and `t3_per_class_cer.json` to `$SST_CONCEAL_OUT`.

The taxonomy depth statistics (`analysis.mc_campaign_analysis`) need the original `slice_*/steps.jsonl` inputs, which are not distributed; they cannot be reconstructed by passing a merged `.pt` file to that script. The layer-localization CSV is a stored table; this command replays it rather than rerunning layer extraction. The curve figure does not reproduce the density-scan or refinement-comparison results. No reproducer for those results is included. `python -m analysis.crossing_vs_argmin` computes the crossing-versus-argmin agreement of the supporting table from the campaign labels and step success.

The miss-rate figure reads its three inputs independently. In all three panels the confidence-probe scores are problem-split; the rollout-supervised probe's held-out scores in panel C are the same in both per-seed files. The pit-dip driver reads the stored trace-split confidence curves independently of `SST_CONCEAL_SCORES`.

The problem-split calibration command turns off the re-evaluation gate (`SST_ECE_REEVAL=0`) because the gate's reference file, `runs/conf_probe/grouped_split_reeval.csv`, is trace-split. The detection-controls output reports standard deviations over seeds with the population formula; the paper's tables use the sample formula, which is larger by a factor of √(5/4).

The nonlinear text controls have separate commands, in dependency order. They require stored `cot_dynamics_4b.parquet` and `cot_dynamics_8b.parquet` under `runs/conf_probe/dynamics/`:

```sh
python -m scripts.deconfound_conf_probe --dynamics-dir runs/conf_probe/dynamics --n-boot 2000 --out outputs/logistic_null.json
python -m analysis.text_baseline_gbdt --dynamics-dir runs/conf_probe/dynamics --logistic-ref outputs/logistic_null.json --out outputs/gbdt_null.json
python -m analysis.text_baseline_variants --dynamics-dir runs/conf_probe/dynamics --out outputs/text_null_variants.json
python -m analysis.text_baseline_increment --dynamics-dir runs/conf_probe/dynamics --gbdt-ref outputs/gbdt_null.json --out outputs/incremental_value.json
```

Probe training and feature extraction need a GPU. Use `python -m scripts.train_conf_probe --help`, `python -m scripts.train_value_probe --help` and `python -m scripts.train_pointer_probe --help` for their arguments. Set `--split-by problem --label terminal` for the primary confidence probe.

## vLLM evaluation

`scripts.evaluate_vllm` runs in the base environment and starts a local vLLM server with the vLLM environment's interpreter (`--server-python`), waits for its health endpoint, generates greedy answers for a JSONL file and grades them with the rule checker. Decoding is greedy by default (`--temperature 0.0`); pass `--temperature` and `--seed` to sample. Each input row has `id`, `question` and `ground_truth_solution`. `ground_truth_solution` must contain the answer in `\boxed{…}` or `$…$`; the script stops and lists the rows whose gold cannot be parsed. The `gt_answer` column of the evaluation parquets is the grader's normalised answer string and is not a valid `ground_truth_solution` input. Questions are chat-templated by `StudentModel`; do not pass rendered prompts. The client keeps `--pipeline-depth` batches submitted, so the server decodes the next batch while the previous one is graded on CPU. The client batch size equals `--max-num-seqs`. The released `evaluation/` files are single sampled generations; their `decoding` column says which.

```sh
python -m scripts.evaluate_vllm --input data/questions.jsonl --out outputs/evaluation.jsonl \
  --model models/student/Qwen3-4B --server-python .venv-vllm/bin/python --max-new-tokens 4096 --max-model-len 8192 \
  --max-num-seqs 512 --pipeline-depth 2 --gpu-memory-utilization 0.75 --tensor-parallel-size 1 \
  --port 8000 --startup-timeout-s 1800 --verify-timeout-s 60 --verify-workers 8
```

`StudentModel` also loads the student weights on CPU for tokenization, which uses host memory. See `docs/throughput.md` for the settings.

## Monte Carlo rollouts

`scripts.run_step_rollouts` runs in the vLLM environment and runs vLLM in process. For every step of every requested trace it builds the prefix question plus trace up to the end of that step, using merged character spans that must match the corpus step count. It samples one greedy continuation and `k - 1` sampled continuations per prefix, sends a whole batch of prefixes to vLLM in one call, grades every continuation with the rule checker, and writes the per-step success count and success rate.

```sh
python -m scripts.run_step_rollouts --corpus-dir runs/hf_pull/mc_rollouts_locus_k16/qwen3-4b/corpus \
  --trace-ids trace_ids.json --model models/student/Qwen3-4B --out outputs/rollouts.jsonl --k 16 \
  --max-new-tokens 4096 --temperature 0.7 --top-p 0.95 --seed 0 --max-model-len 24576 --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 --prefix-batch 64 --step-k 400 --verify-timeout-s 60 --verify-workers 8
```

The corpus directory holds the rollout corpus shards (one dict per trace with `trace_id`, `problem_id`, `question`, `trace`, `canonical_length` and `ground_truth`; other fields are ignored). The longest prefix in the 16k corpus is about 16,600 tokens, so a 4,096-token cap needs `--max-model-len` of at least about 20,700. This command generates new rollouts. It does not regenerate the released rollout labels: the temperature, top-p and seed of the original run are not recoverable, and this script is not the original generator. The values above are examples.

## Transcoder failure-type classifier

The classifier reads pretrained transcoders at layers 19 to 24 of Qwen3-4B and Qwen3-8B (MIT licence). Layer `L` reads the file `layer_{L-1}.safetensors`. Download the revisions used here:

```sh
hf download mwhanna/qwen3-4b-transcoders layer_18.safetensors layer_19.safetensors layer_20.safetensors \
  layer_21.safetensors layer_22.safetensors layer_23.safetensors \
  --revision 94d176260ac39ce2f882b8b09aba8c118df29bb3 --local-dir transcoders/qwen3-4b
hf download mwhanna/qwen3-8b-transcoders layer_18.safetensors layer_19.safetensors layer_20.safetensors \
  layer_21.safetensors layer_22.safetensors layer_23.safetensors \
  --revision dc677109cde096a85d03fff4f73a3ec88e7e2105 --local-dir transcoders/qwen3-8b
```

Extraction runs one forward pass over the lowest-trace_id failed trace of each problem and stores, per trace, the sum over generated tokens of the ReLU transcoder latents (983,040 dimensions over six layers) and of the ReLU hidden states (15,360 at 4B, 24,576 at 8B). It needs a GPU; on one 48 GB L40S it took about 40 minutes at 4B and about 3 hours at 8B. The log-probability statistics used by two of the baselines need a second GPU pass. Both run in the base environment.

```sh
I=runs/failure_type_classifier/inputs/4b
python -m analysis.transcoder_presence_extract --corpus-dir runs/hf_pull/mc_rollouts_locus_k16/qwen3-4b/corpus \
  --problem-regimes $I/problem_regimes.json --trace-ids $I/trace_ids.json --student-model models/student/Qwen3-4B \
  --transcoder-dir transcoders/qwen3-4b --layers 19 20 21 22 23 24 --out stores/presence_4b
python -m analysis.trace_logprob_extract --corpus-dir runs/hf_pull/mc_rollouts_locus_k16/qwen3-4b/corpus \
  --problem-regimes $I/problem_regimes.json --trace-ids $I/trace_ids.json --student-model models/student/Qwen3-4B \
  --out stores/trace_logprobs_4b.json
```

The fit runs on CPU and requires the extraction store. Its rows are ordered by problem id, one trace per problem. One call fits one fold partition (`--partition 0` to `5`) for one label set; add `--remove-problems $I/defects_removed.json` for the set with defect-exposed problems removed. It writes the score file and its out-of-fold scores, which the table command reads (name them `gd_4b_s_seed<k>` and `gd_4b_d_seed<k>`). One partition took between 40 minutes and 3 hours at 4B and between 3 and 4.5 hours at 8B.

```sh
python -m analysis.failure_type_classifier_fit --store stores/presence_4b --questions $I/questions.json \
  --problem-lookup data/problem_lookup.parquet --tokenizer models/student/Qwen3-4B \
  --trace-logprobs stores/trace_logprobs_4b.json --trace-texts $I/trace_texts.json.gz --partition 0 \
  --out outputs/failure_type_classifier/gd_4b_s_seed0.json
python -m analysis.failure_type_classifier_table --scores-dir outputs/failure_type_classifier --out outputs/failure_type_classifier.json
```

## Limitations

- Released labels are the stored labels. Corrections for binary answers, letter-coded answers and unparseable answers are separate flags. The shipped grader is the corrected implementation, so regrading can change labels. Probe training used the stored labels.
- The released rollouts used 16 continuations (1 greedy, 15 sampled) at refined steps and 8 elsewhere, with a 4,096-token continuation cap. Their temperature, top-p and seed are not recoverable from the artifact. The rollout script uses one `--k` for every step.
- At 8B the localization evidence is a value probe on the pit steps of the broad collection, paired at the seed level. There is no 8B gold collection.
- Class-dependent miss rates do not establish the residualized decision at both scales. The supervision comparison contrasts two training designs that differ in pool, mask, weighting and optimization. It does not isolate the effect of the labels alone.
- Read positions. The features were not read exactly at the last token of each step. The stored step positions (`step_spans`) were computed from each step tokenized on its own, and the features were extracted in left-padded batches without shifting those positions. Each read therefore sits earlier in the trace than the step's last token, by the same number of tokens for every step of a trace: a median of 157 tokens at Qwen3-4B and 148 at Qwen3-8B, at most 512, and zero for about 22% of traces (pad widths reconstructed from the batching rule). For the final step, the read falls inside the intended step in 51.0% (4B) and 51.6% (8B) of traces and in an earlier step in 48.9% and 48.3% (the remaining 0.1% fall between steps). Tokenizing steps separately adds a smaller drift (a median of 3 tokens on a 40-trace Qwen3-4B sample). The probes in the paper were trained and evaluated on these features, and `scripts/precompute_p1_features.py` reads at the same positions. No result was re-scored with corrected positions. To read exact last-token states, locate steps with `merged_step_char_spans` and map their character spans to the token offsets of the joint prompt-and-trace tokenization.
- `python -m analysis.rollout_vs_outcome_probe_scores --model {4b,8b}`, which writes the per-seed files read by the supervision comparison, requires a GPU and has not been run in this release; the shipped per-seed outputs (`runs/mc_conf_probe/mc_vs_bce_{4b,8b}_perseed.parquet`) are the result.
- `scripts/precompute_p1_features.py` has not been run on a GPU in this release. It reads each step at the same positions as the stored features (the last token of each stored step span, under the same left-padded batching).
- The transcoder extraction and log-probability scripts have not been run on a GPU in this release. Run on the original activation stores, the fit reproduces the stored scores exactly on partition 0 for Qwen3-4B with both label sets and for Qwen3-8B with stored labels; the other partitions and the Qwen3-8B defects-removed set were not refitted.
- The localization figure's right panel plots, per layer, the AUROC of an untrained distance score: the distance of each step's last-token hidden state from the pooled step states of the same problem's correct traces, scored by how well it separates the pit step from the other steps of the trace, on 76 Qwen3-4B pit traces. Each point is the midpoint and each bar the range over four distance measures; the bars are not confidence intervals. The table is shipped as data; the script that computed it is not included.
- The base environment pins torch 2.10.0, transformers 5.3.0, accelerate 1.13.0 and peft 0.18.1. The vLLM environment pins vllm 0.19.1 with transformers 5.13.0 (`requirements-vllm.txt`). Neither environment has a full lock. GPU execution has not been tested.
