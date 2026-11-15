"""Pydantic models, constants and citation text for the Reasoning-Failures MCP server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DATASET = "ssubhnil/Reasoning-Failures"

COLLECTIONS: tuple[str, ...] = (
    "eval_aime",
    "eval_competition_math",
    "eval_deepmath",
    "eval_gsm_plus",
    "eval_math500",
    "eval_math_perturb",
    "eval_math_test",
    "eval_math_train_l1_3_indist",
    "eval_putnam_axiom",
)

Collection = Literal[
    "eval_aime",
    "eval_competition_math",
    "eval_deepmath",
    "eval_gsm_plus",
    "eval_math500",
    "eval_math_perturb",
    "eval_math_test",
    "eval_math_train_l1_3_indist",
    "eval_putnam_axiom",
]

GroupBy = Literal["model", "subject", "level", "token_budget", "finish_reason", "dataset"]
TraceView = Literal["full", "tail", "steps"]
CheckReason = Literal["ok", "no_boxed_answer", "unparseable", "timeout"]

CITATION = (
    "Subhnil, Belkhiter, Miehling, Dahlgren Lindström and Daly (2026). Effectiveness of Hidden-State "
    "Probes for Localizing Reasoning Failures. Interpretability as a Science: NeurIPS 2026 Workshop. "
    "Data: huggingface.co/datasets/ssubhnil/Reasoning-Failures (CC BY 4.0)."
)

BIBTEX = r"""@inproceedings{
subhnil2026effectiveness,
title={Effectiveness of Hidden-State Probes for Localizing Reasoning Failures},
author={Shubham Subhnil and Yannis Belkhiter and Erik Miehling and Adam Dahlgren Lindstr{\"o}m and Elizabeth M. Daly},
booktitle={Interpretability as a Science: NeurIPS 2026 Workshop},
year={2026},
url={https://openreview.net/forum?id=X1D0ZeEeZ4}
}"""


class CollectionInfo(BaseModel):
    name: str
    rows: int
    columns: list[str]
    filter_supported: bool = Field(description="find_traces with structured filters works on this collection.")
    search_supported: bool = Field(description="find_traces with text_query works on this collection.")
    rows_supported: bool = Field(
        description="get_trace can fetch any row_idx. If false, get_trace serves only rows that "
        "find_traces returned in this session."
    )
    notes: list[str]


class ListCollectionsOutput(BaseModel):
    collections: list[CollectionInfo]
    citation: str


class TraceMeta(BaseModel):
    row_idx: int
    problem_id: str | None = None
    model: str | None = None
    dataset: str | None = None
    subject: str | None = None
    level: str | None = None
    difficulty: float | None = None
    token_budget: int | None = None
    correct_strict: bool | None = None
    finish_reason: str | None = None
    predicted_answer: str | None = None
    gt_answer: str | None = None
    question_preview: str = Field(description="First 300 characters of the question.")
    trace_chars: int | None = Field(
        default=None,
        description="Length of the full trace in characters; null if the upstream cell was truncated.",
    )


class FindTracesOutput(BaseModel):
    collection: str
    total_matches: int | None = Field(default=None, description="Total rows matching the filters, if reported.")
    offset: int
    items: list[TraceMeta]
    next_offset: int | None = Field(description="Pass as offset to get the next page; null when there are no more.")
    partial: bool = Field(description="True if the upstream index covers only part of the collection.")
    source: Literal["dataset_viewer", "parquet"] = Field(
        description="Where the page came from: the dataset viewer, or DuckDB over the parquet files when the "
        "viewer was failing."
    )
    notes: list[str]
    citation: str


class Step(BaseModel):
    index: int
    start_char: int
    end_char: int
    text: str


class GetTraceOutput(BaseModel):
    meta: TraceMeta
    view: TraceView
    text: str | None = Field(description="Trace text for view='full' or 'tail'; null for 'steps'.")
    steps: list[Step] | None = Field(description="Steps for view='steps'; null otherwise.")
    total_steps: int | None = Field(default=None, description="Number of steps in the whole trace (view='steps').")
    truncated: bool = Field(description="True if any text was cut to fit max_chars.")
    dropped_step_indices: list[int] = Field(description="Indices of whole steps left out to fit max_chars.")
    notes: list[str] = Field(default_factory=list)
    citation: str


class CheckAnswerOutput(BaseModel):
    correct: bool
    extracted_answer: str | None = Field(description="The answer parsed from the last \\boxed{...}, if any.")
    gt_used: list[str] = Field(description="The normalised gold answers the response was checked against.")
    reason: CheckReason


class BreakdownRow(BaseModel):
    group: str | int | None = Field(description="Value of the group_by column (null means missing).")
    n: int
    accuracy_strict: float | None
    accuracy_answer_forced: float | None


class AccuracyBreakdownOutput(BaseModel):
    collection: str
    group_by: GroupBy
    rows: list[BreakdownRow]
    source: Literal["duckdb"] = "duckdb"
    notes: list[str]
    citation: str
