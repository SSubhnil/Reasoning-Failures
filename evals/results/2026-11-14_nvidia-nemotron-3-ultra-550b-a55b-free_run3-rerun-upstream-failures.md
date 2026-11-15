# Tool-calling eval

- Model: nvidia/nemotron-3-ultra-550b-a55b:free
- Date (UTC): 2026-11-14T23:25:06+00:00
- Tasks run: 5
- Tool-selection accuracy (tool tasks): 4/5
- Argument accuracy (tool tasks): 4/5
- Refuse tasks passed: 0/0
- Tasks passed overall: 4/5

| id | kind | tools called | tool selection | args | result |
|---|---|---|---|---|---|
| t09 | tool | list_collections, accuracy_breakdown, find_traces, find_traces, get_trace, get_trace, get_trace, check_answer | fail | fail | ERROR |
| t11 | tool | list_collections, find_traces, get_trace | pass | pass | PASS |
| t12 | tool | list_collections, accuracy_breakdown, find_traces, find_traces, get_trace | pass | pass | PASS |
| t13 | tool | list_collections, find_traces, find_traces, get_trace, get_trace, get_trace, get_trace | pass | pass | PASS |
| t17 | tool | list_collections, accuracy_breakdown, find_traces, get_trace | pass | pass | PASS |
