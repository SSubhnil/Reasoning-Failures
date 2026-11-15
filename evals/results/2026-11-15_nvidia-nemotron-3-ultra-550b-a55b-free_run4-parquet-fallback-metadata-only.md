# Tool-calling eval

- Model: nvidia/nemotron-3-ultra-550b-a55b:free
- Date (UTC): 2026-11-15T11:12:58+00:00
- Tasks run: 20
- Tool-selection accuracy (tool tasks): 15/17
- Argument accuracy (tool tasks): 15/17
- Refuse tasks passed: 3/3
- Tasks passed overall: 18/20

| id | kind | tools called | tool selection | args | result |
|---|---|---|---|---|---|
| t01 | tool | accuracy_breakdown | pass | pass | PASS |
| t02 | tool | accuracy_breakdown | pass | pass | PASS |
| t03 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t04 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t05 | tool | accuracy_breakdown | pass | pass | PASS |
| t06 | tool | list_collections | pass | pass | PASS |
| t07 | tool | list_collections, list_collections, list_collections, list_collections, list_collections, list_collections, list_collections | pass | pass | PASS |
| t08 | tool | find_traces | pass | pass | PASS |
| t09 | tool | list_collections, accuracy_breakdown, find_traces, get_trace, get_trace, get_trace | pass | pass | PASS |
| t10 | tool | find_traces, list_collections, find_traces, find_traces, find_traces, find_traces, find_traces, find_traces | fail | fail | ERROR |
| t11 | tool | list_collections, find_traces, get_trace | pass | pass | PASS |
| t12 | tool | list_collections, find_traces, get_trace, get_trace | pass | pass | PASS |
| t13 | tool | list_collections, find_traces, get_trace, get_trace, get_trace | pass | pass | PASS |
| t14 | tool | check_answer | pass | pass | PASS |
| t15 | tool | check_answer | pass | pass | PASS |
| t16 | tool | check_answer | pass | pass | PASS |
| t17 | tool | list_collections, accuracy_breakdown, find_traces, get_trace, check_answer, get_trace, get_trace, check_answer | fail | fail | ERROR |
| t18 | refuse | list_collections, accuracy_breakdown | pass | pass | PASS |
| t19 | refuse | list_collections | pass | pass | PASS |
| t20 | refuse | - | pass | pass | PASS |
