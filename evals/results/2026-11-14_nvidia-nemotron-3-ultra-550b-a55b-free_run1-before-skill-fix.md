# Tool-calling eval

- Model: nvidia/nemotron-3-ultra-550b-a55b:free
- Date (UTC): 2026-11-14T22:41:08+00:00
- Tasks run: 20
- Tool-selection accuracy (tool tasks): 17/17
- Argument accuracy (tool tasks): 17/17
- Refuse tasks passed: 2/3
- Tasks passed overall: 19/20

| id | kind | tools called | tool selection | args | result |
|---|---|---|---|---|---|
| t01 | tool | accuracy_breakdown | pass | pass | PASS |
| t02 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t03 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t04 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t05 | tool | accuracy_breakdown | pass | pass | PASS |
| t06 | tool | list_collections | pass | pass | PASS |
| t07 | tool | list_collections | pass | pass | PASS |
| t08 | tool | list_collections, find_traces | pass | pass | PASS |
| t09 | tool | list_collections, find_traces, get_trace, get_trace, get_trace | pass | pass | PASS |
| t10 | tool | find_traces, find_traces, find_traces | pass | pass | PASS |
| t11 | tool | list_collections, find_traces, get_trace | pass | pass | PASS |
| t12 | tool | list_collections, accuracy_breakdown, find_traces, get_trace, get_trace, get_trace | pass | pass | PASS |
| t13 | tool | list_collections, find_traces, get_trace | pass | pass | PASS |
| t14 | tool | check_answer | pass | pass | PASS |
| t15 | tool | check_answer | pass | pass | PASS |
| t16 | tool | check_answer | pass | pass | PASS |
| t17 | tool | list_collections, accuracy_breakdown, find_traces, get_trace | pass | pass | PASS |
| t18 | refuse | list_collections, accuracy_breakdown, find_traces, accuracy_breakdown, get_trace | pass | pass | PASS |
| t19 | refuse | list_collections | pass | pass | PASS |
| t20 | refuse | list_collections, find_traces, find_traces, get_trace, check_answer, get_trace, check_answer, get_trace | fail | fail | ERROR |
