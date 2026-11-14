# Tool-calling eval

- Model: nvidia/nemotron-3-ultra-550b-a55b:free
- Date (UTC): 2026-11-14T23:11:46+00:00
- Tasks run: 20
- Tool-selection accuracy (tool tasks): 12/17
- Argument accuracy (tool tasks): 12/17
- Refuse tasks passed: 3/3
- Tasks passed overall: 15/20

| id | kind | tools called | tool selection | args | result |
|---|---|---|---|---|---|
| t01 | tool | accuracy_breakdown | pass | pass | PASS |
| t02 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t03 | tool | accuracy_breakdown | pass | pass | PASS |
| t04 | tool | list_collections, accuracy_breakdown | pass | pass | PASS |
| t05 | tool | accuracy_breakdown | pass | pass | PASS |
| t06 | tool | list_collections | pass | pass | PASS |
| t07 | tool | list_collections, list_collections, list_collections, list_collections | pass | pass | PASS |
| t08 | tool | find_traces, find_traces, find_traces, find_traces, find_traces, find_traces, find_traces | pass | pass | PASS |
| t09 | tool | list_collections, find_traces, find_traces, accuracy_breakdown, find_traces, find_traces, find_traces, find_traces | fail | fail | ERROR |
| t10 | tool | find_traces, find_traces, find_traces | pass | pass | PASS |
| t11 | tool | list_collections, find_traces, find_traces, find_traces, find_traces, accuracy_breakdown, find_traces, find_traces | fail | fail | ERROR |
| t12 | tool | list_collections, accuracy_breakdown, find_traces, find_traces, find_traces, find_traces, find_traces | fail | fail | FAIL |
| t13 | tool | list_collections, accuracy_breakdown, find_traces, find_traces, find_traces | fail | fail | FAIL |
| t14 | tool | check_answer | pass | pass | PASS |
| t15 | tool | check_answer | pass | pass | PASS |
| t16 | tool | check_answer | pass | pass | PASS |
| t17 | tool | list_collections, accuracy_breakdown, find_traces, find_traces, find_traces, find_traces, find_traces, find_traces | fail | fail | ERROR |
| t18 | refuse | list_collections, accuracy_breakdown, find_traces | pass | pass | PASS |
| t19 | refuse | list_collections, accuracy_breakdown | pass | pass | PASS |
| t20 | refuse | - | pass | pass | PASS |
