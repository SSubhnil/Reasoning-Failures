import re

class ThoughtExtractor:
    def extract_steps(self, response: str) -> list[str]:
        raise NotImplementedError

def _merge_steps_by_min_tokens(trace, k, tokenizer, delimiter):
    steps_raw = trace.split(delimiter)

    merged_steps = []
    buffer = ""
    buffer_tokens = 0

    for step in steps_raw:
        step_tokens = len(tokenizer(step)['input_ids'])

        if buffer_tokens + step_tokens < k:

            buffer += (' ' if buffer else '') + step
            buffer_tokens += step_tokens
        else:

            if buffer:
                merged_steps.append(buffer+" "+step)
                buffer = ""
                buffer_tokens = 0

            else:
                merged_steps.append(step)
                buffer = ""
                buffer_tokens = 0

    if buffer:
        merged_steps.append(buffer)

    return merged_steps

def merged_step_char_spans(trace, k, tokenizer, delimiter="\n\n"):
    chunks = trace.split(delimiter)
    starts, off = [], 0
    for c in chunks:
        starts.append(off)
        off += len(c) + len(delimiter)
    ends = [starts[i] + len(chunks[i]) for i in range(len(chunks))]

    spans = []
    buffer = ""
    buffer_tokens = 0
    buffered: list[int] = []
    for i, step in enumerate(chunks):
        step_tokens = len(tokenizer(step)['input_ids'])
        if buffer_tokens + step_tokens < k:
            buffer += (' ' if buffer else '') + step
            buffer_tokens += step_tokens
            buffered.append(i)
        else:
            if buffer:

                first = next(j for j in buffered if chunks[j])
                spans.append((starts[first], ends[i]))
            else:
                spans.append((starts[i], ends[i]))
            buffer = ""
            buffer_tokens = 0
            buffered = []

    if buffer:
        first = next(j for j in buffered if chunks[j])
        last = max(j for j in buffered if chunks[j])
        spans.append((starts[first], ends[last]))

    return spans

class SkyworkExtractor(ThoughtExtractor):
    def __init__(self,
                 option: str = 'original',
                 max_steps: int = 10,
                 minimal_number_tokens_k: int = 1):
        self.option = option
        self.max_steps = max_steps
        self.minimal_number_tokens_k = minimal_number_tokens_k

    def extract_steps(self, response: str, tokenizer=None):
        option = self.option

        if option == "original":
            preliminary_steps = [s.strip() for s in response.split("\n") if s.strip()]
            step_mappings = [[i] for i in range(len(preliminary_steps))]

            close_tag_re = re.compile(r"<\s*/\s*(think|thinking)\s*>", re.I)

            end_idx = None
            for i, step in enumerate(preliminary_steps):
                if close_tag_re.search(step):
                    end_idx = i

            if end_idx is not None:
                main_steps = preliminary_steps[: end_idx + 1]
                main_map   = step_mappings[: end_idx + 1]

                post_steps = preliminary_steps[end_idx + 1 :]
                post_map   = step_mappings[end_idx + 1 :]

                main_cap = max(self.max_steps - 2, 1)
                post_cap = 2

                if len(post_steps) == 0:
                    main_cap = self.max_steps

                if len(main_steps) > main_cap:
                    main_steps, main_map = self._enforce_max_steps_with_mapping(main_steps, main_map, main_cap)

                if len(post_steps) > post_cap:
                    post_steps, post_map = self._enforce_max_steps_with_mapping(post_steps, post_map, post_cap)

                steps = main_steps + post_steps
                mappings = main_map + post_map
            else:
                steps, mappings = preliminary_steps, step_mappings
                if len(steps) > self.max_steps:
                    steps, mappings = self._enforce_max_steps_with_mapping(steps, mappings, self.max_steps)

            return steps

        elif option == "back_line_k":
            if tokenizer is None:
                raise ValueError("back_line_k needs a tokenizer: pass tokenizer= to extract_steps.")
            return _merge_steps_by_min_tokens(response, self.minimal_number_tokens_k, tokenizer, delimiter="\n\n")

    def _enforce_max_steps_with_mapping(self, steps: list[str], mappings: list[list[int]], max_steps: int):
        max_iterations = len(steps)
        iterations = 0

        while len(steps) > max_steps and iterations < max_iterations:
            shortest_idx = min(range(len(steps) - 1), key=lambda i: len(steps[i]))

            steps[shortest_idx] += ' ' + steps[shortest_idx + 1]

            mappings[shortest_idx].extend(mappings[shortest_idx + 1])

            del steps[shortest_idx + 1]
            del mappings[shortest_idx + 1]

            iterations += 1

        return steps, mappings
