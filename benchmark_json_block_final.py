import timeit
import re
from typing import Optional

def original_find_json_block(text: str) -> Optional[str]:
    start = text.find('{')
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    return None

def optimized_find_json_block(text: str) -> Optional[str]:
    start = text.find('{')
    if start == -1:
        return None

    brace_count = 0
    next_open = start
    next_close = text.find('}', start)

    while next_close != -1:
        if next_open != -1 and next_open < next_close:
            brace_count += 1
            next_open = text.find('{', next_open + 1)
        else:
            brace_count -= 1
            if brace_count == 0:
                return text[start:next_close+1]
            next_close = text.find('}', next_close + 1)

    return None

test_cases = {
    "large_prefix_suffix": "Some random text " * 1000 + "{\"key\": \"value\", \"nested\": {\"key2\": \"value2\"}}" + " more text " * 1000,
    "pure_json": "{\"key\": \"value\", \"nested\": {\"key2\": \"value2\"}}",
    "no_braces": "Just some text without any braces." * 1000,
    "large_spaces": "{" + " " * 10000 + "}",
    "many_braces": 'Here is a JSON: {"data": [' + '{"id": 1}, ' * 500 + '{"id": 501}]}'
}

print("Running benchmarks...")
print("="*40)

for name, text in test_cases.items():
    assert original_find_json_block(text) == optimized_find_json_block(text), f"Mismatch for {name}"

    t_orig = timeit.timeit(lambda: original_find_json_block(text), number=1000)
    t_opt = timeit.timeit(lambda: optimized_find_json_block(text), number=1000)

    print(f"Test Case: {name}")
    print(f"  Baseline (Old): {t_orig:.5f}s")
    print(f"  Optimized:      {t_opt:.5f}s")
    if t_opt > 0:
        print(f"  Improvement:    {t_orig/t_opt:.2f}x faster")
    print("-" * 40)
