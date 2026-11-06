#!/usr/bin/env python3
"""Coding evaluation: compare "before fine-tuning" and "after" by unit-test pass rate.

Same principle as the architecture: the model does not grade itself — tests decide correctness.
Each task gives the model a function-signature description, it writes the implementation,
then we exec + assert.

The 31 tasks cover: strings, arrays/lists, math, sorting/searching, dicts, structure
transforms, classic algorithms.
The sample grew from 8 tasks to 31 because with 8 tasks a single task is 12.5% swing —
noise swamps signal and no conclusion can stand on it.

Usage:
  # baseline (no adapter)
  python3 eval_coding.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX
  # fine-tuned
  python3 eval_coding.py --model /Volumes/LLM/hf/openbmb__MiniCPM5-2B-MLX \
      --adapter /Volumes/LLM/adapters/executor-v1
"""

import argparse
import json
import re
import sys

TASKS = [
    # ---------- arrays / lists ----------
    {
        "name": "two_sum",
        "prompt": "Write a Python function two_sum(nums, target) that returns the indices of the two numbers that add up to target, as a list of two indices. Output only the Python function, no explanation.",
        "tests": "assert two_sum([2,7,11,15], 9) == [0, 1]\nassert two_sum([3,2,4], 6) == [1, 2]\nassert two_sum([3,3], 6) == [0, 1]\n",
    },
    {
        "name": "dedup_keep_order",
        "prompt": "Write a Python function dedup(items) that removes duplicates from a list while preserving the original order. Output only the Python function, no explanation.",
        "tests": "assert dedup([3,1,3,2,1]) == [3,1,2]\nassert dedup([]) == []\nassert dedup(['a','b','a']) == ['a','b']\n",
    },
    {
        "name": "flatten_nested",
        "prompt": "Write a Python function flatten(nested) that flattens an arbitrarily nested list into a single flat list. Output only the Python function, no explanation.",
        "tests": "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]\nassert flatten([]) == []\nassert flatten([[1],[2]]) == [1,2]\n",
    },
    {
        "name": "max_subarray_sum",
        "prompt": "Write a Python function max_subarray_sum(nums) that returns the largest sum of any contiguous subarray. Output only the Python function, no explanation.",
        "tests": "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6\nassert max_subarray_sum([-1,-2]) == -1\nassert max_subarray_sum([5]) == 5\n",
    },
    {
        "name": "rotate_list",
        "prompt": "Write a Python function rotate(nums, k) that returns a new list with the elements rotated to the right by k positions. Output only the Python function, no explanation.",
        "tests": "assert rotate([1,2,3,4,5], 2) == [4,5,1,2,3]\nassert rotate([1], 0) == [1]\nassert rotate([1,2], 5) == [2,1]\n",
    },
    {
        "name": "chunk_list",
        "prompt": "Write a Python function chunk(lst, size) that splits a list into consecutive chunks of the given size; the last chunk may be shorter. Output only the Python function, no explanation.",
        "tests": "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunk([1,2,3], 1) == [[1],[2],[3]]\nassert chunk([], 3) == []\n",
    },
    {
        "name": "list_intersection",
        "prompt": "Write a Python function intersection(a, b) that returns a list of the unique elements present in both lists, in any order. Output only the Python function, no explanation.",
        "tests": "assert sorted(intersection([1,2,2,3],[2,3,4])) == [2,3]\nassert sorted(intersection([1,2],[3,4])) == []\nassert sorted(intersection([1,1],[1])) == [1]\n",
    },
    {
        "name": "max_pair_product",
        "prompt": "Write a Python function max_product(nums) that returns the largest product of any two distinct elements in the list. Output only the Python function, no explanation.",
        "tests": "assert max_product([1,2,3,4]) == 12\nassert max_product([-10,-10,1,2]) == 100\nassert max_product([-5,2]) == -10\n",
    },
    {
        "name": "find_missing_number",
        "prompt": "Write a Python function find_missing(nums) that takes a list containing n distinct numbers from 0..n with one missing, and returns the missing number. Output only the Python function, no explanation.",
        "tests": "assert find_missing([0,1,3]) == 2\nassert find_missing([1,2,3]) == 0\nassert find_missing([0,1,2]) == 3\n",
    },
    # ---------- strings ----------
    {
        "name": "is_palindrome",
        "prompt": "Write a Python function is_palindrome(s) that returns True if the string s is a palindrome ignoring case and non-alphanumeric characters, else False. Output only the Python function, no explanation.",
        "tests": "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('race a car') is False\nassert is_palindrome('') is True\n",
    },
    {
        "name": "word_count",
        "prompt": "Write a Python function word_count(text) that returns a dict mapping each lowercase word to its number of occurrences. Output only the Python function, no explanation.",
        "tests": "assert word_count('a b a') == {'a':2,'b':1}\nassert word_count('') == {}\nassert word_count('Hi hi') == {'hi':2}\n",
    },
    {
        "name": "reverse_string",
        "prompt": "Write a Python function reverse_string(s) that returns the reversed string. Output only the Python function, no explanation.",
        "tests": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'\n",
    },
    {
        "name": "count_vowels",
        "prompt": "Write a Python function count_vowels(s) that returns the number of vowels (a,e,i,o,u) in the string, case-insensitive. Output only the Python function, no explanation.",
        "tests": "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0\n",
    },
    {
        "name": "longest_common_prefix",
        "prompt": "Write a Python function common_prefix(strs) that returns the longest common prefix of a list of strings, or an empty string if there is none. Output only the Python function, no explanation.",
        "tests": "assert common_prefix(['flower','flow','flight']) == 'fl'\nassert common_prefix(['dog','cat']) == ''\nassert common_prefix(['a']) == 'a'\n",
    },
    {
        "name": "is_anagram",
        "prompt": "Write a Python function is_anagram(s, t) that returns True if the two strings are anagrams of each other, else False. Output only the Python function, no explanation.",
        "tests": "assert is_anagram('listen','silent') is True\nassert is_anagram('hello','world') is False\nassert is_anagram('','') is True\n",
    },
    {
        "name": "caesar_cipher",
        "prompt": "Write a Python function caesar(s, shift) that returns the string with each lowercase letter shifted by shift positions in the alphabet, wrapping around from z to a. Non-lowercase characters are unchanged. Output only the Python function, no explanation.",
        "tests": "assert caesar('abc', 1) == 'bcd'\nassert caesar('xyz', 3) == 'abc'\nassert caesar('a b', 0) == 'a b'\n",
    },
    {
        "name": "valid_parentheses",
        "prompt": "Write a Python function valid_parens(s) that returns True if the brackets in the string are properly matched and nested. Only round, square and curly brackets appear. Output only the Python function, no explanation.",
        "tests": "assert valid_parens('()[]{}') is True\nassert valid_parens('([)]') is False\nassert valid_parens('') is True\nassert valid_parens('(') is False\n",
    },
    # ---------- math ----------
    {
        "name": "fibonacci",
        "prompt": "Write a Python function fib(n) that returns the n-th Fibonacci number with fib(0)=0 and fib(1)=1. Output only the Python function, no explanation.",
        "tests": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55\nassert fib(20) == 6765\n",
    },
    {
        "name": "factorial",
        "prompt": "Write a Python function factorial(n) that returns n factorial, with factorial(0)=1. Output only the Python function, no explanation.",
        "tests": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(1) == 1\n",
    },
    {
        "name": "is_prime",
        "prompt": "Write a Python function is_prime(n) that returns True if n is a prime number, else False. Output only the Python function, no explanation.",
        "tests": "assert is_prime(2) is True\nassert is_prime(1) is False\nassert is_prime(97) is True\nassert is_prime(100) is False\n",
    },
    {
        "name": "gcd",
        "prompt": "Write a Python function gcd(a, b) that returns the greatest common divisor of a and b. Output only the Python function, no explanation.",
        "tests": "assert gcd(12, 18) == 6\nassert gcd(7, 3) == 1\nassert gcd(0, 5) == 5\n",
    },
    {
        "name": "sum_of_digits",
        "prompt": "Write a Python function sum_digits(n) that returns the sum of the digits of the non-negative integer n. Output only the Python function, no explanation.",
        "tests": "assert sum_digits(123) == 6\nassert sum_digits(0) == 0\nassert sum_digits(999) == 27\n",
    },
    {
        "name": "is_power_of_two",
        "prompt": "Write a Python function is_power_of_two(n) that returns True if the positive integer n is a power of two, else False. Output only the Python function, no explanation.",
        "tests": "assert is_power_of_two(8) is True\nassert is_power_of_two(6) is False\nassert is_power_of_two(1) is True\n",
    },
    {
        "name": "climbing_stairs",
        "prompt": "Write a Python function climb(n) that returns the number of distinct ways to climb n stairs taking 1 or 2 steps at a time. Output only the Python function, no explanation.",
        "tests": "assert climb(1) == 1\nassert climb(2) == 2\nassert climb(5) == 8\n",
    },
    {
        "name": "roman_to_int",
        "prompt": "Write a Python function roman_to_int(s) that converts a Roman numeral string to an integer. Output only the Python function, no explanation.",
        "tests": "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('MCMXCIV') == 1994\n",
    },
    {
        "name": "fizzbuzz",
        "prompt": "Write a Python function fizzbuzz(n) that returns a list of strings for the numbers 1 through n, where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', and all other numbers are their decimal string. Output only the Python function, no explanation.",
        "tests": "assert fizzbuzz(3) == ['1','2','Fizz']\nassert fizzbuzz(5)[4] == 'Buzz'\nassert fizzbuzz(15)[14] == 'FizzBuzz'\nassert fizzbuzz(15)[2] == 'Fizz'\n",
    },
    # ---------- sorting / searching ----------
    {
        "name": "binary_search",
        "prompt": "Write a Python function binary_search(arr, target) that returns the index of target in the sorted list arr, or -1 if not found. Output only the Python function, no explanation.",
        "tests": "assert binary_search([1,3,5,7,9], 7) == 3\nassert binary_search([1,3,5], 2) == -1\nassert binary_search([], 1) == -1\n",
    },
    {
        "name": "merge_sorted",
        "prompt": "Write a Python function merge_sorted(a, b) that merges two already-sorted lists into one sorted list. Output only the Python function, no explanation.",
        "tests": "assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]\nassert merge_sorted([], [1]) == [1]\nassert merge_sorted([2], [1]) == [1,2]\n",
    },
    {
        "name": "sort_list",
        "prompt": "Write a Python function sort_list(nums) that returns a new list containing the elements of nums sorted in ascending order. Do not call the built-in sorted() or list.sort(). Output only the Python function, no explanation.",
        "tests": "assert sort_list([3,1,2]) == [1,2,3]\nassert sort_list([]) == []\nassert sort_list([5,5,1]) == [1,5,5]\n",
    },
    # ---------- structure transforms ----------
    {
        "name": "matrix_transpose",
        "prompt": "Write a Python function transpose(m) that returns the transpose of the 2D list m. Output only the Python function, no explanation.",
        "tests": "assert transpose([[1,2],[3,4]]) == [[1,3],[2,4]]\nassert transpose([[1,2,3]]) == [[1],[2],[3]]\nassert transpose([[]]) == []\n",
    },
    {
        "name": "dict_invert",
        "prompt": "Write a Python function invert(d) that returns a new dict with keys and values swapped. Values are unique. Output only the Python function, no explanation.",
        "tests": "assert invert({'a':1,'b':2}) == {1:'a',2:'b'}\nassert invert({}) == {}\nassert invert({'x':0}) == {0:'x'}\n",
    },
]

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract code from model output: prefer a fenced block, else everything from the first def."""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()


def run_task(model, tokenizer, task: dict, max_tokens: int = 512,
             use_chat_template: bool = False) -> dict:
    from mlx_lm import generate

    # Passing a raw string to generate() does not apply the chat template. SFT/instruct
    # models must be templated by hand or they degenerate into raw continuation
    # (they parrot the prompt); base models are more natural with raw continuation.
    prompt = task["prompt"]
    if use_chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=False, add_generation_prompt=True)

    try:
        raw = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
    except Exception as exc:  # noqa: BLE001
        return {"name": task["name"], "passed": False, "error": f"generation failed: {exc}"}

    code = extract_code(raw)
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {"name": task["name"], "passed": False,
                "error": f"code cannot execute: {type(exc).__name__}: {exc}"}

    try:
        exec(task["tests"], namespace)  # noqa: S102
    except AssertionError:
        return {"name": task["name"], "passed": False, "error": "tests failed"}
    except Exception as exc:  # noqa: BLE001
        return {"name": task["name"], "passed": False,
                "error": f"tests raised: {type(exc).__name__}: {exc}"}

    return {"name": task["name"], "passed": True, "error": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit to evaluate the base model")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="path for the result JSON")
    ap.add_argument("--chat-template", action="store_true",
                    help="wrap the prompt in the chat template (required for SFT/instruct models, not for base)")
    args = ap.parse_args()

    from mlx_lm import load

    print(f"model: {args.model}")
    print(f"adapter: {args.adapter or '(none, base model)'}")
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    results = []
    for task in TASKS:
        r = run_task(model, tokenizer, task, args.max_tokens, args.chat_template)
        results.append(r)
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{flag}] {r['name']}" + (f"  {r['error']}" if r["error"] else ""),
              flush=True)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\npass rate: {passed}/{total} = {passed / total:.1%}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "adapter": args.adapter,
                       "passed": passed, "total": total, "results": results},
                      fh, ensure_ascii=False, indent=2)
        print(f"results written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
