#!/usr/bin/env python3
"""Build reviewer fine-tuning data.

Idea: start from clean functions and inject known bugs via controlled string
replacement. Because the bugs are ours, labels are 100% correct — no annotation.

Produces mlx-lm format:
  user      -> asks the model to review the code
  assistant -> issue list (one "type: description" per line), or "no issue"
"""

import argparse
import json
import os
import random

# ---------------------------------------------------------------- seed functions
# Clean, short, common Python functions — the hosts for injected bugs
SEEDS = {
    "sum_list": '''def sum_list(nums):
    total = 0
    for x in nums:
        total += x
    return total''',

    "find_max": '''def find_max(nums):
    if not nums:
        return None
    m = nums[0]
    for x in nums:
        if x > m:
            m = x
    return m''',

    "count_even": '''def count_even(nums):
    c = 0
    for x in nums:
        if x % 2 == 0:
            c += 1
    return c''',

    "average": '''def average(nums):
    if not nums:
        return 0
    return sum(nums) / len(nums)''',

    "reverse_string": '''def reverse_string(s):
    result = ""
    for ch in s:
        result = ch + result
    return result''',

    "is_palindrome": '''def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True''',

    "count_vowels": '''def count_vowels(s):
    c = 0
    for ch in s.lower():
        if ch in "aeiou":
            c += 1
    return c''',

    "factorial": '''def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result''',

    "fibonacci": '''def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b''',

    "is_prime": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True''',

    "gcd": '''def gcd(a, b):
    while b:
        a, b = b, a % b
    return a''',

    "binary_search": '''def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1''',

    "remove_duplicates": '''def remove_duplicates(nums):
    seen = []
    for x in nums:
        if x not in seen:
            seen.append(x)
    return seen''',

    "word_count": '''def word_count(text):
    counts = {}
    for w in text.split():
        counts[w] = counts.get(w, 0) + 1
    return counts''',

    "clamp": '''def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x''',

    "sum_digits": '''def sum_digits(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total''',

    "linear_search": '''def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1''',

    "bubble_sort": '''def bubble_sort(nums):
    n = len(nums)
    for i in range(n):
        for j in range(n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums''',

    "max_subarray_sum": '''def max_subarray_sum(nums):
    best = nums[0]
    cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best''',

    "valid_parens": '''def valid_parens(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return len(stack) == 0''',

    "merge_sorted": '''def merge_sorted(a, b):
    i, j = 0, 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out''',

    "chunk": '''def chunk(lst, size):
    out = []
    for i in range(0, len(lst), size):
        out.append(lst[i:i + size])
    return out''',

    "rotate": '''def rotate(nums, k):
    k = k % len(nums)
    return nums[-k:] + nums[:-k]''',

    "is_anagram": '''def is_anagram(s, t):
    return sorted(s) == sorted(t)''',

    "common_prefix": '''def common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
    return prefix''',

    "climbing_stairs": '''def climbing_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    return b''',

    "flatten": '''def flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out''',

    "dict_invert": '''def dict_invert(d):
    out = {}
    for k, v in d.items():
        out[v] = k
    return out''',

    "is_power_of_two": '''def is_power_of_two(n):
    if n < 1:
        return False
    while n % 2 == 0:
        n = n // 2
    return n == 1''',

    "missing_number": '''def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)''',

    "list_intersection": '''def list_intersection(a, b):
    out = []
    for x in a:
        if x in b and x not in out:
            out.append(x)
    return out''',

    "fizzbuzz": '''def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out''',

    "longest_word": '''def longest_word(s):
    best = ""
    for w in s.split():
        if len(w) > len(best):
            best = w
    return best''',

    "matrix_transpose": '''def matrix_transpose(m):
    if not m:
        return []
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]''',

    "count_words": '''def count_words(text):
    return len(text.split())''',

    "remove_negatives": '''def remove_negatives(nums):
    out = []
    for x in nums:
        if x >= 0:
            out.append(x)
    return out''',

    "sum_matrix": '''def sum_matrix(m):
    total = 0
    for row in m:
        for x in row:
            total += x
    return total''',

    "first_duplicate": '''def first_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return x
        seen.add(x)
    return None''',

    "merge_dicts": '''def merge_dicts(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = v
    return out''',

    "insert_sorted": '''def insert_sorted(nums, x):
    for i in range(len(nums)):
        if nums[i] > x:
            nums.insert(i, x)
            return nums
    nums.append(x)
    return nums''',

    "count_char": '''def count_char(s, ch):
    c = 0
    for x in s:
        if x == ch:
            c += 1
    return c''',

    "power": '''def power(base, exp):
    result = 1
    for i in range(exp):
        result *= base
    return result''',

    "count_occurrences": '''def count_occurrences(lst, target):
    c = 0
    for x in lst:
        if x == target:
            c += 1
    return c''',

    "product_list": '''def product_list(nums):
    p = 1
    for x in nums:
        p *= x
    return p''',

    "find_min": '''def find_min(nums):
    if not nums:
        return None
    m = nums[0]
    for x in nums:
        if x < m:
            m = x
    return m''',

    "join_strings": '''def join_strings(parts, sep):
    result = ""
    for i in range(len(parts)):
        result += parts[i] + sep
    return result''',

    "to_upper_list": '''def to_upper_list(words):
    out = []
    for w in words:
        out.append(w.upper())
    return out''',

    "filter_positive": '''def filter_positive(nums):
    out = []
    for x in nums:
        if x > 0:
            out.append(x)
    return out''',

    "sum_of_squares": '''def sum_of_squares(n):
    total = 0
    for i in range(1, n + 1):
        total += i * i
    return total''',

    "remove_spaces": '''def remove_spaces(s):
    out = ""
    for ch in s:
        if ch != " ":
            out += ch
    return out''',

    "count_upper": '''def count_upper(s):
    c = 0
    for ch in s:
        if ch.isupper():
            c += 1
    return c''',

    "list_difference": '''def list_difference(a, b):
    out = []
    for x in a:
        if x not in b:
            out.append(x)
    return out''',

    "median": '''def median(nums):
    s = sorted(nums)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2''',

    "is_sorted": '''def is_sorted(nums):
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            return False
    return True''',

    "reverse_words": '''def reverse_words(s):
    words = s.split()
    out = []
    for i in range(len(words)):
        out.append(words[len(words) - i - 1])
    return out''',

    "count_substring": '''def count_substring(s, sub):
    c = 0
    for i in range(len(s)):
        if s[i:i + len(sub)] == sub:
            c += 1
    return c''',

    "sum_positive": '''def sum_positive(nums):
    total = 0
    for x in nums:
        if x > 0:
            total += x
    return total''',

    "delete_key": '''def delete_key(d, key):
    if key in d:
        del d[key]
    return d''',

    "merge_lists": '''def merge_lists(a, b):
    out = list(a)
    for x in b:
        out.append(x)
    return out''',

    "second_largest": '''def second_largest(nums):
    s = sorted(set(nums))
    if len(s) < 2:
        return None
    return s[len(s) - 2]''',

    "count_letters": '''def count_letters(s):
    c = 0
    for ch in s:
        if ch.isalpha():
            c += 1
    return c''',

    "shift_list": '''def shift_list(nums):
    if len(nums) < 2:
        return nums
    first = nums[0]
    for i in range(1, len(nums)):
        nums[i - 1] = nums[i]
    nums[len(nums) - 1] = first
    return nums''',

    "is_empty_str": '''def is_empty_str(s):
    if s is None:
        return True
    return len(s) == 0''',

    "max_abs": '''def max_abs(nums):
    best = 0
    for x in nums:
        if abs(x) > best:
            best = abs(x)
    return best''',

    "count_below": '''def count_below(nums, threshold):
    c = 0
    for x in nums:
        if x < threshold:
            c += 1
    return c''',

    "string_to_list": '''def string_to_list(s):
    out = []
    for ch in s:
        out.append(ch)
    return out''',

    "add_matrices": '''def add_matrices(a, b):
    out = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[0])):
            row.append(a[i][j] + b[i][j])
        out.append(row)
    return out''',

    "capitalize_words": '''def capitalize_words(s):
    words = s.split()
    out = []
    for w in words:
        out.append(w.capitalize())
    return out''',

    "index_of_max": '''def index_of_max(nums):
    best = 0
    for i in range(len(nums)):
        if nums[i] > nums[best]:
            best = i
    return best''',

    "repeat_string": '''def repeat_string(s, n):
    out = ""
    for i in range(n):
        out += s
    return out''',

    "count_in_range": '''def count_in_range(nums, low, high):
    c = 0
    for x in nums:
        if low <= x and x <= high:
            c += 1
    return c''',

    # ---- seeds with null guards ----
    # The original 71 seeds had only 7 guards in total, leaving null-type bugs at 6%.
    # Null bugs can only be made by deleting an existing guard, so guarded seeds had
    # to be added.
    "first_element": '''def first_element(nums):
    if not nums:
        return None
    return nums[0]''',

    "last_item": '''def last_item(nums):
    if len(nums) == 0:
        return None
    return nums[-1]''',

    "list_average": '''def list_average(nums):
    if not nums:
        return 0
    return sum(nums) / len(nums)''',

    "normalize_max": '''def normalize_max(nums):
    if not nums:
        return []
    m = max(nums)
    out = []
    for x in nums:
        out.append(x / m)
    return out''',

    "dict_lookup": '''def dict_lookup(d, key, default):
    if d is None:
        return default
    return d.get(key, default)''',

    "concat_all": '''def concat_all(parts):
    if not parts:
        return ""
    out = ""
    for p in parts:
        out += p
    return out''',

    "list_max": '''def list_max(nums):
    if nums is None:
        return None
    best = nums[0]
    for x in nums:
        if x > best:
            best = x
    return best''',

    "item_count": '''def item_count(items):
    if items is None:
        return 0
    return len(items)''',

    "first_token": '''def first_token(s):
    if not s:
        return ""
    return s.split()[0]''',

    "join_csv": '''def join_csv(parts):
    if not parts:
        return ""
    return ",".join(parts)''',

    "stack_pop": '''def stack_pop(stack):
    if not stack:
        return None
    return stack.pop()''',

    "string_trunc": '''def string_trunc(s, n):
    if s is None:
        return ""
    return s[:n]''',

    "matrix_first_row": '''def matrix_first_row(m):
    if not m:
        return []
    return m[0]''',

    "safe_index": '''def safe_index(nums, idx):
    if nums is None or idx is None:
        return None
    if idx < 0 or idx >= len(nums):
        return None
    return nums[idx]''',

    "sum_matrix": '''def sum_matrix(m):
    if not m:
        return 0
    total = 0
    for row in m:
        for x in row:
            total += x
    return total''',
}

# ---------------------------------------------------------------- mutation operators
# (find string, replace string, issue type, issue description)
# Exact string matching, so nothing outside the intended site is touched.
# NOTE: types and descriptions stay in Chinese — they are training-data content for a
# Chinese-heavy SFT base; translating them means regenerating data and retraining.
MUTATIONS = [
    # ---- boundary / off-by-one ----
    ("range(len(arr))", "range(len(arr) - 1)", "边界条件", "循环少执行一次，最后一个元素被漏掉"),
    ("for x in nums:", "for x in nums[:-1]:", "边界条件", "循环排除了最后一个元素"),
    ("range(1, n + 1)", "range(1, n)", "边界条件", "循环少执行一次，累乘会漏掉 n"),
    ("range(2, n + 1)", "range(2, n)", "边界条件", "循环少执行一次，结果偏小"),
    ("range(3, n + 1)", "range(3, n)", "边界条件", "循环少执行一次，结果偏小"),
    ("int(n ** 0.5) + 1", "int(n ** 0.5)", "边界条件", "循环上界少 1，完全平方数会被误判为质数"),
    ("for i in range(n):", "for i in range(n - 1):", "边界条件", "外层循环少执行一次，排序可能未完成"),
    ("range(n - i - 1)", "range(n - i)", "边界条件", "内层循环上界多 1，会访问到已排好的位置"),
    ("range(0, len(lst), size)", "range(size, len(lst), size)", "边界条件", "切分从下标 size 开始，前面的元素被丢掉"),
    ("for r in range(rows)", "for r in range(rows - 1)", "边界条件", "遍历少一行，最后一行数据丢失"),
    ("for i in range(exp):", "for i in range(exp - 1):", "边界条件", "循环少执行一次，结果少乘一次底数"),
    ("range(a, b + 1)", "range(a, b)", "边界条件", "区间右端点少 1，右边界值被漏掉"),

    # ---- guard deletion / null handling ----
    ("if not nums:\n        return None\n", "", "空值处理", "缺少空输入守卫，nums 为空时访问 nums[0] 会抛 IndexError"),
    ("if not nums:\n        return 0\n", "", "空值处理", "缺少空输入守卫，空列表会导致除零错误"),
    ("if not strs:\n        return \"\"\n", "", "空值处理", "缺少空输入守卫，空列表会抛 IndexError"),
    ("if n <= 1:\n        return n\n", "", "边界条件", "缺少 n<=1 的基准情况，结果会出错"),
    ("if n <= 2:\n        return n\n", "", "边界条件", "缺少 n<=2 的基准情况，结果会出错"),
    ("if n < 2:\n        return False\n", "", "边界条件", "缺少 n<2 的守卫，0 和 1 会被误判"),
    ("if n < 1:\n        return False\n", "", "边界条件", "缺少 n<1 的守卫，0 会进入循环"),
    ("if not m:\n        return []\n", "", "空值处理", "缺少空矩阵守卫，访问 m[0] 会抛 IndexError"),
    ("k = k % len(nums)", "k = k", "空值处理", "缺少取模，k 大于长度时切片结果异常"),
    ("if not stack or stack.pop() != pairs[ch]:", "if stack.pop() != pairs[ch]:",
     "空值处理", "缺少空栈判断，右括号多余时会抛异常"),

    # ---- logic / operators ----
    ("total += x", "total -= x", "逻辑错误", "累加写成了累减，结果符号相反"),
    ("c += 1", "c += 2", "逻辑错误", "计数步长写成 2，结果是实际的两倍"),
    ("if x > m:", "if x >= m:", "逻辑错误", "比较用错，相等时也更新，逻辑冗余"),
    ("if x % 2 == 0:", "if x % 2 == 1:", "逻辑错误", "判断条件反了，统计的是奇数"),
    ("if x not in seen:", "if x in seen:", "逻辑错误", "判断反了，只保留重复元素"),
    ("if x >= 0:", "if x > 0:", "边界条件", "把 0 也当成负数排除了"),
    ("if a[i] < b[j]:", "if a[i] > b[j]:", "逻辑错误", "比较方向反了，合并结果会逆序"),
    ("if nums[j] > nums[j + 1]:", "if nums[j] < nums[j + 1]:", "逻辑错误", "比较方向反了，排序结果变成降序"),
    ("if i % 15 == 0:", "if i % 15 == 1:", "逻辑错误", "FizzBuzz 的判断条件错误"),
    ("if len(w) > len(best):", "if len(w) >= len(best):", "逻辑错误", "相等时也更新，结果不稳定"),
    ("a, b = b, a + b", "a, b = b, a - b", "逻辑错误", "递推写成减法，结果完全错误"),
    ("a, b = b, a % b", "a, b = b, a // b", "逻辑错误", "取模写成了整除，欧几里得算法失效"),
    ("n = n // 2", "n = n / 2", "逻辑错误", "整除改成了真除法，会陷入死循环"),
    ("return sum(nums) / len(nums)", "return sum(nums) // len(nums)", "逻辑错误",
     "真除法改成整除，小数部分被截掉"),
    ("result = ch + result", "result = result + ch", "逻辑错误", "拼接顺序反了，等于没反转"),
    ("left += 1", "left += 2", "边界条件", "左指针步长错误，会跳过字符"),
    ("right -= 1", "right -= 2", "边界条件", "右指针步长错误，会跳过字符"),
    ("lo = mid + 1", "lo = mid", "逻辑错误", "左边界没有推进，可能死循环"),
    ("hi = mid - 1", "hi = mid", "逻辑错误", "右边界没有推进，可能死循环"),
    ("while lo <= hi:", "while lo < hi:", "边界条件", "循环条件少了等号，单元素区间会被漏掉"),
    ("while n > 0:", "while n > 1:", "边界条件", "循环提前结束，最后一位数字被漏掉"),
    ("n //= 10", "n //= 100", "边界条件", "每次去掉两位数字，会漏掉一半"),
    ("total += n % 10", "total += n % 100", "逻辑错误", "取的是两位数字而不是个位"),
    ("counts[w] = counts.get(w, 0) + 1", "counts[w] = 1", "逻辑错误", "计数被重置，永远统计为 1"),
    ("if x in b and x not in out:", "if x in b:", "逻辑错误", "缺少去重判断，结果会有重复元素"),
    ("return len(stack) == 0", "return True", "逻辑错误", "没有检查栈是否清空，右括号缺失时会误判通过"),
    ("return n * (n + 1) // 2 - sum(nums)", "return n * (n + 1) // 2 - sum(nums) - 1",
     "逻辑错误", "结果多减了 1"),
    ('if ch in "aeiou":', 'if ch in "aeiouy":', "逻辑错误", "把 y 也算成元音"),
    ("while b:", "while a:", "逻辑错误", "循环条件变量写错，求余过程错误"),
    ("out.extend(a[i:])", "out.extend(a[i + 1:])", "边界条件", "剩余元素少取一个"),
    ("out.append(lst[i:i + size])", "out.append(lst[i:i + size - 1])", "边界条件", "每个分块少取一个元素"),
    ("cur = max(x, cur + x)", "cur = cur + x", "逻辑错误", "缺少与当前元素的比较，Kadane 算法失效"),
    ("best = max(best, cur)", "best = cur", "逻辑错误", "直接覆盖最优值，结果错误"),
    ("out[v] = k", "out[k] = v", "逻辑错误", "键值写反了，没有真正反转字典"),
    ("if isinstance(item, list):", "if isinstance(item, str):", "逻辑错误",
     "递归判断类型错误，嵌套列表不会被展开"),
    ("if x == ch:", "if x != ch:", "逻辑错误", "判断反了，统计的是不匹配的字符"),
    ("result *= base", "result += base", "逻辑错误", "累乘写成了累加"),
    ("total += i", "total -= i", "逻辑错误", "累加写成了累减"),
    ("for k, v in b.items():\n        out[k] = v", "for k, v in b.items():\n        out[k] = a.get(k, v)",
     "逻辑错误", "合并时没有覆盖旧值，b 的新值不生效"),
    ("if nums[i] > x:", "if nums[i] < x:", "逻辑错误", "比较方向反了，插入位置错误"),

    # ---- 针对扩展种子的变异 ----
    ("if x == target:", "if x != target:", "逻辑错误", "判断反了，统计的是不匹配的元素"),
    ("p *= x", "p += x", "逻辑错误", "累乘写成了累加"),
    ("if x < m:", "if x <= m:", "逻辑错误", "比较用错，相等时也更新"),
    ("for i in range(len(parts)):", "for i in range(len(parts) - 1):", "边界条件",
     "循环少执行一次，最后一项被漏掉"),
    ("result += parts[i] + sep", "result += parts[i]", "逻辑错误", "拼接时丢了分隔符"),
    ("for w in words:", "for w in words[:-1]:", "边界条件", "循环排除了最后一个单词"),
    ("out.append(w.upper())", "out.append(w.lower())", "逻辑错误", "转大写写成了转小写"),
    ("if x > 0:", "if x >= 0:", "边界条件", "比较包含边界，0 也被算进去了"),
    ("total += i * i", "total += i", "逻辑错误", "累加的是 i 而不是 i 的平方"),
    ('if ch != " ":', 'if ch != "":', "逻辑错误", "过滤条件写错，空格没被去掉"),
    ("out += ch", "out = ch + out", "逻辑错误", "拼接顺序反了，字符串被反转"),
    ("if ch.isupper():", "if ch.islower():", "逻辑错误", "判断反了，统计的是小写字母"),
    ("if x not in b:", "if x in b:", "逻辑错误", "判断反了，保留的是 b 中有的元素"),
    ("if n % 2 == 1:", "if n % 2 == 0:", "边界条件", "奇偶判断反了，偶数会走进奇数分支"),
    ("return s[n // 2]", "return s[n // 2 - 1]", "边界条件", "中位数索引偏了一位"),
    ("for i in range(len(nums) - 1):", "for i in range(len(nums)):", "边界条件",
     "循环多执行一次，会访问越界下标"),
    ("words[len(words) - i - 1]", "words[len(words) - i]", "边界条件", "索引没减 1，会越界"),
    ("if s[i:i + len(sub)] == sub:", "if s[i:i + len(sub) - 1] == sub:", "边界条件",
     "切片短了一个字符，永远匹配不上"),
    ("for i in range(len(s)):", "for i in range(len(s) - 1):", "边界条件",
     "循环少执行一次，最后一个位置被漏掉"),
    ("if key in d:", "if key not in d:", "逻辑错误", "判断反了，键不存在时反而去删"),
    ("out = list(a)", "out = list(b)", "逻辑错误", "起始列表写错，a 的元素会全部丢失"),
    ("return s[len(s) - 2]", "return s[len(s) - 1]", "逻辑错误", "返回的是最大值而不是第二大值"),
    ("if ch.isalpha():", "if ch.isdigit():", "逻辑错误", "判断反了，统计的是数字"),
    ("for i in range(1, len(nums)):", "for i in range(2, len(nums)):", "边界条件",
     "循环起点偏后，前面的元素不会被移动"),
    ('if s is None:', 'if s == "":', "空值处理", "只判断了空串，None 会漏过去导致抛异常"),
    ("if abs(x) > best:", "if abs(x) >= best:", "逻辑错误", "比较用错，相等时也更新"),
    ("if x < threshold:", "if x <= threshold:", "边界条件", "比较包含边界，等于阈值时也被计入"),
    ("out.append(ch)", "out.append(ch.upper())", "逻辑错误", "多做了大写转换，改变了原字符"),
    ("for i in range(len(a)):", "for i in range(len(a) - 1):", "边界条件",
     "循环少执行一次，最后一行丢失"),
    ("for j in range(len(a[0])):", "for j in range(len(a[0]) - 1):", "边界条件",
     "内层循环少执行一次，最后一列丢失"),
    ("row.append(a[i][j] + b[i][j])", "row.append(a[i][j] - b[i][j])", "逻辑错误",
     "矩阵加法写成了减法"),
    ("out.append(w.capitalize())", "out.append(w.upper())", "逻辑错误", "首字母大写写成了全大写"),
    ("if nums[i] > nums[best]:", "if nums[i] < nums[best]:", "逻辑错误",
     "比较方向反了，找的是最小值下标"),
    ("for i in range(n):", "for i in range(n - 1):", "边界条件", "循环少执行一次，重复次数少一次"),
    ("if low <= x and x <= high:", "if low < x and x < high:", "边界条件",
     "区间改成开区间，边界值被排除"),

    # ---- 空值处理：针对新增带守卫种子的变异 ----
    # 这批把「空值处理」从 6% 拉到接近 20%。分两类：
    #   (a) 直接删掉守卫；(b) 把 None 检查改写成等价性检查（ None 会漏过去）
    ("if len(nums) == 0:\n        return None\n", "", "空值处理",
     "缺少空列表守卫，nums 为空时访问 nums[-1] 会抛 IndexError"),
    ("if not nums:\n        return []\n", "", "空值处理",
     "缺少空列表守卫，max() 对空序列会抛 ValueError"),
    ("if d is None:\n        return default\n", "", "空值处理",
     "缺少 None 判断，d 为 None 时调用 .get 会抛 AttributeError"),
    ("if not parts:\n        return \"\"\n", "", "空值处理",
     "缺少空列表守卫，空输入下行为不确定"),
    ("if nums is None:\n        return None\n", "", "空值处理",
     "缺少 None 判断，nums 为 None 时访问 nums[0] 会抛 TypeError"),
    ("if items is None:\n        return 0\n", "", "空值处理",
     "缺少 None 判断，len(None) 会抛 TypeError"),
    ("if not s:\n        return \"\"\n", "", "空值处理",
     "缺少空串守卫，s 为空时 split()[0] 会抛 IndexError"),
    ("if s is None:\n        return \"\"\n", "", "空值处理",
     "缺少 None 判断，None 做切片会抛 TypeError"),
    ("if nums is None or idx is None:\n        return None\n", "", "空值处理",
     "缺少 None 判断，None 参与下标比较会抛 TypeError"),
    ("if not m:\n        return 0\n", "", "空值处理",
     "缺少空矩阵守卫，未显式处理空输入"),
    ("if nums is None:", "if nums == []:", "空值处理",
     "None 判断写成了空列表判断，None 会漏过去导致抛异常"),
    ("if items is None:", "if items == []:", "空值处理",
     "None 判断写成了空列表判断，None 会漏过去"),
    ("if d is None:", "if d == {}:", "空值处理",
     "None 判断写成了空字典判断，None 会漏过去"),
]

PROMPT_TMPL = '''审查以下 Python 函数，找出其中的问题。

```python
{code}
```

请列出所有问题，每行一条，格式为「类型: 描述」。如果没有任何问题，只回答「无问题」。'''


def build_samples(seed_names, rng):
    """按种子 × 变异 生成样本；负样本（干净代码）单独生成。"""
    positive, negative = [], []
    seen_pairs = set()

    for name in seed_names:
        code = SEEDS[name]
        # 负样本：干净代码 -> 无问题
        negative.append({"code": code, "issues": [], "seed": name})

        for search, replace, cat, desc in MUTATIONS:
            if search not in code:
                continue
            mutated = code.replace(search, replace, 1)
            if mutated == code:
                continue
            key = (name, mutated)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            positive.append({
                "code": mutated,
                "issues": [{"type": cat, "desc": desc}],
                "seed": name,
            })

    # 组合变异：同一函数注入两个 bug，更接近真实代码
    combined = []
    for name in seed_names:
        code = SEEDS[name]
        applicable = [m for m in MUTATIONS if m[0] in code]
        if len(applicable) < 2:
            continue
        rng.shuffle(applicable)
        # 取两两组合，但两个变异不能互相包含（避免替换冲突）
        pairs_made = 0
        for i in range(len(applicable)):
            for j in range(i + 1, len(applicable)):
                if pairs_made >= 6:
                    break
                m1, m2 = applicable[i], applicable[j]
                if m1[0] in m2[0] or m2[0] in m1[0]:
                    continue
                mutated = code.replace(m1[0], m1[1], 1)
                if m2[0] not in mutated:
                    continue
                mutated = mutated.replace(m2[0], m2[1], 1)
                key = (name, mutated)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                combined.append({
                    "code": mutated,
                    "issues": [{"type": m1[2], "desc": m1[3]},
                               {"type": m2[2], "desc": m2[3]}],
                    "seed": name,
                })
                pairs_made += 1

    return positive, combined, negative


def to_messages(sample):
    issues = sample["issues"]
    if not issues:
        answer = "无问题"
    else:
        answer = "\n".join(f"{i['type']}: {i['desc']}" for i in issues)
    return {
        "messages": [
            {"role": "user", "content": PROMPT_TMPL.format(code=sample["code"])},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"seed": sample["seed"], "n_issues": len(issues),
                 "types": [i["type"] for i in issues]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/Volumes/LLM/datasets/reviewer_mlx")
    ap.add_argument("--valid-seeds", type=int, default=6, help="留出多少种子做验证集")
    ap.add_argument("--test-seeds", type=int, default=14,
                    help="完全独立留出的种子数，只用于测试集，绝不参与训练")
    ap.add_argument("--test-names-file", default=None,
                    help="固化测试集种子名单文件；给了就不再随机划分，保证跨实验可比")
    ap.add_argument("--neg-ratio", type=float, default=0.25,
                    help="负样本（无问题）在总样本中的目标占比")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    names = sorted(SEEDS)
    if args.test_names_file:
        # 固化测试集。往 SEEDS 里加新种子会打乱 shuffle 顺序，导致每次生成的
        # 测试集都不同，跨实验就无法对比——所以一旦定了名单就锁死。
        test_names = [s.strip() for s in open(args.test_names_file) if s.strip()]
        missing = set(test_names) - set(names)
        if missing:
            raise SystemExit(f"测试种子不存在于 SEEDS: {missing}")
    else:
        rng.shuffle(names)
        test_names = names[: args.test_seeds]
    # 三个 split 的种子互不重叠：测试集的函数绝不出现在训练数据里，
    # 否则测到的是"背住了这个函数"，而不是审查能力
    rest = [n for n in names if n not in set(test_names)]
    rng.shuffle(rest)
    valid_names = rest[: args.valid_seeds]
    train_names = rest[args.valid_seeds:]

    os.makedirs(args.out_dir, exist_ok=True)

    def assemble(seed_names):
        pos, comb, neg = build_samples(seed_names, rng)
        all_pos = pos + comb
        rng.shuffle(all_pos)
        # 按目标比例配平负样本
        n_neg = max(1, int(len(all_pos) * args.neg_ratio / (1 - args.neg_ratio)))
        # 负样本可重复采样（干净代码数量有限），但要覆盖不同种子
        neg_pool = neg * (n_neg // max(1, len(neg)) + 1)
        rng.shuffle(neg_pool)
        samples = all_pos + neg_pool[:n_neg]
        rng.shuffle(samples)
        return samples

    train = assemble(train_names)
    valid = assemble(valid_names)
    test = assemble(test_names)

    for split, samples in (("train", train), ("valid", valid), ("test", test)):
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(to_messages(s), ensure_ascii=False) + "\n")
        pos_n = sum(1 for s in samples if s["issues"])
        print(f"{split}.jsonl: {len(samples)} 条  "
              f"(有问题 {pos_n} / 无问题 {len(samples) - pos_n})  -> {path}")

    # 统计描述
    total_issues = sum(len(s["issues"]) for s in train + valid if s["issues"])
    cats = {}
    for s in train + valid:
        for i in s["issues"]:
            cats[i["type"]] = cats.get(i["type"], 0) + 1
    print(f"\n注入的 bug 总数: {total_issues}")
    print("问题类型分布:", {k: v for k, v in sorted(cats.items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
