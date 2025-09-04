import numpy as np
import sys
sys.path.append("../utils/")
from utils import print_with_separation
from tqdm import tqdm

KEYPRESS_TIME = 60 / float((90 * 5)) # rough WPM to keypress time

class EditOp:
    def __init__(self, start_index, content, is_deletion):
        self.start_index = start_index
        self.content = content
        self.length = len(content)
        self.is_deletion = is_deletion
    
    def __str__(self):
        return f"EditOp:\t{"delete" if self.is_deletion else "add"}\tstart_index {self.start_index}\tlength {self.length}\tcontent \"{self.content}\"\t"
    
    def __hash__(self):
        return hash((self.start_index, self.content, self.is_deletion))


def levenshtein_with_alignment(a: str, b: str) -> list:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1])
    pos_map = [None] * len(a)
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pos_map[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:  # Deletion
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:  # Insertion
            j -= 1
    return dp[m][n], pos_map

def get_edited_indices(pos_map: list, text2: str) -> tuple[list, list]:
    deleted_indices = []
    added_indices = []

    for i, item in enumerate(pos_map):
        if item is None:
            deleted_indices.append(i)

    for i in range(len(text2)):
        if i not in pos_map:
            added_indices.append(i)
    
    return deleted_indices, added_indices

RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'
def print_char_diff(text1: str, text2: str):
    distance, pos_map = levenshtein_with_alignment(text1, text2)

    deleted_indices = []
    added_indices = []

    for i, item in enumerate(pos_map):
        if item is None:
            deleted_indices.append(i)

    for i in range(len(text2)):
        if i not in pos_map:
            added_indices.append(i)

    text1_i = 0
    text2_i = 0
    while text1_i < len(text1) or text2_i < len(text2):
        if text2_i not in added_indices and text1_i not in deleted_indices:
            print(text1[text1_i], end="")
            text1_i += 1
            text2_i += 1
        if text1_i in deleted_indices:
            while text1_i in deleted_indices:
                print(f"{RED}{text1[text1_i]}{RESET}", end="")
                text1_i += 1
        if text2_i in added_indices:
            while text2_i in added_indices:
                print(f"{GREEN}{text2[text2_i]}{RESET}", end="")
                text2_i += 1


def get_edit_ops(a: str, b: str):
    edit_ops = []
    _, pos_map = levenshtein_with_alignment(a, b)
    deleted_indices, added_indices = get_edited_indices(pos_map, b)

    deleted_runs = []
    if deleted_indices:
        start = deleted_indices[0]
        length = 1

        for i in range(1, len(deleted_indices)):
            if deleted_indices[i] == deleted_indices[i-1] + 1:
                length += 1
            else:
                deleted_runs.append((start, length))
                start = deleted_indices[i]
                length = 1
        deleted_runs.append((start, length))

    added_runs = []
    if added_indices:
        start = added_indices[0]
        length = 1

        for i in range(1, len(added_indices)):
            if added_indices[i] == added_indices[i-1] + 1:
                length += 1
            else:
                added_runs.append((start, length))
                start = added_indices[i]
                length = 1
        added_runs.append((start, length))

    for start, length in deleted_runs:
        edit_ops.append(EditOp(start, a[start:start+length], True))
    for start, length in added_runs:
        edit_ops.append(EditOp(start, b[start:start+length], False))

    return edit_ops

# util
def index_to_line_char(text: str, index: int) -> tuple[int, int]:
    lines = [line + "\n" for line in text.splitlines()]
    chars_in_line = index
    for i, line in enumerate(lines):
        if chars_in_line < len(line):
            return i, chars_in_line
        chars_in_line -= len(line)
    return len(lines) - 1, len(lines[-1])

# util
def line_char_to_index(text: str, line: int, char: int):
    index = 0
    lines = [line + "\n" for line in text.splitlines()]
    for _, l in enumerate(lines[:line]):
        index += len(l)
    index += char
    return index


def cursor_jump_cost(edit_op: EditOp, cursor_pos: int, text: str) -> float:
    # cost of jumping from cursor_pos to edit_op's index in the text
    prev_line, prev_char = index_to_line_char(text, cursor_pos)
    post_line, post_char = index_to_line_char(text, edit_op.start_index)
    dist = np.sqrt(
        (float(prev_line) - post_line) ** 2 + (float(prev_char) - post_char) ** 2
    )

    # if highlighted deletion, cursor be on either end
    if edit_op.is_deletion and edit_op.length > 2:
        post_line, post_char = index_to_line_char(text, edit_op.start_index + edit_op.length)
        dist2 = np.sqrt(
            (float(prev_line) - post_line) ** 2 + (float(prev_char) - post_char) ** 2
        )
        dist = dist2 if dist2 < dist else dist

    if dist == 1: # if one char over, use arrow key
        time = KEYPRESS_TIME
    else:
        time = 0.2 + 0.025 * dist
    return time # fixed cost of moving hand to cursor plus distance traveled


def add_cost(edit_op: EditOp) -> float:
    # assuming 60 WPM
    return KEYPRESS_TIME * edit_op.length


def delete_cost(edit_op: EditOp, cursor_pos: int, text: str) -> float:
    if edit_op.length <= 2:
        return KEYPRESS_TIME * edit_op.length
    else:
        prev_line, prev_char = index_to_line_char(text, edit_op.start_index)
        post_line, post_char = index_to_line_char(text, edit_op.start_index + edit_op.length)
        dist = np.sqrt(
            (float(prev_line) - post_line) ** 2 + (float(prev_char) - post_char) ** 2
        )
        # fixed cost for delete keypress and linear scale for highlight
        return KEYPRESS_TIME + 0.025 * dist
    
def translate_insertion_index(insertion_op: EditOp, current_text: str, text2: str) -> int:
    if insertion_op.is_deletion:
        return insertion_op.start_index

    text2_index = insertion_op.start_index
    if text2_index == 0:
        return 0  # Insert at beginning

    current_pos = 0
    text2_pos = 0
    while text2_pos < text2_index and current_pos < len(current_text):
        if (
            text2_pos < len(text2)
            and current_pos < len(current_text)
            and text2[text2_pos] == current_text[current_pos]
        ):
            text2_pos += 1
            current_pos += 1
        else:
            current_pos += 1

    return current_pos


def apply_edit_op(
    edit_op: EditOp,
    cursor_pos: int,
    text: str,
    remaining_edit_ops: list[EditOp],
    pos_map: list,
    text1: str,
    text2: str,
    applied_ops: list[EditOp] = None,
):
    if applied_ops is None:
        applied_ops = []

    jump_cost = cursor_jump_cost(edit_op, cursor_pos, text)
    edit_cost = delete_cost(edit_op, cursor_pos, text) if edit_op.is_deletion else add_cost(edit_op)

    if edit_op.is_deletion:
        actual_index = edit_op.start_index
        new_text = text[:actual_index] + text[actual_index + edit_op.length :]
        new_cursor_pos = actual_index
        index_shift = -edit_op.length
        shift_threshold = actual_index
    else:
        actual_index = translate_insertion_index(
            edit_op, text, text2,
        )
        new_text = text[:actual_index] + edit_op.content + text[actual_index:]
        new_cursor_pos = actual_index + edit_op.length
        index_shift = edit_op.length
        shift_threshold = actual_index

    new_remaining_ops = []
    for op in remaining_edit_ops:
        if op != edit_op:
            new_start_index = op.start_index
            if op.is_deletion and op.start_index > shift_threshold:
                new_start_index += index_shift

            new_remaining_ops.append(
                EditOp(new_start_index, op.content, op.is_deletion)
            )

    new_applied_ops = applied_ops + [edit_op]

    return (
        jump_cost + edit_cost,
        new_text,
        new_cursor_pos,
        new_remaining_ops,
        new_applied_ops,
    )


def keystroke_distance_with_backtrack(
    edit_ops: list[EditOp], cursor_pos: int, text: str, text1: str, text2: str
) -> tuple[float, list]:
    if len(edit_ops) == 0:
        return 0, []

    _, pos_map = levenshtein_with_alignment(text1, text2)

    memo = {}
    best_sequence = {}  # Store the optimal sequence for each state

    def dp(
        cursor_pos: int,
        text: str,
        remaining_ops: tuple[EditOp],
        applied_ops: tuple[EditOp],
    ) -> float:
        if not remaining_ops:
            best_sequence[(cursor_pos, hash(text), hash(remaining_ops))] = []
            return 0

        key = (cursor_pos, hash(text), hash(remaining_ops))
        if key in memo:
            return memo[key]

        min_cost = float("inf")
        best_op = None
        best_next_sequence = []

        for edit_op in tqdm(remaining_ops, leave=False):
            cost, new_text, new_cursor_pos, new_remaining_ops, new_applied_ops = (
                apply_edit_op(
                    edit_op,
                    cursor_pos,
                    text,
                    list(remaining_ops),
                    pos_map,
                    text1,
                    text2,
                    list(applied_ops),
                )
            )

            remaining_cost = dp(
                new_cursor_pos,
                new_text,
                tuple(new_remaining_ops),
                tuple(new_applied_ops),
            )
            total_cost = cost + remaining_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_op = edit_op
                next_key = (
                    new_cursor_pos,
                    hash(new_text),
                    hash(tuple(new_remaining_ops)),
                )
                best_next_sequence = best_sequence.get(next_key, [])

        # Store the best sequence for this state
        best_sequence[key] = [best_op] + best_next_sequence
        memo[key] = min_cost
        return min_cost

    initial_key = (cursor_pos, hash(text), hash(tuple(edit_ops)))
    min_cost = dp(cursor_pos, text, tuple(edit_ops), tuple())
    optimal_sequence = best_sequence.get(initial_key, [])

    return min_cost, optimal_sequence

def get_keystroke_distance(text1: str, text2: str):
    print("\n\n\n")
    print("="*80)
    print_char_diff(text1, text2)
    print("\n" + "="*80)
    edit_ops = get_edit_ops(text1, text2)
    initial_cursor_pos = len(text1.splitlines()[0]) + 2
    min_cost, _ = keystroke_distance_with_backtrack(edit_ops, initial_cursor_pos, text1, text1, text2)
    return min_cost