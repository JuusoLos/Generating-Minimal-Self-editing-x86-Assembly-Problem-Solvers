# Numba simulation core for the simplified mov-only search environment.
#
# The search represents the initial mutable instruction block as destination-
# source pairs and evaluates candidates over multiple generations. During each
# generation:
#   - the initial mutable instruction block is executed,
#   - all previously appended instructions are executed,
#   - one new appended instruction is derived from op0/op1,
#   - invalid memory-to-memory appended instructions stop the simulation.
#
# The target pipeline mask summarizes which source locations in the mutable
# memory tape influence output, op0, or op1 in the last executed generation.


from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # type: ignore

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

if not HAVE_NUMBA:
    raise RuntimeError("Numba not available. Install numba and re-run.")


# =============================================================================
# Simplified search environment constants
# =============================================================================
TAPE_DWORDS = 11
OPERAND_VOCAB_SIZE = 11
STOP_ID = 11

FREE0 = 0
FREE1 = 1
FREE2 = 2
INPUT = 3
OUTPUT = 4
DATA0 = 5
DATA1 = 6
OP0 = 7
OP1 = 8
EAX_SYMBOL = 9
EBX_SYMBOL = 10

BIAS_OPERANDS = [
    FREE0,
    FREE1,
    FREE2,
    INPUT,
    OUTPUT,
    DATA0,
    DATA1,
    OP0,
    OP1,
    EAX_SYMBOL,
    EBX_SYMBOL,
]


# =============================================================================
# Low-level helpers
# =============================================================================
@njit(cache=True)
def is_memory_symbol(symbol: np.int32) -> np.int32:
    return np.int32(1) if (symbol >= 0 and symbol <= 8) else np.int32(0)


@njit(cache=True)
def get_source_value(tape_u32, regs_u32, src: np.int32) -> np.uint32:
    if is_memory_symbol(src) != 0:
        return tape_u32[src]
    return regs_u32[src - np.int32(9)]


@njit(cache=True)
def execute_initial_instruction_actual(tape_u32, regs_u32, dst: np.int32, src: np.int32) -> None:
    # In the concrete x86 realization, memory-to-memory instructions from the
    # initial mutable instruction block are implemented through eax.
    if is_memory_symbol(dst) != 0 and is_memory_symbol(src) != 0:
        regs_u32[0] = tape_u32[src]
        tape_u32[dst] = regs_u32[0]
        return

    src_value = get_source_value(tape_u32, regs_u32, src)

    if is_memory_symbol(dst) != 0:
        tape_u32[dst] = src_value
    else:
        regs_u32[dst - np.int32(9)] = src_value


@njit(cache=True)
def execute_appended_instruction_actual(tape_u32, regs_u32, dst: np.int32, src: np.int32) -> None:
    src_value = get_source_value(tape_u32, regs_u32, src)
    if is_memory_symbol(dst) != 0:
        tape_u32[dst] = src_value
    else:
        regs_u32[dst - np.int32(9)] = src_value


@njit(cache=True)
def update_usage_masks(
    dst: np.int32,
    src: np.int32,
    source_used_mask: np.uint8,
    root_destination_mask: np.uint8,
    free_source_mask: np.uint8,
    free_destination_mask: np.uint8,
):
    if src == np.int32(INPUT):
        source_used_mask |= np.uint8(1)
    elif src == np.int32(DATA0):
        source_used_mask |= np.uint8(2)
    elif src == np.int32(DATA1):
        source_used_mask |= np.uint8(4)

    if dst == np.int32(OUTPUT):
        root_destination_mask |= np.uint8(1)
    elif dst == np.int32(OP0):
        root_destination_mask |= np.uint8(2)
    elif dst == np.int32(OP1):
        root_destination_mask |= np.uint8(4)

    if src == np.int32(FREE0):
        free_source_mask |= np.uint8(1)
    elif src == np.int32(FREE1):
        free_source_mask |= np.uint8(2)
    elif src == np.int32(FREE2):
        free_source_mask |= np.uint8(4)

    if dst == np.int32(FREE0):
        free_destination_mask |= np.uint8(1)
    elif dst == np.int32(FREE1):
        free_destination_mask |= np.uint8(2)
    elif dst == np.int32(FREE2):
        free_destination_mask |= np.uint8(4)

    return source_used_mask, root_destination_mask, free_source_mask, free_destination_mask


@njit(cache=True)
def wanted_pipeline_bit(symbol: np.int32) -> np.uint16:
    if symbol == np.int32(FREE0):
        return np.uint16(1 << 0)
    if symbol == np.int32(FREE1):
        return np.uint16(1 << 1)
    if symbol == np.int32(FREE2):
        return np.uint16(1 << 2)
    if symbol == np.int32(INPUT):
        return np.uint16(1 << 3)
    if symbol == np.int32(DATA0):
        return np.uint16(1 << 5)
    if symbol == np.int32(DATA1):
        return np.uint16(1 << 6)
    return np.uint16(0)


# =============================================================================
# Pipeline computation
# =============================================================================
@njit(cache=True)
def build_full_pipeline_mask_from_last_executed_generation(
    initial_dst_row,
    initial_src_row,
    initial_len_row: np.int32,
    history_dst_row,
    history_src_row,
    active_history_len: np.int32,
) -> np.uint16:
    """
    Compute the repeated-generation pipeline implied by the last executed
    generation's full logical program:

        initial mutable instruction block
        + active appended instructions from that generation

    The computation works by:
      1) building a dependency map where dep[x] is the previous-state location
         from which the final value of location x is derived,
      2) starting from {output, op0, op1},
      3) following dependencies transitively,
      4) recording which relevant memory locations are reached.
    """
    dependency = np.empty((TAPE_DWORDS,), dtype=np.int32)
    seen_write = np.zeros((TAPE_DWORDS,), dtype=np.uint8)

    for i in range(TAPE_DWORDS):
        dependency[i] = np.int32(i)

    for j in range(active_history_len - 1, -1, -1):
        dst = history_dst_row[j]
        src = history_src_row[j]
        if seen_write[dst] == np.uint8(0):
            dependency[dst] = src
            seen_write[dst] = np.uint8(1)

    for i in range(initial_len_row - 1, -1, -1):
        dst = initial_dst_row[i]
        src = initial_src_row[i]
        if seen_write[dst] == np.uint8(0):
            dependency[dst] = src
            seen_write[dst] = np.uint8(1)

    seen = np.zeros((TAPE_DWORDS,), dtype=np.uint8)
    stack = np.empty((32,), dtype=np.int32)
    stack_pointer = np.int32(0)

    stack[stack_pointer] = np.int32(OUTPUT)
    stack_pointer += np.int32(1)
    stack[stack_pointer] = np.int32(OP0)
    stack_pointer += np.int32(1)
    stack[stack_pointer] = np.int32(OP1)
    stack_pointer += np.int32(1)

    pipeline_mask = np.uint16(0)

    while stack_pointer > np.int32(0):
        stack_pointer -= np.int32(1)
        x = stack[stack_pointer]

        if seen[x] != np.uint8(0):
            continue
        seen[x] = np.uint8(1)

        y = dependency[x]
        pipeline_mask |= wanted_pipeline_bit(y)

        if seen[y] == np.uint8(0):
            stack[stack_pointer] = y
            stack_pointer += np.int32(1)

    return pipeline_mask


# =============================================================================
# Batch simulation
# =============================================================================
@njit(parallel=True, cache=True)
def run_batch_env_4runs(
    tape_templates,
    initial_dst,
    initial_src,
    initial_len,
    n_generations,
):
    n_runs = tape_templates.shape[0]

    out_history_length = np.zeros((n_runs,), dtype=np.int32)
    out_history_dst = np.full((n_runs, n_generations), np.int32(-1), dtype=np.int32)
    out_history_src = np.full((n_runs, n_generations), np.int32(-1), dtype=np.int32)

    out_final_out = np.zeros((n_runs,), dtype=np.uint32)
    out_final_tape = np.zeros((n_runs, TAPE_DWORDS), dtype=np.uint32)

    out_source_used_mask = np.zeros((n_runs,), dtype=np.uint8)
    out_root_destination_mask = np.zeros((n_runs,), dtype=np.uint8)
    out_free_source_mask = np.zeros((n_runs,), dtype=np.uint8)
    out_free_destination_mask = np.zeros((n_runs,), dtype=np.uint8)
    out_target_pipeline_mask = np.zeros((n_runs,), dtype=np.uint16)

    for run_index in prange(n_runs):
        tape = tape_templates[run_index].copy()
        regs = np.zeros((2,), dtype=np.uint32)

        history_dst_local = np.empty((n_generations,), dtype=np.int32)
        history_src_local = np.empty((n_generations,), dtype=np.int32)
        history_length_local = np.int32(0)

        source_used_mask = np.uint8(0)
        root_destination_mask = np.uint8(0)
        free_source_mask = np.uint8(0)
        free_destination_mask = np.uint8(0)
        stopped = np.uint8(0)

        executed_generations = np.int32(0)

        for _ in range(n_generations):
            if stopped != np.uint8(0):
                break

            executed_generations += np.int32(1)

            for i in range(initial_len[run_index]):
                dst = initial_dst[run_index, i]
                src = initial_src[run_index, i]

                (
                    source_used_mask,
                    root_destination_mask,
                    free_source_mask,
                    free_destination_mask,
                ) = update_usage_masks(
                    dst,
                    src,
                    source_used_mask,
                    root_destination_mask,
                    free_source_mask,
                    free_destination_mask,
                )

                execute_initial_instruction_actual(tape, regs, dst, src)

            for j in range(history_length_local):
                dst = history_dst_local[j]
                src = history_src_local[j]

                (
                    source_used_mask,
                    root_destination_mask,
                    free_source_mask,
                    free_destination_mask,
                ) = update_usage_masks(
                    dst,
                    src,
                    source_used_mask,
                    root_destination_mask,
                    free_source_mask,
                    free_destination_mask,
                )

                execute_appended_instruction_actual(tape, regs, dst, src)

            candidate_dst = np.int32(tape[OP0] % np.uint32(11))
            candidate_src = np.int32(tape[OP1] % np.uint32(11))

            if is_memory_symbol(candidate_dst) != 0 and is_memory_symbol(candidate_src) != 0:
                stopped = np.uint8(1)
            else:
                if history_length_local < n_generations:
                    history_dst_local[history_length_local] = candidate_dst
                    history_src_local[history_length_local] = candidate_src
                    out_history_dst[run_index, history_length_local] = candidate_dst
                    out_history_src[run_index, history_length_local] = candidate_src
                    history_length_local += np.int32(1)

        active_history_len_last_generation = np.int32(0)
        if executed_generations > np.int32(0):
            x = executed_generations - np.int32(1)
            if history_length_local < x:
                active_history_len_last_generation = history_length_local
            else:
                active_history_len_last_generation = x

        target_pipeline_mask_local = build_full_pipeline_mask_from_last_executed_generation(
            initial_dst[run_index],
            initial_src[run_index],
            initial_len[run_index],
            history_dst_local,
            history_src_local,
            active_history_len_last_generation,
        )

        out_history_length[run_index] = history_length_local
        out_final_out[run_index] = tape[OUTPUT]
        out_final_tape[run_index, :] = tape
        out_source_used_mask[run_index] = source_used_mask
        out_root_destination_mask[run_index] = root_destination_mask
        out_free_source_mask[run_index] = free_source_mask
        out_free_destination_mask[run_index] = free_destination_mask
        out_target_pipeline_mask[run_index] = target_pipeline_mask_local

    return (
        out_history_length,
        out_history_dst,
        out_history_src,
        out_final_out,
        out_final_tape,
        out_source_used_mask,
        out_root_destination_mask,
        out_free_source_mask,
        out_free_destination_mask,
        out_target_pipeline_mask,
    )
