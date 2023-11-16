import time
import numpy as np
import concurrent.futures
from numba import njit

# optimized algorithm
@njit
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray: 
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the output element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    """
    current_ts2 = 0
    len_timestamps2 = len(timestamps2)
    matching = np.zeros(len(timestamps1), dtype=np.int32)
    for i, ts1 in enumerate(timestamps1):
        current_diff = np.abs(timestamps2[current_ts2] - ts1)
        current_ts2 += 1
        while current_ts2 < len_timestamps2 and np.abs(timestamps2[current_ts2]-ts1) < current_diff:
            current_diff = np.abs(timestamps2[current_ts2]-ts1)
            current_ts2 += 1
        current_ts2 -= 1
        matching[i] = current_ts2

    return matching

# not optimized algorithm
def match_timestamps_(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    matching = np.zeros(len(timestamps1), dtype=int)
    for i, ts1 in enumerate(timestamps1):
        min_index = np.argmin(np.abs(timestamps2 - ts1))
        matching[i] = min_index
    return matching

# parallel algorithm
def match_timestamps_parallel(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    def find_min_index(i):
        ts1 = timestamps1[i]
        min_index = np.argmin(np.abs(timestamps2 - ts1))
        return min_index
    matching = np.zeros(len(timestamps1), dtype=int)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        matching = list(executor.map(find_min_index, range(len(timestamps1))))
    return np.array(matching)

# make timestamps
def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps

# time check
def main():
    # generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)
    average_time = 0
    number_of_iterations = 1000
    for iter in range(number_of_iterations):
        start_time = time.time()
        matching = match_timestamps(timestamps1, timestamps2)
        end_time = time.time()
        average_time += end_time - start_time
    print(f"Average time unparalled algorithm: {average_time/number_of_iterations}")


if __name__ == '__main__':
    main()