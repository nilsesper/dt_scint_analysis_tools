###########################################
### MATCHING FITS OF DIFFERENT SUPERLAYERS IN TIME
###########################################

import numpy as np

# -----------------------------------------

### pair every fit a with the closest unused fit b in time (|t0_b - t0_a| <= tolerance)
# fits a are taken in the order of their t0 (sort_kind of np.argsort); a fit b is used at most once
# is_candidate(i, j): additional condition for b = j to be considered for a = i
# accept_pair(i, j): final check of the closest candidate; if it fails, a = i stays unmatched (and j stays unused)
# returns [(i, j, number of candidates for i)]
def match_nearest_in_time(t0_a, t0_b, tolerance, *, sort_kind="stable", is_candidate=None, accept_pair=None):
    order_a, order_b = np.argsort(t0_a, kind=sort_kind), np.argsort(t0_b, kind=sort_kind)
    t0_b_sorted = t0_b[order_b]
    used_b = np.zeros(len(t0_b), dtype=bool)
    matches = []
    window_start = 0
    for i in order_a:
        while window_start < len(t0_b) and t0_b_sorted[window_start] < t0_a[i] - tolerance:
            window_start += 1
        closest, closest_distance, n_candidates = None, None, 0
        k = window_start
        while k < len(t0_b) and t0_b_sorted[k] <= t0_a[i] + tolerance:
            j = order_b[k]
            k += 1
            if used_b[j] or (is_candidate is not None and not is_candidate(i, j)):
                continue
            n_candidates += 1
            distance = np.abs(t0_b[j] - t0_a[i])
            if closest_distance is None or distance < closest_distance:
                closest, closest_distance = j, distance
        if closest is None or (accept_pair is not None and not accept_pair(i, closest)):
            continue
        used_b[closest] = True
        matches.append((i, closest, n_candidates))
    return matches
