import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numba

# --- Global cache for STC matrices and effects ---
# This cache stores the computationally expensive results of STC matrix construction.
# Key: (c, h, n, matrix_seed, method) -> Value: (effects, check_effects, check_lengths, submatrix_cols)
_stc_matrix_cache = {}


# --- Static Helper Functions ---

def int_to_bits(value: int, n_bits: int) -> np.ndarray:
    """Converts an integer to a binary numpy array of a specific length."""
    return np.array([int(b) for b in bin(value)[2:].zfill(n_bits)], dtype=np.int32)

def bits_to_int(bits: np.ndarray) -> int:
    """Converts a binary numpy array to an integer."""
    if bits.ndim == 0: # Handle scalar array
        return int(bits)
    # Using a faster method for array to int conversion
    return np.dot(bits.astype(np.int32), 1 << np.arange(bits.size)[::-1])

# The 'mats' array ported from the C++ reference implementation (text_sample/src/common.cpp).
# This is a pre-computed pool of submatrices used to construct the full parity-check matrix H.
# It's indexed by (height - 7) and (width - 1).
MATS = np.array([
    # h=7, w=1..20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    109, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    109, 79, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    89, 127, 99, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    95, 75, 121, 71, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    71, 117, 127, 75, 89, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    111, 83, 127, 97, 77, 117, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    113, 111, 87, 93, 99, 73, 117, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    89, 97, 115, 81, 77, 117, 87, 127, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    95, 107, 109, 79, 117, 67, 121, 123, 103, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    117, 71, 109, 79, 101, 115, 123, 81, 77, 95, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    119, 73, 81, 125, 123, 103, 99, 127, 109, 69, 89, 107, 0, 0, 0, 0, 0, 0, 0, 0,
    87, 127, 117, 81, 97, 67, 101, 93, 105, 109, 75, 115, 123, 0, 0, 0, 0, 0, 0, 0,
    93, 107, 115, 95, 121, 81, 75, 99, 111, 85, 79, 119, 105, 65, 0, 0, 0, 0, 0, 0,
    123, 85, 79, 87, 127, 65, 115, 93, 101, 111, 73, 119, 105, 99, 91, 0, 0, 0, 0, 0,
    127, 99, 121, 111, 71, 109, 103, 117, 113, 65, 105, 87, 101, 75, 93, 123, 0, 0, 0, 0,
    89, 93, 111, 117, 103, 127, 77, 95, 85, 105, 67, 69, 113, 123, 99, 75, 119, 0, 0, 0,
    65, 99, 77, 85, 101, 91, 125, 103, 127, 111, 69, 93, 75, 95, 119, 113, 105, 115, 0, 0,
    91, 117, 77, 107, 101, 127, 115, 83, 85, 119, 105, 113, 93, 71, 111, 121, 97, 73, 81, 0,
    95, 111, 117, 83, 97, 75, 87, 127, 85, 93, 105, 115, 77, 101, 99, 89, 71, 121, 67, 123,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # h=8, w=1..20
    247, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    143, 187, 233, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    235, 141, 161, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    219, 185, 151, 255, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    251, 159, 217, 167, 221, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    201, 143, 231, 251, 189, 169, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    143, 245, 177, 253, 217, 163, 155, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    233, 145, 219, 185, 231, 215, 173, 129, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    139, 201, 177, 167, 213, 253, 227, 199, 185, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    183, 145, 223, 199, 245, 139, 187, 157, 217, 237, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    223, 145, 137, 219, 197, 243, 247, 189, 135, 181, 207, 235, 0, 0, 0, 0, 0, 0, 0, 0,
    229, 205, 237, 187, 135, 241, 183, 163, 151, 243, 213, 137, 159, 0, 0, 0, 0, 0, 0, 0,
    205, 165, 239, 211, 231, 247, 133, 227, 219, 189, 249, 185, 149, 129, 0, 0, 0, 0, 0, 0,
    131, 213, 255, 207, 227, 221, 173, 185, 197, 147, 235, 247, 217, 143, 229, 0, 0, 0, 0, 0,
    247, 139, 157, 223, 187, 147, 177, 249, 165, 153, 161, 227, 237, 255, 207, 197, 0, 0, 0, 0,
    205, 139, 239, 183, 147, 187, 249, 225, 253, 163, 173, 233, 209, 159, 255, 149, 197, 0, 0, 0,
    177, 173, 195, 137, 211, 249, 191, 135, 175, 155, 229, 215, 203, 225, 247, 237, 221, 227, 0, 0,
    159, 189, 195, 163, 255, 147, 219, 247, 231, 157, 139, 173, 185, 197, 207, 245, 193, 241, 233, 0,
    235, 179, 219, 253, 241, 131, 213, 231, 247, 223, 201, 193, 191, 249, 145, 237, 155, 165, 141, 173,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # h=9, w=1..20
    339, 489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    469, 441, 379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    371, 439, 277, 479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    413, 489, 443, 327, 357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    509, 453, 363, 409, 425, 303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    377, 337, 443, 487, 467, 421, 299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    497, 349, 279, 395, 365, 427, 399, 297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    435, 373, 395, 507, 441, 325, 279, 289, 319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    301, 379, 509, 411, 293, 467, 455, 261, 343, 447, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    367, 289, 445, 397, 491, 279, 373, 315, 435, 473, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    465, 379, 319, 275, 293, 407, 373, 427, 445, 497, 347, 417, 0, 0, 0, 0, 0, 0, 0, 0,
    473, 401, 267, 311, 359, 347, 333, 441, 405, 381, 497, 463, 269, 0, 0, 0, 0, 0, 0, 0,
    467, 283, 405, 303, 269, 337, 385, 441, 511, 361, 455, 355, 353, 311, 0, 0, 0, 0, 0, 0,
    489, 311, 259, 287, 445, 471, 419, 345, 289, 391, 405, 411, 371, 457, 331, 0, 0, 0, 0, 0,
    493, 427, 305, 309, 339, 447, 381, 335, 323, 423, 453, 457, 443, 313, 371, 353, 0, 0, 0, 0,
    271, 301, 483, 401, 369, 367, 435, 329, 319, 473, 441, 491, 325, 455, 389, 341, 317, 0, 0, 0,
    333, 311, 509, 319, 391, 441, 279, 467, 263, 487, 393, 405, 473, 303, 353, 337, 451, 365, 0, 0,
    301, 477, 361, 445, 505, 363, 375, 277, 271, 353, 337, 503, 457, 357, 287, 323, 435, 345, 497, 0,
    281, 361, 413, 287, 475, 359, 483, 351, 337, 425, 453, 423, 301, 309, 331, 499, 507, 277, 375, 471,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # h=10, w=1..20
    519, 885, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    579, 943, 781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    685, 663, 947, 805, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    959, 729, 679, 609, 843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    959, 973, 793, 747, 573, 659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    631, 559, 1023, 805, 709, 913, 979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    607, 867, 731, 1013, 625, 973, 825, 925, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    743, 727, 851, 961, 813, 605, 527, 563, 867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    863, 921, 943, 523, 653, 969, 563, 597, 753, 621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    729, 747, 901, 839, 815, 935, 777, 641, 1011, 603, 973, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    581, 831, 659, 877, 781, 929, 1003, 1021, 655, 729, 983, 611, 0, 0, 0, 0, 0, 0, 0, 0,
    873, 1013, 859, 887, 579, 697, 769, 927, 679, 683, 911, 753, 733, 0, 0, 0, 0, 0, 0, 0,
    991, 767, 845, 977, 923, 609, 633, 769, 533, 829, 859, 759, 687, 657, 0, 0, 0, 0, 0, 0,
    781, 663, 731, 829, 851, 941, 601, 997, 719, 675, 947, 939, 657, 549, 647, 0, 0, 0, 0, 0,
    619, 879, 681, 601, 1015, 797, 737, 841, 839, 869, 931, 789, 767, 547, 823, 635, 0, 0, 0, 0,
    855, 567, 591, 1019, 745, 945, 769, 671, 803, 799, 925, 701, 517, 653, 885, 731, 581, 0, 0, 0,
    887, 643, 785, 611, 905, 669, 703, 1017, 575, 763, 625, 869, 731, 861, 847, 941, 933, 577, 0, 0,
    867, 991, 1021, 709, 599, 741, 933, 921, 619, 789, 957, 791, 969, 525, 591, 763, 657, 683, 829, 0,
    1009, 1003, 901, 715, 643, 803, 805, 975, 667, 619, 569, 769, 685, 767, 853, 671, 881, 907, 955, 523,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # h=11, w=1..20
    1655, 1493, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1859, 1481, 1119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1395, 1737, 1973, 1259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1339, 1067, 1679, 1641, 2021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1657, 1331, 1783, 2043, 1097, 1485, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1611, 1141, 1849, 2001, 1511, 1359, 1245, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1215, 1733, 1461, 2025, 1251, 1945, 1649, 1851, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1275, 1373, 1841, 1509, 1631, 1737, 1055, 1891, 1041, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1715, 1117, 1503, 2025, 1027, 1959, 1365, 1739, 1301, 1233, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1101, 1127, 1145, 1157, 1195, 1747, 1885, 1527, 1325, 2033, 1935, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1369, 1255, 1809, 1889, 1183, 1495, 1223, 1781, 2029, 1327, 1075, 1065, 0, 0, 0, 0, 0, 0, 0, 0,
    1157, 1499, 1871, 1365, 1559, 1149, 1293, 1571, 1641, 1971, 1807, 1673, 2023, 0, 0, 0, 0, 0, 0, 0,
    1929, 1533, 1135, 1359, 1547, 1723, 1529, 1107, 1273, 1879, 1709, 1141, 1897, 1161, 0, 0, 0, 0, 0, 0,
    1861, 1801, 1675, 1699, 1103, 1665, 1657, 1287, 1459, 2047, 1181, 1835, 1085, 1377, 1511, 0, 0, 0, 0, 0,
    1915, 1753, 1945, 1391, 1205, 1867, 1895, 1439, 1719, 1185, 1685, 1139, 1229, 1791, 1821, 1295, 0, 0, 0, 0,
    1193, 1951, 1469, 1737, 1047, 1227, 1989, 1717, 1735, 1643, 1857, 1965, 1405, 1575, 1907, 1173, 1299, 0, 0, 0,
    1641, 1887, 1129, 1357, 1543, 1279, 1687, 1975, 1839, 1775, 1109, 1337, 1081, 1435, 1603, 2037, 1249, 1153, 0, 0,
    1999, 1065, 1387, 1977, 1555, 1915, 1219, 1469, 1889, 1933, 1819, 1315, 1319, 1693, 1143, 1361, 1815, 1109, 1631, 0,
    1253, 1051, 1827, 1871, 1613, 1759, 2015, 1229, 1585, 1057, 1409, 1831, 1943, 1491, 1557, 1195, 1339, 1449, 1675, 1679,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # h=12, w=1..20
    3475, 2685, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3865, 2883, 2519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4019, 3383, 3029, 2397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2725, 3703, 3391, 2235, 2669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2489, 3151, 2695, 3353, 4029, 3867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2467, 2137, 3047, 3881, 3125, 2683, 3631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2739, 3163, 2137, 4031, 2967, 3413, 3749, 2301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3443, 2305, 3365, 2231, 2127, 3697, 3535, 4041, 2621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3641, 2777, 2789, 2357, 3003, 2729, 3229, 2925, 3443, 2291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    3567, 2361, 2061, 2219, 3905, 2285, 2871, 3187, 2455, 2783, 2685, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4043, 2615, 2385, 3911, 3267, 2871, 3667, 3037, 2905, 2921, 2129, 2299, 0, 0, 0, 0, 0, 0, 0, 0,
    2315, 2997, 3743, 2729, 3117, 2297, 2585, 3141, 3283, 3943, 3613, 3345, 4047, 0, 0, 0, 0, 0, 0, 0,
    3967, 3069, 3377, 3909, 3691, 2439, 2533, 3075, 2129, 3319, 3433, 3035, 2745, 2631, 0, 0, 0, 0, 0, 0,
    3023, 3349, 2111, 2385, 3907, 3959, 3425, 3801, 2135, 2671, 2637, 2977, 2999, 3107, 2277, 0, 0, 0, 0, 0,
    2713, 2695, 3447, 2537, 2685, 3755, 3953, 3901, 3193, 3107, 2407, 3485, 2097, 3091, 2139, 2261, 0, 0, 0, 0,
    3065, 4059, 2813, 3043, 2849, 3477, 3205, 3381, 2747, 3203, 3937, 3603, 3625, 3559, 3831, 2243, 2343, 0, 0, 0,
    3999, 3183, 2717, 2307, 2103, 3353, 2761, 2541, 2375, 2327, 3277, 2607, 3867, 3037, 2163, 2261, 3649, 2929, 0, 0,
    2543, 2415, 3867, 3709, 3161, 2369, 4087, 2205, 3785, 2515, 2133, 2913, 3941, 3371, 2605, 3269, 3385, 3025, 2323, 0,
    2939, 2775, 3663, 2413, 2573, 2205, 3821, 3513, 2699, 3379, 2479, 2663, 2367, 2517, 3027, 3201, 3177, 3281, 4069, 2069
], dtype=np.int32).reshape(6, 20, 20) # (height-7), (width-1), submatrix_cols

class STCSampler:
    """
    Implements the Syndrome-Trellis Coded Sampler algorithm.
    This version is adapted for bit-plane based steganography. It constructs a large,
    sparse parity-check matrix H from submatrices and performs sampling on a long
    sequence of bits (a bit-plane).

    The H matrix can be constructed in three ways:
    1. 'from_precomputed': Using pre-computed non-singular submatrices from a static pool (MATS).
       This is fast and deterministic.
    2. 'random_walk': Generating submatrices on-the-fly using a pseudo-random process
       that ensures certain desirable properties (non-singular, full rank).
    3. 'convolutional': Using a single submatrix whose columns are used cyclically, with a
       state transition typical for convolutional codes (y_next = (y_prev << 1) ^ ...).
       The submatrix width 'w' is determined by the payload c/n.

    The complexity is O(n * 2^h), where n is the block length and h is the
    constraint height of the parity check matrix H.
    This version uses a NON-WRAP-AROUND (non-toroidal) parity-check matrix.

    Optimizations:
    1. JIT Compilation: Core forward/backward passes are compiled with Numba.
    2. Memory Efficiency: Does not store the full H matrix in memory.
    3. Caching: Caches the generated matrix parameters to speed up repeated runs
       with the same configuration.
    """
    def __init__(self, c: int, h: int, n: int, matrix_seed: int = 42, sample_seed: int = 123, H_construction_method: str = 'convolutional'):
        if not (7 <= h <= 12):
            raise ValueError(f"Constraint height 'h' must be between 7 and 12 for all construction methods.")
        if not (1 <= c <= n):
            raise ValueError(f"Payload 'c' must be between 1 and {n}.")
        if H_construction_method not in ['random_walk', 'from_precomputed', 'convolutional']:
            raise ValueError("H_construction_method must be one of 'random_walk', 'from_precomputed', or 'convolutional'.")

        self.n = n
        self.c = c
        self.h = h
        self.rng_matrix = np.random.default_rng(matrix_seed)
        self.rng_sample = np.random.default_rng(sample_seed)
        self.h_mask = (1 << h) - 1
        self.num_states = 1 << h
        self.H_construction_method = H_construction_method
        self.submatrix_cols = None
        self.effects, self.check_effects, self.check_lengths = None, None, None

        # --- Transition parameters ---
        # For block methods, 'b' defines the sliding window. For conv method, it defines when to embed.
        self.b = np.floor(np.arange(self.n + 1) * self.c / self.n).astype(np.int32)

        cache_key = (c, h, n, matrix_seed, H_construction_method)
        if cache_key in _stc_matrix_cache:
            self.effects, self.check_effects, self.check_lengths, self.submatrix_cols = _stc_matrix_cache[cache_key]
        else:
            if self.H_construction_method == 'convolutional':
                w_avg = self.n / self.c
                k = int(np.floor(w_avg))
                k1 = k + 1

                # Determine the number of blocks of width k and k+1 to fill the width n.
                # We solve the system of linear equations:
                # 1) num_k * k + num_k1 * (k + 1) = n
                # 2) num_k + num_k1 = c
                # The unique integer solution is:
                num_k1_blocks = self.n - self.c * k
                num_k_blocks = self.c * (k + 1) - self.n

                if num_k_blocks < 0 or num_k1_blocks < 0:
                     raise ValueError(f"Could not find a valid mixture of submatrix widths for n={n}, c={c}.")

                self.block_widths = np.array([k] * num_k_blocks + [k1] * num_k1_blocks, dtype=int)
                self.rng_matrix.shuffle(self.block_widths)
                
                # Overwrite self.b for this specific mode to reflect the diagonal block structure
                b_conv = np.zeros(self.n + 1, dtype=np.int32)
                col_idx = 0
                row_offset = 0
                for block_width in self.block_widths:
                    # For all columns within this block, the row_offset is the same.
                    for _ in range(block_width):
                        if col_idx < self.n:
                            b_conv[col_idx] = row_offset
                            col_idx += 1
                        else:
                            break # Should not happen if block_widths sum to n
                    row_offset += 1 # Move down by 1 for the next block
                    if col_idx >= self.n:
                        break
                
                # The final b[n] should be the offset for the block that would start after n-1
                while col_idx <= self.n:
                    b_conv[col_idx] = row_offset
                    col_idx += 1
                self.b = b_conv

                self.submatrix_k = self._get_submatrix_as_matrix(k, h)
                self.submatrix_k1 = self._get_submatrix_as_matrix(k1, h)
                self.submatrix_cols = None # Not used
                
                # For the new convolutional mode, we must compute effects from the full H matrix.
                self.effects, self.check_effects, self.check_lengths = self._precompute_effects_convolutional()
                _stc_matrix_cache[cache_key] = (self.effects, self.check_effects, self.check_lengths, None)

            else: # 'random_walk' or 'from_precomputed'
                w = self.h * 2
                self.submatrix_cols = self._get_submatrix_cols_for_width(w, self.h)
                self.effects, self.check_effects, self.check_lengths = self._precompute_effects_from_submatrices_block()
                _stc_matrix_cache[cache_key] = (self.effects, self.check_effects, self.check_lengths, self.submatrix_cols)

    def _get_submatrix_as_matrix(self, w: int, h: int) -> np.ndarray:
        """Gets a submatrix as a full h x w numpy array."""
        if w <= 0:
            return np.zeros((h, 0), dtype=np.int32)
        
        cols_1d = self._get_submatrix_cols_for_width(w, h)
        matrix = np.zeros((h, w), dtype=np.int32)
        for i, col_val in enumerate(cols_1d):
            for j in range(h):
                if (col_val >> (h - 1 - j)) & 1:
                    matrix[j, i] = 1
        return matrix

    def _get_submatrix_cols_for_width(self, w: int, h: int) -> np.ndarray:
        """Helper to get columns for a specific width, from MATS or random."""
        if w == 0:
            return np.array([], dtype=np.int32)
        if 2 <= w <= 20 and 7 <= h <= 12:
            return MATS[h - 7, w - 1, :w].copy()
        else:
            return self._construct_H_random_walk(w, h)

    def _construct_H_random_walk(self, width: int, height: int) -> np.ndarray:
        """
        Generates a random submatrix pool using the 'random walk' method.
        This method ensures that each column has the first and last bit set to 1,
        and all columns within the submatrix are unique. This creates a matrix
        with good pseudo-random properties, suitable for steganography.
        """
        cols = np.zeros(width, dtype=np.int32)
        
        # Mask for the middle h-2 bits
        middle_mask = (1 << (height - 2)) - 1
        
        # Bitmask to set the top (MSB) and bottom (LSB) bits to 1
        top_bottom_bop = (1 << (height - 1)) | 1

        # Check if we can generate enough unique columns
        max_unique_cols = 1 << (height - 2)
        if max_unique_cols < width:
            # Not enough unique variations for the middle bits.
            # The C++ code falls back to allowing duplicates. We will do the same.
            random_middle_parts = self.rng_matrix.integers(0, middle_mask + 1, size=width, dtype=np.int32)
            cols = (random_middle_parts << 1) | top_bottom_bop
        else:
            # Generate unique columns
            generated_cols = set()
            for i in range(width):
                while True:
                    random_middle = self.rng_matrix.integers(0, middle_mask + 1, dtype=np.int32)
                    col = (random_middle << 1) | top_bottom_bop
                    if col not in generated_cols:
                        cols[i] = col
                        generated_cols.add(col)
                        break
        return cols

    def _precompute_effects_convolutional(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precomputes effects for the 'convolutional' quasi-cyclic block code.
        This requires materializing the full H matrix in memory to derive the effects
        for the sliding window sampler.
        """
        effects = np.zeros(self.n, dtype=np.int32)
        check_effects = np.zeros(self.n, dtype=np.int32)
        check_lengths = np.zeros(self.n, dtype=np.int32)

        H = self.get_H() # Materialize the full c x n matrix

        for i in range(self.n):
            b_i = self.b[i]
            b_i_plus_1 = self.b[i+1]

            # Pruning check effect
            check_len = b_i_plus_1 - b_i
            check_lengths[i] = check_len
            if check_len > 0:
                # Extract the relevant bits from the full H column
                check_col_slice = H[b_i : b_i_plus_1, i]
                check_effects[i] = bits_to_int(check_col_slice)

            # State update effect
            val = 0
            for j in range(self.h):
                row_in_H = b_i_plus_1 + j
                bit = 0
                if row_in_H < self.c:
                    bit = H[row_in_H, i]
                val = (val << 1) | bit
            effects[i] = val
            
        return effects, check_effects, check_lengths

    def _precompute_effects_from_submatrices_block(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precomputes effects for block-based codes ('random_walk', 'from_precomputed')
        that use the b[i] sliding window.
        """
        effects = np.zeros(self.n, dtype=np.int32)
        check_effects = np.zeros(self.n, dtype=np.int32)
        check_lengths = np.zeros(self.n, dtype=np.int32)
        
        pool = self.submatrix_cols
        pool_width = len(pool)

        for i in range(self.n):
            h_col_i_int = pool[i % pool_width]
            
            b_i = self.b[i]
            b_i_plus_1 = self.b[i+1]

            check_len = b_i_plus_1 - b_i
            check_lengths[i] = check_len
            if check_len > 0:
                val = (h_col_i_int >> (self.h - check_len))
                check_effects[i] = val

            val = 0
            for j in range(self.h):
                row_in_H = b_i_plus_1 + j
                bit = 0
                if row_in_H < self.c:
                    k_in_h_col = row_in_H - b_i
                    if 0 <= k_in_h_col < self.h:
                        bit = (h_col_i_int >> (self.h - 1 - k_in_h_col)) & 1
                val = (val << 1) | bit
            effects[i] = val
        return effects, check_effects, check_lengths

    def get_H(self) -> np.ndarray:
        """
        Constructs and returns the full c x n parity-check matrix.
        """
        if self.H_construction_method in ['random_walk', 'from_precomputed']:
            # This logic correctly implements the sliding window for block codes
            H = np.zeros((self.c, self.n), dtype=int)
            pool = self.submatrix_cols
            pool_width = len(pool)
            for i in range(self.n):
                col_from_pool_int = pool[i % pool_width]
                row_start_H = self.b[i]
                for k in range(self.h):
                    if (col_from_pool_int >> (self.h - 1 - k)) & 1:
                        row_H = row_start_H + k
                        if row_H < self.c:
                            H[row_H, i] = 1
            return H

        elif self.H_construction_method == 'convolutional':
            H = np.zeros((self.c, self.n), dtype=np.int32)
            col_offset = 0
            row_offset = 0

            for w in self.block_widths:
                if w == 0: continue
                submatrix = self.submatrix_k if w == self.submatrix_k.shape[1] else self.submatrix_k1
                
                h_sub, w_sub = submatrix.shape
                
                # Determine the size of the block that can actually fit.
                w_fit = min(w_sub, self.n - col_offset)
                h_fit = min(h_sub, self.c - row_offset)

                # Stop if no part of the block can fit.
                if w_fit <= 0 or h_fit <= 0:
                    break

                # Place the part of the submatrix that fits.
                H[row_offset : row_offset + h_fit, col_offset : col_offset + w_fit] = submatrix[:h_fit, :w_fit]
                
                col_offset += w_sub
                row_offset += 1 # Move down by 1 for the next block (creates diagonal band)

            return H

    def matvec(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates H @ y % 2 without materializing H.
        This logic is now unified for all block-based methods.
        """
        # A bit slower, but guaranteed to match get_H(). For debugging and correctness.
        H = self.get_H()
        return (H @ y) % 2

    def _matvec_conv(self, stego_bits: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: This method was for a streaming convolutional code model.
        The 'convolutional' mode is now a block-based code. Use matvec instead.
        """
        raise DeprecationWarning("'_matvec_conv' is deprecated. Use 'matvec' for all H construction methods.")

    def sample_bit_plane(self, marginal_probs: np.ndarray, message_chunk: np.ndarray, calculate_posterior: bool = True, verbose: bool = False) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        # The sampling logic is now unified for all methods.
        log_marginal_probs = np.log(marginal_probs + 1e-30).astype(np.float64)

        if verbose:
            print(f"STC Fwd Sweep (c={self.c}, h={self.h}, method='{self.H_construction_method}')...")

        # --- 1. Forward Sweep (Alpha calculation) ---
        log_d, preds_y_prev, preds_s_val = stc_forward_pass(
            self.n, self.h, self.num_states, self.h_mask,
            log_marginal_probs, self.b, self.check_lengths, self.check_effects,
            self.effects, message_chunk
        )

        if np.isneginf(log_d[self.n, 0]):
            print("Sampling failed: final state y_n=0 is not reachable.")
            return None, None

        # --- 2. Backward Pass for Posteriors (Optional) ---
        posterior_probs = None
        if calculate_posterior:
            log_z_m = log_d[self.n, 0]
            if verbose:
                print("STC Bwd & Posterior...")

            posterior_probs = stc_posterior_pass(
                self.n, self.h, self.num_states, self.h_mask, log_d, log_z_m,
                log_marginal_probs, self.b, self.check_lengths, self.check_effects,
                self.effects, message_chunk
            )
            if posterior_probs is None:
                print("Sampling failed: posterior calculation returned None.")
                return None, None

        # --- 3. Backward Sweep for Sampling ---
        stego_bits = np.zeros(self.n, dtype=np.int32)
        current_y = 0

        for i in range(self.n - 1, -1, -1):
            log_weights = []
            valid_preds = []

            # Unpack predecessors for state current_y at step i+1
            y_prev_s0, y_prev_s1 = preds_y_prev[i, current_y]
            s_val_s0, s_val_s1 = preds_s_val[i, current_y]

            # Path from s_val = 0
            if y_prev_s0 != -1:
                log_d_prev = log_d[i, y_prev_s0]
                if not np.isneginf(log_d_prev):
                    log_p_trans = log_marginal_probs[i, s_val_s0]
                    log_weights.append(log_d_prev + log_p_trans)
                    valid_preds.append({'y_prev': y_prev_s0, 'val': s_val_s0})

            # Path from s_val = 1
            if y_prev_s1 != -1:
                log_d_prev = log_d[i, y_prev_s1]
                if not np.isneginf(log_d_prev):
                    log_p_trans = log_marginal_probs[i, s_val_s1]
                    log_weights.append(log_d_prev + log_p_trans)
                    valid_preds.append({'y_prev': y_prev_s1, 'val': s_val_s1})

            if not valid_preds:
                print(f"Sampling failed at bwd i={i}: no valid predecessor for state {current_y}.")
                return None, None

            log_weights = np.array(log_weights, dtype=np.float64)
            max_log_w = np.max(log_weights)
            if np.isneginf(max_log_w):
                print(f"Sampling failed at bwd i={i}: all paths have zero probability.")
                return None, None

            probabilities = np.exp(log_weights - max_log_w)
            sum_probs = np.sum(probabilities)
            if sum_probs < 1e-9:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities /= sum_probs

            chosen_idx = self.rng_sample.choice(len(valid_preds), p=probabilities)
            chosen_pred = valid_preds[chosen_idx]
            stego_bits[i] = chosen_pred['val']
            current_y = chosen_pred['y_prev']

        return stego_bits, posterior_probs

    def _sample_bit_plane_convolutional(self, marginal_probs: np.ndarray, message_chunk: np.ndarray, calculate_posterior: bool = True, verbose: bool = False) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        raise DeprecationWarning("'_sample_bit_plane_convolutional' is deprecated. Use 'sample_bit_plane' for all methods.")

# ===================================================================
# == Numba JIT-Compiled Functions
# ===================================================================

@numba.jit(nopython=True, cache=True)
def _log_sum_exp_numba(log_probs):
    if log_probs.size == 0:
        return -np.inf
    max_log = np.max(log_probs)
    if np.isneginf(max_log):
        return -np.inf
    return max_log + np.log(np.sum(np.exp(log_probs - max_log)))

@numba.jit(nopython=True, cache=True)
def stc_forward_pass(n, h, num_states, h_mask, log_marginal_probs, b,
                     check_lengths, check_effects, effects, message_chunk):

    log_d = np.full((n + 1, num_states), -np.inf, dtype=np.float64)

    preds_y_prev = np.full((n, num_states, 2), -1, dtype=np.int64)
    preds_s_val = np.full((n, num_states, 2), -1, dtype=np.int64)

    log_d[0, 0] = 0.0

    for i in range(n):
        db = b[i+1] - b[i]
        check_len = check_lengths[i]

        msg_check_val = 0
        if check_len > 0:
            msg_slice = message_chunk[b[i]:b[i+1]]
            val = 0
            for bit in msg_slice:
                val = (val << 1) | bit
            msg_check_val = val

        next_layer_log_probs_sum = np.full(num_states, -np.inf, dtype=np.float64)

        # This array is used to store predecessors for the log-sum-exp operation
        # For each (y_next, s_val), we find the most likely y_prev
        best_pred_log_prob = np.full((num_states, 2), -np.inf, dtype=np.float64)

        active_prev_states = np.where(log_d[i] > -np.inf)[0]

        for y_prev in active_prev_states:
            log_d_prev = log_d[i, y_prev]

            for s_val in range(2):
                log_p_trans = log_marginal_probs[i, s_val]
                if np.isneginf(log_p_trans): continue

                if check_len > 0:
                    state_check_val = y_prev >> (h - check_len)
                    syndrome_check = state_check_val ^ (s_val * check_effects[i])
                    if syndrome_check != msg_check_val: continue

                y_next = ((y_prev << db) & h_mask) ^ (s_val * effects[i])

                log_d_path = log_d_prev + log_p_trans

                # Accumulate for log-sum-exp
                current_sum = next_layer_log_probs_sum[y_next]
                if np.isneginf(current_sum):
                    next_layer_log_probs_sum[y_next] = log_d_path
                else:
                    if log_d_path > current_sum:
                        next_layer_log_probs_sum[y_next] = log_d_path + np.log(1 + np.exp(current_sum - log_d_path))
                    else:
                        next_layer_log_probs_sum[y_next] = current_sum + np.log(1 + np.exp(log_d_path - current_sum))

                # Find the best predecessor for this specific path (y_prev, s_val -> y_next)
                if log_d_path > best_pred_log_prob[y_next, s_val]:
                    best_pred_log_prob[y_next, s_val] = log_d_path
                    preds_y_prev[i, y_next, s_val] = y_prev
                    preds_s_val[i, y_next, s_val] = s_val

        log_d[i+1] = next_layer_log_probs_sum

    return log_d, preds_y_prev, preds_s_val


@numba.jit(nopython=True, cache=True)
def stc_posterior_pass(n, h, num_states, h_mask, log_d, log_z_m,
                       log_marginal_probs, b, check_lengths, check_effects,
                       effects, message_chunk):

    log_beta = np.full((n + 1, num_states), -np.inf, dtype=np.float64)
    all_q_posterior = np.zeros((n, 2), dtype=np.float64)
    log_beta[n, 0] = 0.0

    for i in range(n - 1, -1, -1):
        db = b[i+1] - b[i]
        check_len = check_lengths[i]
        msg_check_val = 0
        if check_len > 0:
            msg_slice = message_chunk[b[i]:b[i+1]]
            val = 0
            for bit in msg_slice:
                val = (val << 1) | bit
            msg_check_val = val

        posterior_log_prob_paths = np.full(2, -np.inf, dtype=np.float64)

        # Correct Beta Calculation
        for y_prev in range(num_states):
            paths = np.full(2, -np.inf, dtype=np.float64)
            for s_val in range(2):
                log_p_trans = log_marginal_probs[i, s_val]
                if np.isneginf(log_p_trans): continue
                if check_len > 0:
                    state_check_val = y_prev >> (h - check_len)
                    if state_check_val ^ (s_val * check_effects[i]) != msg_check_val: continue
                y_next = ((y_prev << db) & h_mask) ^ (s_val * effects[i])
                if not np.isneginf(log_beta[i+1, y_next]):
                    paths[s_val] = log_p_trans + log_beta[i+1, y_next]
            log_beta[i, y_prev] = _log_sum_exp_numba(paths)

        # Correct Posterior Calculation
        for y_prev in range(num_states):
             log_alpha_prev = log_d[i, y_prev]
             if np.isneginf(log_alpha_prev): continue
             for s_val in range(2):
                log_p_trans = log_marginal_probs[i, s_val]
                if np.isneginf(log_p_trans): continue
                if check_len > 0:
                    state_check_val = y_prev >> (h - check_len)
                    if state_check_val ^ (s_val * check_effects[i]) != msg_check_val: continue
                y_next = ((y_prev << db) & h_mask) ^ (s_val * effects[i])
                if not np.isneginf(log_beta[i+1, y_next]):
                    full_path_prob = log_alpha_prev + log_p_trans + log_beta[i+1, y_next]
                    current_sum = posterior_log_prob_paths[s_val]
                    if np.isneginf(current_sum):
                        posterior_log_prob_paths[s_val] = full_path_prob
                    else:
                        if full_path_prob > current_sum:
                            posterior_log_prob_paths[s_val] = full_path_prob + np.log(1 + np.exp(current_sum - full_path_prob))
                        else:
                            posterior_log_prob_paths[s_val] = current_sum + np.log(1 + np.exp(full_path_prob - current_sum))

        # Finalize posterior for step i
        log_q0 = posterior_log_prob_paths[0] - log_z_m
        log_q1 = posterior_log_prob_paths[1] - log_z_m

        q0 = np.exp(log_q0)
        q1 = np.exp(log_q1)
        q_sum = q0 + q1
        if q_sum > 1e-9:
            all_q_posterior[i, 0] = q0 / q_sum
            all_q_posterior[i, 1] = q1 / q_sum
        else:
            all_q_posterior[i, 0] = 0.5
            all_q_posterior[i, 1] = 0.5

    return all_q_posterior

# ===================================================================
# == Numba JIT-Compiled Functions for Convolutional STC
# ===================================================================

@numba.jit(nopython=True, cache=True)
def stc_forward_pass_conv(n, c, h, num_states, h_mask, log_marginal_probs,
                          submatrix_cols, message_chunk):
    # This function is now deprecated as the 'convolutional' model has been unified
    # with the block-based model. This stub is left to prevent import errors but should not be called.
    raise RuntimeError("stc_forward_pass_conv is deprecated.")

@numba.jit(nopython=True, cache=True)
def stc_posterior_pass_conv(n, c, h, num_states, h_mask, log_d, log_z_m,
                            log_marginal_probs, submatrix_cols, message_chunk):
    # This function is now deprecated as the 'convolutional' model has been unified
    # with the block-based model. This stub is left to prevent import errors but should not be called.
    raise RuntimeError("stc_posterior_pass_conv is deprecated.")


if __name__ == "__main__":
    # ===================================================================
    # == Example code to generate and visualize H matrices ==
    # ===================================================================
    # --- 1. Set parameters for the H matrix you want to see ---
    n_param = 10      # Matrix width (block length)
    c_param = 10       # Matrix height (payload)
    h_param = 7        # Constraint height
    matrix_seed_param = 42 # Seed for matrix construction

    # --- 2. Choose the construction method ---
    # Options: 'random_walk', 'from_precomputed', 'convolutional'
    method = 'convolutional'

    print(f"Preparing H matrix with '{method}' method...")
    print(f"Parameters: n={n_param}, c={c_param}, h={h_param}, seed={matrix_seed_param}")

    # --- 3. Instantiate STCSampler ---
    # This is fast as H is not stored in memory.
    try:
        sampler_for_test = STCSampler(
            c=c_param,
            h=h_param,
            n=n_param,
            matrix_seed=matrix_seed_param,
            H_construction_method=method
        )

        # --- 4. Get the H matrix (generated on demand) ---
        print("\nGenerating H matrix on demand...")
        H_matrix = sampler_for_test.get_H()

        # --- 5. (Optional) Print H matrix to console ---
        if n_param <= 64 and c_param <= 64:
            print("\nGenerated H matrix:")
            # Format printing for better readability
            for row in H_matrix:
                print("".join(map(str, row)))
        else:
            print("\nH matrix is too large to print to console.")

        # --- 6. Save H matrix to a text file ---
        output_filename = f"H_matrix/H_matrix_{method}_n{n_param}_c{c_param}_h{h_param}_seed{matrix_seed_param}.txt"
        np.savetxt(output_filename, H_matrix, fmt='%d', delimiter='')
        print(f"\nH matrix successfully saved to: {output_filename}")
        print("You can open this file to see the sparse, banded structure.")

    except (ValueError, ImportError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure 'numba' is installed: pip install numba") 