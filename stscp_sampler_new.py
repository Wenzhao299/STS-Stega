import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numba

# --- Global cache for STC matrices and effects ---
# This cache stores the computationally expensive results of STC matrix construction.
# Key: (c, h, n, matrix_seed) -> Value: (effects, check_effects, check_lengths, submatrix_cols)
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

class STSSampler:
    """
    Implements the Syndrome-Trellis Sampler algorithm.
    This version is adapted for bit-plane based steganography. It constructs a large,
    sparse parity-check matrix H from pre-computed non-singular submatrices and
    performs sampling on a long sequence of bits (a bit-plane).
    The complexity is O(n * 2^h), where n is the block length and h is the
    constraint height of the parity check matrix H.
    This version uses a NON-WRAP-AROUND (non-toroidal) parity-check matrix.

    Optimizations:
    1. JIT Compilation: Core forward/backward passes are compiled with Numba.
    2. Memory Efficiency: Does not store the full H matrix in memory. Trellis paths
       are stored in compact NumPy arrays instead of Python dicts/lists.
    3. Enhanced Randomness: Uses separate seeds for matrix construction and path sampling.
    """
    def __init__(self, c: int, h: int, n: int, matrix_seed: int = 42, sample_seed: int = 123):
        if not (7 <= h <= 12):
            raise ValueError(f"Constraint height 'h' must be between 7 and 12 to use the precomputed MATS.")
        if not (1 <= c <= n):
            raise ValueError(f"Payload 'c' must be between 1 and {n}.")
        
        self.n = n
        self.c = c
        self.h = h
        self.rng_sample = np.random.default_rng(sample_seed)
        self.h_mask = (1 << h) - 1
        self.num_states = 1 << h

        # --- Transition parameters ---
        self.b = np.floor(np.arange(self.n + 1) * self.c / self.n).astype(np.int32)
        
        cache_key = (c, h, n, matrix_seed)
        if cache_key in _stc_matrix_cache:
            self.effects, self.check_effects, self.check_lengths, self.submatrix_cols = _stc_matrix_cache[cache_key]
        else:
            self.rng_matrix = np.random.default_rng(matrix_seed)
            w = self.h * 2
            if not (1 <= w <= 20):
                raise ValueError(f"Submatrix width 'w'={w} must be between 1 and 20 for MATS.")
            
            submatrix_cols = MATS[self.h - 7, w - 1].copy()
            # self.rng_matrix.shuffle(submatrix_cols) # DO NOT shuffle. Original implementation does not. This was the bug.
            self.submatrix_cols = submatrix_cols

            self.effects, self.check_effects, self.check_lengths = self._precompute_effects_from_submatrices()
            _stc_matrix_cache[cache_key] = (self.effects, self.check_effects, self.check_lengths, self.submatrix_cols)

    def _precompute_effects_from_submatrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precomputes the effects directly from the submatrix pool without building the full H.
        This is crucial for memory efficiency.
        """
        effects = np.zeros(self.n, dtype=np.int32)
        check_effects = np.zeros(self.n, dtype=np.int32)
        check_lengths = np.zeros(self.n, dtype=np.int32)
        w = self.h * 2
        
        for i in range(self.n):
            h_col_i = self.submatrix_cols[i % w]
            b_i = self.b[i]
            b_i_plus_1 = self.b[i+1]

            # Pruning check is on rows [b_i, b_{i+1})
            check_len = b_i_plus_1 - b_i
            check_lengths[i] = check_len
            if check_len > 0:
                val = (h_col_i >> (self.h - check_len))
                check_effects[i] = val

            # State update effect is from H[:,i] on the next window [b_{i+1}, b_{i+1}+h)
            val = 0
            for j in range(self.h): # j is the bit index in the effect vector
                row_in_H = b_i_plus_1 + j
                bit = 0
                if row_in_H < self.c: # Check boundary against total rows c
                    k_in_h_col = row_in_H - b_i
                    if 0 <= k_in_h_col < self.h:
                        # If the row is within this column's non-zero entries
                        bit = (h_col_i >> (self.h - 1 - k_in_h_col)) & 1
                val = (val << 1) | bit
            effects[i] = val
        
        return effects.astype(np.int32), check_effects.astype(np.int32), check_lengths.astype(np.int32)

    def get_H(self) -> np.ndarray:
        """
        Constructs and returns the full H matrix on-demand.
        Warning: This can be very slow and memory-intensive for large n and c.
        It is primarily for debugging or verification.
        """
        H = np.zeros((self.c, self.n), dtype=int)
        w = self.h * 2
        for i in range(self.n):
            col_from_pool = self.submatrix_cols[i % w]
            row_start_H = self.b[i]
            for k in range(self.h):
                if (col_from_pool >> (self.h - 1 - k)) & 1:
                    row_H = row_start_H + k
                    if row_H < self.c:
                        H[row_H, i] = 1
        return H

    def matvec(self, y: np.ndarray) -> np.ndarray:
        """Calculates H @ y % 2 without materializing H."""
        res = np.zeros(self.c, dtype=np.int32)
        w = self.h * 2
        for i in np.flatnonzero(y): # Iterate only over non-zero elements of y
            col_from_pool = self.submatrix_cols[i % w]
            row_start_H = self.b[i]
            for k in range(self.h):
                if (col_from_pool >> (self.h - 1 - k)) & 1:
                    row_H = row_start_H + k
                    if row_H < self.c:
                        res[row_H] ^= 1
        return res

    def sample_bit_plane(self, marginal_probs: np.ndarray, message_chunk: np.ndarray, calculate_posterior: bool = True, verbose: bool = False) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        log_marginal_probs = np.log(marginal_probs + 1e-30).astype(np.float64)
        
        if verbose:
            print(f"STS Fwd Sweep (c={self.c}, h={self.h})...")

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
                print("STS Bwd & Posterior...")

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
                    path_log_prob = log_p_trans + log_beta[i+1, y_next]
                    
                    # This calculation is for beta[i, y_prev], but we are iterating y_prev.
                    # This part needs restructuring to calculate beta efficiently.
                    # The current posterior calculation is also flawed due to this loop structure.
                    # For now, let's focus on a correct beta calculation first.
                    pass # Placeholder

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

if __name__ == "__main__":
    # ===================================================================
    # == 用于生成和可视化H矩阵的示例代码 ==
    # ===================================================================
    # --- 1. 设置您想查看的H矩阵的参数 ---
    n_param = 10      # 矩阵宽度 (块长)
    c_param = 10       # 矩阵高度 (负载)
    h_param = 7       # 约束高度
    matrix_seed_param = 42 # 用于矩阵构造的种子

    print(f"正在准备 H 矩阵 (非回绕版本)，参数: n={n_param}, c={c_param}, h={h_param}, matrix_seed={matrix_seed_param}")

    # --- 2. 实例化 STSSampler ---
    # 它不会在内存中存储H，所以即使n,c很大，实例化也很快。
    try:
        sampler_for_test = STSSampler(c=c_param, h=h_param, n=n_param, matrix_seed=matrix_seed_param)

        # --- 3. 获取 H 矩阵 (按需生成) ---
        # 注意: get_H() 会在调用时动态构建完整的H矩阵，可能消耗大量内存和时间！
        print("\n按需生成 H 矩阵...")
        H_matrix = sampler_for_test.get_H()

        # --- 4. (可选) 在控制台打印 H 矩阵 ---
        # 注意：如果 n 和 c 很大，打印会非常慢且难以阅读。
        # 建议只对 n < 50, c < 50 的情况使用。
        if n_param <= 50 and c_param <= 50:
            print("\n生成的 H 矩阵:")
            print(H_matrix)
        else:
            print("\nH 矩阵尺寸过大，不建议在控制台打印。")

        # --- 5. 将 H 矩阵保存到文本文件 ---
        # 这是最有用的部分，您可以随时打开文件查看矩阵结构。
        # 矩阵中的 0 和 1 会被保存，非常直观。
        output_filename = f"H_matrix/H_matrix_n{n_param}_c{c_param}_h{h_param}_seed{matrix_seed_param}_non_wrap.txt"
        np.savetxt(output_filename, H_matrix, fmt='%d', delimiter='')
        print(f"\nH 矩阵已成功保存到文件: {output_filename}")
        print("您可以打开此文件查看矩阵的稀疏带状结构。")

    except (ValueError, ImportError) as e:
        print(f"\n发生错误: {e}")
        print("请确保已安装 'numba' 库: pip install numba")