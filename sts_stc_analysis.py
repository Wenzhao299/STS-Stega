import numpy as np
from stc_sampler import STCSampler, stc_forward_pass as stc_forward_pass_stc
from stscp_sampler_new import STSSampler, stc_forward_pass as stc_forward_pass_stscp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 统计函数：第step_i步嵌入mi时所有满足H*y=mi的概率和，并输出所有可达状态的概率和、所有满足隐写条件的路径数量、满足隐写约束节点的归一化概率和
def step_prob_sum(type, sampler, marginal_probs, message_chunk, step_i, mi, print_states_sum=True):
    n, h, num_states, h_mask = sampler.n, sampler.h, sampler.num_states, sampler.h_mask
    b = sampler.b
    check_lengths = sampler.check_lengths
    check_effects = sampler.check_effects
    effects = sampler.effects
    
    log_marginal_probs = np.log(marginal_probs + 1e-30).astype(np.float64)
    if type == 'stc':
        log_d, preds_y_prev, preds_s_val = stc_forward_pass_stc(
            n, h, num_states, h_mask,
            log_marginal_probs, b, check_lengths, check_effects,
            effects, message_chunk
        )
    elif type == 'sts':
        log_d, preds_y_prev, preds_s_val = stc_forward_pass_stscp(
            n, h, num_states, h_mask,
            log_marginal_probs, b, check_lengths, check_effects,
            effects, message_chunk
        )

    prob_sum = 0.0
    for y_prev in range(num_states):
        log_d_prev = log_d[step_i, y_prev]
        if np.isneginf(log_d_prev):
            continue
        check_len = check_lengths[step_i]
        msg_check_val = 0
        valid = True
        if check_len > 0:
            msg_slice = message_chunk[b[step_i]:b[step_i+1]]
            val = 0
            for bit in msg_slice:
                val = (val << 1) | bit
            msg_check_val = val
            state_check_val = y_prev >> (h - check_len)
            syndrome_check = state_check_val ^ (mi * check_effects[step_i])
            if syndrome_check != msg_check_val:
                valid = False
        if valid:
            prob_sum += np.exp(log_d_prev) * marginal_probs[step_i, mi]
    # 统计所有满足隐写条件的路径数量（128*2条转移中syndrome校验通过的条数）
    valid_transition_count = 0
    db = b[step_i+1] - b[step_i]
    check_len = check_lengths[step_i]
    msg_check_val = 0
    if check_len > 0:
        msg_slice = message_chunk[b[step_i]:b[step_i+1]]
        val = 0
        for bit in msg_slice:
            val = (val << 1) | bit
        msg_check_val = val
    for y_prev in range(num_states):
        for s_val in [0, 1]:
            valid = True
            if check_len > 0:
                state_check_val = y_prev >> (h - check_len)
                syndrome_check = state_check_val ^ (s_val * check_effects[step_i])
                if syndrome_check != msg_check_val:
                    valid = False
            if valid:
                valid_transition_count += 1
    if print_states_sum:
        state_probs = np.exp(log_d[step_i, :])
        total_state_prob = np.sum(state_probs)
        # 统计满足隐写约束的节点的概率和
        valid_state_prob_sum = 0.0
        for y_prev in range(num_states):
            log_d_prev = log_d[step_i, y_prev]
            if np.isneginf(log_d_prev):
                continue
            check_len = check_lengths[step_i]
            msg_check_val = 0
            valid = True
            if check_len > 0:
                msg_slice = message_chunk[b[step_i]:b[step_i+1]]
                val = 0
                for bit in msg_slice:
                    val = (val << 1) | bit
                msg_check_val = val
                state_check_val = y_prev >> (h - check_len)
                syndrome_check = state_check_val ^ (mi * check_effects[step_i])
                if syndrome_check != msg_check_val:
                    valid = False
            if valid:
                valid_state_prob_sum += np.exp(log_d_prev)
        norm_valid_state_prob = valid_state_prob_sum / total_state_prob if total_state_prob > 0 else 0.0
        print(f"{type.upper()}第{step_i}步所有满足隐写条件的路径数量: {valid_transition_count}")
        print(f"{type.upper()}第{step_i}步所有可达状态的概率之和: {total_state_prob}")
        print(f"{type.upper()}第{step_i}步所有满足隐写约束节点的归一化概率和: {norm_valid_state_prob}")
    return prob_sum

def get_secret_bit_steps(sampler):
    check_lengths = sampler.check_lengths
    return [i for i, clen in enumerate(check_lengths) if clen > 0]

def get_valid_state_prob_sum(type, sampler, marginal_probs, message_chunk, step_i, mi):
    n, h, num_states, h_mask = sampler.n, sampler.h, sampler.num_states, sampler.h_mask
    b = sampler.b
    check_lengths = sampler.check_lengths
    check_effects = sampler.check_effects
    effects = sampler.effects
    log_marginal_probs = np.log(marginal_probs + 1e-30).astype(np.float64)
    if type == 'stc':
        log_d, _, _ = stc_forward_pass_stc(
            n, h, num_states, h_mask,
            log_marginal_probs, b, check_lengths, check_effects,
            effects, message_chunk
        )
    elif type == 'sts':
        log_d, _, _ = stc_forward_pass_stscp(
            n, h, num_states, h_mask,
            log_marginal_probs, b, check_lengths, check_effects,
            effects, message_chunk
        )
    state_probs = np.exp(log_d[step_i, :])
    total_state_prob = np.sum(state_probs)
    valid_state_prob_sum = 0.0
    valid_states = []
    valid_probs = []
    for y_prev in range(num_states):
        log_d_prev = log_d[step_i, y_prev]
        if np.isneginf(log_d_prev):
            continue
        check_len = check_lengths[step_i]
        msg_check_val = 0
        valid = True
        if check_len > 0:
            msg_slice = message_chunk[b[step_i]:b[step_i+1]]
            val = 0
            for bit in msg_slice:
                val = (val << 1) | bit
            msg_check_val = val
            state_check_val = y_prev >> (h - check_len)
            syndrome_check = state_check_val ^ (mi * check_effects[step_i])
            if syndrome_check != msg_check_val:
                valid = False
        if valid:
            prob = np.exp(log_d_prev)
            valid_state_prob_sum += prob
            valid_states.append(y_prev)
            valid_probs.append(prob)
    norm_valid_state_prob = valid_state_prob_sum / total_state_prob if total_state_prob > 0 else 0.0
    print(f"满足隐写约束的状态编号: {valid_states}")
    print(f"满足隐写约束的状态概率: {valid_probs}")
    print(f"满足隐写约束的概率和: {valid_state_prob_sum}")
    print(f"当前步所有状态的概率和: {total_state_prob}")
    # 可选：打印所有状态的概率（只打印前10个）
    print("当前步所有状态的概率(前10个):", state_probs[:10])
    return norm_valid_state_prob

def single_bit_prob(type, sampler, marginal_probs, message_chunk, secret_idx):
    secret_steps = get_secret_bit_steps(sampler)
    if secret_idx >= len(secret_steps):
        raise ValueError(f'秘密信息位编号超出范围，最大为{len(secret_steps)-1}')
    step_i = secret_steps[secret_idx]
    mi = message_chunk[secret_idx]
    norm_prob = get_valid_state_prob_sum(type, sampler, marginal_probs, message_chunk, step_i, mi)
    return norm_prob

def full_message_stats(type, sampler, marginal_probs, message_chunk):
    secret_steps = get_secret_bit_steps(sampler)
    valid_state_probs = []
    for idx, step_i in enumerate(secret_steps):
        mi = message_chunk[idx]
        norm_prob = get_valid_state_prob_sum(type, sampler, marginal_probs, message_chunk, step_i, mi)
        valid_state_probs.append(norm_prob)
    mean_val = np.mean(valid_state_probs)
    var_val = np.var(valid_state_probs)
    return mean_val, var_val

def trial_worker(type, c, h, n, matrix_seed, sample_seed, marginal_probs, seed):
    np.random.seed(seed)
    message_chunk = np.random.randint(0, 2, size=(c,))
    if type == 'sts':
        sampler = STSSampler(c=c, h=h, n=n, matrix_seed=matrix_seed, sample_seed=sample_seed)
    else:
        sampler = STCSampler(c=c, h=h, n=n, matrix_seed=matrix_seed, sample_seed=sample_seed)
    mean_val, _ = full_message_stats(type, sampler, marginal_probs, message_chunk)
    return mean_val

def multi_trial_stats(type, c, h, n, matrix_seed, sample_seed, marginal_probs, num_trials=100):
    seeds = np.random.randint(0, 2**32-1, size=num_trials)
    mean_list = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(trial_worker, type, c, h, n, matrix_seed, sample_seed, marginal_probs, int(seed))
            for seed in seeds
        ]
        for f in tqdm(as_completed(futures), total=num_trials):
            mean_val = f.result()
            mean_list.append(mean_val)
    return np.mean(mean_list), np.var(mean_list)

def main():
    h = 7
    n = 1000
    c = 1000
    p = 0.3
    matrix_seed = 42
    sample_seed = 123
    marginal_probs = np.tile([[1-p, p]], (n, 1))
    # 随机消息
    # np.random.seed(42424242)
    message_chunk = np.random.randint(0, 2, size=(c,))
    print('--- 单个秘密信息位归一化概率 ---')
    sts_sampler = STSSampler(c=c, h=h, n=n, matrix_seed=matrix_seed, sample_seed=sample_seed)
    stc_sampler = STCSampler(c=c, h=h, n=n, matrix_seed=matrix_seed, sample_seed=sample_seed)
    secret_idx = 100  # 可修改
    prob_sts = single_bit_prob('sts', sts_sampler, marginal_probs, message_chunk, secret_idx)
    prob_stc = single_bit_prob('stc', stc_sampler, marginal_probs, message_chunk, secret_idx)
    print(f'STS第{secret_idx}位归一化概率: {prob_sts}')
    print(f'STC第{secret_idx}位归一化概率: {prob_stc}')
    # print('\n--- 完整隐写均值和方差 ---')
    # mean_sts, var_sts = full_message_stats('sts', sts_sampler, marginal_probs, message_chunk)
    # mean_stc, var_stc = full_message_stats('stc', stc_sampler, marginal_probs, message_chunk)
    # print(f'STS: 均值={mean_sts}, 方差={var_sts}')
    # print(f'STC: 均值={mean_stc}, 方差={var_stc}')
    # print('\n--- 多轮次均值的均值和方差 ---')
    # num_trials = 100
    # mean_sts_mt, var_sts_mt = multi_trial_stats('sts', c, h, n, matrix_seed, sample_seed, marginal_probs, num_trials)
    # mean_stc_mt, var_stc_mt = multi_trial_stats('stc', c, h, n, matrix_seed, sample_seed, marginal_probs, num_trials)
    # print(f'STS多轮次: 均值的均值={mean_sts_mt}, 方差={var_sts_mt}')
    # print(f'STC多轮次: 均值的均值={mean_stc_mt}, 方差={var_stc_mt}')

if __name__ == '__main__':
    main()