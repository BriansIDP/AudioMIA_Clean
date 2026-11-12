import torch
import numpy as np


def get_meta_metrics(input_ids, probabilities, log_probabilities, q=2.0, r=1.0):
    entropies = []
    all_prob = []
    modified_entropies = []
    max_prob = []
    gap_prob = []
    renyi_05 = []
    renyi_2 = []
    losses = []
    modified_entropies_alpha05 = []
    modified_entropies_alpha2 = []
    sharma_mittal_entropies = []
    epsilon = 1e-10

    input_ids_processed = input_ids
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :] 
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)
        
        entropy = -(token_probs * token_log_probs).sum().item() 
        entropies.append(entropy)

        token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

        alpha = 0.5
        renyi_05_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_05.append(renyi_05_)
        alpha = 2
        renyi_2_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_2.append(renyi_2_)

        max_p = token_log_probs.max().item()
        vals = token_log_probs[token_log_probs != token_log_probs.max()]

        if vals.numel() == 0:
            second_p = max_p
        else:
            second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()

        gap_p = max_p - second_p
        gap_prob.append(gap_p)
        max_prob.append(max_p)

        mink_p = token_log_probs[token_id].item()
        all_prob.append(mink_p)

        cross_entropy_loss = -mink_p
        losses.append(cross_entropy_loss)

        p_y = token_probs_safe[token_id].item()
        modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
        modified_entropies.append(modified_entropy)

        token_probs_remaining = torch.cat((token_probs_safe[:token_id], token_probs_safe[token_id+1:]))
        
        for alpha in [0.5,2]:
            entropy = - (1 / abs(1 - alpha)) * (
                (1-p_y)* p_y**(abs(1-alpha))\
                    - (1-p_y)
                    + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                    - torch.sum(token_probs_remaining)
                    ).item() 
            if alpha==0.5:
                modified_entropies_alpha05.append(entropy)
            if alpha==2:
                modified_entropies_alpha2.append(entropy)
        epsilon = 1e-10 
        if abs(q - r) < epsilon: 
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sm_entropy = (1 / (1 - q + epsilon)) * (sum_q - 1)
        elif abs(r - 1) < epsilon:  
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sum_q = torch.clamp(sum_q, min=epsilon)  
            sm_entropy = (1 / (1 - q + epsilon)) * torch.log(sum_q)
        elif abs(q - 1) < epsilon and abs(r - 1) < epsilon: 
            probs_safe = torch.clamp(token_probs_safe, min=epsilon)
            sm_entropy = -torch.sum(probs_safe * torch.log(probs_safe))
        else:  
            sum_q = torch.sum(torch.pow(torch.clamp(token_probs_safe, min=epsilon), q))
            sum_q = torch.clamp(sum_q, min=epsilon) 
            exponent = (1 - r) / (1 - q + epsilon)
            if abs(exponent) > 100:  
                exponent = torch.tensor(exponent, dtype=torch.float32)
                exponent = torch.clamp(exponent, -100, 100)
            sm_entropy = (1 / (1 - r + epsilon)) * (sum_q ** exponent - 1)
        
        sharma_mittal_entropies.append(sm_entropy.item())
    overall_sm_entropy = get_overall_sharma_mittal_entropy(probabilities, q, r)
    loss = np.nanmean(losses)

    return {
        "ppl": np.exp(loss),
        "all_prob": all_prob,
        "loss": loss,
        "entropies": entropies,
        "modified_entropies": modified_entropies,
        "max_prob": max_prob,
        "probabilities": probabilities,
        "log_probs" : log_probabilities,
        "gap_prob": gap_prob,
        "renyi_05": renyi_05,
        "renyi_2": renyi_2,
        "mod_renyi_05" : modified_entropies_alpha05,
        "mod_renyi_2" : modified_entropies_alpha2,
        "sequence_sm_entropy": overall_sm_entropy,
        "sm_entropy": sharma_mittal_entropies
    }


def get_overall_sharma_mittal_entropy(probabilities, q, r):
    avg_probs = torch.mean(probabilities, dim=0)      
    avg_probs = avg_probs / avg_probs.sum()       
    sm_entropy_all = compute_sharma_mittal_entropy(avg_probs, q, r)

    return sm_entropy_all.item()


def compute_sharma_mittal_entropy(prob_dist, q, r, epsilon=1e-10):
    prob_dist = torch.clamp(prob_dist, min=epsilon, max=1 - epsilon)
    if abs(q - r) < epsilon:
        sum_q = torch.sum(prob_dist ** q)
        sm_entropy = (1 / (1 - q + epsilon)) * (sum_q - 1)
    elif abs(r - 1) < epsilon:
        sum_q = torch.sum(prob_dist ** q)
        sum_q = torch.clamp(sum_q, min=epsilon)
        sm_entropy = (1 / (1 - q + epsilon)) * torch.log(sum_q)
    elif abs(q - 1) < epsilon and abs(r - 1) < epsilon:
        sm_entropy = -torch.sum(prob_dist * torch.log(prob_dist))
    else:
        sum_q = torch.sum(prob_dist ** q)
        sum_q = torch.clamp(sum_q, min=epsilon)
        exponent = (1 - r) / (1 - q + epsilon)
        if abs(exponent) > 100:
            exponent = torch.tensor(exponent, dtype=torch.float32)
            exponent = torch.clamp(exponent, -100, 100)
        sm_entropy = (1 / (1 - r + epsilon)) * (sum_q ** exponent - 1)

    return sm_entropy


def get_img_metric(
    ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, 
    gap_p, renyi_05, renyi_2, log_probs, mod_renyi_05, mod_renyi_2, sequence_vetp, vetp, sequence_vetp_inverse, vetp_inverse
):
    pred = {}

    # Perplexity
    pred["ppl"] = ppl

    # Min-K% Probability Scores
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9]:
        k_length = max(1, int(len(all_prob) * ratio))
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    # Modified entropy metrics
    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()
    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()
    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    # Probability gap metric
    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()

    # Max-K% Renyi Entropy Values
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(renyi_05) * ratio))

        pred[f"Max_{ratio*100}% renyi_05"] = np.mean(np.sort(renyi_05)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_1"] = np.mean(np.sort(entropies)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_2"] = np.mean(np.sort(renyi_2)[-k_length:]).item()
        pred[f"Max_{ratio*100}% renyi_inf"] = np.mean(np.sort(-np.array(max_p))[-k_length:]).item()

    # Min-K% Renyi Entropy Values
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(renyi_05) * ratio))

        pred[f"Min_{ratio*100}% renyi_05"] = np.mean(np.sort(renyi_05)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_1"] = np.mean(np.sort(entropies)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_2"] = np.mean(np.sort(renyi_2)[:k_length]).item()
        pred[f"Min_{ratio*100}% renyi_inf"] = np.mean(np.sort(-np.array(max_p))[:k_length]).item()

    # Sharma–Mittal Entropy
    vetp_diff = [x - y for x, y in zip(vetp, vetp_inverse)]
    pred["Max_vetp"] = np.max(vetp_diff).item()
    pred["Min_vetp"] = np.min(vetp_diff).item()
    pred["Mean_vetp"] = np.mean(vetp_diff).item()
    for ratio in [0, 0.05, 0.1, 0.3, 0.6, 0.9, 1]:
        k_length = max(1, int(len(vetp) * ratio))
        pred[f"Max_{ratio*100}% vetp"] = np.mean(np.sort(vetp_diff)[-k_length:]).item()
        pred[f"Min_{ratio*100}% vetp"] = np.mean(np.sort(vetp_diff)[:k_length]).item()
    pred["sequence_vetp"] = sequence_vetp - sequence_vetp_inverse
    return pred
