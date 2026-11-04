import json
import sys, os
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
import random
# from bert_score import score


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
report_metrics = [
    "ppl",
    "Min_10.0% Prob",
    "Min_30.0% Prob",
    "Min_60.0% Prob",
    "Modified_entropy",
    "Max_10.0% renyi_05",
    "Max_30.0% renyi_05",
    "Max_60.0% renyi_05",
    "Max_10.0% renyi_2",
    "Max_30.0% renyi_2",
    "Max_60.0% renyi_2",
    "Max_10.0% vetp",
    "Max_30.0% vetp",
    "Max_60.0% vetp",
]
# combined metrics
to_combine = [
    "Min_30.0% Prob",
    "Min_60.0% Prob",
    "Max_30.0% renyi_2",
    "Max_60.0% renyi_2",
]

try:
    infile = sys.argv[1]
except:
    raise Exception("Usage: python score.py /path/to/output/file.json [optional: /path/to/reference/file.json]")
useref = False
per_audio = True
use_max = True
use_ratio = True
score_topk = 2
start_topk = 0
lse_tau = 0.02

# Change total questions to be >0 to enable top-M most similar questions to training set
total_questions = 0
sim_questions = {}
with open("data/question_to_affinity_qwen25omni.json") as fin:
    affinity = json.load(fin)
topk_affinity = {}
for audio, questions in affinity.items():
    qscores = []
    for question, score in questions.items():
        qscores.append([question, score])
    qscores = sorted(qscores, key=lambda x: x[1], reverse=True)
    sim_questions[audio] = [q[0] for q in qscores[:total_questions]]


if len(sys.argv) > 2:
    reffile = sys.argv[2]
    with open(reffile) as fin:
        refdata = json.load(fin)
outfile = infile.replace(".json", "_results.json")

def compare_with_orig(metric, orig_metric):
    if use_ratio:
        return metric / (orig_metric + 1e-5)
    else:
        return metric - orig_metric


def process_score(scores):
    distance_k = 200
    return 1 - np.sort(scores)[:distance_k].sum() / distance_k


def get_tpr_at_fpr(y_true, y_scores, target_fpr=0.05):
    # Get the FPR and TPR at various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the index where FPR is closest to or just under the target
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]


with open(infile) as fin:
    data = json.load(fin)

ref_all_metrics = {}

if refdata is not None:
    for datapiece in tqdm(refdata):
        key = datapiece["audio"] + datapiece["question"]
        ref_all_metrics[key] = {}
        speed_factor = datapiece["pred"].pop("speed_factor", None)
        noise_snr = datapiece["pred"].pop("noise_snr", None)
        for metric, value in datapiece["pred"].items():
            if metric != "pred":
                value = min(100000, max(-10000, value))
                ref_all_metrics[key][metric] = value

labels = []
all_metrics = {}
all_metrics_audio = {}
audio_to_label = {}

for idx, datapiece in enumerate(tqdm(data)):
    if datapiece["audio"] in sim_questions and datapiece["question"] not in sim_questions[datapiece["audio"]]:
        continue
    labels.append(datapiece["label"])
    if datapiece["audio"] not in audio_to_label:
        audio_to_label[datapiece["audio"]] = datapiece["label"]
    speed_factor = datapiece["pred"].pop("speed_factor", None)
    noise_snr = datapiece["pred"].pop("noise_snr", None)
    for metric, value in datapiece["pred"].items():
        key = datapiece["audio"] + datapiece["question"]
        audiokey = datapiece["audio"]
        if metric == "pred":
            continue
        else:
            if metric not in all_metrics:
                all_metrics[metric] = []
            if per_audio:
                if audiokey not in all_metrics_audio:
                    all_metrics_audio[audiokey] = {}
                if metric not in all_metrics_audio[audiokey]:
                    all_metrics_audio[audiokey][metric] = []
            value = min(100000, max(-10000, value))
            if "min_k_pp" in metric:
                value = - value
            if key in ref_all_metrics and metric in ref_all_metrics[key]:
                if ref_all_metrics[key][metric] == 0:
                    ref_all_metrics[key][metric] += 1e-5
                if per_audio:
                    all_metrics_audio[audiokey][metric].append(compare_with_orig(value, ref_all_metrics[key][metric]))
                all_metrics[metric].append(compare_with_orig(value, ref_all_metrics[key][metric]))
            else:
                if per_audio:
                    all_metrics_audio[audiokey][metric].append(value)
                all_metrics[metric].append(value)

all_metrics_audio_mean = {}
all_metrics_audio_max = {}
print(len(all_metrics_audio))
all_audios = []
if per_audio:
    labels = []
    for audiokey, metrics in all_metrics_audio.items():
        all_audios.append(audiokey)
        labels.append(audio_to_label[audiokey])
        for metric, values in metrics.items():
            if metric not in all_metrics_audio_mean:
                all_metrics_audio_mean[metric] = []
                all_metrics_audio_max[metric] = []
            all_metrics_audio_mean[metric].append(sum(values) / len(values))
            if score_topk > 0:
                if total_questions > 0:
                    random.shuffle(values)
                    values = values[:total_questions]
                # local_score_topk = min(len(values), score_topk)
                end = min(len(values), start_topk + score_topk)
                start = start_topk
                values = np.sort(values)
                values_to_consider =  values[start:end]
                all_metrics_audio_max[metric].append(values_to_consider.sum() / len(values_to_consider))
            elif lse_tau > 0:
                values = np.array(values)
                log_sum_exp = lse_tau * np.log(np.sum(np.exp(np.clip(values / lse_tau, -500, 500))) / len(values) + 1e-5)
                all_metrics_audio_max[metric].append(log_sum_exp)
            else:
                all_metrics_audio_max[metric].append(min(values))
    if score_topk > 0 or use_max:
        all_metrics = all_metrics_audio_max
    else:
        all_metrics = all_metrics_audio_mean

results = {}
audio_score = {}

print("Metric\tAUROC\tTPR @5% FPR")
for metric in report_metrics:
    values = all_metrics[metric]
    score = roc_auc_score(labels, values)
    tpr_at_low_fpr = get_tpr_at_fpr(labels, values)
    print("{}\t{:.4f}\t{:.4f}".format(metric, score, tpr_at_low_fpr))

combined = np.array(all_metrics["ppl"]) * 0
for metric in to_combine:
    combined += np.array(all_metrics[metric])
score = roc_auc_score(labels, combined)
tpr_at_low_fpr = get_tpr_at_fpr(labels, combined)
print("AUROC combined:", score)
print("TPR @5% FPR combined:", tpr_at_low_fpr)
