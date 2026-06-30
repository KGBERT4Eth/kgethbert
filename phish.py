import argparse
import os
import datetime
import platform
import statistics
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# ==============  ============== #
class DevDataset(Dataset):
    """
     dev.tsv :
    - : (0/1)
    - : 
    """

    def __init__(self, tsv_file, tokenizer, max_len=64):
        """
        :param tsv_file: dev.tsv 
        :param tokenizer: BERT tokenizer
        :param max_len: 
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(tsv_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx == 0:
                    # , 
                    # 
                    continue
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    continue
                label = int(cols[1])
                text = cols[2]
                self.samples.append((label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        #  BERT tokenizer 
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ==============  ============== #
class BertClassifier(nn.Module):
    """
     BERT :
    -  freeze_bert  BERT 
    """

    def __init__(self, pretrained_model_path, num_labels=2, freeze_bert=False):
        """
        :param pretrained_model_path: BERT
        :param num_labels: (2)
        :param freeze_bert:  BERT 
        """
        super(BertClassifier, self).__init__()
        # BERT
        self.bert = BertModel.from_pretrained(pretrained_model_path)

        # 
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        :
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len]
        :return: logits [batch_size, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        # outputs.pooler_output: [batch_size, hidden_size] (CLS)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output


# ==============  ============== #
def compute_metrics(labels, preds, probs):
    """
    : Precision, Recall, F1, AUC-ROC, FNR
    :param labels: (listnumpy array)
    :param preds: (listnumpy array)
    :param probs: (listnumpy array)
    :return: 
    """
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    #  (probs)  AUC-ROC
    try:
        auc_roc = roc_auc_score(labels, probs)
    except ValueError:
        auc_roc = 0.0  # 

    #  FNR (False Negative Rate)
    # FNR = FN / (FN + TP)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

    #  FPR (False Positive Rate)
    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC-ROC': auc_roc,
        'FNR': fnr,
        'FPR': fpr
    }


# ==============  ============== #
def plot_training_curve(train_losses, val_losses, save_path):
    """
    
    :param train_losses:  loss (list)
    :param val_losses:  loss (list)
    :param save_path: 
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f": {save_path}")


# ============== () ============== #
def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024.0


def percentile(values, percent):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent / 100.0
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def summarize_values(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_memory_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    trainable_memory_bytes = sum(
        param.numel() * param.element_size() for param in model.parameters() if param.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    buffer_memory_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "total_param_count": total_params,
        "trainable_param_count": trainable_params,
        "non_trainable_param_count": total_params - trainable_params,
        "buffer_count": buffer_count,
        "parameter_memory_bytes": param_memory_bytes,
        "trainable_parameter_memory_bytes": trainable_memory_bytes,
        "buffer_memory_bytes": buffer_memory_bytes,
    }


def get_module_counts(model):
    counts = {}
    for module in model.modules():
        name = module.__class__.__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def profile_flops(model, sample_batch, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)
    batch_size = int(input_ids.size(0))
    result = {
        "available": False,
        "error": None,
        "profiled_batch_size": batch_size,
        "flops_per_batch": 0,
        "flops_per_sample": 0.0,
    }

    try:
        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                model(input_ids, attention_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        flops = sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        result["available"] = flops > 0
        result["flops_per_batch"] = flops
        result["flops_per_sample"] = flops / batch_size if batch_size else 0.0
    except Exception as exc:
        result["error"] = repr(exc)

    return result


def measure_inference_efficiency(model, dataloader, device, warmup_batches, max_timed_batches):
    model.eval()
    batch_latencies = []
    sample_latencies = []
    timed_samples = 0
    timed_batches = 0
    all_batches_seen = 0
    all_samples_seen = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    total_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Efficiency Inference")):
            all_batches_seen += 1
            batch_size = int(batch["input_ids"].size(0))
            all_samples_seen += batch_size
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(input_ids, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                if max_timed_batches > 0 and timed_batches >= max_timed_batches:
                    continue
                batch_latencies.append(elapsed)
                sample_latencies.append(elapsed / batch_size if batch_size else 0.0)
                timed_batches += 1
                timed_samples += batch_size

    total_wall_seconds = time.perf_counter() - total_start
    timed_inference_seconds = sum(batch_latencies)
    samples_per_second = timed_samples / timed_inference_seconds if timed_inference_seconds > 0 else 0.0
    batches_per_second = timed_batches / timed_inference_seconds if timed_inference_seconds > 0 else 0.0

    gpu_memory = {
        "available": device.type == "cuda" and torch.cuda.is_available(),
        "device_name": "",
        "allocated_bytes_after": 0,
        "reserved_bytes_after": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "total_memory_bytes": 0,
        "peak_allocated_percent": 0.0,
    }
    if gpu_memory["available"]:
        props = torch.cuda.get_device_properties(device)
        total_memory = int(props.total_memory)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        gpu_memory.update({
            "device_name": torch.cuda.get_device_name(device),
            "allocated_bytes_after": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_after": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_memory_bytes": total_memory,
            "peak_allocated_percent": 100.0 * peak_allocated / total_memory if total_memory else 0.0,
        })

    return {
        "all_eval_batches": all_batches_seen,
        "all_eval_samples": all_samples_seen,
        "warmup_batches": warmup_batches,
        "timed_batches_after_warmup": timed_batches,
        "timed_samples_after_warmup": timed_samples,
        "eval_total_wall_seconds": total_wall_seconds,
        "timed_inference_seconds": timed_inference_seconds,
        "samples_per_second": samples_per_second,
        "batches_per_second": batches_per_second,
        "milliseconds_per_batch": 1000.0 / batches_per_second if batches_per_second > 0 else 0.0,
        "milliseconds_per_sample": 1000.0 / samples_per_second if samples_per_second > 0 else 0.0,
        "batch_latency_seconds": summarize_values(batch_latencies),
        "sample_latency_seconds": summarize_values(sample_latencies),
        "gpu_memory": gpu_memory,
    }


def write_efficiency_report(model, dataloader, device, args, run_dir, phase, model_checkpoint_path=None):
    report_path = os.path.join(run_dir, args.efficiency_report_name)
    sample_batch = next(iter(dataloader), None)
    if sample_batch is None:
        raise ValueError("cannot write efficiency report for an empty dataloader")

    param_stats = count_parameters(model)
    module_counts = get_module_counts(model)
    flops = profile_flops(model, sample_batch, device)
    throughput = measure_inference_efficiency(
        model=model,
        dataloader=dataloader,
        device=device,
        warmup_batches=args.efficiency_warmup_batches,
        max_timed_batches=args.efficiency_max_batches,
    )
    estimated_flops_per_second = (
        flops["flops_per_sample"] * throughput["samples_per_second"] if flops["available"] else 0.0
    )
    checkpoint_size_bytes = (
        os.path.getsize(model_checkpoint_path)
        if model_checkpoint_path and os.path.exists(model_checkpoint_path)
        else 0
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Phish Model Efficiency Report\n")
        file.write("=" * 80 + "\n")
        file.write(f"generated_at: {datetime.datetime.now().isoformat()}\n")
        file.write(f"phase: {phase}\n")
        file.write(f"python: {sys.version}\n")
        file.write(f"platform: {platform.platform()}\n")
        file.write(f"pytorch: {torch.__version__}\n")
        file.write(f"cuda_available: {torch.cuda.is_available()}\n")
        file.write(f"device: {device}\n")

        file.write("\n[Config]\n")
        for key, value in sorted(vars(args).items()):
            file.write(f"{key}: {value}\n")

        file.write("\n[Model Size]\n")
        for key, value in param_stats.items():
            file.write(f"{key}: {value}\n")
        file.write(f"parameter_memory_readable: {format_bytes(param_stats['parameter_memory_bytes'])}\n")
        file.write(
            "trainable_parameter_memory_readable: "
            f"{format_bytes(param_stats['trainable_parameter_memory_bytes'])}\n"
        )
        file.write(f"buffer_memory_readable: {format_bytes(param_stats['buffer_memory_bytes'])}\n")
        file.write(f"checkpoint_path: {model_checkpoint_path or ''}\n")
        file.write(f"checkpoint_size_bytes: {checkpoint_size_bytes}\n")
        file.write(f"checkpoint_size_readable: {format_bytes(checkpoint_size_bytes)}\n")

        file.write("\n[FLOPs]\n")
        file.write(f"available: {flops['available']}\n")
        file.write(f"error: {flops['error']}\n")
        file.write(f"profiled_batch_size: {flops['profiled_batch_size']}\n")
        file.write(f"flops_per_batch: {flops['flops_per_batch']}\n")
        file.write(f"flops_per_sample: {flops['flops_per_sample']:.3f}\n")
        file.write(f"estimated_flops_per_second: {estimated_flops_per_second:.3f}\n")
        file.write(f"estimated_gflops_per_second: {estimated_flops_per_second / 1e9:.6f}\n")

        file.write("\n[Inference Throughput]\n")
        for key in [
            "all_eval_batches",
            "all_eval_samples",
            "warmup_batches",
            "timed_batches_after_warmup",
            "timed_samples_after_warmup",
            "eval_total_wall_seconds",
            "timed_inference_seconds",
            "samples_per_second",
            "batches_per_second",
            "milliseconds_per_batch",
            "milliseconds_per_sample",
        ]:
            file.write(f"{key}: {throughput[key]}\n")

        file.write("\n[GPU Memory]\n")
        gpu_memory = throughput["gpu_memory"]
        for key, value in gpu_memory.items():
            file.write(f"{key}: {value}\n")
        file.write(f"allocated_after_readable: {format_bytes(gpu_memory['allocated_bytes_after'])}\n")
        file.write(f"reserved_after_readable: {format_bytes(gpu_memory['reserved_bytes_after'])}\n")
        file.write(f"peak_allocated_readable: {format_bytes(gpu_memory['peak_allocated_bytes'])}\n")
        file.write(f"peak_reserved_readable: {format_bytes(gpu_memory['peak_reserved_bytes'])}\n")
        file.write(f"total_memory_readable: {format_bytes(gpu_memory['total_memory_bytes'])}\n")

        file.write("\n[Latency Seconds Per Batch]\n")
        for key, value in throughput["batch_latency_seconds"].items():
            file.write(f"{key}: {value:.9f}\n" if isinstance(value, float) else f"{key}: {value}\n")

        file.write("\n[Latency Seconds Per Sample]\n")
        for key, value in throughput["sample_latency_seconds"].items():
            file.write(f"{key}: {value:.9f}\n" if isinstance(value, float) else f"{key}: {value}\n")

        file.write("\n[Model Modules]\n")
        file.write(f"total_modules: {sum(module_counts.values())}\n")
        file.write(f"unique_module_types: {len(module_counts)}\n")
        file.write("top_module_types:\n")
        for name, count in list(module_counts.items())[:30]:
            file.write(f"  {name}: {count}\n")

    print(f"Efficiency report saved to: {report_path}")
    return report_path


def direct_evaluate(model, dataloader, device, run_dir, save_results=True):
    """
    dev
    :param model:  BertClassifier
    :param dataloader: devdataloader
    :param device: 
    :param run_dir: 
    """
    model.eval()
    preds_list = []
    labels_list = []
    probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Direct)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits,_ = model(input_ids, attention_mask)
            # 
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()

            preds_list.extend(predicted)
            labels_list.extend(labels)
            probs_list.extend(probs)

    metrics_result = compute_metrics(labels_list, preds_list, probs_list)
    result_path = os.path.join(run_dir, "direct_evaluate_results.txt")
    if save_results:
        with open(result_path, "w") as f:
            f.write(str(metrics_result))

    print(f"[Direct Evaluate] : {metrics_result}")
    print(f": {result_path}")
    return metrics_result


# ==============  ============== #
def finetune(model, train_loader, val_loader, device, epochs, lr, run_dir, save_model=True):
    """
    
    :param model:  BertClassifier
    :param train_loader:  dataloader
    :param val_loader:  dataloader
    :param device: 
    :param epochs: 
    :param lr: 
    :param run_dir: 
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, "best_ft_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits,_ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        # loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits,_ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        # 
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            if save_model:
                torch.save(model, best_model_path)
                print(f"Best model saved with val loss: {best_val_loss:.4f} in {best_model_path}")
    # 
    # curve_path = os.path.join(run_dir, "finetune_curve.png")
    # plot_training_curve(train_losses, val_losses, curve_path)

    # ( .pth )
    model_save_path = os.path.join(run_dir, "finetuned_model.pth")
    torch.save(model, model_save_path)
    print(f": {model_save_path}")


# ==============  ============== #
def linear_probe(model, train_loader, val_loader, device, epochs, lr, run_dir):
    """
     BERT 
    :param model: BertClassifier (freeze_bert=True )
    :param train_loader: 
    :param val_loader: 
    :param device: 
    :param epochs: 
    :param lr: ( BERT )
    :param run_dir: 
    """
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, "best_linear_probe_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Linear Probe Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits,_ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        # loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits,_ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f} in {best_model_path}")

    # 
    # curve_path = os.path.join(run_dir, "linear_probe_curve.png")
    # plot_training_curve(train_losses, val_losses, curve_path)

    # ( .pth )
    model_save_path = os.path.join(run_dir, "linear_probe_model.pth")
    torch.save(model, model_save_path)
    print(f": {model_save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="finetune",
                        choices=["direct", "finetune", "linear" , "aft_ft"])
    parser.add_argument("--dev_tsv", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--efficiency_warmup_batches", type=int, default=5)
    parser.add_argument("--efficiency_max_batches", type=int, default=0)
    parser.add_argument("--efficiency_report_name", type=str, default="efficiency_report.txt")
    args = parser.parse_args()

    # : Eval__data/{task}/{}/
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    run_dir = f"Eval_output/{args.task}/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running task: {args.task}")
    #  config.txt
    config_txt_path = os.path.join(run_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write("Running Configurations:\n")
        f.write(str(args) + "\n")

    # 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(config_txt_path, "a") as f:
        f.write(f"Device: {device}\n")

    #  tokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    # 
    dataset = DevDataset(tsv_file=args.dev_tsv, tokenizer=tokenizer, max_len=args.max_length)
    print(f" {args.dev_tsv}  {len(dataset)} ")
    with open(config_txt_path, "a") as f:
        f.write(f"Loaded dataset from {args.dev_tsv}, total samples: {len(dataset)}\n")

    # /
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 
    freeze_bert_flag = True if args.task == "linear" else False
    model = BertClassifier(
        pretrained_model_path=args.pretrained_path,
        num_labels=2,
        freeze_bert=freeze_bert_flag
    )
    print(f", freeze_bert={freeze_bert_flag}")
    with open(config_txt_path, "a") as f:
        f.write(f"Model loaded. freeze_bert={freeze_bert_flag}\n")

    # 
    if args.task == "direct":
        # 1) 
        model.eval()
        model.to(device)
        direct_metrics = direct_evaluate(model, val_loader, device, run_dir)
        print(f": {direct_metrics}")

        # txt
        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Direct Evaluate Results:\n")
            f.write(str(direct_metrics) + "\n")
        write_efficiency_report(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
            run_dir=run_dir,
            phase="direct",
            model_checkpoint_path=None,
        )

    elif args.task == "aft_ft":
        try:
            model = torch.load(args.model_path, map_location=device, weights_only=False)
        except TypeError:
            model = torch.load(args.model_path, map_location=device)
        model.eval()
        model.to(device)
        direct_metrics = direct_evaluate(model, val_loader, device, run_dir)
        print(f": {direct_metrics}")

        # txt
        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Evaluate Results After Supervised Training:\n")
            f.write(str(direct_metrics) + "\n")
        write_efficiency_report(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
            run_dir=run_dir,
            phase="aft_ft",
            model_checkpoint_path=args.model_path,
        )

    elif args.task == "finetune":
        # 2) 
        finetune(model, train_loader, val_loader, device,
                 epochs=args.epochs,
                 lr=args.learning_rate,
                 run_dir=run_dir)

        # :
        model.eval()
        model.to(device)
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()

                logits,_ = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(predicted)
                labels_list.extend(labels)
                probs_list.extend(probs)

        metrics_result = compute_metrics(labels_list, preds_list, probs_list)
        print(f"[Finetune Evaluate] {metrics_result}")

        #  result.txt
        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Finetune Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")
        write_efficiency_report(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
            run_dir=run_dir,
            phase="finetune",
            model_checkpoint_path=os.path.join(run_dir, "finetuned_model.pth"),
        )

    elif args.task == "linear":
        # 3) 
        linear_probe(model, train_loader, val_loader, device,
                     epochs=args.epochs,
                     lr=1e-3,  # : 
                     run_dir=run_dir)

        # :
        model.eval()
        model.to(device)
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()

                logits,_ = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(predicted)
                labels_list.extend(labels)
                probs_list.extend(probs)

        metrics_result = compute_metrics(labels_list, preds_list, probs_list)
        print(f"[Linear Probe Evaluate] {metrics_result}")

        #  result.txt
        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Linear Probe Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")
        write_efficiency_report(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
            run_dir=run_dir,
            phase="linear",
            model_checkpoint_path=os.path.join(run_dir, "linear_probe_model.pth"),
        )


if __name__ == "__main__":
    main()
