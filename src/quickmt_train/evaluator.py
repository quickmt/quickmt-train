import math
import sacrebleu
import torch
import torch.distributed as dist

from .utils import unwrap_model


def validate(
    model,
    loader,
    src_sp,
    tgt_sp,
    device,
    train_cfg,
    data_cfg,
    model_cfg,
    get_time_info,
    use_autoregressive=True,
):
    """
    Validate the model on the dev/validation set.
    """
    # Free up memory before validation (especially if it involves generation)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    total_loss_sum = 0
    total_tokens = 0
    correct_tokens = 0

    hypotheses = []
    references = []

    autocast_dtype = torch.float32
    if device.type == "cuda":
        if train_cfg.precision in ("bf16", "bfloat16"):
            autocast_dtype = torch.bfloat16
        elif train_cfg.precision in ("fp16", "float16"):
            autocast_dtype = torch.float16

    with torch.inference_mode():
        for batch_idx, (src, tgt) in enumerate(loader, start=1):
            src, tgt = (
                src.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss_sum, (logits, num_tokens_batch) = model(
                    src, tgt, return_outputs=True
                )

            if loss_sum.ndim > 0:
                loss_sum = loss_sum.sum()
            if num_tokens_batch.ndim > 0:
                num_tokens_batch = num_tokens_batch.sum()

            total_loss_sum += loss_sum.item()
            total_tokens += num_tokens_batch.item()

            tgt_labels = tgt[:, 1:]
            preds = logits.argmax(dim=-1)
            mask_acc = tgt_labels != model_cfg.pad_id
            correct_tokens += ((preds == tgt_labels) & mask_acc).sum().item()

            if use_autoregressive:
                raw_model = unwrap_model(model)
                enc = raw_model.encode(src)
                generated_ids = raw_model.generate(
                    src,
                    max_len=model_cfg.max_len,
                    enc_output=enc,
                    bos_id=model_cfg.bos_id,
                    eos_id=model_cfg.eos_id,
                )
            else:
                generated_ids = preds

            for i in range(src.size(0)):
                # Stop at EOS or PAD tokens
                def cleanup_ids(ids_list, pad_id, eos_id):
                    for idx, token_id in enumerate(ids_list):
                        if token_id == eos_id or token_id == pad_id:
                            return ids_list[:idx]
                    return ids_list

                ids = cleanup_ids(
                    generated_ids[i].tolist(), model_cfg.pad_id, model_cfg.eos_id
                )
                ref_ids = cleanup_ids(
                    tgt[i].tolist(), model_cfg.pad_id, model_cfg.eos_id
                )

                hyp = tgt_sp.decode(ids)
                ref = tgt_sp.decode(ref_ids)
                hypotheses.append(hyp)
                references.append(ref)

    if dist.is_initialized():
        sync_t = torch.tensor(
            [total_loss_sum, float(total_tokens), float(correct_tokens)], device=device
        )
        dist.all_reduce(sync_t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_tokens, correct_tokens = sync_t.tolist()

        all_h, all_r = [None] * dist.get_world_size(), [None] * dist.get_world_size()
        dist.all_gather_object(all_h, hypotheses)
        dist.all_gather_object(all_r, references)
        hypotheses, references = [i for s in all_h for i in s], [
            i for s in all_r for i in s
        ]

    avg_loss = total_loss_sum / max(1, total_tokens)
    ppl, acc = math.exp(min(avg_loss, 100)), correct_tokens / max(1, total_tokens)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"\n{get_time_info()} [Validation] Loss: {avg_loss:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}"
        )
        for i in range(min(train_cfg.quick_test_samples, len(hypotheses))):
            print(
                f"Sample {i}: Ref: {references[i][:100]}... | Hyp: {hypotheses[i][:100]}..."
            )
        print("-" * 30)

    # Free up memory after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    return metrics


def run_quick_test(
    model, loader, src_sp, tgt_sp, device, model_cfg, train_cfg, get_time_info
):
    """
    Quick Test with examples from dev data, displaying target outputs.
    """
    print(
        f"\n{get_time_info()} Running final quick test on {train_cfg.quick_test_samples} dev samples:"
    )
    model.eval()

    samples_found = 0
    with torch.inference_mode():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            # Process up to n samples from this batch
            n = min(train_cfg.quick_test_samples - samples_found, src.size(0))

            for i in range(n):
                s_tensor = src[i : i + 1]
                t_tensor = tgt[i : i + 1]

                # Generate
                raw_model = unwrap_model(model)
                generated_ids = raw_model.generate(
                    s_tensor,
                    max_len=model_cfg.max_len,
                    bos_id=model_cfg.bos_id,
                    eos_id=model_cfg.eos_id,
                )

                # Decoding
                # Helper to remove padding and decode
                def cleanup_and_decode(ids_tensor, sp, pad_id, eos_id):
                    ids = ids_tensor[0].tolist()
                    # Stop at EOS or PAD tokens
                    for idx, token_id in enumerate(ids):
                        if token_id == eos_id or token_id == pad_id:
                            ids = ids[:idx]
                            break
                    return sp.decode(ids)

                s_text = cleanup_and_decode(
                    s_tensor, src_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_ref = cleanup_and_decode(
                    t_tensor, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_hyp = cleanup_and_decode(
                    generated_ids, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )

                print(f"Example {samples_found + 1}:")
                print(f"  Input:  {s_text}")
                print(f"  Ref:    {t_ref}")
                print(f"  Output: {t_hyp}")
                print()

                samples_found += 1

            if samples_found >= train_cfg.quick_test_samples:
                break
