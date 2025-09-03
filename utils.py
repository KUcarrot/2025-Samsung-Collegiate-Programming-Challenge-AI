# -*- coding: utf-8 -*-


import os
import random
import numpy as np
import torch
from sentence_transformers import util

# 결과 재현을 위한 시드 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 중복 캡션을 제거하는 함수
def remove_redundant_captions(captions, model, threshold=0.85):
    if not captions: return []
    embeddings = model.encode(captions, convert_to_tensor=True)
    filtered = []
    for i, caption in enumerate(captions):
        is_similar = False
        for j in range(i):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                is_similar = True
                break
        if not is_similar:
            filtered.append(caption)
    return filtered

# 모델의 총 파라미터 수를 계산하는 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 학습된 모델들의 파라미터 개수를 출력하는 함수
def print_parameter_summary(final_model, caption_model, similarity_model):
    total_params = count_parameters(final_model) + count_parameters(caption_model) + count_parameters(similarity_model)
    print("\n" + "="*50)
    print("모델 파라미터 개수 검증")
    print("-"*50)
    print(f"  - VQA Model (LoRA-tuned): {count_parameters(final_model):,}")
    print(f"  - Caption Model: {count_parameters(caption_model):,}")
    print(f"  - Similarity Model: {count_parameters(similarity_model):,}")
    print("-"*50)
    print(f"   Total Parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    print("="*50)