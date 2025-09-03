# -*- coding: utf-8 -*-

import os
import re
import warnings
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import argparse


import torch
import torch.nn as nn


from transformers import (
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    BlipProcessor, BlipForConditionalGeneration,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model, TaskType


from sentence_transformers import SentenceTransformer


from utils import set_seed, remove_redundant_captions, print_parameter_summary


# 추론 실행 함수
def run_prediction_batch(batch_df, seed, data_base_path, models_and_processors, device):
    # 모델 및 프로세서 언패킹
    caption_model = models_and_processors['caption_model']
    caption_processor = models_and_processors['caption_processor']
    similarity_model = models_and_processors['similarity_model']
    final_model = models_and_processors['final_model']
    processor = models_and_processors['processor']

    set_seed(seed)
    questions = batch_df["Question"].tolist()
    image_paths = [os.path.join(data_base_path, path) for path in batch_df['img_path']]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    batch_size = len(batch_df)

    # 캡션 생성
    caption_inputs = caption_processor(images=images, return_tensors="pt").to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        caption_outputs = caption_model.generate(**caption_inputs, do_sample=True, temperature=1.2, top_k=50, top_p=0.95, max_length=50, num_return_sequences=5)
    decoded_captions = caption_processor.batch_decode(caption_outputs, skip_special_tokens=True)
    image_captions = []
    for i in range(batch_size):
        captions_per_image = decoded_captions[i * 5 : (i + 1) * 5]
        diverse_captions = remove_redundant_captions(captions_per_image, similarity_model)
        image_captions.append(". ".join(diverse_captions))

    # VQA 정답 생성
    choices_list = batch_df[["A", "B", "C", "D"]].values.tolist()
    prompts = [f"Based on the image and the caption: \"{cap}\", answer the following question.\nQuestion: {q}\nChoices:\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(ch)]) + "\nAnswer:" for q, ch, cap in zip(questions, choices_list, image_captions)]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = final_model.generate(**inputs, max_length=10)
    raw_answers = processor.batch_decode(outputs, skip_special_tokens=True)
    final_predictions = [re.search(r"[A-D]", ans.strip().upper()).group() if re.search(r"[A-D]", ans.strip().upper()) else "A" for ans in raw_answers]
    return final_predictions


# 메인 실행 함수
def main(args):
    warnings.filterwarnings('ignore')
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 모델 로딩
    print("\n모델을 로딩합니다...")
    vqa_model_path = os.path.join(args.model_dir, "instructblip-flan-t5-xl")
    lm_path = os.path.join(args.model_dir, "flan-t5-large")
    caption_model_path = os.path.join(args.model_dir, "blip-image-captioning-large")
    similarity_model_path = os.path.join(args.model_dir, "all-mpnet-base-v2")

    instructblip_xl = InstructBlipForConditionalGeneration.from_pretrained(vqa_model_path, torch_dtype=torch.bfloat16)
    print("1. InstructBLIP-XL (뼈대) 로딩 완료")

    flan_large = AutoModelForSeq2SeqLM.from_pretrained(lm_path, torch_dtype=torch.bfloat16, use_safetensors=True)
    print("2. Flan-T5-Large (언어모델) 로딩 완료")

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    flan_large_lora = get_peft_model(flan_large, lora_config)
    qformer_dim, lm_dim = instructblip_xl.qformer.config.hidden_size, flan_large_lora.config.hidden_size
    instructblip_xl.language_projection = nn.Linear(qformer_dim, lm_dim, bias=False)
    instructblip_xl.language_model = flan_large_lora
    instructblip_xl.config.text_config = flan_large.config
    print("3. 모델 아키텍처 수정 완료")

    final_model = instructblip_xl.to(dtype=torch.bfloat16, device=device)
    processor = InstructBlipProcessor.from_pretrained(vqa_model_path)
    print("4. 최종 VQA 모델 및 프로세서 준비 완료")

    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_path, torch_dtype=torch.bfloat16).to(device)
    caption_processor = BlipProcessor.from_pretrained(caption_model_path)
    print("5. 캡셔닝 모델 및 프로세서 준비 완료")

    similarity_model = SentenceTransformer(similarity_model_path).to(device)
    print("6. 유사도 모델 준비 완료")

    models_and_processors = {
        'final_model': final_model,
        'processor': processor,
        'caption_model': caption_model,
        'caption_processor': caption_processor,
        'similarity_model': similarity_model
    }
    print("모든 모델 로딩 완료!")

    # 추론 실행
    print("\n메인 추론을 시작합니다...")
    test_path = os.path.join(args.data_dir, "test.csv")
    test_df = pd.read_csv(test_path).reset_index(drop=True)

    for model in models_and_processors.values():
        if hasattr(model, 'eval'):
            model.eval()

    BATCH_SIZE = 64
    SEEDS = [42, 123, 1024, 2048, 4242]
    all_seed_predictions = []

    for seed in SEEDS:
        print(f"\n   Running Predictions for SEED: {seed}")
        current_seed_preds = []
        for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc=f" Inference (Seed: {seed})"):
            batch_df = test_df.iloc[i:i+BATCH_SIZE]
            preds = run_prediction_batch(batch_df, seed, args.data_dir, models_and_processors, device)
            current_seed_preds.extend(preds)
        all_seed_predictions.append(current_seed_preds)

    # 결과 종합 및 저장
    print("\n앙상블 예측 결과를 Voting 방식으로 종합합니다...")
    pred_df = pd.DataFrame(all_seed_predictions).T
    pred_df.columns = [f'seed_{s}' for s in SEEDS]
    final_ensemble_preds = pred_df.mode(axis=1)[0].tolist()

    submission_df = pd.DataFrame({'ID': test_df['ID'], 'answer': final_ensemble_preds})

    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)

    print(f"\n 제출 파일이 저장되었습니다: {submission_path}")
    print("\n[제출 파일 샘플]")
    print(submission_df.head())

    # 파라미터 검증
    print_parameter_summary(
        models_and_processors['final_model'],
        models_and_processors['caption_model'],
        models_and_processors['similarity_model']
    )


# 스크립트 실행 지점
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA Challenge Inference Script")
    parser.add_argument('--data_dir', type=str, default='./data', help='test.csv와 이미지 폴더가 있는 데이터 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='./models', help='사전 학습된 모델 가중치가 저장된 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='./', help='결과물(submission.csv)을 저장할 디렉토리 경로')

    args = parser.parse_args()
    main(args)