import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

parser = argparse.ArgumentParser(description='Simple LLM Finetuner')

parser.add_argument('--models',
    nargs='+',
    default=[
        # 'kfkas/Llama-2-ko-7b-Chat',
        'beomi/llama-2-ko-7b',
        'meta-llama/Llama-2-7b-hf',
    ],
    help='사용가능한 모델 리스트 (gpu T4 기준으로는 7b를 추천드립니다.)'
)

# train confing
parser.add_argument('--num_train_epochs', type=int, default=1, help='epochs 수')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='./result', help='저장위치')
parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='훈련당 train gpu batch size')
parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help='저장되는 모델이름 설정')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--optim', type=str, default='paged_adamw_32bit', help='사용할 Optimizer 선택')
parser.add_argument('--save_steps', type=int, default=0, help='업데이트시 체크포인트 저장')
parser.add_argument('--logging_steps', type=int, default=25, help='log 업데이트 step')
parser.add_argument('--weight_decay', type=float, default=0.001, help='레이어에 적용할 가중치 감소율')
parser.add_argument('--fp16', type=bool, default=False, help='T4는 지원안함 A100은 지원')
parser.add_argument('--bf16', type=bool, default=False, help='T4는 지원안함 A100은 지원')
parser.add_argument('--max_grad_norm', type=float, default= 0.3, help='Maximum gradient normal')
parser.add_argument('--max_steps', type=float, default=-1, help='Number of training steps (overrides num_train_epochs)')
parser.add_argument('--warmup_ratio', type=float, default= 0.03, help='선형 준비 단계 비율')
parser.add_argument('--group_by_length', type=bool, default=True, help='시퀀스 같은 길이의 배치로 그룹화 시켜줌, 메모리 절약 + 훈련속도 높여줌')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate schedule 선택 linear 등 존재')
parser.add_argument('--report_to', type=str, default='tensorboard', help='학습상황 monitoring ("azure_ml","clearml","codecarbon","comet_ml" 등 존재)')

# lora confing
parser.add_argument('--lora_r', type=int, default=64, help='LORA r')
parser.add_argument('--lora_alpha', type=int, default=16, help='LORA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LORA dropout')
parser.add_argument('--lora_bias', type=str, default="none", help='LORA bias')
parser.add_argument('--lora_task_type', type=str, default="CAUSAL_LM", help='LORA task_type')

# SFT confing
parser.add_argument('--SFT_dataset_text_field', type=str, default='text', help='dataset_text_field')
parser.add_argument('--SFT_packing', type=bool, default=False, help='입력 순서로 여러 개의 짧은 예제를 압축할지 여부 선택')

# gradio 관련
parser.add_argument('--share', action='store_true', default=False, help='Whether to deploy the interface with Gradio')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host name or IP to launch Gradio webserver on')
parser.add_argument('--port', type=int, default=8000, help='Host port to launch Gradio webserver on')

# other
parser.add_argument('--dataset_name', type=str, default='mlabonne/guanaco-llama2-1k', help='사용할 데이터셋 선택')
parser.add_argument('--bnb_4bit_compute_dtype', type=str, default='float16', help='dtype 설정')
parser.add_argument('--use_4bit', type=bool, default=True, help='4bit 사용 여부 설정')
parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', help='fp4 or nf4')
parser.add_argument('--use_nested_quant', type=bool, default=False, help='중첩 양자화 활성여부')
parser.add_argument('--save_pretrained_name', type=str, default='llama2-7b-burgerking', help='저장되는 모델이름 설정')
parser.add_argument('--device-map', type=str, default='', help='사용할 GPU 선택 없으면 자동 0')
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='사용할 모델 선택')


# GENERATION_PARAMS
parser.add_argument('--max_length', type=int, default=150 , help='max_length 설정')
parser.add_argument('--top_p', type=float, default=0 , help='top_p 설정')
parser.add_argument('--top_k', type=int, default=3, help='top_k 설정')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature 설정')

args = parser.parse_args()



################### 위의 인자값 기준으로 조립

GENERATION_PARAMS = {
    'max_length' : args.max_length,
    'top_p' : args.top_p,
    'top_k' : args.top_k,
    'temperature' : args.temperature
}


train_confing_dict = {
    'num_train_epochs' : args.num_train_epochs,
    'learning_rate' : args.learning_rate,
    'output_dir' : args.output_dir,
    'per_device_train_batch_size' : args.per_device_train_batch_size,
    'per_device_eval_batch_size' : args.per_device_eval_batch_size,
    'gradient_accumulation_steps' : args.gradient_accumulation_steps,
    'optim' : args.optim,
    'save_steps' : args.save_steps,
    'logging_steps' : args.logging_steps,
    'weight_decay' : args.weight_decay,
    'fp16' : args.fp16,
    'bf16' : args.bf16,
    'max_grad_norm' : args.max_grad_norm,
    'max_steps' : args.max_steps,
    'warmup_ratio' : args.warmup_ratio,
    'group_by_length' : args.group_by_length,
    'lr_scheduler_type' : args.lr_scheduler_type,
    'report_to' : args.report_to
}

lora_confing_dict = {
    'lora_r' : args.lora_r,
    'lora_alpha' : args.lora_alpha,
    'lora_dropout' : args.lora_dropout,
    'lora_bias' : args.lora_bias,
    'lora_task_type' : args.lora_task_type
}

sft_confing_dict = {
    'SFT-dataset_text_field' : args.SFT_dataset_text_field,
    'SFT-packing' : args.SFT_packing
}

other_conding_dict = {
    'dataset_name' : args.dataset_name,
    'bnb_4bit_compute_dtype' : args.bnb_4bit_compute_dtype,
    'use_4bit' : args.use_4bit,
    'bnb_4bit_quant_type' : args.bnb_4bit_quant_type,
    'use_nested_quant' : args.use_nested_quant,
    'save_pretrained_name' : args.save_pretrained_name,
}




SHARE = args.share
SERVER_HOST = args.host
SERVER_PORT = args.port

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

MODELS = args.models
DEVICE_MAP = {'': 0} if not args.device_map else args.device_map
MODEL = args.model

