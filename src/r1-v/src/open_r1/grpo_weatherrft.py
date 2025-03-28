# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset
from transformers import Qwen2VLForConditionalGeneration
from PIL import Image

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainerWeatherRFT, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerWeatherRFT
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import sys

sys.path.append('/home/kaiyu/Graduation/WeatherRFT/src')

from eval.weather_rft_dataset_loader import WeatherRFTDataset
from api.call_api import api_client
from utils.data_process import force_parse_json


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "length", "related", "fluency_logical"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_language: Optional[str] = field(
        default="cn",
        metadata={"help": "Language of the dataset"},
    )
    weather_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the data"},
    )
    weather_image_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the image data"},
    )
    exclude_category: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of reward functions. Possible values: '500hpa_situation', '850hpa_situation', 'land_situation', 'rain', 'phenomena', 'max_temp', 'min_temp'"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion matches the correct answer choice (A/B/C/D)."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            # solution 本来就是 A/B/C/D 中的一个
            ground_truth = sol
            
            # 使用 re.findall 获取所有匹配项
            matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)

            # 检查是否有匹配项
            if matches:
                # 取最后一个匹配项
                content_answer = matches[-1].strip()
            else:
                sentences = re.split(r'[。！？.!?]', content)
                # 去除空的句子
                sentences = [s.strip() for s in sentences if s.strip()]
                content_answer = sentences[-1] if sentences else ""
            
            # 正则表达式提取出 A/B/C/D 中的一个
            student_answer_match = re.search(r'[A-D]', content_answer)
            student_answer = student_answer_match.group(0) if student_answer_match else ""

            # 将 ground_truth 和 student_answer 转换为大写
            ground_truth = ground_truth.upper().strip()
            student_answer = student_answer.upper().strip()
            
            # 判断对错
            if student_answer == ground_truth:
                # 若 content_answer 就已经是 A/B/C/D 中的一个，则直接奖励 1.0
                if content_answer == student_answer:
                    reward = 1.0
                else:
                    reward = 0.5
                    
        except Exception:
            pass  # Keep reward as 0.0 if matching fails
            
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    def compute_score(content):
        # 计算 <think> 和 <answer> 的数量
        single_think = content.count("<think>") == 1 and content.count("</think>") == 1
        single_answer = content.count("<answer>") == 1 and content.count("</answer>") == 1

        # 判断是否有 <think> 和 <answer> 标签
        has_think = bool(re.search(r'^<think>.*?</think>', content, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>$', content, re.DOTALL))

        # 判断是否同时有 <think> 和 <answer> 标签
        has_think_answer = bool(re.search(r"^<think>.*?</think>\s*<answer>.*?</answer>$", content, re.DOTALL))
        
        # 计算分数
        score = 0.0
        if has_think and single_think:
            score += 0.25
        if has_answer and single_answer:
            score += 0.25
        if has_think_answer and single_think and single_answer:
            score += 0.5
        
        return score

    return [compute_score(completion[0]["content"]) for completion in completions]


def length_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific length."""

    completion_contents = [completion[0]["content"] for completion in completions]
    
    # Calculate lengths and extract answers in a single pass
    think_lengths = [
        len(content.strip()) - len(matches[-1].strip() if (matches := re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)) else "")
        for content in completion_contents
    ]
    
    # Calculate think lengths
    min_think = min(think_lengths)
    max_think = max(think_lengths)

    # Normalize think lengths
    return [(think_len - min_think) / max_think for think_len in think_lengths]


def call_local_api(query):
    local_client = api_client("local")
    model_name = "/home/kaiyu/model/Qwen/Qwen2.5-32B-Instruct-AWQ/"
    response = local_client.call_text_api(query=query, 
                                            temperature=0, 
                                            return_json=True, 
                                            model=model_name)
    return response


def call_local_api_return_list(query, num):
    # print(query)
    response = call_local_api(query)
    response = force_parse_json(response)
    if isinstance(response, dict) and "scores" in response and isinstance(response["scores"], list) and len(response["scores"]) == num:
        for score in response["scores"]:
            if not isinstance(score, (int, float)):
                return [0] * num
        return response["scores"]
    else: 
        # 返回长度为 num 的全 0 列表
        return [0] * num


RELATED_SCORE_PROMPT_CN = """你是一个气象领域的资深评估专家，请针对答案与分析的相关程度，对以下{num}个作答进行打分。
分数可选：0分，不相关；0.5分，部分相关，1分，高度相关。
请以 json 格式直接输出分数，输出格式：{{"scores": [xxx, xxx, xxx, xxx, xxx, ...]}}
原始题目：{prompt}
输入：{contents}
输出："""

RELATED_SCORE_PROMPT_EN = """You are a senior evaluation expert in the meteorological field. Please rate the relevance of the answer and analysis for the following {num} answers.
Score options: 0 points, irrelevant; 0.5 points, partially relevant; 1 point, highly relevant.
Please output the scores directly in json format, output format: {{"scores": [xxx, xxx, xxx, xxx, xxx, ...]}}
Prompt: {prompt}
Input: {contents}
Output: """

FLUENCY_LOGICAL_SCORE_PROMPT_CN = """你是一个气象领域的资深评估专家，请针对文本的流畅度与逻辑性，对以下{num}个作答进行打分。
分数可选：0分，语言混乱、多语言混杂、逻辑不通；0.5分，语言流畅、逻辑不通，1分，语言流畅，逻辑合理。
请以 json 格式直接输出分数，输出格式：{{"scores": [xxx, xxx, xxx, xxx, xxx, ...]}}
原始题目：{prompt}
输入：{contents}
输出："""

FLUENCY_LOGICAL_SCORE_PROMPT_EN = """You are a senior evaluation expert in the meteorological field. Please rate the fluency and logic of the text for the following {num} answers.
Score options: 0 points, multilingualism and language confusion, illogical; 0.5 points, fluent language, illogical; 1 point, fluent language, logical.
Please output the scores directly in json format, output format: {{"scores": [xxx, xxx, xxx, xxx, xxx, ...]}}
Prompt: {prompt}
Input: {contents}
Output: """


def related_reward(completions, prompts, **kwargs):
    prompt = prompts[0][0]['content'][1]['text']
    contents = [completion[0]["content"] for completion in completions]
    num = len(contents)
    query = RELATED_SCORE_PROMPT_EN.format(prompt=prompt, contents=contents, num=num)
    return call_local_api_return_list(query, num)


def fluency_logical_reward(completions, prompts, **kwargs):
    prompt = prompts[0][0]['content'][1]['text']
    contents = [completion[0]["content"] for completion in completions]
    num = len(contents)
    query = FLUENCY_LOGICAL_SCORE_PROMPT_EN.format(prompt=prompt, contents=contents, num=num)
    return call_local_api_return_list(query, num)


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
    "related": related_reward,
    "fluency_logical": fluency_logical_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    weather_rft_dataset_train = WeatherRFTDataset(weather_rft_path=script_args.weather_path, image_rft_path=script_args.weather_image_path, split='train', prompt_type='r1', language=script_args.data_language, exclude_category=script_args.exclude_category)
    # weather_rft_dataset_validation = WeatherRFTDataset(weather_rft_path=script_args.weather_path, image_rft_path=script_args.weather_image_path, split='validation', prompt_type='r1', language=script_args.data_language, exclude_category=script_args.exclude_category)


    train_dataset = Dataset.from_dict({key: [d[key] for d in weather_rft_dataset_train] for key in weather_rft_dataset_train[0]})
    # eval_dataset = Dataset.from_dict({key: [d[key] for d in weather_rft_dataset_validation] for key in weather_rft_dataset_validation[0]})


    def process_example(example):
        prompt = example.get("prompt")
        image = Image.open(example.get("image_path"))
        solution = example.get("answer")
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "image": image,
            "solution": solution
        }


    train_dataset = train_dataset.map(process_example)
    eval_dataset = eval_dataset.map(process_example)

    
    trainer_cls = Qwen2VLGRPOTrainerWeatherRFT if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerWeatherRFT
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
