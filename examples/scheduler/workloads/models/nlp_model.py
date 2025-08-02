import os
import pickle
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AdamW,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    GPT2LMHeadModel, GPT2Config,
    BertForMaskedLM, BertConfig,
)

# 仅保留需要的类
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel),
    "bert": (BertConfig, BertForMaskedLM),
}

class TextDataset(Dataset):
    """
    数据加载逻辑基本不变，但可以简化，不再需要处理 HDFS。
    注意：在分布式环境中，缓存文件可能需要每个 worker 单独处理或使用共享存储。
    为简化起见，这里假设使用本地文件系统。
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path), f"Input file not found at {file_path}"
        
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, f"cached_lm_{block_size}_{filename}"
        )

        if os.path.exists(cached_features_file):
            print(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file.")
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            self.examples = []
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                )
            
            print(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def get_nlp_dataset(args, sargs):
    """加载并返回 NLP 的 Dataset 对象。"""
    model_name = sargs["model_name"]
    tokenizer_name = "bert-base-uncased" if model_name == "bert" else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 将 block_size 移到 sargs 中以保持一致
    sargs["block_size"] = tokenizer.model_max_length
    
    return TextDataset(
        tokenizer, file_path=sargs["train_dir"], block_size=sargs["block_size"]
    )

class NLPModel(nn.Module):
    def __init__(self, args, sargs):
        super().__init__()
        self.args = args
        self.sargs = sargs
        self.model_name = sargs["model_name"]
        
        config_class, model_class = MODEL_CLASSES[self.model_name]
        self.config = config_class()
        self.model = model_class(config=self.config)

        # Tokenizer 需要在外部创建，但这里保存一份用于 collate 和 mask
        tokenizer_name = "bert-base-uncased" if self.model_name == "bert" else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.mlm = self.model_name == "bert"

    def get_optimizer_and_scheduler(self, model_to_optimize, sargs):
        """定义并返回优化器和调度器。"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_to_optimize.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": sargs["weight_decay"],
            },
            {
                "params": [p for n, p in model_to_optimize.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=sargs["learning_rate"], eps=sargs["adam_epsilon"]
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=sargs["warmup_iters"], num_training_steps=sargs["iters"]
        )
        return optimizer, scheduler

    def forward_pass(self, model, inputs, labels):
        """执行前向传播并返回损失。"""
        outputs = model(inputs, labels=labels)
        return outputs.loss

    def forward(self, inputs, labels=None):
        """标准的 forward 方法，用于 DDP。"""
        return self.model(inputs, labels=labels)
    
    def collate(self, examples: List[torch.Tensor]):
        """用于 DataLoader 的 collate_fn。"""
        # MLM 逻辑移到这里，直接在批次上操作
        batch = pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return inputs, labels
        else:
            return batch, batch
            
    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为 MLM 准备掩码输入和标签。"""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels

    def print_info(self):
        print(f"Model: {self.model_name}, Batch Size: {self.sargs['batch_size']}")