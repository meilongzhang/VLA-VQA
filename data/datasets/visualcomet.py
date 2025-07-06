import os
import json
import random
from PIL import Image
import torch

from data.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from data.datasets.base_dataset import BaseDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict({
            "file": ann["img_fn"],
            "question": ann["question"],
            "question_id": ann["question_id"],
            "answers": "; ".join(ann["answer"]),
            "image": sample["image"],
        })


class VisualCOMETDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            answer_weight[answer] = answer_weight.get(answer, 0) + 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class VisualCOMETDataset_Raw(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        if 'test' in ann_paths[0]:
            self.jsonl_paths = os.path.join(vis_root.replace('images/visualcomet', 'visualcomet_gt'), 'test.jsonl')
        elif 'train' in ann_paths[0]:
            self.jsonl_paths = os.path.join(vis_root.replace('images/visualcomet', 'visualcomet_gt'), 'train.jsonl')
        elif 'val' in ann_paths[0]:
            self.jsonl_paths = os.path.join(vis_root.replace('images/visualcomet', 'visualcomet_gt'), 'val.jsonl')
        
        self.full_jsonl = []
        with open(self.jsonl_paths, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.full_jsonl.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print("Line content:", line)

        # index by img_fn
        self.jsonl_by_img_fn = {}
        for item in self.full_jsonl:
            fn = item["img_fn"]
            self.jsonl_by_img_fn.setdefault(fn, []).append(item)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["img_fn"])
        image_raw = Image.open(image_path).convert("RGB")

        entries = self.jsonl_by_img_fn.get(ann["img_fn"], [])
        qa = random.choice(entries) if entries else None
        
        if qa is None:
            return {
                "image_raw": image_raw,
                "text_input_raw": "",
                "question_id":"",
                "answers": [],
                "weights": [],
            }

        question_tokens = qa["question"]
        question_str = " ".join(
            str(tok) if isinstance(tok, (str, int)) else f"[{','.join(map(str, tok))}]"
            for tok in question_tokens
        )
        question = self.text_processor(question_str)
        
        answer_list = [" ".join(map(str, ans)) if isinstance(ans, list) else str(ans) for ans in qa["answer_choices"]]

        answer_weight = {}
        for answer in answer_list:
            answer_weight[answer] = answer_weight.get(answer, 0) + 1 / len(answer_list)

        return {
            "image_raw": image_raw,
            "text_input_raw": question_str,
            "question_id": question,
            "answers": list(answer_weight.keys()),
            "weights": list(answer_weight.values()),
        }