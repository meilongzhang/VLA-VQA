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
            "file": ann["image"],
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

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")

        answer_weight = {}
        for answer in ann["answer"]:
            answer_weight[answer] = answer_weight.get(answer, 0) + 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        multiple_choice_answer = max(set(ann["answer"]), key=ann["answer"].count)

        return {
            "answers": answers,
            "multiple_choice_answer": multiple_choice_answer,
            "weights": weights,
            "image_raw": image_raw,
            "text_input_raw": ann["question"],
        }
