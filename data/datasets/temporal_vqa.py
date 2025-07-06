from datasets import load_dataset
import base64
import requests
import os
from io import BytesIO
from PIL import Image
import json

from data.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from data.datasets.multiframe_datasets import MultiframeDataset 
from data.datasets.base_dataset import BaseDataset
from data.datasets.coco_vqa import VQADataset_Raw
from collections import OrderedDict

class __DisplMixinHF:
    def displ_item(self, index):
        sample = self.__getitem__(index)

        return OrderedDict(
            {
                "file": "huggingface",
                "question": self.question,
                "question_id": 0,
                "answers": "; ".join(sample["answers"]),
                "image": sample["image"],
            }
        )

# def concatenate_images_side_by_side(image1, image2):
#     # Make sure both images have the same height
#     if image1.height != image2.height:
#         target_height = min(image1.height, image2.height)
#         image1 = image1.resize((int(image1.width * target_height / image1.height), target_height))
#         image2 = image2.resize((int(image2.width * target_height / image2.height), target_height))

#     total_width = image1.width + image2.width
#     combined = Image.new("RGB", (total_width, image1.height))
#     combined.paste(image1, (0, 0))
#     combined.paste(image2, (image1.width, 0))
#     return combined

# from PIL import Image

def concatenate_images_side_by_side(image1, image2, final_size=(384, 384)):
    """
    Concatenates two PIL Images side by side and resizes the result to final_size (default: 384x384).
    """
    # Resize both images to have the same height
    target_height = min(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * target_height / image1.height), target_height), Image.BICUBIC)
    image2 = image2.resize((int(image2.width * target_height / image2.height), target_height), Image.BICUBIC)

    # Concatenate horizontally
    total_width = image1.width + image2.width
    combined = Image.new("RGB", (total_width, target_height))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))

    # Resize the final image to desired size (384x384)
    combined_resized = combined.resize(final_size, Image.BICUBIC)

    return combined_resized



class TemporalVQADataset_Raw(VQADataset_Raw, __DisplMixinHF):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor)

        self.subset = 'temporal_order'
        self.dataset = load_dataset('fazliimam/temporal-vqa', self.subset, split='test')
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        if self.subset == 'temporal_order':
            self.question = "Between these two images, which one depicts the event that happened first? Provide your answer in dictionary format: {'Answer':'First image or Second image', 'Reasoning':'Brief explanation of your choice'}"
        elif self.subset == 'timelapse_estimation':
            self.question = "In the given image, estimate the time that has passed between the first image (left) and the second image (right). Choose one of the following options: A. Less than 15 seconds B. Between 2 minutes to 15 minutes C. Between 1 hour to 12 hours D. Between 2 days to 30 days E. Between 4 months to 12 months F. More than 3 years. Provide your answer in dictionary format: {'Answer':'Selected option', 'Reasoning':'Brief explanation of your choice'}"
        else:
            raise ValueError(f"Unsupported subset: {self.subset}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        # Load image (assuming it's a URL or path string)
        image_1 = Image.open(item['image_1']).convert("RGB")
        image_2 = Image.open(item['image_2']).convert("RGB")

        image_raw = concatenate_images_side_by_side(image_1, image_2)

        answer = item['label']

        answers = [answer] * 10
        weights = [1] * 10

        multiple_choice_answer = answer

        return {
            "image_raw": image_raw,
            "text_input_raw": self.question,
            'multiple_choice_answer': multiple_choice_answer,
            "answers": answers,
            "weights": weights,
        }
    
# class TemporalVQADataset(VQADataset):
#     def __init__(self, dataset_name, subset, split, vis_processor, text_processor):
#         super().__init__(vis_processor, text_processor)
#         self.dataset = load_dataset(dataset_name, subset, split=split)
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor
#         self.subset = subset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         item = self.dataset[index]

#         # Load image (assuming it's a URL or path string)
#         image_1 = Image.open(item['image_1']).convert("RGB")
#         image_2 = Image.open(item['image_2']).convert("RGB")

#         image = concatenate_images_side_by_side(image_1, image_2)

#         image = self.vis_processor(image)

#         #image_1 = self.vis_processor(image_1)
#         #image_2 = self.vis_processor(image_2)


#         if self.subset == 'temporal_order':
#             question = "Between these two images, which one depicts the event that happened first? Provide your answer in dictionary format: {'Answer':'First image or Second image', 'Reasoning':'Brief explanation of your choice'}"
#         elif self.subset == 'timelapse_estimation':
#             question = "In the given image, estimate the time that has passed between the first image (left) and the second image (right). Choose one of the following options: A. Less than 15 seconds B. Between 2 minutes to 15 minutes C. Between 1 hour to 12 hours D. Between 2 days to 30 days E. Between 4 months to 12 months F. More than 3 years. Provide your answer in dictionary format: {'Answer':'Selected option', 'Reasoning':'Brief explanation of your choice'}"
#         else:
#             raise ValueError(f"Unsupported subset: {self.subset}")
        
#         question = self.text_processor(question)

#         answer = item['label']

#         answers = [answer] * 10
#         weights = [1] * 10

#         return {
#             "image": image,
#             "text_input": question,
#             "answers": answers,
#             "weights": weights,
#         }
    
class TemporalVQAEvalDataset_Raw(VQAEvalDataset, __DisplMixinHF):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):

        self.subset = 'temporal_order'
        self.dataset = load_dataset('fazliimam/temporal-vqa', self.subset, split='test')
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        if self.subset == 'temporal_order':
            self.question = "In the given image, which side depicts the event that happened first? The first image corresponds to the left side and the second image corresponds to the right side. Provide your answer in dictionary format: {'Answer':'First image or Second image', 'Reasoning':'Brief explanation of your choice'}"
        elif self.subset == 'timelapse_estimation':
            self.question = "In the given image, estimate the time that has passed between the first image (left) and the second image (right). Choose one of the following options: A. Less than 15 seconds B. Between 2 minutes to 15 minutes C. Between 1 hour to 12 hours D. Between 2 days to 30 days E. Between 4 months to 12 months F. More than 3 years. Provide your answer in dictionary format: {'Answer':'Selected option', 'Reasoning':'Brief explanation of your choice'}"
        else:
            raise ValueError(f"Unsupported subset: {self.subset}")
        
        self.output_dir = 'datasets_src/temporal_vqa_gt'
        os.makedirs(self.output_dir, exist_ok=True)

        self.ann_file = os.path.join(self.output_dir, 'temporal_vqa_val_annotations.json')
        self.ques_file = os.path.join(self.output_dir, 'temporal_vqa_val_questions.json')

        if not os.path.exists(self.ann_file) or not os.path.exists(self.ques_file):
            self.create_annotation_and_question_json()
        
        # gt_json_path = 'datasets_src/temporal_vqa_gt/temporal_vqa_val_annotations.json'
        # if not os.path.exists(gt_json_path):
        #     self.create_gt_json(gt_json_path)
        
    def __len__(self):
        return len(self.dataset)
    
    # def create_gt_json(self, output_path):
    #     """
    #     Create a ground truth JSON file for evaluation.
    #     """
    #     gt_data = []
    #     for index in range(len(self.dataset)):
    #         item = self.dataset[index]
    #         answer = item['label']
    #         gt_data.append({
    #             "question_id": index,
    #             "answer": answer
    #         })
        
    #     with open(output_path, 'w') as f:
    #         json.dump(gt_data, f, indent=4)

    def create_annotation_and_question_json(self):
        annotations, questions = [], []

        for idx, item in enumerate(self.dataset):
            answer_text = item['label']

            answers = [{
                "answer": answer_text.lower() + ' image',
                "answer_confidence": "yes",
                "answer_id": i
            } for i in range(10)]

            annotations.append({
                "question_id": idx,
                "image_id": idx,
                "question_type": "unknown",
                "answer_type": "other",
                "answers": answers,
                "multiple_choice_answer": answer_text.lower() + ' image'
            })

            questions.append({
                "question_id": idx,
                "image_id": idx,
                "question": self.question,
                "multiple_choices": ["First image", "Second image"]
            })

        with open(self.ann_file, "w") as f:
            json.dump({"annotations": annotations}, f, indent=4)

        with open(self.ques_file, "w") as f:
            json.dump({
                "questions": questions,
                "task_type": "Open-Ended",
                "data_type": "temporal-vqa",
                "data_subtype": "test",
                "license": {},
                "info": {}
            }, f, indent=4)
        
    def __getitem__(self, index):
        item = self.dataset[index]

        # Load image (assuming it's a URL or path string)
        image_1 = item['image_1'] #Image.open(item['image_1']).convert("RGB")
        image_2 = item['image_2'] #Image.open(item['image_2']).convert("RGB")

        image_raw = concatenate_images_side_by_side(image_1, image_2)

        answer = item['label']
        multiple_choice_answer = answer

        return {
            "question_id": index,
            "instance_id": index,
            "image_raw": image_raw,
            "image_path": "",
            "text_input_raw": self.question,
            'multiple_choice_answer': multiple_choice_answer
        }