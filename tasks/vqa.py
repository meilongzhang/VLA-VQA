import logging
import json
import os
import torch
from tqdm import tqdm

from common.utils import is_convertible_to_int
import common.dist_utils as dist_utils
from common.registry import registry
from common.vqa_tools.vqa import VQA
from common.vqa_tools.vqa_eval import VQAEval
from tasks.base_task import BaseTask

from common.logger import MetricLogger, SmoothedValue
import torch.distributed as dist
from common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from data.data_utils import prepare_sample


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        sample_id_key = "",
        ques_files=dict(),
        anno_files=dict(),
        valid_splits=['val']
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = ques_files
        self.anno_files = anno_files

        # generalize to non coco data
        self.sample_id_key = sample_id_key

        self.valid_splits = valid_splits

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        prompt = run_cfg.get("prompt", "")

        # generalize to non coco data
        sample_id_key = run_cfg.get("sample_id_key", "instance_id")
        ques_files = run_cfg.get("ques_files", dict())
        anno_files = run_cfg.get("anno_files", dict())
        valid_splits = run_cfg.get("valid_splits", ["val"])


        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            sample_id_key = sample_id_key,
            ques_files=ques_files,
            anno_files=anno_files,
            valid_splits=valid_splits
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in self.valid_splits:
                if split not in dataset:
                    print(f"Split {split} not found in {ds_name}.")
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file
                else:
                    if split not in self.ques_files: # precomputed and passed in task builder
                        self.ques_files[split] = os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt', f'{ds_name}_{split}_questions.json')
                        self.anno_files[split] = os.path.join(registry.get_path("cache_root"), f'{ds_name}_gt', f'{ds_name}_{split}_annotations.json')
                        if dist_utils.get_rank() == 0:
                            os.makedirs(os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt'), exist_ok=True)
                            try:
                                convert_to_coco_gt(dataset, self.ques_files[split], self.anno_files[split], split, self.sample_id_key)
                            except:
                                pass # tasks like vizwiz with no gt answer
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        # predict_answers function
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            if ques_id != int and is_convertible_to_int(ques_id):
                ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )
            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics
    
    """
    Adding get cross attention
    """

    def get_xattn(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = {} 
        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        for samples in tqdm(data_loader):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # print("model device :", model.device)
            # print("samples  device: ", samples.device)
            
            # print(samples)
            instance_ids = samples["instance_id"] #(1, batch size)
            img_ratio_batch, txt_ratio_batch = model.attn_scores(samples)
            
            for idx, instance_id in enumerate(instance_ids) : 
                if instance_id in results : 
                    raise Exception("duplicate instance id")
                
                results[instance_id] = {}
                results[instance_id]['img_ratio'] = img_ratio_batch[idx]
                results[instance_id]['txt_ratio'] = txt_ratio_batch[idx]
                results[instance_id]['image_path'] = samples['image_path'][idx]
                results[instance_id]['text_input_raw'] = samples['text_input_raw'][idx]
                results[instance_id]['multiple_choice_answer'] = samples['multiple_choice_answer'][idx]

        if is_dist_avail_and_initialized():
            dist.barrier()
            print("merging results across the gpus")

            all_results = [{} for _ in range(dist.get_world_size())]
            print("size ", dist.get_world_size())
            dist.all_gather_object(all_results, results)

            # Combine results into a single dictionary
            merged_results = {}
            for gpu_results in all_results:
                merged_results.update(gpu_results)

        #stack the results
    
        return merged_results 

def convert_to_coco_gt(data, outpath_questions, outpath_annotations, split, sample_id_key):
    if split not in data:
        return
    questions_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'questions':[]}
    annotations_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'annotations':[]}
    print("Generating ground truth annotations...")
    for ann in tqdm(data[split]):
        if ann == None:
            continue
        # if ann[sample_id_key] not in img_ids:
        #     continue
        ques_id = ann["question_id"]
        ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
        if ques_id != int and is_convertible_to_int(ques_id):
            ques_id = int(ques_id)
        questions_data["questions"].append({"question": ann["text_input"], "image_id": ann[sample_id_key], "question_id": ques_id})
        annotations_data["annotations"].append({
            "question_type": "" if "question_type" not in ann else ann["question_type"],
            "multiple_choice_answer": ann["answers"][0] if isinstance(ann["answers"], list) else ann["answers"],
            "answers": [{"answer":ans, "answer_id":i} for i,ans in enumerate(ann["answers"])] if isinstance(ann["answers"], list) else [{"answer":ann["answers"], "answer_id":0}], 
            "image_id": ann[sample_id_key], 
            "question_id": ques_id,
            "answer_type": "" if "answer_type" not in ann else ann["answer_type"],
        })
       
    json.dump(questions_data, open(outpath_questions, 'w'))
    print(f"Saved questions data at {outpath_questions}")
    json.dump(annotations_data, open(outpath_annotations, 'w'))
    print(f"Saved annotation data at {outpath_annotations}")


@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
    
    def build_datasets(self, cfg):
        datasets = BaseTask.build_datasets(self,cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            # added to ensure that the ground truth format of answers is as expected for non-gqa but similar tasks
            gt_ans = vqa_tool.processPunctuation(gt_ans)
            gt_ans = vqa_tool.processDigitArticle(gt_ans)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics