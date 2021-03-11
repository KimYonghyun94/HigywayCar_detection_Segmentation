import torch, torchvision
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

register_coco_instances("vehicle_test_BB", {}, 
                        "/home/super/Desktop/yh/test_BB10.json", #테스트셋 jSON파일 경로지정
                        "/home/super/Desktop/yh/test_BB")        #테스트셋 이미지폴더 경로지정

vehicle_test_metadata = MetadataCatalog.get("vehicle_test_BB")
dataset_dicts = DatasetCatalog.get("vehicle_test_BB")

cfg = get_cfg()
cfg.DATASETS.TRAIN = ("vehicle_test_BB",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
trainer = DefaultTrainer(cfg)

cfg.MODEL.WEIGHTS = "/home/super/Desktop/model_finalBB.pth"    #검증모델 경로지정
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

evaluator = COCOEvaluator("vehicle_test_BB", cfg,False, output_dir="/home/super/Desktop/yh/detectron2_git/detectron2/output_eval_BB/")
test_loader = build_detection_test_loader(cfg, "vehicle_test_BB")
print(inference_on_dataset(trainer.model, test_loader, evaluator))