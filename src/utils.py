# src/utils.py
import logging
from datetime import datetime
import torchvision

def setup_logging():
    """Sets up the logging configuration."""
    log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def apply_nms_gpu_workaround():
    """Applies a workaround for NMS on certain GPU environments."""
    original_nms = torchvision.ops.nms

    def nms_gpu_workaround(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            boxes_cpu = boxes.cpu()
            scores_cpu = scores.cpu()
            keep = original_nms(boxes_cpu, scores_cpu, iou_threshold)
            return keep.to(boxes.device)
        return original_nms(boxes, scores, iou_threshold)

    torchvision.ops.nms = nms_gpu_workaround
    logging.info("Applied NMS GPU workaround.")