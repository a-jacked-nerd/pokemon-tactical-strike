#!/usr/bin/env python3
"""
ENHANCED POKEMON TARGETING CV MODULE - COMPETITION OPTIMIZED
================================================================

Major improvements:
1. Advanced data augmentation for orientation robustness (especially Bulbasaur)
2. Color-based filtering for Charizard false positives
3. Ensemble detection with multiple models
4. Smart heuristics for 2-miss advantage
5. Adaptive NMS with class-specific thresholds
6. Test-Time Augmentation (TTA) optimization
7. Updated for separate images/annotations folder structure

Author: Enhanced CV System v3.1
"""

import torch
import numpy as np
import cv2
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict
import shutil
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== POKEMON CLASS MAPPING ======================
POKEMON_CLASSES = {
    0: "Pikachu",
    1: "Charizard", 
    2: "Bulbasaur",
    3: "Mewtwo"
}

POKEMON_CLASS_IDS = {v: k for k, v in POKEMON_CLASSES.items()}

# Class-specific configuration for handling known issues
CLASS_CONFIG = {
    "Pikachu": {
        "color_range": [(200, 150, 0), (255, 255, 100)],  # Yellow range in RGB
        "conf_threshold": 0.25,
        "nms_threshold": 0.4,
        "max_aspect_ratio": 1.5
    },
    "Charizard": {
        "color_range": [(180, 50, 0), (255, 150, 50)],  # Orange range
        "conf_threshold": 0.35,  # Higher threshold due to false positives
        "nms_threshold": 0.3,  # Stricter NMS
        "max_aspect_ratio": 2.0,
        "color_filter": True  # Enable color filtering
    },
    "Bulbasaur": {
        "color_range": [(0, 100, 0), (100, 255, 100)],  # Green/Teal range
        "conf_threshold": 0.20,  # Lower threshold for orientation issues
        "nms_threshold": 0.5,
        "max_aspect_ratio": 1.8,
        "use_rotation_tta": True  # Extra rotation augmentation
    },
    "Mewtwo": {
        "color_range": [(150, 100, 150), (255, 200, 255)],  # Purple range
        "conf_threshold": 0.25,
        "nms_threshold": 0.4,
        "max_aspect_ratio": 2.5
    }
}

# ====================== DATASET CONVERTER ======================
class DatasetConverter:
    """Convert annotations to YOLO format and create dataset structure"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.annotations_path = self.base_path / "annotations"
        self.train_prompts_path = self.base_path / "train_prompts.json"
        
        # Create YOLO dataset structure
        self.yolo_dataset_path = self.base_path / "yolo_dataset"
        self.yolo_images_path = self.yolo_dataset_path / "images"
        self.yolo_labels_path = self.yolo_dataset_path / "labels"
        
    def convert_to_yolo_format(self):
        """Convert dataset to YOLO format"""
        logger.info("Converting dataset to YOLO format...")
        
        # Create directory structure
        for split in ['train', 'val']:
            (self.yolo_images_path / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_labels_path / split).mkdir(parents=True, exist_ok=True)
        
        # Load training prompts
        with open(self.train_prompts_path, 'r') as f:
            train_prompts = json.load(f)
        
        # Create mapping from image_id to pokemon class
        image_to_class = {}
        for item in train_prompts:
            image_id = item['image_id']
            pokemon_name = item['prompt'].split(': ')[1].lower()
            # Capitalize first letter for consistency
            pokemon_name = pokemon_name.capitalize()
            image_to_class[image_id] = pokemon_name
        
        # Get all image files
        image_files = list(self.images_path.glob("*.png"))
        
        # Split into train/val
        train_files, val_files = train_test_split(
            image_files, test_size=0.2, random_state=42
        )
        
        # Convert annotations and copy files
        self._process_split(train_files, 'train', image_to_class)
        self._process_split(val_files, 'val', image_to_class)
        
        # Create dataset YAML file
        self._create_dataset_yaml()
        
        logger.info("Dataset conversion completed!")
    
    def _process_split(self, image_files: List[Path], split: str, image_to_class: Dict):
        """Process a dataset split"""
        for img_file in image_files:
            # Copy image
            dst_img = self.yolo_images_path / split / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Find corresponding annotation file
            ann_file = self.annotations_path / img_file.name.replace('.png', '.txt')
            
            if ann_file.exists():
                # Convert annotation to YOLO format
                self._convert_annotation(ann_file, img_file, split, image_to_class)
            else:
                # Create empty annotation file if no annotation exists
                label_file = self.yolo_labels_path / split / img_file.name.replace('.png', '.txt')
                label_file.touch()
    
    def _convert_annotation(self, ann_file: Path, img_file: Path, split: str, image_to_class: Dict):
        """Convert single annotation file to YOLO format"""
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not read image: {img_file}")
            return
        
        img_height, img_width = img.shape[:2]
        
        # Get class ID from prompts
        pokemon_name = image_to_class.get(img_file.name, "Pikachu")  # Default fallback
        class_id = POKEMON_CLASS_IDS.get(pokemon_name, 0)
        
        # Read annotation file
        yolo_annotations = []
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse annotation (assuming format: x_center y_center width height)
                # If your annotations are in a different format, modify this section
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        # Assuming normalized coordinates (0-1)
                        x_center = float(parts[0])
                        y_center = float(parts[1]) 
                        width = float(parts[2])
                        height = float(parts[3])
                        
                        # If coordinates are not normalized, normalize them
                        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height
                        
                        # Create YOLO annotation line
                        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        yolo_annotations.append(yolo_line)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing annotation in {ann_file}: {line} - {e}")
                    continue
        
        # Write YOLO annotation file
        label_file = self.yolo_labels_path / split / img_file.name.replace('.png', '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    def _create_dataset_yaml(self):
        """Create YOLO dataset YAML file"""
        yaml_content = {
            'path': str(self.yolo_dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',  # Use val as test for now
            'nc': len(POKEMON_CLASSES),
            'names': list(POKEMON_CLASSES.values())
        }
        
        yaml_file = self.yolo_dataset_path / "pokemon.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.info(f"Dataset YAML created: {yaml_file}")

# ====================== ADVANCED DATA AUGMENTATION ======================
class AdvancedAugmentation:
    """Advanced augmentation pipeline for orientation and noise robustness"""
    
    @staticmethod
    def get_training_augmentation():
        """Strong augmentations for training"""
        return A.Compose([
            # Geometric transforms for orientation robustness
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.Perspective(scale=(0.05, 0.15), p=0.3),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=(-0.2, 0.2),
                rotate=(-30, 30),
                shear=(-15, 15),
                p=0.5
            ),
            
            # Flip augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Color augmentations for noise robustness
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise augmentations specifically for Charizard issues
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ], p=0.3),
            
            # Add orange/yellow noise patches for Charizard training
            A.CoarseDropout(
                max_holes=8,
                max_height=30,
                max_width=30,
                min_holes=1,
                min_height=10,
                min_width=10,
                fill_value=[255, 165, 0],  # Orange color
                p=0.2
            ),
            
            # Blur augmentations
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.2),
            
            # Distortion for Bulbasaur orientation issues
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            
            # Cutout/Erasing
            A.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    @staticmethod
    def get_tta_augmentation():
        """Test-time augmentation for better detection"""
        return [
            A.Compose([]),  # Original
            A.Compose([A.HorizontalFlip(p=1)]),
            A.Compose([A.VerticalFlip(p=1)]),
            A.Compose([A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT)]),
            A.Compose([A.Rotate(limit=180, p=1, border_mode=cv2.BORDER_CONSTANT)]),
            A.Compose([A.Rotate(limit=270, p=1, border_mode=cv2.BORDER_CONSTANT)]),
        ]

# ====================== ENHANCED SYNTHETIC DATA GENERATOR ======================
class EnhancedSyntheticDataGenerator:
    """Generate synthetic data with focus on problem cases"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.augmenter = AdvancedAugmentation()
        
    def generate_hard_negatives_for_charizard(self, output_dir: Path, num_samples: int = 500):
        """Generate hard negative samples with orange/yellow noise"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for i in range(num_samples):
            # Create image with orange/yellow noise
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Add random orange/yellow patches
            for _ in range(random.randint(10, 30)):
                x = random.randint(0, 600)
                y = random.randint(0, 600)
                w = random.randint(20, 100)
                h = random.randint(20, 100)
                
                # Random orange/yellow color
                color = (
                    random.randint(200, 255),
                    random.randint(100, 200),
                    random.randint(0, 50)
                )
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
            
            # Add Gaussian blur
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Save image
            cv2.imwrite(str(output_dir / f"hard_negative_{i}.jpg"), img)
            
            # Create empty annotation
            with open(output_dir / f"hard_negative_{i}.txt", 'w') as f:
                f.write("")  # Empty annotation for negative samples
    
    def augment_bulbasaur_orientations(self, bulbasaur_images: List[Path], output_dir: Path):
        """Special augmentation for Bulbasaur orientation issues"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for img_path in bulbasaur_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Generate multiple orientations
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            for angle in angles:
                # Rotate image
                center = (img.shape[1]//2, img.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
                # Apply additional transformations
                if random.random() > 0.5:
                    rotated = cv2.flip(rotated, random.choice([0, 1, -1]))
                
                # Save augmented image
                output_name = f"{img_path.stem}_rot{angle}.jpg"
                cv2.imwrite(str(output_dir / output_name), rotated)

# ====================== COLOR-BASED FILTER ======================
class ColorBasedFilter:
    """Filter detections based on color distribution"""
    
    @staticmethod
    def filter_by_color(image: np.ndarray, bbox: List[float], 
                        expected_color_range: Tuple, threshold: float = 0.3) -> bool:
        """
        Check if bbox region contains expected colors
        
        Args:
            image: Input image
            bbox: [x, y, w, h] in pixels
            expected_color_range: ((min_r, min_g, min_b), (max_r, max_g, max_b))
            threshold: Minimum percentage of pixels in color range
        """
        x, y, w, h = [int(v) for v in bbox]
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False
        
        # Convert to RGB if needed
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
        elif roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Create mask for color range
        lower = np.array(expected_color_range[0])
        upper = np.array(expected_color_range[1])
        mask = cv2.inRange(roi, lower, upper)
        
        # Calculate percentage of pixels in range
        percentage = np.sum(mask > 0) / mask.size
        
        return percentage >= threshold

# ====================== ENSEMBLE DETECTOR ======================
class EnsembleDetector:
    """Ensemble of multiple YOLO models for robust detection"""
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        self.models = [YOLO(path) for path in model_paths if Path(path).exists()]
        self.weights = weights or [1.0] * len(self.models)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not self.models:
            raise ValueError("No valid model paths provided!")
        
    def detect_ensemble(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """Run ensemble detection with weighted voting"""
        all_detections = defaultdict(list)
        
        for model, weight in zip(self.models, self.weights):
            results = model(image_path, conf=conf_threshold, device=self.device, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        det = {
                            'bbox': box.xywh[0].cpu().numpy(),
                            'conf': box.conf.item() * weight,
                            'class': int(box.cls.item())
                        }
                        all_detections[det['class']].append(det)
        
        # Merge detections using weighted NMS
        final_detections = []
        for class_id, class_dets in all_detections.items():
            merged = self._weighted_nms(class_dets)
            final_detections.extend(merged)
        
        return final_detections
    
    def _weighted_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Weighted Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
                else:
                    # Merge confidence scores
                    best['conf'] = max(best['conf'], det['conf'])
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes in xywh format"""
        # Convert to xyxy
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        box1_xyxy = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
        box2_xyxy = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]
        
        # Calculate intersection
        x_left = max(box1_xyxy[0], box2_xyxy[0])
        y_top = max(box1_xyxy[1], box2_xyxy[1])
        x_right = min(box1_xyxy[2], box2_xyxyxy[2])
        y_bottom = min(box1_xyxy[3], box2_xyxy[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

# ====================== SMART HEURISTIC OPTIMIZER ======================
class HeuristicOptimizer:
    """Optimize shot selection using the 2-miss advantage"""
    
    @staticmethod
    def optimize_shots(detections: List[Dict], max_shots: int = 10) -> List[Dict]:
        """
        Optimize shot selection using heuristics
        
        Strategy:
        - Take top confidence detections first
        - Add 2 strategic "buffer" shots for high-value targets
        - Use spatial clustering to avoid redundant shots
        """
        if not detections:
            return []
        
        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Apply spatial clustering to avoid shooting same Pokemon twice
        clustered = HeuristicOptimizer._cluster_detections(sorted_dets)
        
        selected = []
        buffer_shots = 2  # We can afford 2 misses
        
        for cluster in clustered:
            if len(selected) >= max_shots:
                break
                
            # Take the highest confidence detection from cluster
            best = max(cluster, key=lambda x: x['confidence'])
            selected.append(best)
            
            # Add buffer shot for high-confidence targets
            if best['confidence'] > 0.7 and buffer_shots > 0 and len(selected) < max_shots:
                # Add a slightly offset shot as insurance
                buffer_shot = best.copy()
                buffer_shot['center'] = [
                    best['center'][0] + random.uniform(-5, 5),
                    best['center'][1] + random.uniform(-5, 5)
                ]
                buffer_shot['is_buffer'] = True
                selected.append(buffer_shot)
                buffer_shots -= 1
        
        return selected[:max_shots]
    
    @staticmethod
    def _cluster_detections(detections: List[Dict], distance_threshold: float = 50) -> List[List[Dict]]:
        """Cluster nearby detections"""
        if not detections:
            return []
        
        clusters = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det]
            used.add(i)
            
            for j, other in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                # Calculate distance between centers
                dist = np.linalg.norm(
                    np.array(det['center']) - np.array(other['center'])
                )
                
                if dist < distance_threshold:
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters

# ====================== OPTIMIZED POKEMON DETECTOR ======================
class OptimizedPokemonDetector:
    """Main detector with all optimizations"""
    
    def __init__(self, model_paths: List[str], use_ensemble: bool = True):
        if use_ensemble and len(model_paths) > 1:
            self.detector = EnsembleDetector(model_paths)
            self.use_ensemble = True
        else:
            # Use single model
            valid_paths = [p for p in model_paths if Path(p).exists()]
            if not valid_paths:
                raise ValueError("No valid model paths found!")
            self.model = YOLO(valid_paths[0])
            self.use_ensemble = False
        
        self.color_filter = ColorBasedFilter()
        self.optimizer = HeuristicOptimizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tta_transforms = AdvancedAugmentation.get_tta_augmentation()
        
        logger.info(f"Optimized detector initialized (ensemble={use_ensemble})")
    
    def detect_with_tta(self, image_path: str, target_class: str, 
                       apply_color_filter: bool = True) -> List[Dict]:
        """
        Detect with Test-Time Augmentation and filtering
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        class_config = CLASS_CONFIG.get(target_class, {})
        all_detections = []
        
        # Apply TTA if needed (especially for Bulbasaur)
        if class_config.get('use_rotation_tta', False):
            transforms_to_use = self.tta_transforms
        else:
            transforms_to_use = self.tta_transforms[:3]  # Only basic flips
        
        for transform in transforms_to_use:
            # Apply transformation
            transformed = transform(image=image)['image']
            
            # Save temporary image
            temp_path = "temp_tta.jpg"
            cv2.imwrite(temp_path, transformed)
            
            # Run detection
            if self.use_ensemble:
                results = self.detector.detect_ensemble(
                    temp_path, 
                    class_config.get('conf_threshold', 0.25)
                )
            else:
                results = self._single_model_detect(
                    temp_path,
                    class_config.get('conf_threshold', 0.25)
                )
            
            # Filter for target class
            class_id = POKEMON_CLASS_IDS[target_class]
            filtered = [r for r in results if r.get('class', r.get('class_id')) == class_id]
            
            # Reverse transformation for coordinates (simplified)
            for det in filtered:
                det = self._reverse_transform_detection(det, transform, image.shape)
                all_detections.append(det)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Apply NMS on all TTA detections
        merged_detections = self._merge_tta_detections(
            all_detections, 
            class_config.get('nms_threshold', 0.4)
        )
        
        # Apply color filter for Charizard
        if apply_color_filter and target_class == "Charizard" and class_config.get('color_filter', False):
            filtered_detections = []
            for det in merged_detections:
                bbox = det.get('bbox', [])
                if bbox and self.color_filter.filter_by_color(
                    image, bbox, class_config['color_range'], threshold=0.2
                ):
                    filtered_detections.append(det)
            merged_detections = filtered_detections
        
        return merged_detections
    
    def _single_model_detect(self, image_path: str, conf_threshold: float) -> List[Dict]:
        """Single model detection"""
        results = self.model(image_path, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xywh[0].cpu().numpy()
                    x_center, y_center, width, height = bbox
                    
                    det = {
                        'class_id': int(box.cls.item()),
                        'class': int(box.cls.item()),
                        'confidence': box.conf.item(),
                        'bbox': [x_center - width/2, y_center - height/2, width, height],
                        'center': [x_center, y_center]
                    }
                    detections.append(det)
        
        return detections
    
    def _reverse_transform_detection(self, detection: Dict, transform, 
                                   original_shape: Tuple) -> Dict:
        """Reverse coordinate transformation from TTA"""
        # Simplified - would need full implementation based on transform type
        return detection
    
    def _merge_tta_detections(self, detections: List[Dict], 
                            nms_threshold: float) -> List[Dict]:
        """Merge detections from TTA using NMS"""
        if not detections:
            return []
        
        # Convert to numpy arrays
        boxes = []
        scores = []
        for det in detections:
            x, y, w, h = det['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(det.get('confidence', det.get('conf', 0.5)))
        
        if not boxes:
            return []
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.1, nms_threshold)
        
        if len(indices) == 0:
            return []
        
        # Return filtered detections
        return [detections[i] for i in indices.flatten()]
    
    def get_optimized_targets(self, image_path: str, target_class: str, 
                            prompt: str = "") -> List[List[float]]:
        """
        Get optimized targeting coordinates
        
        Args:
            image_path: Path to image
            target_class: Pokemon to target
            prompt: Original prompt for context
        
        Returns:
            List of [x, y] coordinates in COCO format (top-left origin)
        """
        # Detect with all optimizations
        detections = self.detect_with_tta(image_path, target_class)
        
        if not detections:
            logger.warning(f"No {target_class} detected in {image_path}")
            return []
        
        # Apply heuristic optimization
        optimized = self.optimizer.optimize_shots(detections, max_shots=10)
        
        # Extract coordinates
        coordinates = []
        for det in optimized:
            # Use center coordinates in COCO format
            x_center, y_center = det['center']
            coordinates.append([float(x_center), float(y_center)])
            
            if det.get('is_buffer', False):
                logger.info(f"Added buffer shot at ({x_center:.1f}, {y_center:.1f})")
        
        logger.info(f"Generated {len(coordinates)} optimized shots for {target_class}")
        return coordinates

# ====================== TRAINING PIPELINE ======================
class OptimizedTrainingPipeline:
    """Complete training pipeline with all optimizations"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.converter = DatasetConverter(dataset_path)
        self.augmenter = AdvancedAugmentation()
        self.synthetic_gen = EnhancedSyntheticDataGenerator(dataset_path)
        
    def prepare_enhanced_dataset(self):
        """Prepare dataset with augmentations and synthetic data"""
        logger.info("Preparing enhanced dataset...")
        
        # First convert to YOLO format
        self.converter.convert_to_yolo_format()
        
        # Generate hard negatives for Charizard
        synthetic_path = self.dataset_path / "yolo_dataset" / "synthetic"
        self.synthetic_gen.generate_hard_negatives_for_charizard(
            synthetic_path / "charizard_negatives",
            num_samples=500
        )
        
        # Augment Bulbasaur orientations
        bulbasaur_images = []
        images_path = self.dataset_path / "images"
        
        # Load prompts to find Bulbasaur images
        with open(self.dataset_path / "train_prompts.json", 'r') as f:
            train_prompts = json.load(f)
        
        for item in train_prompts:
            if 'bulbasaur' in item['prompt'].lower():
                img_path = images_path / item['image_id']
                if img_path.exists():
                    bulbasaur_images.append(img_path)
        
        self.synthetic_gen.augment_bulbasaur_orientations(
            bulbasaur_images,
            synthetic_path / "bulbasaur_augmented"
        )
        
        logger.info("Enhanced dataset prepared")
    
    def train_models(self, num_models: int = 3) -> List[str]:
        """Train multiple models for ensemble"""
        model_paths = []
        
        # Get the YOLO dataset path
        yolo_yaml = self.dataset_path / "yolo_dataset" / "pokemon.yaml"
        
        if not yolo_yaml.exists():
            raise FileNotFoundError(f"YOLO dataset file not found: {yolo_yaml}")
        
        for i in range(num_models):
            logger.info(f"Training model {i+1}/{num_models}")
            
            # Use different model sizes for diversity
            sizes = ['s', 'm', 'l']
            model_size = sizes[i % len(sizes)]
            
            model = YOLO(f"yolov8{model_size}.pt")
            
            # Train with specific optimizations
            results = model.train(
                data=str(yolo_yaml),
                epochs=150,
                imgsz=640,
                batch=16,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,
                warmup_momentum=0.8,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                degrees=90.0 if i == 0 else 45.0,  # More rotation for first model
                translate=0.3,
                scale=0.5,
                shear=15.0,
                perspective=0.0002,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.3 if i == 1 else 0.2,  # Different mixup
                copy_paste=0.3,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                erasing=0.4,
                crop_fraction=0.8,
                name=f"pokemon_model_{i}",
                device=0 if torch.cuda.is_available() else "cpu",
                amp=True,
                patience=25,
                save=True,
                exist_ok=True
            )
            
            model_path = f"runs/detect/pokemon_model_{i}/weights/best.pt"
            model_paths.append(model_path)
            logger.info(f"Model {i+1} saved to {model_path}")
        
        return model_paths
    
    def train_single_model(self, model_size: str = 's') -> str:
        """Train a single model (faster option)"""
        logger.info(f"Training single YOLOv8{model_size} model...")
        
        # Get the YOLO dataset path
        yolo_yaml = self.dataset_path / "yolo_dataset" / "pokemon.yaml"
        
        if not yolo_yaml.exists():
            raise FileNotFoundError(f"YOLO dataset file not found: {yolo_yaml}")
        
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train with optimizations
        results = model.train(
            data=str(yolo_yaml),
            epochs=200,
            imgsz=640,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            degrees=45.0,
            translate=0.3,
            scale=0.5,
            shear=15.0,
            perspective=0.0002,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            crop_fraction=0.8,
            name="pokemon_single_model",
            device=0 if torch.cuda.is_available() else "cpu",
            amp=True,
            patience=30,
            save=True,
            exist_ok=True
        )
        
        model_path = "runs/detect/pokemon_single_model/weights/best.pt"
        logger.info(f"Model saved to {model_path}")
        return model_path

# ====================== MAIN EXECUTION ======================
def main():
    """Main function to run the complete optimized pipeline"""
    logger.info("🚀 Starting Optimized Pokemon Detection System")
    
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR STRUCTURE
    dataset_path = r"D:\IIT Madras\Competitions\dataset\dataset"
    
    try:
        # Verify dataset structure
        base_path = Path(dataset_path)
        images_path = base_path / "images"
        annotations_path = base_path / "annotations"
        prompts_path = base_path / "train_prompts.json"
        
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_path}")
        if not prompts_path.exists():
            raise FileNotFoundError(f"Training prompts file not found: {prompts_path}")
        
        logger.info(f"Dataset structure verified:")
        logger.info(f"  Images: {len(list(images_path.glob('*.png')))} files")
        logger.info(f"  Annotations: {len(list(annotations_path.glob('*.txt')))} files")
        
        # Initialize training pipeline
        pipeline = OptimizedTrainingPipeline(dataset_path)
        
        # Prepare enhanced dataset
        pipeline.prepare_enhanced_dataset()
        
        # Option 1: Train ensemble models (slower but better)
        # model_paths = pipeline.train_models(num_models=3)
        
        # Option 2: Train single model (faster)
        model_path = pipeline.train_single_model(model_size='m')  # or 's', 'l', 'x'
        model_paths = [model_path]
        
        # Initialize optimized detector
        detector = OptimizedPokemonDetector(model_paths, use_ensemble=len(model_paths) > 1)
        
        # Example usage
        test_images = list(images_path.glob("*.png"))[:5]  # Test on first 5 images
        
        for test_image in test_images:
            logger.info(f"Testing on: {test_image.name}")
            
            # For each Pokemon type
            for pokemon_name in POKEMON_CLASSES.values():
                try:
                    coordinates = detector.get_optimized_targets(
                        str(test_image),
                        pokemon_name,
                        f"Kill: {pokemon_name.lower()}"
                    )
                    
                    if coordinates:
                        logger.info(f"  {pokemon_name}: {len(coordinates)} targets found")
                    else:
                        logger.info(f"  {pokemon_name}: No targets detected")
                        
                except Exception as e:
                    logger.error(f"Error detecting {pokemon_name} in {test_image.name}: {e}")
        
        logger.info("✅ System ready for competition!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# ====================== UTILITY FUNCTIONS ======================
def test_single_image(model_path: str, image_path: str, target_pokemon: str):
    """Test detection on a single image"""
    detector = OptimizedPokemonDetector([model_path], use_ensemble=False)
    
    coordinates = detector.get_optimized_targets(
        image_path,
        target_pokemon,
        f"Kill: {target_pokemon.lower()}"
    )
    
    print(f"Detected {len(coordinates)} {target_pokemon} targets:")
    for i, (x, y) in enumerate(coordinates):
        print(f"  Target {i+1}: ({x:.1f}, {y:.1f})")
    
    return coordinates

def validate_dataset_structure(dataset_path: str):
    """Validate that dataset has the correct structure"""
    base_path = Path(dataset_path)
    
    required_paths = [
        base_path / "images",
        base_path / "annotations", 
        base_path / "train_prompts.json"
    ]
    
    for path in required_paths:
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False
        else:
            print(f"✅ Found: {path}")
    
    # Count files
    num_images = len(list((base_path / "images").glob("*.png")))
    num_annotations = len(list((base_path / "annotations").glob("*.txt")))
    
    print(f"📊 Dataset statistics:")
    print(f"  Images: {num_images}")
    print(f"  Annotations: {num_annotations}")
    
    return True

if __name__ == "__main__":
    # Uncomment to validate dataset structure first
    # validate_dataset_structure(r"D:\IIT Madras\Competitions\dataset\dataset")
    
    # Run main pipeline
    main()