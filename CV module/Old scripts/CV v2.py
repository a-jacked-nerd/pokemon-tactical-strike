#!/usr/bin/env python3
"""
FINAL REFINED CV MODULE FOR POKEMON DETECTION
===============================================

This module is optimized for the Pokemon targeting hackathon with the following constraints:
- Only 4 Pokemon classes: Pikachu, Charizard, Bulbasaur, Mewtwo
- Test images are noisy with floating particles and rotated Pokemon
- Need maximum precision for accurate center targeting
- Enhanced data augmentation to simulate test conditions

Author: Enhanced CV System for Pokemon Hackathon
Version: 2.0 - Single Target Optimized (cause that is what was asked in the PS)
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import json
import yaml
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A
import random
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration for the enhanced CV module"""
    # Training parameters - optimized for single target detection
    YOLO_EPOCHS = 200
    YOLO_IMG_SIZE = 640
    YOLO_BATCH = 8  # Smaller batch for better gradients
    YOLO_PATIENCE = 30
    
    # Dataset paths
    IMAGES_DIR = r"D:\IIT Madras\Competitions\dataset\dataset\images"
    INSTANCES_FILE = r"D:\IIT Madras\Competitions\dataset\dataset\annotations\instances_train.json"
    
    # Detection parameters - very sensitive to catch all instances
    DETECTION_CONFIDENCE = 0.08  # Ultra low for maximum recall
    NMS_THRESHOLD = 0.25
    
    # Augmentation parameters - aggressive to simulate test conditions
    AUGMENT_FACTOR = 8  # More augmentation for robustness

config = Config()

class NoiseAugmentation:
    """Specialized augmentation to simulate noisy test conditions"""
    
    @staticmethod
    def get_noise_augmentations():
        """Augmentations specifically designed for noisy test images"""
        return A.Compose([
            # Simulate test image noise patterns
            A.OneOf([
                A.GaussNoise(var_limit=(15.0, 60.0), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.8, 1.2], elementwise=True, p=0.4),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.15, 0.6), p=0.4),
            ], p=0.8),
            
            # Motion blur and quality degradation
            A.OneOf([
                A.MotionBlur(blur_limit=9, p=0.4),
                A.MedianBlur(blur_limit=9, p=0.3),
                A.GaussianBlur(blur_limit=9, p=0.4),
                A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=0.3),
            ], p=0.7),
            
            # Rotation and orientation changes (critical for test images)
            A.OneOf([
                A.Rotate(limit=60, border_mode=cv2.BORDER_REFLECT, p=0.6),
                A.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.25, rotate_limit=60, 
                    border_mode=cv2.BORDER_REFLECT, p=0.6
                ),
                A.Affine(rotate=(-60, 60), p=0.5),
            ], p=0.9),
            
            # Lighting and color variations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.4),
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.4),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.4),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            ], p=0.8),
            
            # Geometric transformations
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Transpose(p=0.2),
                A.RandomRotate90(p=0.3),
            ], p=0.7),
            
            # Quality degradation to match test conditions
            A.OneOf([
                A.Downscale(scale_min=0.6, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=0.4),
                A.ImageCompression(quality_lower=40, quality_upper=95, p=0.4),
                A.Posterize(num_bits=4, p=0.2),
            ], p=0.5),
            
            # Weather and environment effects
            A.OneOf([
                A.RandomRain(slant_lower=-15, slant_upper=15, drop_length=25, drop_width=2, 
                           drop_color=(180, 180, 180), blur_value=2, brightness_coefficient=0.7, p=0.3),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1, p=0.2),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.2),
            ], p=0.4),
            
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.3))
    
    @staticmethod
    def create_floating_particle_backgrounds():
        """Create backgrounds with floating particles like in test images"""
        logger.info("Creating synthetic particle backgrounds...")
        backgrounds = []
        
        for _ in range(30):  # More backgrounds for variety
            # Create base background
            bg = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            # Add floating particles (yellow/orange like test images)
            particle_count = random.randint(100, 400)
            for _ in range(particle_count):
                x = random.randint(0, 639)
                y = random.randint(0, 639)
                
                # Particle colors (yellow/orange spectrum)
                colors = [
                    (255, 255, 0),    # Pure yellow
                    (255, 215, 0),    # Gold
                    (255, 165, 0),    # Orange
                    (255, 140, 0),    # Dark orange
                    (255, 100, 0),    # Red-orange
                    (200, 200, 50),   # Dim yellow
                ]
                color = random.choice(colors)
                
                # Particle size
                radius = random.randint(1, 4)
                cv2.circle(bg, (x, y), radius, color, -1)
                
                # Add some glow
                if random.random() < 0.3:
                    cv2.circle(bg, (x, y), radius + 1, 
                             tuple(int(c * 0.5) for c in color), 1)
            
            # Add larger noise blobs
            blob_count = random.randint(20, 50)
            for _ in range(blob_count):
                center = (random.randint(30, 610), random.randint(30, 610))
                axes = (random.randint(8, 40), random.randint(8, 40))
                angle = random.randint(0, 180)
                
                # Random color with some transparency effect
                color = tuple(random.randint(80, 255) for _ in range(3))
                cv2.ellipse(bg, center, axes, angle, 0, 360, color, -1)
                
                # Add some texture
                if random.random() < 0.5:
                    cv2.ellipse(bg, center, 
                              (axes[0]//2, axes[1]//2), angle, 0, 360, 
                              tuple(int(c * 0.7) for c in color), -1)
            
            backgrounds.append(bg)
        
        logger.info(f"Created {len(backgrounds)} synthetic backgrounds")
        return backgrounds

class EnhancedDataPreparation:
    """Enhanced data preparation with aggressive augmentation"""
    
    def __init__(self):
        self.backgrounds = NoiseAugmentation.create_floating_particle_backgrounds()
        self.augment_pipeline = NoiseAugmentation.get_noise_augmentations()
    
    def prepare_yolo_dataset(self):
        """Prepare YOLO dataset with enhanced augmentation"""
        logger.info("Starting enhanced YOLO data preparation...")
        
        base_dir = Path.cwd()
        images_dir = base_dir / config.IMAGES_DIR
        annotations_file = base_dir / config.INSTANCES_FILE
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Setup directories
        yolo_dir = base_dir / "final_yolo_data"
        train_dir = yolo_dir / "train"
        val_dir = yolo_dir / "val"
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        class_mapping = {cat['id']: cat['id'] - 1 for cat in coco_data['categories']}
        
        # Split data
        all_images = coco_data['images']
        train_size = int(0.88 * len(all_images))  # 88% train, 12% val
        train_images = all_images[:train_size]
        val_images = all_images[train_size:]
        
        logger.info(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
        
        # Process splits
        self._process_split(train_images, coco_data, images_dir, train_dir, 
                          class_mapping, "train", use_augmentation=True)
        self._process_split(val_images, coco_data, images_dir, val_dir, 
                          class_mapping, "val", use_augmentation=False)
        
        # Create dataset config
        self._create_dataset_yaml(yolo_dir)
        
        logger.info(f"✅ Dataset prepared at {yolo_dir}")
        return yolo_dir
    
    def _process_split(self, image_list, coco_data, images_dir, split_dir, 
                      class_mapping, split_name, use_augmentation=True):
        """Process train/val split with augmentation"""
        
        for idx, image_info in enumerate(image_list):
            if idx % 50 == 0:
                logger.info(f"Processing {split_name}: {idx+1}/{len(image_list)}")
            
            image_id = image_info['id']
            image_name = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Get annotations
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            
            # Load image
            img_path = images_dir / image_name
            if not img_path.exists():
                continue
                
            original_image = cv2.imread(str(img_path))
            if original_image is None:
                continue
            
            # Prepare bboxes for augmentation
            bboxes = []
            class_labels = []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])  # Convert to x1, y1, x2, y2
                class_labels.append(class_mapping[ann['category_id']])
            
            # Save original
            self._save_image_and_labels(original_image, bboxes, class_labels, 
                                      split_dir, image_name, img_width, img_height)
            
            # Create augmented versions for training
            if use_augmentation and split_name == "train":
                self._create_augmented_samples(original_image, bboxes, class_labels, 
                                             split_dir, image_name)
    
    def _create_augmented_samples(self, image, bboxes, class_labels, split_dir, base_name):
        """Create augmented samples with various transformations"""
        base_stem = Path(base_name).stem
        
        for aug_idx in range(config.AUGMENT_FACTOR):
            try:
                # Apply standard augmentation
                augmented = self.augment_pipeline(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
                
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                if len(aug_bboxes) == 0:
                    continue
                
                # Save augmented image
                aug_name = f"{base_stem}_aug_{aug_idx}.jpg"
                self._save_image_and_labels(aug_image, aug_bboxes, aug_labels,
                                          split_dir, aug_name, aug_image.shape[1], aug_image.shape[0])
                
                # Create particle background version (30% chance)
                if random.random() < 0.3:
                    bg = random.choice(self.backgrounds)
                    bg_resized = cv2.resize(bg, (aug_image.shape[1], aug_image.shape[0]))
                    
                    # Blend with background
                    alpha = random.uniform(0.75, 0.92)
                    composite = cv2.addWeighted(aug_image, alpha, bg_resized, 1-alpha, 0)
                    
                    # Save composite
                    comp_name = f"{base_stem}_comp_{aug_idx}.jpg"
                    self._save_image_and_labels(composite, aug_bboxes, aug_labels,
                                              split_dir, comp_name, composite.shape[1], composite.shape[0])
                
            except Exception as e:
                logger.warning(f"Augmentation failed for {base_name}_{aug_idx}: {e}")
                continue
    
    def _save_image_and_labels(self, image, bboxes, class_labels, split_dir, img_name, img_w, img_h):
        """Save image and corresponding YOLO labels"""
        # Save image
        img_path = split_dir / "images" / img_name
        cv2.imwrite(str(img_path), image)
        
        # Save labels
        label_path = split_dir / "labels" / f"{Path(img_name).stem}.txt"
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                # Convert to YOLO format
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
    
    def _create_dataset_yaml(self, yolo_dir):
        """Create dataset configuration file"""
        dataset_config = {
            'path': str(yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'Pikachu', 1: 'Charizard', 2: 'Bulbasaur', 3: 'Mewtwo'},
            'nc': 4
        }
        
        with open(yolo_dir / 'pokemon_final.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

def train_final_yolo_model():
    """Train the final optimized YOLO model"""
    logger.info("🚀 Starting final YOLO training...")
    
    # Prepare dataset
    data_prep = EnhancedDataPreparation()
    yolo_data_dir = data_prep.prepare_yolo_dataset()
    
    # Initialize model - use YOLOv8m for better accuracy
    model = YOLO('yolov8m.pt')
    
    # Optimized training configuration
    training_config = {
        'data': str(yolo_data_dir / 'pokemon_final.yaml'),
        'epochs': config.YOLO_EPOCHS,
        'imgsz': config.YOLO_IMG_SIZE,
        'batch': config.YOLO_BATCH,
        'patience': config.YOLO_PATIENCE,
        
        # Learning rate schedule
        'lr0': 0.008,  # Lower initial LR for stability
        'lrf': 0.0008,  # Final LR
        'momentum': 0.94,
        'weight_decay': 0.0008,
        'warmup_epochs': 8,
        'warmup_momentum': 0.85,
        'warmup_bias_lr': 0.08,
        
        # Augmentation (in addition to our custom)
        'hsv_h': 0.012,
        'hsv_s': 0.6,
        'hsv_v': 0.35,
        'degrees': 8.0,  # Lower since we handle rotation in custom aug
        'translate': 0.08,
        'scale': 0.4,
        'shear': 1.5,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.05,
        'copy_paste': 0.0,
        
        # Training settings
        'optimizer': 'AdamW',
        'close_mosaic': 20,  # Disable mosaic in last epochs
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        
        # Validation
        'val': True,
        'split': 'val',
        'save': True,
        'save_period': 15,
        'cache': False,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': 'pokemon_final',
        'name': 'single_target_detection',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        
        # Detection thresholds
        'conf': 0.05,  # Very low for training
        'iou': 0.6,
    }
    
    # Train model
    logger.info("Starting training with optimized hyperparameters...")
    results = model.train(**training_config)
    
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    logger.info(f"✅ Training completed! Best model: {best_model_path}")
    
    return str(best_model_path)

class FinalPokemonDetector:
    """Final production-ready Pokemon detector"""
    
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on {self.device}")
        
        self.model = YOLO(model_path)
        self.class_map = {0: 'Pikachu', 1: 'Charizard', 2: 'Bulbasaur', 3: 'Mewtwo'}
        
        # Validation parameters optimized for single targets
        self.validation_params = {
            'min_confidence': 0.08,
            'size_ranges': {
                'Pikachu': (300, 12000),
                'Charizard': (800, 20000),
                'Bulbasaur': (400, 15000),
                'Mewtwo': (600, 18000)
            },
            'aspect_ratios': {
                'Pikachu': (0.6, 1.8),
                'Charizard': (0.5, 2.2),
                'Bulbasaur': (0.6, 1.9),
                'Mewtwo': (0.4, 2.5)
            }
        }
    
    def preprocess_test_image(self, image):
        """Enhanced preprocessing for noisy test images"""
        # Multi-stage denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 12, 12, 7, 21)
        
        # Adaptive contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(10, 10))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Gentle sharpening
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend for natural look
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def detect_single_target(self, image_path, target_pokemon, confidence_threshold=0.08):
        """Detect specific target Pokemon with maximum precision"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            # Preprocess image
            processed = self.preprocess_test_image(image)
            
            # Multi-scale detection for robustness
            all_detections = []
            
            scales = [1.0, 0.92, 1.08]  # Multiple scales
            for scale in scales:
                if scale != 1.0:
                    h, w = processed.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_img = cv2.resize(processed, (new_w, new_h))
                    
                    if scale > 1.0:
                        # Crop back to original size
                        start_y = (new_h - h) // 2
                        start_x = (new_w - w) // 2
                        scaled_img = scaled_img[start_y:start_y+h, start_x:start_x+w]
                    else:
                        # Pad to original size
                        pad_y = (h - new_h) // 2
                        pad_x = (w - new_w) // 2
                        scaled_img = cv2.copyMakeBorder(
                            scaled_img, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, 
                            cv2.BORDER_REFLECT
                        )
                else:
                    scaled_img = processed
                
                # Run detection
                results = self.model(scaled_img, conf=confidence_threshold, 
                                   iou=config.NMS_THRESHOLD, verbose=False)
                
                # Process results for target Pokemon only
                scale_detections = self._extract_target_detections(
                    results, target_pokemon, image
                )
                all_detections.extend(scale_detections)
            
            # Merge and validate detections
            final_detections = self._merge_and_validate(all_detections, target_pokemon)
            
            return final_detections
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    def _extract_target_detections(self, results, target_pokemon, original_image):
        """Extract only target Pokemon detections"""
        detections = []
        target_class_id = None
        
        # Find target class ID
        for class_id, name in self.class_map.items():
            if name == target_pokemon:
                target_class_id = class_id
                break
        
        if target_class_id is None:
            return detections
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    
                    # Only process target Pokemon
                    if class_id == target_class_id:
                        confidence = float(box.conf.item())
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        
                        # Validate detection
                        if self._validate_detection(bbox, target_pokemon, confidence):
                            detections.append({
                                'name': target_pokemon,
                                'x_center': x_center,
                                'y_center': y_center,
                                'confidence': confidence,
                                'bbox': bbox.tolist()
                            })
        
        return detections
    
    def _validate_detection(self, bbox, pokemon_name, confidence):
        """Validate detection using size and aspect ratio constraints"""
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Size validation
        min_area, max_area = self.validation_params['size_ranges'].get(
            pokemon_name, (200, 25000)
        )
        
        # Aspect ratio validation
        min_aspect, max_aspect = self.validation_params['aspect_ratios'].get(
            pokemon_name, (0.3, 3.0)
        )
        
        size_valid = min_area <= area <= max_area
        aspect_valid = min_aspect <= aspect_ratio <= max_aspect
        conf_valid = confidence >= self.validation_params['min_confidence']
        
        return size_valid and aspect_valid and conf_valid
    
    def _merge_and_validate(self, detections, target_pokemon):
        """Merge overlapping detections and return best candidates"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove overlapping detections
        merged = []
        for detection in detections:
            is_duplicate = False
            
            for existing in merged:
                # Check if too close to existing detection
                dist = np.sqrt(
                    (detection['x_center'] - existing['x_center']) ** 2 +
                    (detection['y_center'] - existing['y_center']) ** 2
                )
                
                if dist < 60:  # 60 pixel threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(detection)
        
        return merged[:8]  # Return top 8 candidates

# Main execution
def main():
    """Main training function"""
    logger.info("🎯 Starting Final Pokemon CV Training System")
    
    try:
        # Train the final model
        best_model_path = train_final_yolo_model()
        
        # Test the detector
        logger.info("Testing final detector...")
        detector = FinalPokemonDetector(best_model_path)
        
        logger.info("✅ Final CV Module Ready for Production!")
        logger.info(f"📁 Model saved at: {best_model_path}")
        
        return best_model_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()