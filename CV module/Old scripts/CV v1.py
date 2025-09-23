#!/usr/bin/env python3
"""
COMPLETE COMPUTER VISION MODULE FOR POKEMON TARGETING SYSTEM
================================================================

This module handles Pokemon detection, classification, and targeting coordinate generation.
Optimized for noisy battlefield conditions with varied Pokemon orientations.

Key Features:
- YOLOv8-based detection with synthetic data augmentation
- Robust to background noise and orientation variations
- Confidence-based targeting with heuristic optimization
- COCO-format coordinate output
- Integration with NLP module for target identification

Author: Enhanced CV System for Pokemon Hackathon
Version: 1.0
"""

import torch
import numpy as np
import cv2
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

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

# ====================== DATASET CONFIGURATION ======================
class PokemonDatasetConfig:
    """Configuration for the Pokemon dataset"""
    
    def __init__(self, base_path: str = "D:\\IIT Madras\\Competitions\\dataset\\dataset"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.annotations_path = self.base_path / "annotations"
        self.train_annotations = self.annotations_path / "instances_train.json"
        self.train_prompts = self.annotations_path / "train_prompts.json"
        
        # Verify paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images path not found: {self.images_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations path not found: {self.annotations_path}")
        
        logger.info(f"Dataset configured with base path: {self.base_path}")

# ====================== DATA LOADER AND PROCESSING ======================
class PokemonDataLoader:
    """Loads and processes Pokemon dataset for YOLO training"""
    
    def __init__(self, config: PokemonDatasetConfig):
        self.config = config
        self.coco_data = self._load_coco_annotations()
        self.class_stats = self._analyze_class_distribution()
        
    def _load_coco_annotations(self) -> Dict:
        """Load COCO format annotations"""
        with open(self.config.train_annotations, 'r') as f:
            return json.load(f)
    
    def _analyze_class_distribution(self) -> Dict:
        """Analyze class distribution in the dataset"""
        class_counts = {class_id: 0 for class_id in POKEMON_CLASSES.keys()}
        
        for ann in self.coco_data['annotations']:
            class_id = ann['category_id'] - 1  # Convert from 1-4 to 0-3
            if class_id in class_counts:
                class_counts[class_id] += 1
        
        logger.info(f"Class distribution: {class_counts}")
        return class_counts
    
    def convert_to_yolo_format(self, output_dir: str = "yolo_dataset") -> None:
        """Convert COCO format to YOLO format"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create directories
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Create dataset YAML file
        yaml_content = f"""path: {output_path.absolute()}
        train: images/train
        val: images/val
        test: images/test

        names:
        0: Pikachu
        1: Charizard
        2: Bulbasaur
        3: Mewtwo
        """
        with open(output_path / "pokemon.yaml", "w") as f:
            f.write(yaml_content)
        
        # Process each image
        image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_id_to_info:
                continue
                
            img_info = image_id_to_info[image_id]
            img_width, img_height = img_info['width'], img_info['height']
            
            # Convert bbox from COCO to YOLO format
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            # Class ID (convert from 1-4 to 0-3)
            class_id = ann['category_id'] - 1
            
            # Create label file
            label_file = labels_dir / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_file, "a") as f:
                f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
            
            # Copy image to appropriate location
            src_image = self.config.images_path / img_info['file_name']
            if src_image.exists():
                dest_image = images_dir / img_info['file_name']
                if not dest_image.exists():
                    # Create a symlink to avoid duplicating images
                    os.symlink(src_image, dest_image)
        
        logger.info(f"YOLO dataset created at {output_path}")
        
        # Split dataset
        self._split_dataset(images_dir, labels_dir)
    
    def _split_dataset(self, images_dir: Path, labels_dir: Path) -> None:
        """Split dataset into train/val/test"""
        image_files = list(images_dir.glob("*.png"))
        image_files = [f for f in image_files if f.exists()]
        
        # Split: 70% train, 20% val, 10% test
        train_files, test_val_files = train_test_split(
            image_files, test_size=0.3, random_state=42
        )
        val_files, test_files = train_test_split(
            test_val_files, test_size=0.33, random_state=42
        )
        
        # Create directories
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(exist_ok=True)
            (labels_dir / split).mkdir(exist_ok=True)
        
        # Move files to appropriate directories
        for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for img_file in files:
                # Move image
                dest_img = images_dir / split / img_file.name
                if not dest_img.exists():
                    os.rename(img_file, dest_img)
                
                # Move corresponding label
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_label = labels_dir / split / f"{img_file.stem}.txt"
                    os.rename(label_file, dest_label)
        
        logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# ====================== SYNTHETIC DATA GENERATOR ======================
class SyntheticDataGenerator:
    """Generates synthetic training data to improve model robustness"""
    
    def __init__(self, original_images_path: str):
        self.original_images_path = Path(original_images_path)
        self.backgrounds = self._load_backgrounds()
        
    def _load_backgrounds(self) -> List[Path]:
        """Load background images for synthetic data generation"""
        # You can add more background images to this directory
        bg_path = Path("backgrounds")
        bg_path.mkdir(exist_ok=True)
        
        # Create some simple backgrounds if none exist
        if not list(bg_path.glob("*.jpg")):
            self._create_default_backgrounds(bg_path)
            
        return list(bg_path.glob("*.jpg"))
    
    def _create_default_backgrounds(self, bg_path: Path) -> None:
        """Create default background images"""
        colors = [
            (50, 50, 50),    # Dark gray
            (200, 200, 200), # Light gray
            (50, 50, 100),   # Dark blue
            (100, 50, 50),   # Dark red
            (50, 100, 50),   # Dark green
        ]
        
        for i, color in enumerate(colors):
            img = Image.new('RGB', (640, 480), color)
            img.save(bg_path / f"bg_{i}.jpg")
    
    def generate_synthetic_image(self, original_image_path: Path, annotations: List, 
                               output_dir: Path, augment: bool = True) -> None:
        """Generate a synthetic image with augmentations"""
        # Load original image
        original_img = Image.open(original_image_path)
        
        # Choose a random background
        bg_path = random.choice(self.backgrounds)
        background = Image.open(bg_path).resize(original_img.size)
        
        # Create a mask for the Pokemon
        mask = Image.new('L', original_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # Draw bounding box as filled rectangle
            draw.rectangle([x, y, x+w, y+h], fill=255)
        
        # Apply augmentations if requested
        if augment:
            # Random rotation
            angle = random.randint(-45, 45)
            original_img = original_img.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)
            
            # Random scaling
            scale = random.uniform(0.8, 1.2)
            new_size = (int(original_img.width * scale), int(original_img.height * scale))
            original_img = original_img.resize(new_size, Image.LANCZOS)
            mask = mask.resize(new_size, Image.LANCZOS)
            
            # Random brightness adjustment
            brightness = random.uniform(0.7, 1.3)
            original_img = Image.blend(original_img, Image.new('RGB', original_img.size, 
                                 (int(255*(1-brightness)),)*3), 0.5)
        
        # Composite Pokemon onto background
        result = Image.composite(original_img, background, mask)
        
        # Save synthetic image
        output_path = output_dir / f"synthetic_{original_image_path.stem}.png"
        result.save(output_path)
        
        # Update annotations for the synthetic image
        self._update_annotations(annotations, output_path, original_img.size, 
                               result.size if augment else original_img.size)
        
        return output_path
    
    def _update_annotations(self, annotations: List, image_path: Path, 
                          original_size: Tuple[int, int], new_size: Tuple[int, int]) -> None:
        """Update annotations for synthetic image"""
        # This would need to be implemented based on your augmentation transformations
        # For simplicity, we're not implementing the full coordinate transformation here
        pass

# ====================== YOLO TRAINER ======================
class YOLOTrainer:
    """Handles YOLO model training and evaluation"""
    
    def __init__(self, dataset_path: str = "yolo_dataset"):
        self.dataset_path = Path(dataset_path)
        self.model = None
        
    def train(self, model_size: str = "m", epochs: int = 100, 
              imgsz: int = 640, patience: int = 20) -> YOLO:
        """Train a YOLO model on the Pokemon dataset"""
        yaml_path = self.dataset_path / "pokemon.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
        
        # Load a pretrained model
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train the model
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            patience=patience,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            close_mosaic=10,
            degrees=45.0,  # Increased rotation for better orientation handling
            translate=0.2,
            scale=0.5,
            shear=0.0,
            perspective=0.0001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,  # Increased mixup for better generalization
            copy_paste=0.2,  # Increased copy-paste for better synthetic data handling
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,  # Random erasing for occlusion handling
            crop_fraction=0.8,  # Random cropping
            name=f"pokemon_yolov8{model_size}",
            save=True,
            save_period=10,
            cache=False,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=8,
            single_cls=False,
            rect=False,
            cos_lr=True,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            deterministic=False,
            verbose=True,
        )
        
        self.model = model
        logger.info("YOLO training completed successfully!")
        return model
    
    def evaluate(self, model_path: Optional[str] = None) -> Dict:
        """Evaluate the trained model"""
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for evaluation")
        
        results = model.val(
            data=str(self.dataset_path / "pokemon.yaml"),
            split='val',
            imgsz=640,
            batch=16,
            save_json=True,
            save_hybrid=False,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            device=0 if torch.cuda.is_available() else "cpu",
            dnn=False,
            plots=True,
            rect=False,
            verbose=True
        )
        
        logger.info(f"Model evaluation results: mAP50-95: {results.box.map:.3f}, mAP50: {results.box.map50:.3f}")
        return results

# ====================== POKEMON DETECTOR ======================
class PokemonDetector:
    """Main class for Pokemon detection and targeting"""
    
    def __init__(self, model_path: str = "best.pt", conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"PokemonDetector initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def detect(self, image_path: str, target_class: Optional[str] = None) -> List[Dict]:
        """
        Detect Pokemon in an image
        
        Args:
            image_path: Path to the image file
            target_class: If specified, only return detections of this class
        
        Returns:
            List of detection results with bounding boxes and confidence scores
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model(
            image_path, 
            conf=self.conf_threshold,
            device=self.device,
            imgsz=640,
            augment=True,  # Test time augmentation for better accuracy
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    bbox = box.xywh[0].cpu().numpy()  # x_center, y_center, width, height
                    
                    # Convert to COCO format (top-left origin)
                    x_center, y_center, width, height = bbox
                    x = x_center - width / 2
                    y = y_center - height / 2
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": POKEMON_CLASSES[class_id],
                        "confidence": confidence,
                        "bbox": [x, y, width, height],
                        "center": [x_center, y_center]
                    }
                    
                    # Filter by target class if specified
                    if target_class is None or detection["class_name"] == target_class:
                        detections.append(detection)
        
        return detections
    
    def get_targeting_coordinates(self, image_path: str, target_class: str, 
                                 max_shots: int = 10, confidence_strategy: str = "adaptive") -> List[List[float]]:
        """
        Get targeting coordinates for a specific Pokemon class
        
        Args:
            image_path: Path to the image file
            target_class: The Pokemon class to target
            max_shots: Maximum number of shots to take
            confidence_strategy: Strategy for confidence thresholding
                - "adaptive": Adjust threshold based on detection count
                - "conservative": High threshold to avoid misses
                - "aggressive": Low threshold to ensure all targets are hit
        
        Returns:
            List of targeting coordinates [[x1, y1], [x2, y2], ...]
        """
        # Get all detections
        all_detections = self.detect(image_path)
        
        # Filter for target class
        target_detections = [d for d in all_detections if d["class_name"] == target_class]
        
        if not target_detections:
            logger.warning(f"No {target_class} detections found in {image_path}")
            return []
        
        # Apply confidence strategy
        if confidence_strategy == "adaptive":
            # Adjust confidence threshold based on number of detections
            # Few detections -> lower threshold, many detections -> higher threshold
            conf_threshold = max(self.conf_threshold, 
                                min(0.7, self.conf_threshold * (1 + len(target_detections) / 10)))
            filtered_detections = [d for d in target_detections if d["confidence"] >= conf_threshold]
        elif confidence_strategy == "conservative":
            filtered_detections = [d for d in target_detections if d["confidence"] >= 0.5]
        elif confidence_strategy == "aggressive":
            filtered_detections = [d for d in target_detections if d["confidence"] >= 0.1]
        else:
            filtered_detections = target_detections
        
        # Sort by confidence (highest first)
        filtered_detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Apply non-maximum suppression to avoid duplicate detections
        filtered_detections = self._apply_nms(filtered_detections, iou_threshold=0.5)
        
        # Limit to max_shots
        if len(filtered_detections) > max_shots:
            logger.info(f"Limiting shots from {len(filtered_detections)} to {max_shots}")
            filtered_detections = filtered_detections[:max_shots]
        
        # Extract center coordinates
        coordinates = [d["center"] for d in filtered_detections]
        
        logger.info(f"Targeting {len(coordinates)} {target_class} instances in {image_path}")
        return coordinates
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to filter overlapping detections"""
        if not detections:
            return []
        
        # Convert to format for NMS
        boxes = []
        confidences = []
        for det in detections:
            x, y, w, h = det["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            confidences.append(det["confidence"])
        
        # Apply NMS
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(), 
            self.conf_threshold, iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Return filtered detections
        return [detections[i] for i in indices.flatten()]
    
    def visualize_detections(self, image_path: str, output_path: str, 
                           target_class: Optional[str] = None) -> None:
        """
        Visualize detections on an image and save the result
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the visualized image
            target_class: If specified, only visualize detections of this class
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get detections
        detections = self.detect(image_path, target_class)
        
        # Draw detections
        for det in detections:
            x, y, w, h = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]
            
            # Draw bounding box
            color = self._get_color_for_class(det["class_id"])
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (int(x), int(y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = det["center"]
            cv2.circle(image, (int(center_x), int(center_y)), 5, color, -1)
        
        # Save image
        cv2.imwrite(output_path, image)
        logger.info(f"Visualization saved to {output_path}")
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for each class"""
        colors = [
            (0, 255, 255),    # Yellow for Pikachu
            (0, 165, 255),    # Orange for Charizard
            (0, 255, 0),      # Green for Bulbasaur
            (255, 0, 255)     # Purple for Mewtwo
        ]
        return colors[class_id % len(colors)]

# ====================== COMPLETE PIPELINE INTEGRATION ======================
class CompletePokemonPipeline:
    """Complete pipeline integrating CV and NLP modules"""
    
    def __init__(self, nlp_parser, cv_detector):
        self.nlp_parser = nlp_parser
        self.cv_detector = cv_detector
        logger.info("Complete Pokemon Pipeline initialized!")
    
    def process_image_with_prompt(self, image_path: str, prompt: str) -> List[List[float]]:
        """
        Process an image with its associated prompt to generate targeting coordinates
        
        Args:
            image_path: Path to the image file
            prompt: The tactical prompt describing the mission
        
        Returns:
            List of targeting coordinates
        """
        # Step 1: Use NLP to extract target Pokemon from prompt
        target_pokemon = self.nlp_parser.predict_target(prompt)
        logger.info(f"NLP identified target: {target_pokemon}")
        
        # Step 2: Use CV to detect target Pokemon and generate coordinates
        coordinates = self.cv_detector.get_targeting_coordinates(
            image_path, target_pokemon, max_shots=10, confidence_strategy="adaptive"
        )
        
        logger.info(f"Generated {len(coordinates)} targeting coordinates")
        return coordinates
    
    def process_batch(self, image_prompt_pairs: List[Tuple[str, str]]) -> Dict[str, List[List[float]]]:
        """
        Process a batch of image-prompt pairs
        
        Args:
            image_prompt_pairs: List of (image_path, prompt) tuples
        
        Returns:
            Dictionary mapping image paths to targeting coordinates
        """
        results = {}
        
        for image_path, prompt in image_prompt_pairs:
            try:
                coordinates = self.process_image_with_prompt(image_path, prompt)
                results[image_path] = coordinates
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = []
        
        return results

# ====================== MAIN EXECUTION ======================
def main():
    """Main function to demonstrate the complete CV module"""
    logger.info("🚀 Starting Complete Pokemon CV System")
    
    try:
        # Initialize dataset configuration
        config = PokemonDatasetConfig()
        
        # Convert dataset to YOLO format
        data_loader = PokemonDataLoader(config)
        data_loader.convert_to_yolo_format()
        
        # Train YOLO model (comment out if using pre-trained weights)
        trainer = YOLOTrainer()
        model = trainer.train(model_size="m", epochs=100)
        
        # Evaluate model
        trainer.evaluate()
        
        # Initialize detector with trained model
        detector = PokemonDetector(model_path="path/to/trained/model.pt")
        
        # Test detection on a sample image
        sample_image = "path/to/sample/image.png"
        detections = detector.detect(sample_image)
        logger.info(f"Detected {len(detections)} Pokemon in sample image")
        
        # Visualize detections
        detector.visualize_detections(sample_image, "detection_results.jpg")
        
        logger.info("✅ CV Module Ready for Production!")
        
    except Exception as e:
        logger.error(f"CV module execution failed: {e}")
        raise

if __name__ == "__main__":
    main()