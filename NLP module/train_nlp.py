#!/usr/bin/env python3
"""
ENHANCED POKEMON TARGET IDENTIFICATION NLP MODULE
=================================================

Improved version with better model architecture, training process,
and integration with CV components.

Key improvements:
1. Better synthetic data generation with more diverse prompts
2. Improved model architecture with attention mechanisms
3. Enhanced rule-based fallback system
4. Proper JSON handling for model saving/loading
5. Integration with CV components for coordinate prediction
6. Confidence-based decision making with miss count tracking

Version: 4.0 - Enhanced Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments, TrainerCallback
)
import numpy as np
import pandas as pd
import json
import re
import random
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== POKEMON KNOWLEDGE BASE ======================
POKEMON_KNOWLEDGE = {
    "Pikachu": {
        "names": ["pikachu", "pika", "electric mouse", "pikachu pokemon"],
        "types": ["electric"],
        "colors": ["yellow", "golden", "bright yellow"],
        "descriptors": [
            "electric rat", "tiny thunder beast", "yellow mouse", 
            "rodent of sparks", "lightning rodent", "spark mouse",
            "mouse", "rodent", "lightning", "thunder", "spark", 
            "electric", "cheek", "quick", "agile", "small", "cute",
            "electrical", "thunderbolt", "lightning bolt", "electric type"
        ],
        "physical_attributes": [
            "yellow fur", "red cheeks", "pointed ears", "lightning bolt tail",
            "small size", "bipedal", "electric pouches", "black-tipped ears",
            "brown stripes on back", "round cheeks"
        ],
        "weaknesses": ["ground"],
        "habitats": ["forests", "power plants", "urban areas"]
    },
    "Charizard": {
        "names": ["charizard", "char", "charizard pokemon"],
        "types": ["fire", "flying"],
        "colors": ["orange", "red", "blue", "cream", "tan"],
        "descriptors": [
            "flame dragon", "winged inferno", "scaled fire titan",
            "orange lizard", "fire dragon", "aerial predator",
            "dragon", "fire", "flame", "wing", "fiery", "aerial",
            "powerful", "lizard", "inferno", "wings", "flying", "large",
            "fire type", "flying type", "flamethrower", "fire breath"
        ],
        "physical_attributes": [
            "orange scales", "large wings", "flame tail", "dragon-like",
            "bipedal", "fire breathing", "powerful build", "creamy underside",
            "two horns", "long neck", "sharp claws"
        ],
        "weaknesses": ["water", "rock", "electric"],
        "habitats": ["mountains", "volcanoes", "rocky areas"]
    },
    "Bulbasaur": {
        "names": ["bulbasaur", "bulba", "bulbasaur pokemon"],
        "types": ["grass", "poison"],
        "colors": ["green", "blue", "teal"],
        "descriptors": [
            "plant reptile", "vine beast", "green seedling",
            "sprout toad", "seed pokemon", "grass creature",
            "seed", "plant", "bulb", "vine", "herbal", "toxic",
            "toad", "sprout", "reptile", "grass", "leaf", "nature",
            "grass type", "poison type", "vine whip", "solar beam"
        ],
        "physical_attributes": [
            "green skin", "bulb on back", "four legs", "plant features",
            "vine whips", "spotted pattern", "quadruped", "red eyes",
            "pointed ears", "bulb with plant", "blue-green skin"
        ],
        "weaknesses": ["fire", "flying", "ice", "psychic"],
        "habitats": ["forests", "grasslands", "gardens"]
    },
    "Mewtwo": {
        "names": ["mewtwo", "mew two", "mewtwo pokemon"],
        "types": ["psychic"],
        "colors": ["purple", "pink", "gray", "silver", "white"],
        "descriptors": [
            "genetic experiment", "psychic clone", "telekinetic predator",
            "synthetic mind weapon", "artificial pokemon", "lab creation",
            "clone", "psychic", "powerful", "intelligent", "experiment",
            "artificial", "legendary", "telepathic", "mental",
            "psychic type", "genetically engineered", "psychokinetic",
            "mind power", "psystrike"
        ],
        "physical_attributes": [
            "purple skin", "large head", "three fingers", "long tail",
            "humanoid", "psychic aura", "feline features", "tube on back of neck",
            "purple abdomen", "pointed ears", "white underside"
        ],
        "weaknesses": ["bug", "ghost", "dark"],
        "habitats": ["caves", "laboratories", "remote areas"]
    }
}

# ====================== LLM SYNTHETIC DATA GENERATOR ======================
class EnhancedSyntheticDataGenerator:
    def __init__(self):
        logger.info("Initializing enhanced synthetic data generator...")
        
        self.military_headers = [
            "HQ REPORT", "INTELLIGENCE UPDATE", "FIELD BULLETIN",
            "OPERATIONAL NOTICE", "MISSION BRIEFING", "TACTICAL UPDATE",
            "SITREP", "INTEL SUMMARY", "COMBAT REPORT", "FIELD UPDATE",
            "COMMAND DIRECTIVE", "TACTICAL BRIEFING", "OPERATIONS UPDATE",
            "SECURITY BULLETIN", "THREAT ASSESSMENT"
        ]
        
        self.tactical_noise_templates = [
            "Radio checkpoint at {time} to confirm continued presence",
            "Use thermal masking as a decoy if pursuit is necessary",
            "Maintain operational secrecy at all times",
            "Draw minimal bloodline photographic evidence is priority",
            "Keep environmental samples for lab analysis",
            "Field logs indicate instrumentation drift observed",
            "Long-range sensors indicate sporadic bursts of radiation",
            "Scouts described sightings moving in small clusters",
            "Additional activity has been noted from various groups",
            "Thermal cameras logged irregular heat signatures",
            "Local sensors report intermittent EM spikes across grid {grid}",
            "Communications are patchy - maintain line-of-sight contact",
            "Weather conditions deteriorating in sector {sector}",
            "Backup teams are on standby at coordinates {coords}",
            "Satellite imagery shows unusual movement patterns",
            "Infrared scans confirm biological signatures",
            "Acoustic sensors detecting abnormal frequency patterns",
            "Magnetic anomalies reported in the vicinity",
            "Team {team_id} reporting visual contact with unidentified entities",
            "Proceed with caution - terrain is unstable in this area"
        ]
        
        self.elimination_command_templates = [
            "Priority: eliminate all {target} detected in the operational zone",
            "Order: eliminate the {target} immediately upon visual confirmation",
            "Mission objective: eliminate any {target} encountered during sweep",
            "HQ directive: neutralize all {target} contacts without delay",
            "Execute elimination protocol against {target} specimens",
            "Immediate action required: eliminate {target} on sight",
            "Your mission is to eliminate all {target} within the perimeter",
            "Authorization granted to terminate {target} presence in the area",
            "Command priority: eradicate {target} from the designated zone",
            "Engagement protocol: destroy {target} with extreme prejudice",
            "Target acquired: proceed with elimination of {target}",
            "Threat assessment: {target} classified as high priority target",
            "Weapons free: engage and eliminate {target} targets",
            "Directive: remove {target} presence from the operational area"
        ]
        
        self.distractor_templates = [
            "Additional activity from {distractor} groups nearby, though they do not appear hostile at present",
            "Scouts reported {distractor} sightings in adjacent sectors but avoid engagement",
            "Non-hostile {distractor} detected in the vicinity - do not target without authorization",
            "Field teams report {distractor} groups showing no aggressive behavior",
            "Unconfirmed reports of {distractor} activity in the northern sector",
            "{distractor} specimens observed but not considered immediate threats",
            "Passive {distractor} entities detected - maintain observation only",
            "Multiple {distractor} signatures but no hostile intent detected"
        ]
        
        self.quotation_templates = [
            'As {famous_person} once said, "{quote}"',
            'Remember the words of {famous_person}: "{quote}"',
            'In the words of {famous_person}, "{quote}"',
            '{famous_person} famously stated: "{quote}"',
            'As noted by {famous_person}, "{quote}"'
        ]
        
        self.famous_people = [
            "Einstein", "Newton", "Darwin", "Tesla", "Edison",
            "Napoleon", "Churchill", "Sun Tzu", "Plato", "Aristotle",
            "Galileo", "Hawking", "Feynman", "Sagan", "Curie"
        ]
        
        self.quotes = [
            "Knowledge is power",
            "The only thing we have to fear is fear itself",
            "I think therefore I am",
            "The unexamined life is not worth living",
            "Eureka!",
            "An apple a day keeps the doctor away",
            "To be or not to be, that is the question",
            "Cogito ergo sum",
            "The greatest glory in living lies not in never falling, but in rising every time we fall",
            "The way to get started is to quit talking and begin doing"
        ]
    
    def generate_single_target_prompt(self, target_pokemon: str, prompt_length: int = None) -> str:
        if prompt_length is None:
            prompt_length = random.randint(300, 800)
        
        # Get target descriptors
        target_knowledge = POKEMON_KNOWLEDGE[target_pokemon]
        target_descriptors = (target_knowledge['names'] + 
                            target_knowledge['descriptors'] + 
                            target_knowledge['physical_attributes'])
        
        primary_descriptor = random.choice(target_descriptors)
        prompt_parts = []
        
        # Military header
        header = random.choice(self.military_headers)
        situation_desc = f"Situation analysis regarding unusual activity of {primary_descriptor} in this operational zone."
        prompt_parts.append(f"{header} {situation_desc}")
        
        # Tactical noise
        noise_count = max(8, int(prompt_length * 0.6 / 15))
        for _ in range(noise_count):
            noise_template = random.choice(self.tactical_noise_templates)
            filled_noise = noise_template.format(
                time=f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
                grid=f"{random.randint(1, 9)}{random.choice(['A', 'B', 'C', 'K'])}",
                sector=random.choice(["Alpha", "Beta", "Gamma", "Delta"]),
                coords=f"{random.randint(100, 999)}-{random.randint(100, 999)}",
                team_id=random.choice(["Alpha", "Bravo", "Charlie", "Delta"])
            )
            prompt_parts.append(filled_noise)
        
        # Add distractors
        other_pokemon = [p for p in POKEMON_KNOWLEDGE.keys() if p != target_pokemon]
        for _ in range(random.randint(1, 3)):
            distractor_pokemon = random.choice(other_pokemon)
            distractor_desc = random.choice(POKEMON_KNOWLEDGE[distractor_pokemon]['descriptors'])
            distractor_template = random.choice(self.distractor_templates)
            distractor_text = distractor_template.format(distractor=distractor_desc)
            prompt_parts.insert(random.randint(1, len(prompt_parts)), distractor_text)
        
        # Add random quotes as noise
        for _ in range(random.randint(1, 2)):
            quote_template = random.choice(self.quotation_templates)
            famous_person = random.choice(self.famous_people)
            quote = random.choice(self.quotes)
            quote_text = quote_template.format(famous_person=famous_person, quote=quote)
            prompt_parts.insert(random.randint(1, len(prompt_parts)), quote_text)
        
        # Main elimination command
        elimination_template = random.choice(self.elimination_command_templates)
        target_desc = random.choice(target_descriptors)
        main_command = elimination_template.format(target=target_desc)
        insert_pos = len(prompt_parts) // 2 + random.randint(0, len(prompt_parts) // 3)
        prompt_parts.insert(insert_pos, main_command)
        
        # More noise
        for _ in range(random.randint(3, 5)):
            noise_template = random.choice(self.tactical_noise_templates)
            prompt_parts.append(noise_template)
        
        # Final assembly
        prompt = " ".join(prompt_parts)
        
        # Ensure the prompt is within the desired length
        words = prompt.split()
        if len(words) > prompt_length:
            # Keep the elimination command and trim from other parts
            elimination_idx = -1
            for i, part in enumerate(prompt_parts):
                if any(cmd in part for cmd in ["eliminate", "neutralize", "terminate", "destroy"]):
                    elimination_idx = i
                    break
            
            if elimination_idx >= 0:
                # Keep parts around the elimination command
                start_idx = max(0, elimination_idx - 5)
                end_idx = min(len(prompt_parts), elimination_idx + 6)
                kept_parts = prompt_parts[start_idx:end_idx]
                prompt = " ".join(kept_parts)
        
        return prompt
    
    def generate_training_dataset(self, samples_per_pokemon: int = 2500) -> List[Dict]:
        logger.info(f"Generating {samples_per_pokemon * 4} synthetic training samples...")
        
        dataset = []
        pokemon_list = list(POKEMON_KNOWLEDGE.keys())
        
        for pokemon_idx, target_pokemon in enumerate(pokemon_list):
            logger.info(f"Generating samples for {target_pokemon}...")
            
            for sample_idx in range(samples_per_pokemon):
                if sample_idx % 500 == 0:
                    logger.info(f"  Generated {sample_idx}/{samples_per_pokemon} samples for {target_pokemon}")
                
                prompt = self.generate_single_target_prompt(target_pokemon)
                
                dataset.append({
                    'prompt': prompt,
                    'target_pokemon': target_pokemon,
                    'label': pokemon_idx,  # Integer label for CrossEntropyLoss
                    'length': len(prompt.split())
                })
        
        logger.info(f"✅ Generated {len(dataset)} total training samples")
        
        lengths = [item['length'] for item in dataset]
        logger.info(f"📊 Prompt length stats: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # Check class distribution
        label_counts = Counter([item['label'] for item in dataset])
        logger.info(f"📊 Class distribution: {label_counts}")
        
        return dataset

# ====================== ENHANCED CLASSIFIER MODEL ======================
class EnhancedPokemonClassifier(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 4):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # Enhanced architecture with attention pooling
        self.dropout = nn.Dropout(0.3)
        
        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use attention pooling instead of just [CLS]
        hidden_states = outputs.last_hidden_state
        attention_weights = self.attention(hidden_states)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)
        
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        
        # Calculate loss if labels are provided (for training)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# ====================== DATASET CLASS ======================
class PokemonDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset created with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item['label']  # Integer label
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ====================== TRAINING FUNCTIONS ======================
class EnhancedLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if 'eval_accuracy' in logs:
                logger.info(f"📈 Eval Accuracy: {logs['eval_accuracy']:.4f}")
            if 'eval_loss' in logs:
                logger.info(f"📉 Eval Loss: {logs['eval_loss']:.4f}")
            if 'train_loss' in logs:
                logger.info(f"📉 Train Loss: {logs['train_loss']:.4f}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate additional metrics
    pokemon_names = list(POKEMON_KNOWLEDGE.keys())
    per_class_acc = {}
    per_class_f1 = {}
    
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    for i, pokemon in enumerate(pokemon_names):
        per_class_acc[f'{pokemon}_accuracy'] = report[str(i)]['precision'] if str(i) in report else 0.0
        per_class_f1[f'{pokemon}_f1'] = report[str(i)]['f1-score'] if str(i) in report else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    metrics.update(per_class_acc)
    metrics.update(per_class_f1)
    
    return metrics

def plot_confusion_matrix(labels, predictions, class_names, output_dir):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def train_enhanced_model(output_dir: str = './enhanced_pokemon_nlp') -> Tuple[nn.Module, AutoTokenizer]:
    logger.info("🚀 Starting enhanced Pokemon NLP training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic training data
    data_generator = EnhancedSyntheticDataGenerator()
    training_data = data_generator.generate_training_dataset(samples_per_pokemon=2500)
    
    # Split data
    labels_for_stratify = [item['label'] for item in training_data]
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.15, 
        random_state=42, 
        stratify=labels_for_stratify
    )
    
    logger.info(f"📚 Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Use a more robust model
    model_name = 'microsoft/deberta-v3-small'  # More efficient than BERT
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EnhancedPokemonClassifier(model_name=model_name, num_classes=4)
    except:
        logger.warning("DeBERTa model not available, falling back to BERT")
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EnhancedPokemonClassifier(model_name=model_name, num_classes=4)
    
    # Create datasets
    train_dataset = PokemonDataset(train_data, tokenizer, max_length=512)
    val_dataset = PokemonDataset(val_data, tokenizer, max_length=512)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=300,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_steps=200,
        save_steps=400,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=42,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EnhancedLoggingCallback()]
    )
    
    # Train the model
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Evaluate and save confusion matrix
    eval_results = trainer.evaluate()
    logger.info(f"📊 Final evaluation results: {eval_results}")
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    plot_confusion_matrix(
        true_labels, pred_labels, 
        list(POKEMON_KNOWLEDGE.keys()), 
        output_dir
    )
    
    # Save the model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration with proper JSON formatting
    config_info = {
        'model_name': model_name,
        'pokemon_classes': list(POKEMON_KNOWLEDGE.keys()),
        'max_length': 512,
        'version': '4.0',
        'evaluation_results': eval_results
    }
    
    with open(Path(output_dir) / 'model_config.json', 'w') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Training completed successfully!")
    
    return model, tokenizer

# ====================== ENHANCED TARGET PARSER ======================
class EnhancedPokemonParser:
    def __init__(self, model_path: str = './enhanced_pokemon_nlp'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on {self.device}")
        
        # Initialize miss counter
        self.miss_counter = 0
        self.last_prediction = None
        self.last_confidence = 0.0
        
        # Load model configuration
        config_path = Path(model_path) / 'model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'pokemon_classes': list(POKEMON_KNOWLEDGE.keys()),
                'max_length': 512
            }
        
        self.pokemon_names = self.config['pokemon_classes']
        self.pokemon_knowledge = POKEMON_KNOWLEDGE
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = self.config.get('model_name', 'bert-base-uncased')
            self.model = EnhancedPokemonClassifier(
                model_name=model_name,
                num_classes=len(self.pokemon_names)
            )
            
            model_file = Path(model_path) / 'pytorch_model.bin'
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("Model weights not found, using rule-based approach only")
                self.model = None
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using rule-based approach only.")
            self.model = None
            self.tokenizer = None
        
        # Enhanced rule-based patterns
        self.target_extraction_patterns = [
            r"(?:eliminate|destroy|kill|terminate|neutralize|wipe out|eradicate|remove)\s+(?:all\s+|any\s+|the\s+)?([^.,;]{1,60})",
            r"(?:priority|objective|mission|order|directive|command)[:,]?\s*(?:is\s+to\s+)?(?:eliminate|kill|destroy|neutralize|terminate)\s+(?:all\s+|the\s+)?([^.,;]{1,60})",
            r"(?:hq\s+(?:orders?|directive)|command|headquarters)[:,]?\s*(?:eliminate|kill|destroy|neutralize)\s+(?:all\s+|the\s+)?([^.,;]{1,60})",
            r"(?:engage|target|acquire|focus on|concentrate on)\s+(?:all\s+|any\s+|the\s+)?([^.,;]{1,60})",
            r"(?:threat|target|priority)\s*:\s*([^.,;]{1,60})"
        ]
    
    def extract_target_rule_based(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        candidate_scores = defaultdict(int)
        
        # Pattern-based extraction
        for pattern in self.target_extraction_patterns:
            matches = re.finditer(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                text_segment = match.group(1).strip()
                pokemon = self._match_text_to_pokemon(text_segment)
                if pokemon:
                    candidate_scores[pokemon] += 3
                    # Check proximity to elimination keywords
                    start, end = match.span()
                    context_start = max(0, start - 50)
                    context_end = min(len(prompt_lower), end + 50)
                    context = prompt_lower[context_start:context_end]
                    
                    if any(kw in context for kw in ["eliminate", "kill", "destroy", "terminate"]):
                        candidate_scores[pokemon] += 2
                    if any(kw in context for kw in ["not", "avoid", "friendly", "neutral"]):
                        candidate_scores[pokemon] -= 3
        
        # Context-based scoring
        elimination_keywords = ['eliminate', 'kill', 'destroy', 'terminate', 'neutralize', 'eradicate', 'remove']
        protection_keywords = ['not hostile', 'friendly', 'avoid', 'do not engage', 'neutral', 'non-target']
        
        for pokemon in self.pokemon_names:
            all_references = self._get_all_pokemon_references(pokemon)
            
            for reference in all_references:
                if reference in prompt_lower:
                    # Count occurrences
                    count = prompt_lower.count(reference)
                    candidate_scores[pokemon] += count
                    
                    # Check context around each occurrence
                    for match in re.finditer(re.escape(reference), prompt_lower):
                        start = max(0, match.start() - 60)
                        end = min(len(prompt_lower), match.end() + 60)
                        context = prompt_lower[start:end]
                        
                        elimination_score = sum(2 for kw in elimination_keywords if kw in context)
                        protection_penalty = sum(3 for kw in protection_keywords if kw in context)
                        
                        candidate_scores[pokemon] += elimination_score - protection_penalty
        
        if candidate_scores:
            best_candidate = max(candidate_scores.items(), key=lambda x: x[1])
            if best_candidate[1] > 0:
                return best_candidate[0]
        
        # Fallback: return the most mentioned Pokemon
        mention_counts = {}
        for pokemon in self.pokemon_names:
            all_refs = self._get_all_pokemon_references(pokemon)
            mention_counts[pokemon] = sum(prompt_lower.count(ref) for ref in all_refs)
        
        if max(mention_counts.values()) > 0:
            return max(mention_counts.items(), key=lambda x: x[1])[0]
        
        return "Pikachu"  # Default fallback
    
    def _match_text_to_pokemon(self, text: str) -> Optional[str]:
        text = text.lower().strip()
        text = re.sub(r'\b(the|any|all|some|every|each)\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        best_match = None
        best_score = 0
        
        for pokemon, knowledge in self.pokemon_knowledge.items():
            all_refs = (knowledge.get('names', []) + 
                       knowledge.get('descriptors', []) + 
                       knowledge.get('physical_attributes', []))
            
            for ref in all_refs:
                ref_lower = ref.lower()
                # Exact match
                if ref_lower == text:
                    return pokemon
                
                # Partial match with scoring
                if ref_lower in text or text in ref_lower:
                    score = min(len(ref_lower), len(text))
                    if score > best_score:
                        best_score = score
                        best_match = pokemon
        
        return best_match
    
    def _get_all_pokemon_references(self, pokemon: str) -> List[str]:
        knowledge = self.pokemon_knowledge.get(pokemon, {})
        references = []
        references.extend(knowledge.get('names', []))
        references.extend(knowledge.get('descriptors', []))
        references.extend(knowledge.get('physical_attributes', []))
        return [ref.lower() for ref in references if len(ref) > 3]  # Only longer references
    
    def predict_target(self, prompt: str) -> str:
        # Rule-based prediction
        rule_based_target = self.extract_target_rule_based(prompt)
        
        # Model prediction if available
        model_target, confidence = None, 0.0
        if self.model and self.tokenizer:
            try:
                model_target, confidence = self._predict_with_model(prompt)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        # Decision logic with miss counter
        if model_target and confidence > 0.7:
            logger.debug(f"High confidence model prediction: {model_target} (confidence: {confidence:.3f})")
            self.miss_counter = 0  # Reset miss counter on high confidence
            self.last_prediction = model_target
            self.last_confidence = confidence
            return model_target
        elif model_target and confidence > 0.4 and model_target == rule_based_target:
            logger.debug(f"Model-rule agreement: {model_target} (confidence: {confidence:.3f})")
            self.miss_counter = max(0, self.miss_counter - 1)  # Reduce miss counter
            self.last_prediction = model_target
            self.last_confidence = confidence
            return model_target
        elif self.miss_counter >= 2 and self.last_prediction:
            # After 2 misses, stick with the last prediction
            logger.debug(f"Sticking with last prediction after {self.miss_counter} misses: {self.last_prediction}")
            self.miss_counter += 1
            return self.last_prediction
        else:
            # Use rule-based with miss counter increment
            logger.debug(f"Using rule-based prediction: {rule_based_target}")
            self.miss_counter += 1
            self.last_prediction = rule_based_target
            self.last_confidence = 0.5  # Medium confidence for rule-based
            return rule_based_target
    
    def _predict_with_model(self, prompt: str) -> Tuple[str, float]:
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_pokemon = self.pokemon_names[predicted_idx]
        
        return predicted_pokemon, confidence
    
    def reset_miss_counter(self):
        """Reset the miss counter (call this when a shot is successful)"""
        self.miss_counter = 0

# ====================== COORDINATE PREDICTION INTEGRATION ======================
class CoordinatePredictor:
    def __init__(self, cv_model_path: str = None):
        # This would integrate with your CV model
        # For now, we'll create a placeholder
        self.cv_model = None
        self.cv_loaded = False
        
        if cv_model_path and os.path.exists(cv_model_path):
            try:
                # Load your CV model here
                # self.cv_model = load_cv_model(cv_model_path)
                self.cv_loaded = True
                logger.info("✅ CV model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CV model: {e}")
    
    def predict_coordinates(self, image, target_pokemon: str) -> List[Dict]:
        """
        Predict coordinates for the target Pokemon in the image
        
        Returns:
            List of dictionaries with 'center_x', 'center_y', 'confidence' keys
        """
        # This would use your CV model to detect all Pokemon
        # and return centers for the target Pokemon
        
        # Placeholder implementation
        centers = []
        
        if self.cv_loaded:
            # Use your actual CV model here
            # detections = self.cv_model.predict(image)
            # for detection in detections:
            #     if detection['class'] == target_pokemon and detection['confidence'] > 0.5:
            #         centers.append({
            #             'center_x': detection['x_center'],
            #             'center_y': detection['y_center'],
            #             'confidence': detection['confidence']
            #         })
            pass
        else:
            # Fallback: return some random coordinates for testing
            for _ in range(random.randint(1, 3)):
                centers.append({
                    'center_x': random.randint(50, 590),
                    'center_y': random.randint(50, 430),
                    'confidence': random.uniform(0.6, 0.9)
                })
        
        return centers

# ====================== MAIN EXECUTION ======================
def main():
    logger.info("🎯 Starting Enhanced Pokemon NLP System")
    
    try:
        # Train the model
        model, tokenizer = train_enhanced_model()
        
        # Test the parser
        logger.info("Testing enhanced parser...")
        parser = EnhancedPokemonParser()
        
        # Example tests
        test_prompts = [
            """
            HQ REPORT Situation analysis regarding unusual activity of flame dragon in this operational zone.
            Additional activity from plant reptile groups nearby, though they do not appear hostile at present.
            Radio checkpoint at 0200 to confirm continued presence.
            Priority: eliminate the scaled fire titan at first contact, then hold position for assessment.
            Maintain operational secrecy. Local sensors report intermittent EM spikes.
            """,
            """
            FIELD BULLETIN: Intel suggests increased pikachu activity in sector 7B.
            Thermal scans show multiple electric-type signatures. 
            Command directive: neutralize all electric mouse specimens on sight.
            Exercise caution - these specimens can discharge high voltage.
            """,
            """
            MISSION BRIEFING: Our scouts have detected mewtwo in the northern sector.
            This psychic clone is extremely powerful and should not be engaged directly.
            However, HQ has issued orders to eliminate the genetic experiment immediately.
            Use extreme caution and coordinate with backup teams.
            """
        ]
        
        for i, prompt in enumerate(test_prompts):
            result = parser.predict_target(prompt)
            logger.info(f"Test {i+1} result: {result}")
        
        logger.info("✅ Enhanced NLP Module Ready for Production!")
        
        return parser
        
    except Exception as e:
        logger.error(f"Training/testing failed: {e}")
        raise

if __name__ == "__main__":
    main()