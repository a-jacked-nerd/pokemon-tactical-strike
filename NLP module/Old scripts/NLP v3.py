#!/usr/bin/env python3
"""
WORKING FINAL NLP MODULE FOR SINGLE TARGET POKEMON IDENTIFICATION
================================================================

Fixed version with proper loss function and training configuration.
This module is optimized for identifying exactly ONE target Pokemon per prompt.

Version: 3
"""

import torch
import torch.nn as nn
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== POKEMON KNOWLEDGE BASE ======================
POKEMON_KNOWLEDGE = {
    "Pikachu": {
        "names": ["pikachu", "pika", "electric mouse"],
        "types": ["electric"],
        "colors": ["yellow", "golden", "bright yellow"],
        "descriptors": [
            "electric rat", "tiny thunder beast", "yellow mouse", 
            "rodent of sparks", "lightning rodent", "spark mouse",
            "mouse", "rodent", "lightning", "thunder", "spark", 
            "electric", "cheek", "quick", "agile", "small", "cute"
        ],
        "physical_attributes": [
            "yellow fur", "red cheeks", "pointed ears", "lightning bolt tail",
            "small size", "bipedal", "electric pouches"
        ]
    },
    "Charizard": {
        "names": ["charizard", "char"],
        "types": ["fire", "flying"],
        "colors": ["orange", "red", "blue", "cream", "tan"],
        "descriptors": [
            "flame dragon", "winged inferno", "scaled fire titan",
            "orange lizard", "fire dragon", "aerial predator",
            "dragon", "fire", "flame", "wing", "fiery", "aerial",
            "powerful", "lizard", "inferno", "wings", "flying", "large"
        ],
        "physical_attributes": [
            "orange scales", "large wings", "flame tail", "dragon-like",
            "bipedal", "fire breathing", "powerful build"
        ]
    },
    "Bulbasaur": {
        "names": ["bulbasaur", "bulba"],
        "types": ["grass", "poison"],
        "colors": ["green", "blue", "teal"],
        "descriptors": [
            "plant reptile", "vine beast", "green seedling",
            "sprout toad", "seed pokemon", "grass creature",
            "seed", "plant", "bulb", "vine", "herbal", "toxic",
            "toad", "sprout", "reptile", "grass", "leaf", "nature"
        ],
        "physical_attributes": [
            "green skin", "bulb on back", "four legs", "plant features",
            "vine whips", "spotted pattern", "quadruped"
        ]
    },
    "Mewtwo": {
        "names": ["mewtwo", "mew two"],
        "types": ["psychic"],
        "colors": ["purple", "pink", "gray", "silver", "white"],
        "descriptors": [
            "genetic experiment", "psychic clone", "telekinetic predator",
            "synthetic mind weapon", "artificial pokemon", "lab creation",
            "clone", "psychic", "powerful", "intelligent", "experiment",
            "artificial", "legendary", "telepathic", "mental"
        ],
        "physical_attributes": [
            "purple skin", "large head", "three fingers", "long tail",
            "humanoid", "psychic aura", "feline features"
        ]
    }
}

# ====================== LLM SYNTHETIC DATA GENERATOR ======================
class LLMSyntheticDataGenerator:
    def __init__(self):
        logger.info("Initializing LLM-based synthetic data generator...")
        
        self.military_headers = [
            "HQ REPORT", "INTELLIGENCE UPDATE", "FIELD BULLETIN",
            "OPERATIONAL NOTICE", "MISSION BRIEFING", "TACTICAL UPDATE",
            "SITREP", "INTEL SUMMARY", "COMBAT REPORT", "FIELD UPDATE"
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
            "Communications are patchy - maintain line-of-sight contact"
        ]
        
        self.elimination_command_templates = [
            "Priority: eliminate all {target} detected in the operational zone",
            "Order: eliminate the {target} immediately upon visual confirmation",
            "Mission objective: eliminate any {target} encountered during sweep",
            "HQ directive: neutralize all {target} contacts without delay",
            "Execute elimination protocol against {target} specimens",
            "Immediate action required: eliminate {target} on sight",
            "Your mission is to eliminate all {target} within the perimeter"
        ]
        
        self.distractor_templates = [
            "Additional activity from {distractor} groups nearby, though they do not appear hostile at present",
            "Scouts reported {distractor} sightings in adjacent sectors but avoid engagement",
            "Non-hostile {distractor} detected in the vicinity - do not target without authorization",
            "Field teams report {distractor} groups showing no aggressive behavior"
        ]
    
    def generate_single_target_prompt(self, target_pokemon: str, prompt_length: int = None) -> str:
        if prompt_length is None:
            prompt_length = random.randint(200, 700)
        
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
        noise_count = max(6, int(prompt_length * 0.5 / 10))
        for _ in range(noise_count):
            noise_template = random.choice(self.tactical_noise_templates)
            filled_noise = noise_template.format(
                time=f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
                grid=f"{random.randint(1, 9)}{random.choice(['A', 'B', 'C', 'K'])}"
            )
            prompt_parts.append(filled_noise)
        
        # Add distractors
        other_pokemon = [p for p in POKEMON_KNOWLEDGE.keys() if p != target_pokemon]
        for _ in range(random.randint(1, 2)):
            distractor_pokemon = random.choice(other_pokemon)
            distractor_desc = random.choice(POKEMON_KNOWLEDGE[distractor_pokemon]['descriptors'])
            distractor_template = random.choice(self.distractor_templates)
            distractor_text = distractor_template.format(distractor=distractor_desc)
            prompt_parts.insert(random.randint(1, len(prompt_parts)), distractor_text)
        
        # Main elimination command
        elimination_template = random.choice(self.elimination_command_templates)
        target_desc = random.choice(target_descriptors)
        main_command = elimination_template.format(target=target_desc)
        insert_pos = len(prompt_parts) // 2 + random.randint(0, len(prompt_parts) // 3)
        prompt_parts.insert(insert_pos, main_command)
        
        # More noise
        for _ in range(random.randint(2, 4)):
            noise_template = random.choice(self.tactical_noise_templates)
            prompt_parts.append(noise_template)
        
        return " ".join(prompt_parts)
    
    def generate_training_dataset(self, samples_per_pokemon: int = 2000) -> List[Dict]:
        logger.info(f"Generating {samples_per_pokemon * 4} synthetic training samples...")
        
        dataset = []
        pokemon_list = list(POKEMON_KNOWLEDGE.keys())
        
        for pokemon_idx, target_pokemon in enumerate(pokemon_list):
            logger.info(f"Generating samples for {target_pokemon}...")
            
            for sample_idx in range(samples_per_pokemon):
                if sample_idx % 400 == 0:
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
        
        return dataset

# ====================== SINGLE TARGET CLASSIFIER MODEL ======================
class SingleTargetPokemonClassifier(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 4):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # Simple but effective architecture
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)  # Raw logits for CrossEntropyLoss
        )
        
        self._init_weights()
    
    def _init_weights(self):
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
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        # Calculate loss if labels are provided (for training)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# ====================== DATASET CLASS ======================
class SingleTargetDataset(Dataset):
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
            'labels': torch.tensor(label, dtype=torch.long)  # Important: torch.long for CrossEntropy
        }

# ====================== TRAINING FUNCTIONS ======================
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if 'eval_accuracy' in logs:
                logger.info(f"📈 Eval Accuracy: {logs['eval_accuracy']:.4f}")
            if 'train_loss' in logs:
                logger.info(f"📉 Train Loss: {logs['train_loss']:.4f}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    pokemon_names = list(POKEMON_KNOWLEDGE.keys())
    per_class_acc = {}
    
    for i, pokemon in enumerate(pokemon_names):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == labels[mask]).mean()
            per_class_acc[f'{pokemon}_accuracy'] = float(class_acc)
    
    metrics = {'accuracy': accuracy}
    metrics.update(per_class_acc)
    
    return metrics

def train_single_target_model(output_dir: str = './final_pokemon_nlp_working') -> Tuple[nn.Module, AutoTokenizer]:
    logger.info("🚀 Starting single target Pokemon NLP training...")
    
    # Generate synthetic training data
    data_generator = LLMSyntheticDataGenerator()
    training_data = data_generator.generate_training_dataset(samples_per_pokemon=2000)  # Reduced for faster training
    
    # Split data
    labels_for_stratify = [item['label'] for item in training_data]
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.15, 
        random_state=42, 
        stratify=labels_for_stratify
    )
    
    logger.info(f"📚 Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Use BERT instead of DeBERTa to avoid tokenizer issues
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SingleTargetPokemonClassifier(model_name=model_name, num_classes=4)
    
    # Create datasets
    train_dataset = SingleTargetDataset(train_data, tokenizer, max_length=512)
    val_dataset = SingleTargetDataset(val_data, tokenizer, max_length=512)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
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
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,
        seed=42,
        fp16=False,  # Disable for stability
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Updated parameter name
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()]
    )
    
    # Train the model
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_info = {
        'model_name': model_name,
        'pokemon_classes': list(POKEMON_KNOWLEDGE.keys()),
        'max_length': 512,
        'version': '3.2'
    }
    
    with open(Path(output_dir) / 'model_config.json', 'w') as f:
        json.dump(config_info, f, indent=2)
    
    logger.info("✅ Training completed successfully!")
    
    return model, tokenizer

# ====================== SINGLE TARGET PARSER ======================
class FinalSingleTargetParser:
    def __init__(self, model_path: str = './final_pokemon_nlp_working'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model on {self.device}")
        
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
            self.model = SingleTargetPokemonClassifier(
                model_name=self.config.get('model_name', 'bert-base-uncased'),
                num_classes=len(self.pokemon_names)
            )
            
            model_file = Path(model_path) / 'pytorch_model.bin'
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning("Model weights not found, using rule-based approach only")
                self.model = None
            
            if self.model:
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using rule-based approach only.")
            self.model = None
            self.tokenizer = None
        
        # Rule-based patterns
        self.target_extraction_patterns = [
            r"(?:eliminate|destroy|kill|terminate|neutralize|wipe out)\\s+(?:all\\s+|any\\s+|the\\s+)?([^.]{1,50})",
            r"(?:priority|objective|mission|order)[:,]?\\s*(?:is\\s+to\\s+)?(?:eliminate|kill|destroy|neutralize)\\s+(?:all\\s+|the\\s+)?([^.]{1,50})",
            r"(?:hq\\s+(?:orders?|directive)|command)[:,]?\\s*(?:eliminate|kill|destroy)\\s+(?:all\\s+|the\\s+)?([^.]{1,50})"
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
        
        # Context-based scoring
        elimination_keywords = ['eliminate', 'kill', 'destroy', 'terminate', 'neutralize']
        
        for pokemon in self.pokemon_names:
            all_references = self._get_all_pokemon_references(pokemon)
            
            for reference in all_references:
                if reference in prompt_lower:
                    for match in re.finditer(re.escape(reference), prompt_lower):
                        start = max(0, match.start() - 60)
                        end = min(len(prompt_lower), match.end() + 60)
                        context = prompt_lower[start:end]
                        
                        elimination_score = sum(2 for kw in elimination_keywords if kw in context)
                        protection_penalty = sum(3 for kw in ['not hostile', 'friendly', 'avoid'] if kw in context)
                        
                        candidate_scores[pokemon] += elimination_score - protection_penalty
        
        if candidate_scores:
            return max(candidate_scores.items(), key=lambda x: x[1])[0]
        
        return "Pikachu"  # Default fallback
    
    def _match_text_to_pokemon(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\\b(the|any|all|some|every)\\b', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        
        for pokemon, knowledge in self.pokemon_knowledge.items():
            all_refs = (knowledge.get('names', []) + 
                       knowledge.get('descriptors', []) + 
                       knowledge.get('physical_attributes', []))
            
            for ref in all_refs:
                if ref.lower() in text or (len(ref) > 4 and text in ref.lower()):
                    return pokemon
        
        return None
    
    def _get_all_pokemon_references(self, pokemon: str) -> List[str]:
        knowledge = self.pokemon_knowledge.get(pokemon, {})
        references = []
        references.extend(knowledge.get('names', []))
        references.extend(knowledge.get('descriptors', []))
        references.extend(knowledge.get('physical_attributes', []))
        return [ref.lower() for ref in references]
    
    def predict_target(self, prompt: str) -> str:
        # Rule-based prediction
        rule_based_target = self.extract_target_rule_based(prompt)
        
        # Model prediction if available
        if self.model and self.tokenizer:
            try:
                model_target, confidence = self._predict_with_model(prompt)
                
                if confidence > 0.7:
                    logger.debug(f"Model prediction: {model_target} (confidence: {confidence:.3f})")
                    return model_target
                elif confidence > 0.4 and model_target == rule_based_target:
                    logger.debug(f"Model-rule agreement: {model_target}")
                    return model_target
            
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        logger.debug(f"Using rule-based prediction: {rule_based_target}")
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

# ====================== MAIN EXECUTION ======================
def main():
    logger.info("🎯 Starting Working Pokemon NLP System")
    
    try:
        # Train the model
        model, tokenizer = train_single_target_model()
        
        # Test the parser
        logger.info("Testing final parser...")
        parser = FinalSingleTargetParser()
        
        # Example test
        test_prompt = """
        HQ REPORT Situation analysis regarding unusual activity of flame dragon in this operational zone.
        Additional activity from plant reptile groups nearby, though they do not appear hostile at present.
        Radio checkpoint at 0200 to confirm continued presence.
        Priority: eliminate the scaled fire titan at first contact, then hold position for assessment.
        Maintain operational secrecy. Local sensors report intermittent EM spikes.
        """
        
        result = parser.predict_target(test_prompt)
        logger.info(f"Test result: {result}")  # Should output: Charizard
        
        logger.info("✅ Working NLP Module Ready for Production!")
        
        return parser
        
    except Exception as e:
        logger.error(f"Training/testing failed: {e}")
        raise

if __name__ == "__main__":
    main()