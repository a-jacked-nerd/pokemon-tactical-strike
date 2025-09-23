# Pokemon Tactical Strike - AI Guild Hackathon Solution 🎯

![Pokemon Hackathon Banner](https://img.shields.io/badge/Pokemon-Tactical%20Strike-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?style=for-the-badge&logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLOv8-Computer%20Vision-green?style=for-the-badge)

## 🎯 Project Overview

This repository contains an AI-powered targeting system that combines **Computer Vision** and **Natural Language Processing** to:

- Parse tactical orders from headquarters containing mission-specific directives
- Analyze battlefield imagery to identify Pokemon species and locations  
- Generate precise targeting coordinates based on mission parameters
- Minimize collateral damage and ammunition waste

### 🏆 Competition Details
- **Event**: AI Guild Hackathon - Pokemon Tactical Strike Challenge
- **Scoring**: 0.7 × Accuracy + 0.2 × Technical Innovation + 0.1 × Code Quality
- **Evaluation**: +1 for correct hits, -1 for collateral damage, -1 for every 3 missed shots

## 🏗️ Architecture

### Computer Vision Module (YOLOv8)
- **Model**: Custom-trained YOLOv8 for Pokemon detection
- **Classes**: Pikachu (0), Charizard (1), Bulbasaur (2), Mewtwo (3)
- **Features**: Robust detection with confidence thresholds and coordinate precision

### NLP Module (BERT-based)
- **Model**: Enhanced BERT classifier with attention pooling
- **Features**: Military-style prompt understanding with rule-based fallback
- **Strategy**: Smart ammunition conservation with miss counter logic

### Integration Pipeline
- **Fusion**: Multi-modal decision engine combining CV and NLP outputs
- **Optimization**: Adaptive confidence strategies for maximum efficiency
- **Visualization**: Real-time result rendering with coordinate overlays

## 📁 Repository Structure

```
pokemon-tactical-strike/
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment setup
├── .gitignore                        # Git ignore patterns
│
├── CV/                               # Computer Vision Module
│   ├── cv_module.py                  # Main CV detection class
│   ├── train_cv.py                   # YOLOv8 training script
│   ├── config.yaml                   # CV configuration
│   └── old_scripts/                  # Previous CV versions
│       ├── cv_v1.py
│       ├── cv_v2.py
│       └── cv_utils.py
│
├── NLP/                              # Natural Language Processing Module
│   ├── nlp_module.py                 # Main NLP classifier class
│   ├── train_nlp.py                  # BERT training script
│   ├── config.json                   # NLP configuration
│   └── old_scripts/                  # Previous NLP versions
│       ├── nlp_v1.py
│       ├── nlp_v2.py
│       └── data_generator.py
│
├── Integration/                      # Combined Pipeline
│   └── integration_pipeline.ipynb    # Main execution notebook
│
├── data/                            # Sample datasets
│   ├── sample_images/
│   └── sample_prompts.json
│
└── utils/                           # Utility scripts
    ├── json_to_csv.py               # Format converter
    └── visualization.py             # Result visualization
```

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM
- **Storage**: 5GB+ free space

### Step 1: Repository Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-tactical-strike.git
cd pokemon-tactical-strike

# Create conda environment
conda create -n pokemon-hackathon python=3.9
conda activate pokemon-hackathon

# Install dependencies
pip install -r requirements.txt

# Alternative: Use conda environment file
conda env create -f environment.yml
conda activate pokemon-hackathon
```

### Step 2: Train Computer Vision Module

```bash
cd CV
python train_cv.py
```

**Expected Output**: 
- Weights saved to: `C:\Users\varin\runs\detect\pokemon_final_model\weights\best.pt`
- Training metrics and validation results
- Model performance evaluation

### Step 3: Train NLP Module

```bash
cd ../NLP
python train_nlp.py
```

**Expected Output**:
- Model files in: `enhanced_pokemon_nlp/` directory
- Contains: `model.safetensors`, `config.json`, `tokenizer.json`, etc.
- Training logs and evaluation metrics

### Step 4: Run Integration Pipeline

```bash
cd ../Integration
jupyter notebook integration_pipeline.ipynb
```

## 📓 Notebook Execution Guide

### Cell 1: Pipeline Initialization
```python
# Update these paths based on your setup
cv_model_path = "C:/Users/varin/runs/detect/pokemon_final_model/weights/best.pt"
nlp_model_path = "enhanced_pokemon_nlp"
```

**Output**: 
- Creates `hackathon_results/` folder
- Generates `performance_report.txt` and `submission.json`
- Processes all test images with coordinate predictions

### Cell 2: Validation & Performance Analysis
**Output**:
- Validates submission format
- Performance metrics analysis
- Error detection and reporting

### Cell 3: Visualization Generation
**Output**:
- Creates `visualizations/` folder
- Sample detection images with bounding boxes
- Coordinate overlay visualizations
- Performance summary charts

### Cell 4: Final CSV Conversion
**Output**:
- Converts `submission.json` to competition format
- Generates `final_submission.csv`
- Ready for hackathon submission

## 🎯 Key Features

### Advanced CV Detection
- **Multi-scale Detection**: Handles varying Pokemon sizes and orientations
- **Confidence Thresholding**: Adaptive confidence strategies
- **Coordinate Precision**: Sub-pixel accurate targeting coordinates

### Intelligent NLP Processing
- **Military Jargon**: Understands tactical communication styles
- **Multi-entity Recognition**: Identifies targets and protected species
- **Fallback Systems**: Rule-based backup for ambiguous orders

### Smart Integration
- **Ammunition Conservation**: Avoids unnecessary shots
- **Collateral Avoidance**: Protects non-target Pokemon
- **Performance Optimization**: Balances speed and accuracy

## 🔧 Configuration Options

### CV Module Settings
```python
confidence_threshold = 0.25        # Detection confidence
max_shots_per_image = 10          # Safety limit
confidence_strategy = "adaptive"   # "conservative", "aggressive"
```

### NLP Module Settings
```python
max_length = 512                  # Input sequence length
batch_size = 16                   # Training batch size
learning_rate = 2e-5              # BERT fine-tuning rate
```

## 📊 Expected Performance

### Training Metrics
- **CV Accuracy**: 95%+ mAP@0.5
- **NLP Accuracy**: 98%+ classification accuracy
- **Integration Speed**: ~1.2 seconds per image

### Competition Scoring
- **Target Hits**: +1 point per correct target
- **Perfect Elimination**: +1 bonus for clearing all enemies
- **Collateral Damage**: -1 point per protected species hit
- **Ammunition Waste**: -1 point per 3 missed shots

## 🐛 Troubleshooting

### Common Issues

**CUDA Not Available**
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Model Loading Errors**
```bash
# Ensure paths are correct
ls -la CV/runs/detect/pokemon_final_model/weights/
ls -la NLP/enhanced_pokemon_nlp/
```

**Memory Issues**
```python
# Reduce batch sizes in config files
batch_size = 8  # Instead of 16
confidence_threshold = 0.35  # Reduce detections
```

### Debug Mode
```python
# Enable verbose logging in notebook
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone your fork
git clone https://github.com/yourusername/pokemon-tactical-strike.git
cd pokemon-tactical-strike

# Create feature branch
git checkout -b feature/amazing-improvement

# Make changes and commit
git add .
git commit -m "Add amazing improvement"

# Push and create pull request
git push origin feature/amazing-improvement
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features

## 📋 Dependencies

### Core Requirements
```
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
transformers>=4.20.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
pillow>=8.3.0
```

### Development Requirements
```
jupyter>=1.0.0
scikit-learn>=1.0.0
seaborn>=0.11.0
tqdm>=4.60.0
safetensors>=0.3.0
```

## 🏆 Competition Strategy

### Scoring Optimization
1. **Maximize Direct Hits**: High-confidence detections only
2. **Avoid Collateral**: Conservative shooting near protected species
3. **Ammunition Management**: Strategic miss counting to avoid penalties
4. **Speed Balance**: Optimize for accuracy over processing speed

### Technical Innovation Points
- Multi-modal fusion architecture
- Adaptive confidence thresholds  
- Military communication understanding
- Real-time visualization system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AI Guild** for organizing the hackathon
- **Ultralytics** for the excellent YOLOv8 framework
- **Hugging Face** for transformer models and tools
- **Pokemon Company** for the inspiration
- **Competition organizers** for the challenging problem statement

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share ideas  
- **Email**: [your-email@domain.com]
- **Discord**: Join our development server

---

**Good luck, soldier! May your targeting be precise and your Pokemon battles victorious! 🎯⚡🔥🌱👻**

---

*Built with ❤️ for the AI Guild Pokemon Hackathon*
