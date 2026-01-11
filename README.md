# MNIST Digit Recognizer 🔢

My first deep learning project! A simple neural network built with PyTorch that can recognize handwritten digits from 0-9.

## 🎯 Results
- **Accuracy:** 96-97% on test data
- **Training time:** ~2-3 minutes on GPU
- **Model size:** Lightweight 3-layer network

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Model
```bash
python trainr.py
```

The script will:
1. Download the MNIST dataset automatically
2. Train the neural network for 5 epochs
3. Show accuracy results
4. Generate prediction visualizations
5. Save the trained model

## 📊 What It Does

This model takes a 28×28 pixel grayscale image of a handwritten digit and predicts which number it is (0-9).

**Architecture:**
- Input Layer: 784 neurons (28×28 flattened image)
- Hidden Layer 1: 128 neurons
- Hidden Layer 2: 64 neurons  
- Output Layer: 10 neurons (one per digit)

## 📁 Output Files

After running, you'll get:
- `digit_reader.pth` - Trained model weights
- `my_predictions.png` - Sample predictions visualization
- `data/` folder - MNIST dataset

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib

## 📝 Usage Example
```python
import torch
from mnist_digit_recognizer import SimpleDigitReader

# Load the trained model
model = SimpleDigitReader()
model.load_state_dict(torch.load('digit_reader.pth'))
model.eval()

# Use it to make predictions
```


## 📜 License

MIT License - Feel free to use this for learning!

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- PyTorch documentation and tutorials
