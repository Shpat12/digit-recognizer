import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Preparing the data
print("📁 Loading dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✅ Loaded {len(train_data)} training images and {len(test_data)} test images\n")

# Build the neural network
class SimpleDigitReader(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten image
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(28*28, 128),  # Input layer: 784 -> 128
            nn.ReLU(),               # Activation
            nn.Linear(128, 64),      # Hidden layer: 128 -> 64
            nn.ReLU(),               # Activation
            nn.Linear(64, 10)        # Output layer: 64 -> 10 digits
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Create the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"💻 Using: {device.upper()}\n")

model = SimpleDigitReader().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
def train_one_epoch():
    model.train()
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Make prediction
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        
        # Learn from mistakes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        correct += (predictions.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    return 100 * correct / total

# Test the model
def test_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total

# Train
print("🚀 Training started...\n")
epochs = 5

for epoch in range(1, epochs + 1):
    train_acc = train_one_epoch()
    test_acc = test_model()
    print(f"Epoch {epoch}/{epochs} - Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print(f"\n✨ Training complete! Final accuracy: {test_acc:.2f}%")

# Show predictions
def show_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        predictions = model(images).argmax(1)
    
    # Show first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().squeeze()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        
        ax.imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'Actual: {true_label}\nGuessed: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('my_predictions.png')
    print("\n📊 Predictions saved to 'my_predictions.png'")

show_predictions()

# Save your model
torch.save(model.state_dict(), 'digit_reader.pth')
print("💾 Model saved to 'digit_reader.pth'")

# How to use your saved model later:
print("\n📝 To use this model later:")
print("   model = SimpleDigitReader()")
print("   model.load_state_dict(torch.load('digit_reader.pth'))")
print("   model.eval()")
