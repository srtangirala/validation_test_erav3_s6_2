from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Net
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import OneCycleLR

# Define the transforms for training data with more controlled augmentations
train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.07, 0.07)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define simpler transforms for test/validation data (no augmentation needed)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Apply transforms when creating datasets
train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    # Track average loss for the epoch
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        pbar.set_description(desc= f'Epoch {epoch}: loss={loss.item():.4f} batch_id={batch_idx}')
    
    # Print average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'\nEpoch {epoch} - Average training loss: {avg_loss:.4f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Calculate total steps for OneCycleLR
    total_steps = 20 * len(train_loader)  # epochs * steps_per_epoch
    
    # Replace the StepLR scheduler with OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,  # maximum learning rate
        total_steps=total_steps,  # you already calculated this correctly above
        pct_start=0.3,  # optional: percentage of steps spent increasing lr
        div_factor=25.0,  # optional: initial_lr = max_lr/div_factor
        final_div_factor=10000.0  # optional: final_lr = initial_lr/final_div_factor
    )
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
        _, accuracy = test(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()  # Call main() instead of train() 