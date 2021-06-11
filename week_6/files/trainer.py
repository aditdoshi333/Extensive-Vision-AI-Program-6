import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F

lambda_l1 = 0.0001

def train(model, device, train_loader, optimizer, epoch, l1_loss):
    training_losses = []
    training_accuracy = []
    model.train()
    correct = 0
    processed = 0
    # lambda_l1 = 0.0001
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if l1_loss:
          l1 = 0
          for p in model.parameters():
            l1 = l1 + p.abs().sum()
          loss = loss + lambda_l1 * l1
        
        total_loss += loss
        loss.backward()
        optimizer.step()
        predictions = output.argmax(dim=1, keepdim=True)
        correct += predictions.eq(target.view_as(predictions)).sum().item()
        processed += len(data)
    training_losses.append(total_loss)
    training_accuracy.append(100*correct/processed)
    if l1_loss:
      print('L1 = ', l1)
    print('Train set: Accuracy={:0.1f}'.format(100*correct/processed))
    return training_losses, training_accuracy



def test(model, device, test_loader):
    testing_losses = []
    testing_accuracy = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()

    test_loss /= len(test_loader.dataset)
    testing_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    testing_accuracy.append(100. * correct / len(test_loader.dataset))
    return testing_losses, testing_accuracy