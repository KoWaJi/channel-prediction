from torch.utils.tensorboard import SummaryWriter
import torch
import os

def savemodel(model, epoch, model_type):
    filename = f'model_epoch_{epoch}.pth'

    if model_type == 'CNN+LSTM':
        directory = os.path.join('channel prediction/save_net/cnn+lstm_network', filename)
    elif model_type == 'LSTM':
        directory = os.path.join('channel prediction/save_net/lstm_network', filename)
    elif model_type == 'AT+CNN+LSTM':
        directory = os.path.join('channel prediction/save_net/at+cnn+lstm_network', filename)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



    torch.save(model.state_dict(), directory)
    print(f'Model saved to {directory}')
 
def train_model(model, train_loader, criterion, optimizer, epochs, device, model_type):
    # 创建TensorBoard的SummaryWriter实例
    if model_type == 'CNN+LSTM':
        writer = SummaryWriter('channel prediction/runs/cnn+lstm')
    elif model_type == 'LSTM':
        writer = SummaryWriter('channel prediction/runs/lstm')
    elif model_type == 'AT+CNN+LSTM':
        writer = SummaryWriter('channel prediction/runs/at+cnn+lstm')
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        if epoch % 25 == 0:
            savemodel(model, epoch, model_type)
        for i, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = torch.squeeze(outputs) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training Loss', loss.item(), (epoch - 1) * len(train_loader) + i)
            total_loss += loss.item()
            print(f'Epoch {epoch}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.8f}')
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.8f}')
    
        # 每个epoch记录一次平均loss
        writer.add_scalar('Average Training Loss', avg_loss, epoch)
    # 关闭SummaryWriter
    writer.close()

