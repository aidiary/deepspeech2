import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from deepspeech2.data import TextTransform, data_processing
from deepspeech2.model import SpeechRecognitionModel


def main():
    hparams = {
        'n_cnn_layers': 3,
        'n_rnn_layers': 5,
        'rnn_dim': 512,
        'n_class': 29,
        'n_feats': 128,
        'stride': 2,
        'dropout': 0.1,
        'learning_rate': 5e-4,
        'batch_size': 20,
        'epochs': 10
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    # load dataset
    os.makedirs('data', exist_ok=True)
    train_dataset = torchaudio.datasets.LIBRISPEECH('./data', url='train-clean-100', download=True)
    valid_dataset = torchaudio.datasets.LIBRISPEECH('./data', url='test-clean', download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: data_processing(x, 'train'))
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=False,
                                               collate_fn=lambda x: data_processing(x, 'valid'))

    # model
    model = SpeechRecognitionModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'],
                                   hparams['rnn_dim'], hparams['n_class'], hparams['n_feats'],
                                   hparams['stride'], hparams['dropout']).to(device)
    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    for epoch in range(1, hparams['epochs'] + 1):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        valid_loss = valid(model, valid_loader, criterion, device)
        print('Epoch: {} train_loss: {} valid_loss: {}'.format(epoch, train_loss, valid_loss))


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        # CTCLoss
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss


def valid(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            spectrograms, labels, input_lengths, label_lengths = batch
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            # CTCLoss
            loss = criterion(output, labels, input_lengths, label_lengths)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
    valid_loss = running_loss / len(valid_loader)
    return valid_loss


if __name__ == '__main__':
    main()
