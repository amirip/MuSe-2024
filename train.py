import os

import numpy as np
import torch
import torch.optim as optim

from eval import evaluate


def train(model, train_loader, optimizer, loss_fn, use_gpu=False):

    train_loss_list = []

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        preds,_ = model(features, feature_lens)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss


def save_model(model, model_folder, id):
    model_file_name = f'model_{id}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            best_model_file = save_model(model, model_path, identifier)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file
