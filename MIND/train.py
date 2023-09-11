from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_type = config.dataset
CUDA_LAUNCH_BLOCKING=1
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    directory = directory
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory) if x.startswith('ckpt')
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    # writer = SummaryWriter(
    #     log_dir=
    #     f"./runs/{model_name}/{str(datetime.datetime.now().replace(microsecond=0).isoformat()).replace(':','')}"
    # )

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(f'./data/{dataset_type}/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    if model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    f'./data/{dataset_type}/train/pretrained_entity_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    f'./data/{dataset_type}/train/pretrained_context_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_context_embedding = None

        model = Model(config, pretrained_word_embedding,
                      pretrained_entity_embedding,
                      pretrained_context_embedding).to(device)
    elif model_name == 'Exp1':
        models = nn.ModuleList([
            Model(config, pretrained_word_embedding).to(device)
            for _ in range(config.ensemble_factor)
        ])
    else:
        model = Model(config, pretrained_word_embedding).to(device)

    if model_name != 'Exp1':
        print(model)
    else:
        print(models[0])

    dataset = BaseDataset(f'data/{dataset_type}/train/behaviors_parsed.tsv',
                          f'data/{dataset_type}/train/news_parsed.tsv')

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=False))
    if model_name != 'Exp1':
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
    else:
        criterion = nn.NLLLoss()
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for model in models
        ]
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()
    if config.checkpoint_num == '':
        checkpoint_dir = os.path.join(f'./checkpoint/{dataset_type}', f'{model_name}{config.training_type}_{config.negative_sampling_ratio}')
    else:
        checkpoint_dir = os.path.join(f'./checkpoint/{dataset_type}_{config.checkpoint_num}', f'{model_name}{config.training_type}_{config.negative_sampling_ratio}')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        if model_name != 'Exp1':
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.train()
        else:
            for model in models:
                model.load_state_dict(checkpoint['model_state_dict'])
                model.train()
            for optimizer in optimizers:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_exhaustion_count = 0
    all_results = []
    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 10),
                  desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=False))
                           
            minibatch = next(dataloader)
            
            (model if model_name != 'Exp1' else models[0]).eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model if model_name != 'Exp1' else models[0], f'./data/{dataset_type}/test',
                config.num_workers)
            (model if model_name != 'Exp1' else models[0]).train()
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )
            result_line = f"{exhaustion_count},{val_auc},{val_mrr},{val_ndcg5},{val_ndcg10}"
            all_results.append(result_line)
            early_stop, get_better = early_stopping(-(val_auc+val_mrr+val_ndcg5+val_ndcg10))
            checkpoint_file = os.path.join(checkpoint_dir,f"ckpt-{step}.pth")
            # if early_stop:
            #     tqdm.write('Early stop.')
            #     break
            # elif get_better:
            try:
                torch.save(
                    {
                        'model_state_dict': (model if model_name != 'Exp1'
                                                else models[0]).state_dict(),
                        'optimizer_state_dict':
                        (optimizer if model_name != 'Exp1' else
                            optimizers[0]).state_dict(),
                        'step':
                        step,
                        'early_stop_value':
                        -val_auc
                    }, checkpoint_file)
            except OSError as error:
                print(f"OS error: {error}")

        step += 1
        if model_name == 'LSTUR':
            y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                           minibatch["candidate_news"],
                           minibatch["clicked_news"])
        elif model_name == 'HiFiArk':
            y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                             minibatch["clicked_news"])
        elif model_name == 'TANR':
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        elif model_name == 'Exp1':
            y_preds = [
                model(minibatch["candidate_news"], minibatch["clicked_news"])
                for model in models
            ]
            y_pred_averaged = torch.stack(
                [F.softmax(y_pred, dim=1) for y_pred in y_preds],
                dim=-1).mean(dim=-1)
            y_pred = torch.log(y_pred_averaged)
        else:
            y_pred = model(minibatch["candidate_news"],
                           minibatch["clicked_news"])

        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)

        if model_name == 'HiFiArk':
            # if i % 10 == 0:
                # writer.add_scalar('Train/BaseLoss', loss.item(), step)
                # writer.add_scalar('Train/RegularizerLoss',
                                #   regularizer_loss.item(), step)
                # writer.add_scalar('Train/RegularizerBaseRatio',
                                #   regularizer_loss.item() / loss.item(), step)
            loss += config.regularizer_loss_weight * regularizer_loss
        elif model_name == 'TANR':
            # if i % 10 == 0:
                # writer.add_scalar('Train/BaseLoss', loss.item(), step)
                # writer.add_scalar('Train/TopicClassificationLoss',
                                #   topic_classification_loss.item(), step)
                # writer.add_scalar(
                    # 'Train/TopicBaseRatio',
                    # topic_classification_loss.item() / loss.item(), step)
            loss += config.topic_classification_loss_weight * topic_classification_loss
        loss_full.append(loss.item())
        if model_name != 'Exp1':
            optimizer.zero_grad()
        else:
            for optimizer in optimizers:
                optimizer.zero_grad()

        loss.backward()

        if model_name != 'Exp1':
            optimizer.step()
        else:
            for optimizer in optimizers:
                optimizer.step()
        # ## test
        # if i == 10:
        #     last_exhaustion_count += 1
        #     (model if model_name != 'Exp1' else models[0]).eval()
        #     val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        #         model if model_name != 'Exp1' else models[0], f'./data/{dataset_type}/val',
        #         config.num_workers, 200000)
            # (model if model_name != 'Exp1' else models[0]).train()
        #     # writer.add_scalar('Validation/AUC', val_auc, step)
        #     # writer.add_scalar('Validation/MRR', val_mrr, step)
        #     # writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
        #     # writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
        #     tqdm.write(
        #         f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
        #     )
        #     break
        # ### test        

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )
  
    result_file_ori = f'./results/{dataset_type}/{model_name}{config.training_type}_{config.negative_sampling_ratio}{config.test_type}_ep{config.num_epochs}-{config.checkpoint_num}'
    number = 1
    while True:
        result_file = f"{result_file_ori}-{str(number)}"
        if os.path.isfile(result_file+".txt"):
            number += 1
        else:
            break
    result_file = result_file +".txt"
    with open(result_file,'w') as wf:
        wf.writelines('\n'.join(all_results))

def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    start = time.time()
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
    train_time = int(time.time()-start)
    with open(os.path.join(f'./checkpoint/{dataset_type}_{config.checkpoint_num}', f'{model_name}{config.training_type}_{config.negative_sampling_ratio}',f'epoch{config.num_epochs}_time.txt'),'w') as wf:
        wf.write(str(train_time))