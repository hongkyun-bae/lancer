from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
import copy
try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=config.early_stop_patience):
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
        if np.isnan(val_loss):
            early_stop = True
        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[0].split('-')[1]): x
        for x in os.listdir(directory)
        if (x.split('.')[0].split('-')[2] == config.candidate_type)
        if (x.split('.')[0].split('-')[3] == config.our_type)
        if (x.split('.')[0].split('-')[4] == config.loss_function)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    # writer = SummaryWriter(
    #     log_dir=
    #     f".\\runs\\{model_name}\\{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    # )
    test_data = config.test_behaviors_file.replace('behaviors_','').split('.')[0]
    result_file = f".\\results\\{config.data}\\{model_name}\\result_ep{config.num_epochs}_{config.candidate_type}_ns{config.negative_sampling_ratio}_{config.loss_function}_lifetime{config.lifetime}_testdata{test_data}_testfilter{config.test_filter}_his{config.history_type}_{config.numbering}"
    number = 1
    while True:
        if os.path.isfile(result_file+".txt"):
            result_file = f"{result_file}-{str(number)}"
            number += 1
        else:
            break
    result_file = f"{result_file}.txt"
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(f'.\\data\\preprocessed_data\\{config.data}\\train\\pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    if model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    f'.\\data\\preprocessed_data\\{config.data}\\train\\pretrained_entity_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    f'.\\data\\preprocessed_data\\{config.data}\\train\\pretrained_context_embedding.npy')).float()
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
    elif model_name == 'Exp2':
        model = Model(config).to(device)
    else:
        model = Model(config, pretrained_word_embedding).to(device)

    if model_name != 'Exp1':
        print(model)
    else:
        print(models[0])

    dataset = BaseDataset(f'.\\data\\preprocessed_data\\{config.data}\\train\\'+config.behaviors_target_file,
                          f'.\\data\\preprocessed_data\\{config.data}\\train\\'+'news_parsed.tsv', f'data\\preprocessed_data\\{config.data}\\train\\roberta')

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    # if config.loss_function == "CEL":
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.learning_rate)
    # elif config.loss_function == "BPR_sig" or config.loss_function == "BPR_soft":
    #     optimizer = torch.optim.SGD(
	# 		model.parameters(), lr=config.learning_rate)
    print(f"Loss function:{config.loss_function}, NS Type: {config.candidate_type}_{config.our_type}")


    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join('.\\checkpoint',config.data, model_name)
    result_dir = os.path.join('.\\results',config.data, model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    checkpoint_path = None

    epoch_result = []
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        #### TEST######
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        exhaustion_count = checkpoint['exhaustion_count']
        epoch_result = [x.split(' ') for x in checkpoint['epoch_result'].split('\n')]
        # (model if model_name != 'Exp1' else models[0]).eval()
        # val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        #     model if model_name != 'Exp1' else models[0], '.\\data\\preprocessed_data\\{config.data}\\test',
        #     200000)
        #### TEST######

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()

    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 1),
                  desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            # tqdm.write(
            #     f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            # )
            # ################################TEST############################
            # torch.save(
            # {
            # 'model_state_dict': (model if model_name != 'Exp1'
            # else models[0]).state_dict(),
            # 'optimizer_state_dict':
            # (optimizer if model_name != 'Exp1' else
            # optimizers[0]).state_dict(),
            # 'step':
            # step
            # # 'early_stop_value':
            # # -sum([val_auc,val_mrr,val_ndcg5,val_ndcg10]),
            # # 'exhaustion_count':
            # # exhaustion_count,
            # # 'epoch_result':
            # # line
            # }, f".\\checkpoint\\{model_name}\\ckpt-{exhaustion_count}-{config.candidate_type}-{config.our_type}-{config.loss_function}.pth")
            ################################TEST############################


            (model if model_name != 'Exp1' else models[0]).eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model if model_name != 'Exp1' else models[0], f'.\\data\\preprocessed_data\\{config.data}\\test')
            (model if model_name != 'Exp1' else models[0]).train()

            # writer.add_scalar('Validation/AUC', val_auc, step)
            # writer.add_scalar('Validation/MRR', val_mrr, step)
            # writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
            # writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )
            print()
            print(exhaustion_count,"Epoch Done!")
            print()
            epoch_result.append([str(val_auc),str(val_mrr),str(val_ndcg5),str(val_ndcg10)])
            with open(result_file,'w') as wf:
                line = '\n'.join([ ' '.join(x) for x in epoch_result])
                wf.write(line)

            early_stop, get_better = early_stopping(-sum([val_auc,val_mrr,val_ndcg5,val_ndcg10]))
            if early_stop:
                tqdm.write(f'{exhaustion_count} Epoch Done! Early stop.')
                break
            # elif get_better:
            #     try:
            #         torch.save(
            #             {
            #                 'model_state_dict': (model if model_name != 'Exp1'
            #                                      else models[0]).state_dict(),
            #                 'optimizer_state_dict':
            #                 (optimizer if model_name != 'Exp1' else
            #                  optimizers[0]).state_dict(),
            #                 'step':
            #                 step,
            #                 'early_stop_value':
            #                 -sum([val_auc,val_mrr,val_ndcg5,val_ndcg10]),
            #                 'exhaustion_count':
            #                 exhaustion_count,
            #                 'epoch_result':
            #                 line
            #             }, f".\\checkpoint\\{model_name}\\ckpt-{exhaustion_count}-{config.candidate_type}-{config.our_type}-{config.loss_function}.pth")
            #     except OSError as error:
            #         print(f"OS error: {error}")
            if exhaustion_count == config.num_epochs:
                break
            ############################
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        # print(minibatch)

        # if config.loss_function.startswith("BPR") or config.loss_function.endswith("pair"):
        #     minibatch['clicked'][0] = minibatch['clicked'][0].repeat(config.negative_sampling_ratio)
        #     minibatch['clicked'][1] = minibatch['clicked'][1].repeat(config.negative_sampling_ratio)
        #     del minibatch['clicked'][2:]

        #     if model_name == 'LSTUR':

        #         minibatch["candidate_news"][0]['title'] = minibatch["candidate_news"][0]['title'].repeat(config.negative_sampling_ratio,1)
        #         minibatch["candidate_news"][0]['subcategory'] = minibatch["candidate_news"][0]['subcategory'].repeat(config.negative_sampling_ratio)
        #         minibatch["candidate_news"][0]['category'] = minibatch["candidate_news"][0]['category'].repeat(config.negative_sampling_ratio)
        #         for _ in range(2,config.negative_sampling_ratio+1):
        #             for k in minibatch["candidate_news"][0].keys():
        #                 minibatch["candidate_news"][1][k] = torch.cat((minibatch["candidate_news"][1][k], minibatch["candidate_news"][2][k]),0)
        #             del minibatch["candidate_news"][2]

        #         for j in range(len(minibatch["clicked_news"])):
        #             minibatch["clicked_news"][j]['title'] = minibatch["clicked_news"][j]['title'].repeat(config.negative_sampling_ratio,1)
        #             minibatch["clicked_news"][j]['subcategory'] = minibatch["clicked_news"][j]['subcategory'].repeat(config.negative_sampling_ratio)
        #             minibatch["clicked_news"][j]['category'] = minibatch["clicked_news"][j]['category'].repeat(config.negative_sampling_ratio)

        #         minibatch["clicked_news_length"] = minibatch["clicked_news_length"].repeat(config.negative_sampling_ratio)
        #         minibatch["user"] = minibatch["user"].repeat(config.negative_sampling_ratio)
        #     else:

        #         for j in range(len(minibatch["clicked_news"])):
        #             minibatch["clicked_news"][j]['title'] = minibatch["clicked_news"][j]['title'].repeat(config.negative_sampling_ratio,1)
        #             # print(minibatch['clicked_news'][0]['title'].shape)

        #         minibatch["candidate_news"][0]['title'] = minibatch["candidate_news"][0]['title'].repeat(config.negative_sampling_ratio,1)
        #         for _ in range(2,config.negative_sampling_ratio+1):
        #             minibatch["candidate_news"][1]['title'] = torch.cat((minibatch["candidate_news"][1]['title'], minibatch["candidate_news"][2]['title']),0)
        #             del minibatch["candidate_news"][2]

            #     y_preds = [
            #         model(minibatch["candidate_news"], minibatch["clicked_news"])
            #         for model in models
            #     ]
            #     y_pred_averaged = torch.stack(
            #         [F.softmax(y_pred, dim=1) for y_pred in y_preds],
            #         dim=-1).mean(dim=-1)
            #     y_pred = torch.log(y_pred_averaged)
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
        # elif model_name == 'Exp1':
        #     y_preds = [
        #         model(minibatch["candidate_news"], minibatch["clicked_news"])
        #         for model in models
        #     ]
        #     y_pred_averaged = torch.stack(
        #         [F.softmax(y_pred, dim=1) for y_pred in y_preds],
        #         dim=-1).mean(dim=-1)
        #     y_pred = torch.log(y_pred_averaged)
        else:
            y_pred = model(minibatch["candidate_news"], minibatch["clicked_news"])

        # if config.loss_function == "CEL" or config.loss_function == "CEL_pair":
        y_true = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y_true)
        # elif config.loss_function == "BPR":
        #     loss = - (y_pred[:,:1] - y_pred[:,1:2]).sigmoid().log().sum()
        # elif config.loss_function == "BPR_soft":
        #     m = torch.nn.Softmax(dim=1)
        #     y_pred = m(y_pred)
        #     loss = - (y_pred[:,:1] - y_pred[:,1:2]).sigmoid().log().sum()
        # elif config.loss_function == "BPR_sig":
        #     y_pred = y_pred.sigmoid()
        #     loss = - (y_pred[:,:1] - y_pred[:,1:2]).sigmoid().log().sum()
        if model_name == 'HiFiArk':
            loss += config.regularizer_loss_weight * regularizer_loss
            # if i % 10 == 0:
            #     writer.add_scalar('Train/BaseLoss', loss.item(), step)
            #     writer.add_scalar('Train/RegularizerLoss',
            #                       regularizer_loss.item(), step)
            #     writer.add_scalar('Train/RegularizerBaseRatio',
            #                       regularizer_loss.item() / loss.item(), step)
        # elif model_name == 'TANR':
        #     if i % 10 == 0:
        #         writer.add_scalar('Train/BaseLoss', loss.item(), step)
        #         writer.add_scalar('Train/TopicClassificationLoss',
        #                           topic_classification_loss.item(), step)
        #         writer.add_scalar(
        #             'Train/TopicBaseRatio',
        #             topic_classification_loss.item() / loss.item(), step)
        #     loss += config.topic_classification_loss_weight * topic_classification_loss
        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # if i % 10 == 0:
        #     writer.add_scalar('Train/Loss', loss.item(), step)

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )
            if np.isnan(loss.item()):
                break
            else:
                pass
    # # evaluate test 
    #     if i == 10:
    #         break
    (model if model_name != 'Exp1' else models[0]).eval()
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        model if model_name != 'Exp1' else models[0], f'.\\data\\preprocessed_data\\{config.data}\\test')
    # (model if model_name != 'Exp1' else models[0]).train()
    if [str(val_auc),str(val_mrr),str(val_ndcg5),str(val_ndcg10)] not in epoch_result:
        epoch_result.append([str(val_auc),str(val_mrr),str(val_ndcg5),str(val_ndcg10)])
        with open(result_file,'w') as wf:
            line = '\n'.join([ ' '.join(x) for x in epoch_result])
            wf.write(line)
        # if i % config.num_batches_validate == 0:
        #     (model if model_name != 'Exp1' else models[0]).eval()
        #     val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        #         model if model_name != 'Exp1' else models[0], '.\\data\\preprocessed_data\\{config.data}\\test',
        #         200000)
        #     (model if model_name != 'Exp1' else models[0]).train()
        #     writer.add_scalar('Validation/AUC', val_auc, step)
        #     writer.add_scalar('Validation/MRR', val_mrr, step)
        #     writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
        #     writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
        #     tqdm.write(
        #         f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
        #     )
        #
        #     early_stop, get_better = early_stopping(-val_auc)
        #     if early_stop:
        #         tqdm.write('Early stop.')
        #         break
        #     elif get_better:
        #         try:
        #             torch.save(
        #                 {
        #                     'model_state_dict': (model if model_name != 'Exp1'
        #                                          else models[0]).state_dict(),
        #                     'optimizer_state_dict':
        #                     (optimizer if model_name != 'Exp1' else
        #                      optimizers[0]).state_dict(),
        #                     'step':
        #                     step,
        #                     'early_stop_value':
        #                     -val_auc
        #                 }, f".\\checkpoint\\{model_name}\\ckpt-{step}.pth")
        #         except OSError as error:
        #             print(f"OS error: {error}")


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
