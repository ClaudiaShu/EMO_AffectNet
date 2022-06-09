import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, EXPR_metric, accuracy_top

torch.manual_seed(0)

class ABAW_trainer(object):
    def __init__(self, best_acc=0, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion = kwargs['criterion']

        self.mode = self.args.mode
        self.best_acc = best_acc

        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'runs', current_time + '_' + socket.gethostname() + self.args.comment)
        dir2log = f'/mnt/d/Data/Yuxuan/downstream/{self.mode}/{log_dir}'
        os.makedirs(dir2log, exist_ok=True)
        self.writer = SummaryWriter(log_dir=dir2log)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.matrix_file = f'{dir2log}/log.txt'

        self.scaler = GradScaler(enabled=self.args.fp16_precision)

        # global instance
        self.n_iter = 0
        self.v_iter = 0

    @staticmethod
    def trange(*args, **kwargs):
        """
        A shortcut for tqdm(xrange(*args), **kwargs).
        On Python3+ range is used instead of xrange.
        """
        return tqdm(range(*args), **kwargs)

    def run(self, train_loader, valid_loader):

        # save config file
        save_config_file(self.writer.log_dir, self.args)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"training on {self.args.device}")
        logging.info(f"training with {self.args.arch}")
        logging.info(f"representation learning with {self.args.rep}")

        logging.info('HP:')
        logging.info(f'learning rate: {self.args.lr}')
        logging.info(f'Batch size: {self.args.batch_size}')
        logging.info(f'Weight decay: {self.args.weight_decay}')

        logfile = open(self.matrix_file, 'w')
        with self.trange(self.args.epochs, total=self.args.epochs, desc='Epoch') as t:
            for epoch_counter in t:
                t.set_description('Epoch %i' % epoch_counter)
                self.train(epoch_counter, train_loader, t, logfile)
                self.valid(epoch_counter, valid_loader, t, logfile)

        logging.info('Training has finished.')
        logfile.close()


    def train(self, epoch, train_loader, t, logfile):
        self.model.train()
        cost_list = 0
        cat_preds = []
        cat_labels = []
        # for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train_mode'):
        for batch_idx, samples in enumerate(train_loader):
            if self.mode == 'image':
                data = samples['images'].to(self.args.device).float()
            elif self.mode == 'audio':
                data = samples['audios'].to(self.args.device).float()
            else:
                raise ValueError
            labels_cat = samples['labels'].to(self.args.device).long()
            # import pdb; pdb.set_trace()

            pred_cat = self.model(data)
            loss = self.criterion(pred_cat, labels_cat)
            if np.isnan(loss.item()):
                raise ValueError
            cost_list += loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # writer accuracy
            if self.n_iter % self.args.log_every_n_steps == 0:
                top1, top5 = accuracy_top(pred_cat, labels_cat, topk=(1, 5))
                self.writer.add_scalar('Train/loss', loss, global_step=self.n_iter)
                self.writer.add_scalar('Train/Avg_loss', cost_list / (batch_idx + 1), global_step=self.n_iter)
                self.writer.add_scalar('Train/acc_top1', top1[0], global_step=self.n_iter)
                self.writer.add_scalar('Train/acc_top5', top5[0], global_step=self.n_iter)
                self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=self.n_iter)

                logging.debug(f"Train mode\tEpoch: {epoch}\tLoss: {loss}\t")

            self.scheduler.step(self.n_iter)
            self.n_iter += 1

            pred_cat = F.softmax(pred_cat)
            pred_cat = torch.argmax(pred_cat, dim=1)
            cat_preds.append(pred_cat.detach().cpu().numpy())
            cat_labels.append(labels_cat.detach().cpu().numpy())
            t.set_postfix(Lr=self.optimizer.param_groups[0]['lr'],
                          Loss=f'{cost_list / (batch_idx + 1):04f}',
                          itr=self.n_iter)

        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_labels = np.concatenate(cat_labels, axis=0)
        cm = confusion_matrix(cat_labels, cat_preds)
        cr = classification_report(cat_labels, cat_preds)
        f1, acc, total = EXPR_metric(cat_preds, cat_labels)
        print('train mode: \n'
              f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion matrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')
        cm_ = np.array2string(cm)
        logfile.write('Train\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm_))
        logfile.write('Title\nf1: {}\nAccuracy: {}\nTotal: {}'.format(f1, acc, total))
        # logging.debug(
        #     f"train mode\tEpoch: {epoch}\tf1:{f1}\tacc:{acc}\ttotal:{total}")


    def valid(self, epoch, valid_loader, t, logfile):
        self.model.eval()
        with torch.no_grad():
            cost_list = 0
            cat_preds = []
            cat_labels = []
            # for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
            for batch_idx, samples in enumerate(valid_loader):
                if self.mode == 'image':
                    data = samples['images'].to(self.args.device).float()
                elif self.mode == 'audio':
                    data = samples['audios'].to(self.args.device).float()
                else:
                    raise ValueError
                labels_cat = samples['labels'].to(self.args.device).long()

                pred_cat = self.model(data)
                loss = self.criterion(pred_cat, labels_cat)
                cost_list += loss.item()
                pred_cat = F.softmax(pred_cat)
                pred_cat = torch.argmax(pred_cat, dim=1)

                cat_preds.append(pred_cat.detach().cpu().numpy())
                cat_labels.append(labels_cat.detach().cpu().numpy())
                t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}')

                # logging.debug(f"Valid mode\tEpoch: {epoch}\tLoss: {loss}")
                self.v_iter += 1

            cat_preds = np.concatenate(cat_preds, axis=0)
            cat_labels = np.concatenate(cat_labels, axis=0)
            cm = confusion_matrix(cat_labels, cat_preds)
            cr = classification_report(cat_labels, cat_preds)
            f1, acc, total = EXPR_metric(cat_preds, cat_labels)
            print('valid mode: \n'
                  f'f1 = {f1} \n'
                  f'acc = {acc} \n'
                  f'total = {total} \n',
                  'confusion matrix: \n', cm, '\n',
                  'classification report: \n', cr, '\n')
            cm_ = np.array2string(cm)
            logfile.write('Valid\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm_))
            logfile.write('Title\nf1: {}\nAccuracy: {}\nTotal: {}'.format(f1, acc, total))

            if f1 >= self.best_acc:
                self.best_acc = f1
                checkpoint_name = 'checkpoint_best.pth.tar'.format(epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'f1': f1,
                    'Accuracy': acc,
                    'Total': total
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

            # logging.debug(
            #     f"valid mode\tEpoch: {epoch}\tf1:{f1}\tacc:{acc}\ttotal:{total}")