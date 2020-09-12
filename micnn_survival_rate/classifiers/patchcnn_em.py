import os
from glob import *
import random

import numpy as np
from sklearn.cluster import KMeans  
import scipy.ndimage as ndimage

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import models.resnet as model_arch

from utils import ensure_dir, get_instance


class PatchCNN_EM:
    def __init__(self, args, device, train_loader=None, eval_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.args = args
        self.device = device
        self.n_epochs = args['n_epochs']
        self.beta1 = args['beta1']
        self.beta2 = args['beta2']
        self.output_dir = args['output_dir']
        self.mask_dir = args['mask_dir']
        self.no_wsi_id = args['no_wsi_id']
        self.verbose = args['verbose']
        self.log_every_iters = args['log_every_iters']
        self.eval_every_iters = args['eval_every_iters']
        self.eval_every_epochs = args['eval_every_epochs']
        self.save_model_every_epochs = args['save_model_every_epochs']
        self.smooth_sigma = args['smooth_sigma']
        self.seg_quantile = args['seg_quantile']
        self.num_cls = args['num_cls'] # number of classes, fill it out later

        manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        cudnn.benchmark = True

        ensure_dir(self.output_dir)

        self.model_path = '{0}/models'.format(self.output_dir)
        ensure_dir(self.model_path)

        self.log_fileName = '{0}/patch_cnn_em_log.txt'.format(self.output_dir)
        with open(self.log_fileName, 'a') as f:
            f.write('\n{}\n'.format(str(args)))

        ###################################################################################
        self.net = get_instance(model_arch, 'model_arch', self.args)
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)

        self.net = self.net.to(self.device)
        net_params = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optimizer = get_instance(optim, 'optimizer', self.args, net_params)
        self.sm = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _load_last_model(self, round_no):
        model_path_round = '{}/round_{}'.format(self.model_path, round_no)
        models = glob('{}/*.pth'.format(model_path_round))
        model_ids = [(int(f.split('_')[2]), f) for f in [p.split('/')[-1].split('.')[0] for p in models]]
        if not model_ids:
            self.epoch = 1
            print('No net for patch cnn loaded!')
        else:
            self.epoch, fn = max(model_ids, key=lambda item: item[0])
            self.net.load_state_dict(torch.load('{}/{}.pth'.format(
                model_path_round, fn))
            )
            print('{}.pth for patch classification loaded!'.format(fn))

    def _save_model(self, epoch, iteration):
        torch.save(self.net.state_dict(), '{0}/net_epoch_{1}_iter_{2}.pth'.format(self.model_path_round, epoch, iteration))

    def set_round_no(self, round_no):
        assert round_no >= 0, 'round_no (round number) should be greater than or equal to 0'
        self.round_no = round_no
        self.model_path_round = '{}/round_{}'.format(self.model_path, self.round_no)
        ensure_dir(self.model_path_round)

    def m_step(self):
        self._load_last_model(self.round_no - 1)
        self.net.train()
        self.epoch = 1
        max_numIters = len(self.train_loader.dataset) // self.train_loader.batch_size
        while self.epoch <= self.n_epochs:
            running_loss = 0.0
            for idx, data in enumerate(self.train_loader, 0):
                
                patches, labels, _, _, _, _, _ = data
                patches = patches.to(self.device)
                labels = labels.to(self.device)

                inputs = patches
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if idx % self.log_every_iters == self.log_every_iters - 1:
                    avg_loss = running_loss / self.log_every_iters
                    log_str = 'Round {:d}, epoch {:d}/{:d}, iter {:d}/{:d}, loss: {:.6f}'.format(self.round_no, self.epoch, self.n_epochs, idx, max_numIters, avg_loss)
                    if self.verbose > 0:
                        print(log_str)
                    with open(self.log_fileName, 'a') as f:
                        f.write('{}\n'.format(log_str))
                    running_loss = 0

            if self.epoch % self.save_model_every_epochs == 0:
                self._save_model(self.epoch, idx)

            self.epoch += 1

    def e_step(self):
        self._load_last_model(self.round_no)
        self.net.eval()
        max_numIters = len(self.eval_loader.dataset) // self.eval_loader.batch_size
        
        prev_wsi_no = -1
        with torch.no_grad():
            for idx, data in enumerate(self.eval_loader, 0):
                patches, labels, cols, rows, n_cols_s, n_rows_s, wsi_nos = data                                        
                patches = patches.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(patches)
                probabilities = self.sm(outputs.detach()).cpu()

                for i in range(wsi_nos.shape[0]):
                    wsi_no = wsi_nos[i].item()
                    label = labels[i].item()
                    col, row = cols[i].item(), rows[i].item()
                    n_cols, n_rows = n_cols_s[i].item(), n_rows_s[i].item()
                    prob = probabilities[i][label].item()
                    
                    if prev_wsi_no != wsi_no:
                        if prev_wsi_no >= 0:
                            self._smooth_save_mask(disc_mask, prev_wsi_no)
                        disc_mask = torch.zeros((n_rows, n_cols))
                        prev_wsi_no = wsi_no

                    disc_mask[row - 1, col - 1] = prob
                    
                if idx % self.log_every_iters == self.log_every_iters - 1:
                    log_str = 'Round {:d}, e_step iter {:d}/{:d}'.format(self.round_no, idx, max_numIters)
                    if self.verbose > 0:
                        print(log_str)

            self._smooth_save_mask(disc_mask, wsi_no)

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader

    def set_eval_loader(self, eval_loader):
        self.eval_loader = eval_loader

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    def _smooth_save_mask(self, disc_mask, wsi_no):
        wsi_id = self.no_wsi_id[wsi_no]
        valid_mask = torch.load('{0}/disc_masks_round_0/{1}_disc_mask.pth'.format(self.mask_dir, wsi_id))
        disc_mask_np = disc_mask.numpy()
        disc_mask_np_sm = ndimage.gaussian_filter(disc_mask_np, sigma=self.smooth_sigma)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(disc_mask_np_sm.reshape((-1, 1)))
        kmeans_thre = kmeans.cluster_centers_.mean().item()
        quantile_thre = np.quantile(disc_mask_np_sm, self.seg_quantile)
        thre = min(kmeans_thre, quantile_thre)
        disc_mask_np_bin = disc_mask_np_sm >= thre
        disc_mask = torch.from_numpy(disc_mask_np_bin.astype(np.uint8)).type(torch.uint8)
        disc_mask = disc_mask * valid_mask
        fn = '{0}/disc_masks_round_{1}/{2}_disc_mask.pth'.format(self.mask_dir, self.round_no, wsi_id)
        # need to debug to see disc_mask data type, debuged, it's bytetensor
        torch.save(disc_mask, fn)

    def _compute_save_feat(self, pred_mat, pred_prob, wsi_no, label, output_dir):
        wsi_id = self.no_wsi_id[wsi_no]
        fn = '{}/{}_hist_feat.npy'.format(output_dir, wsi_id)
        pred_mat = pred_mat.numpy()
        bins = np.arange(-1, self.num_cls + 1)
        hist = np.histogram(pred_mat, bins=bins, weights=pred_prob, density=False)[0]
        hist[0] = label # The first item is the label and the rest is feature
        np.save(fn, hist)
        fn_prob = '{}/{}_prob.npy'.format(output_dir, wsi_id)
        pred_prob = pred_prob.numpy()
        np.save(fn_prob, pred_prob)

    def hist_feat(self, data_loader, output_dir):
        self._load_last_model(self.round_no)
        self.net.eval()

        prev_wsi_no = -1
        prev_label = -1
        with torch.no_grad():
            for idx, data in enumerate(data_loader, 0):
                patches, labels, cols, rows, n_cols_s, n_rows_s, wsi_nos = data
                patches = patches.to(self.device)
                outputs = self.net(patches)
                probs, predict_labels = torch.max(outputs.data, 1)
                probs = self.sm(probs).cpu()

                for i in range(wsi_nos.shape[0]):
                    wsi_no = wsi_nos[i].item()
                    label = labels[i].item()
                    col, row = cols[i].item(), rows[i].item()
                    n_cols, n_rows = n_cols_s[i].item(), n_rows_s[i].item()
                    prob = probs[i].item()
                    predict_label = predict_labels[i].item()

                    if prev_wsi_no != wsi_no:
                        if prev_wsi_no >= 0:
                            self._compute_save_feat(pred_mat, pred_prob, prev_wsi_no, prev_label, output_dir)
                        pred_mat = torch.ones((n_rows, n_cols), dtype=torch.int16)
                        pred_prob = torch.zeros((n_rows, n_cols), dtype=torch.float32)
                        pred_mat.fill_(-1)
                        prev_wsi_no = wsi_no
                        prev_label = label

                    pred_prob[row - 1, col - 1] = prob
                    pred_mat[row - 1, col - 1] = predict_label

            self._compute_save_feat(pred_mat, pred_prob, wsi_no, label, output_dir)


