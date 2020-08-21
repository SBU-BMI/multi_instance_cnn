import os
from glob import glob
import pickle
from copy import copy

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from utils import ensure_dir


class WSI_cls:
    def __init__(self, train_path, test_path, svm_root, wsi_ids_train, wsi_ids_test, scales, patch_size=(224,224), val_test_ratio=0.5):
        # wsi_ids: tuple of wsi_id
        # scales: tuple of scale
        self.patch_size = patch_size
        self.X_train, self.y_train = self._cat_npy_all_scales(train_path, wsi_ids_train, scales)
        X, y = self._cat_npy_all_scales(test_path, wsi_ids_test, scales)
        self.X_val, self.y_val, self.X_test, self.y_test = self._split_test_to_val_and_test(X, y, val_test_ratio=val_test_ratio)

        self.model = None
        self.best_ap = -1
        self.best_acc = -1
        self.best_C = -1
        self.best_gamma = -1
        # best_gamma:
        # -1, for initilization
        # -2, 'scale'
        # -3, 'auto'
        
        self.svm_root = svm_root
        # svm_root: string, path to store trained SVM model
        #     |--linear
        #         |--model.pkl
        #     |--rbf
        #         |--model.pkl
        
    def _cat_npy_all_scales(self, feature_root, wsi_ids, scales):
        n_scales = len(scales)

        data = None
        label = None
        for wsi_id in wsi_ids:
            data_i = None
            label_i = None
            for scale in scales:
                feat_path = '{}/patch_size_{}_{}_scale_{}/{}_hist_feat.npy'.format(feature_root, self.patch_size[0], self.patch_size[1], scale, wsi_id)
                feat = np.load(feat_path)
                if data_i is None:
                    label_i = feat[:1]
                    data_i = feat[1:] / feat[1:].sum()
                else:
                    new_feat = feat[1:] / feat[1:].sum()
                    data_i = np.concatenate((data_i, new_feat), axis=None)
            label_i = label_i[np.newaxis, :]
            data_i = data_i / data_i.sum()
            data_i = data_i[np.newaxis, :]
            if data is None:
                label = copy(label_i)
                data = copy(data_i)
            else:
                label = np.concatenate((label, label_i), axis=0)
                data = np.concatenate((data, data_i), axis=0)
        label = label.ravel()
        
        return data, label

    # split the test set into validation set and real test set.
    # val_ratio = val_set_size / (val_set_size + real_test_set_size)
    def _split_test_to_val_and_test(self, X, y, val_test_ratio=0.5):
        classes = np.unique(y)
        X_val = None
        y_val = None
        X_test = None
        y_test = None

        for c in classes:
            indices = y == c
            X_c = X[indices]
            n = int(X_c.shape[0] * val_test_ratio)
            X_val_c = X_c[:n]
            X_test_c = X_c[n:]
            y_c = y[indices]
            y_val_c = y_c[:n]
            y_test_c = y_c[n:]

            if X_val is None:
                X_val = X_val_c
                y_val = y_val_c
            else:
                X_val = np.concatenate((X_val, X_val_c), axis=0)
                y_val = np.concatenate((y_val, y_val_c), axis=0)

            if X_test is None:
                X_test = X_test_c
                y_test = y_test_c
            else:
                X_test = np.concatenate((X_test, X_test_c), axis=0)
                y_test = np.concatenate((y_test, y_test_c), axis=0)

        return X_val, y_val, X_test, y_test

    def _get_scores(self, y_gt, y_pred_scores, y_pred):
        ap = average_precision_score(y_gt, y_pred)
        accuracy = sum(y_pred == y_gt) / len(y_gt)

        return ap, accuracy

    def _save_model(self, kernel):
        output_dir = '{}/{}'.format(self.svm_root, kernel)
        ensure_dir(output_dir)
        model_path = '{}/model.pkl'.format(output_dir)
        ap_path = '{}/ap.pkl'.format(output_dir)
        acc_path = '{}/acc.pkl'.format(output_dir)
        C_path = '{}/C.pkl'.format(output_dir)
        gamma_path = '{}/gamma.pkl'.format(output_dir)
        
        with open(model_path, 'wb') as output:
            pickle.dump(self.best_model, output, pickle.HIGHEST_PROTOCOL)        
        with open(ap_path, 'wb') as output:
            pickle.dump(self.best_ap, output, pickle.HIGHEST_PROTOCOL)                 
        with open(acc_path, 'wb') as output:
            pickle.dump(self.best_acc, output, pickle.HIGHEST_PROTOCOL)
        with open(C_path, 'wb') as output:
            pickle.dump(self.best_C, output, pickle.HIGHEST_PROTOCOL)
        with open(gamma_path, 'wb') as output:
            pickle.dump(self.best_gamma, output, pickle.HIGHEST_PROTOCOL)

    def _load_model(self, kernel):
        output_dir = '{}/{}'.format(self.svm_root, kernel)
        model_path = '{}/model.pkl'.format(output_dir)
        ap_path = '{}/ap.pkl'.format(output_dir)
        acc_path = '{}/acc.pkl'.format(output_dir)
        C_path = '{}/C.pkl'.format(output_dir)
        gamma_path = '{}/gamma.pkl'.format(output_dir)
        
        with open(model_path, 'rb') as input:
            self.best_model = pickle.load(input)
        with open(apl_path, 'rb') as input:
            self.best_ap = pickle.load(input)
        with open(acc_path, 'rb') as input:
            self.best_acc = pickle.load(input)
        with open(C_path, 'rb') as input:
            self.best_C = pickle.load(input)
        with open(gamma_path, 'rb') as input:
            self.best_gamma = pickle.load(input)            
            
    def train(self, kernel, Cs, gammas=-1):
        # X_tr, X_val, y_tr, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
        X_tr, y_tr, X_val, y_val = self.X_train, self.y_train, self.X_val, self.y_val
        self.kernel = kernel
        use_dual = X_tr.shape[0] <= X_tr.shape[1]
        if kernel.lower() == 'linear':
            for C in Cs:
                model = svm.LinearSVC(dual=use_dual, C=C)
                model.fit(X_tr, y_tr)
                y_val_pred_scores = model.decision_function(X_val)
                y_val_pred = model.predict(X_val)
                ap, acc = self._get_scores(y_val, y_val_pred_scores, y_val_pred)
                if ap > self.best_ap:
                    self.best_model = model
                    self.best_ap = ap
                    self.best_acc = acc
                    self.best_C = C
                print('SVM training, kernel: {}, C: {}, average precision: {}, accuracy: {} '.format(kernel.lower(), C, ap, acc))
                
        elif kernel.lower() == 'rbf':
            gammas = list(gammas) + ['scale', 'auto']
            for C in Cs:
                for gamma in gammas:
                    model = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel='rbf', gamma=gamma))
                    model.fit(X_tr, y_tr)
                    y_val_pred_scores = model.decision_function(X_val)
                    y_val_pred = model.predict(X_val)
                    ap, acc = self._get_scores(y_val, y_val_pred_scores, y_val_pred)
                    if ap > self.best_ap:
                        self.best_model = model
                        self.best_ap = ap
                        self.best_acc = acc
                        self.best_C = C
                        self.best_gamma = gamma
                        if gamma == 'scale':
                            self.best_gamma = -2
                        elif gamma == 'auto':
                            self.best_gamma = -3
                        else:
                            pass
                    print('SVM training, kernel: {}, C: {}, gamma: {}, average precision: {}, accuracy: {} '.format(kernel.lower(), C, gamma, ap, acc))
                        
        else:
            raise NotImplementedError

        self._save_model(kernel)
        
    def test(self):
        if self.best_model is None:
            self._load_model(self.kernel)
    
        y_test_pred_scores = self.best_model.decision_function(self.X_test)
        y_test_pred = self.best_model.predict(self.X_test)
        ap, acc = self._get_scores(self.y_test, y_test_pred_scores, y_test_pred)

        print('Testing Average Precision (AP): {0:.4f}, Accuracy: {1:.4f}'.format(ap, acc))

        return ap, acc
        
