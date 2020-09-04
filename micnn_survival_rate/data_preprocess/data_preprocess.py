import os
import sys
from PIL import Image
from glob import glob
import csv
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from utils import ensure_dir, get_max_xy_in_paths
from utils import find_str_in_list, compute_tile_xys, compute_patch_xys


def get_wsi_id_labels(csv_file_path, survived_days=1825):
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = None
        wsi_labels = dict()
        for row in spamreader:
            if header is None:
                header = row[0]
            else:
                wsi_id = row[6][1:-1]
                if row[3] == 'NA':
                    continue
                days = int(row[3])
                dead = int(row[4][1:-1])
                if days <= survived_days and dead == 1:
                    wsi_labels[wsi_id] = 1
                if days > survived_days:
                    wsi_labels[wsi_id] = 0

    return wsi_labels


class FG_Mask:
    def __init__(self, wsi_root, mask_root, label_file, scale=1, patch_size=(224,224), tile_size=(4000,4000), max_num_patches=5000):
        assert scale >= 1, 'scale should be greater than or equal to 1'
        self.wsi_root = wsi_root
        self.mask_root = mask_root
        self.label_file = label_file
        self.scale = scale
        self.patch_size = patch_size
        self.patch_size_d = (self.patch_size[0] * scale, self.patch_size[1] * scale)
        self.tile_size = tile_size
        self.max_num_patches = max_num_patches

        # cache for fast computation, avoiding loading data repeatly
        self.cache_wsi_id = None
        self.cache_cross_status = None
        self.cache_txy_00 = None
        self.cache_txy_01 = None
        self.cache_txy_10 = None
        self.cache_txy_11 = None
        self.cache_tile_00 = None
        self.cache_tile_01 = None
        self.cache_tile_10 = None
        self.cache_tile_11 = None

    def _reset_cache(self, wsi_id, cross_status,
                      txy_00, txy_01, txy_10, txy_11,
                      pxy_00, pxy_01, pxy_10, pxy_11
    ):
        use_cache = \
            self.cache_wsi_id == wsi_id and \
            self.cache_cross_status == cross_status and \
            self.cache_txy_00 == txy_00 and \
            self.cache_txy_01 == txy_01 and \
            self.cache_txy_10 == txy_10 and \
            self.cache_txy_11 == txy_11

        if not use_cache:
            self.cache_wsi_id = wsi_id
            self.cache_cross_status = cross_status
            self.cache_txy_00 = txy_00
            self.cache_txy_01 = txy_01
            self.cache_txy_10 = txy_10
            self.cache_txy_11 = txy_11
            self.cache_tile_00 = None
            self.cache_tile_01 = None
            self.cache_tile_10 = None
            self.cache_tile_11 = None

        return use_cache

    def _load_rgb(self, wsi_tile_list, txy, pxy):
        subfn = '/{}_{}_{}_{}'.format(txy[0], txy[1], self.tile_size[0], self.tile_size[1])
        
        wsi_tile_path = find_str_in_list(wsi_tile_list, subfn)
        wsi_tile = np.array(Image.open(wsi_tile_path[0]))
        wsi_patch = wsi_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1], :]
        
        return wsi_patch, wsi_tile

    def set_scale(self, scale=1):
        assert scale >= 1, 'scale should be greater than or equal to 1'
        self.scale = scale
        # dilated patch size using scale
        self.patch_size_d = (self.patch_size[0] * scale, self.patch_size[1] * scale)

    def _compute_fg(self, wsi_path):
        wsi_id = wsi_path.split('/')[-1]
        fg_output_dir = '{}/patch_size_{}_{}_scale_{}/fg_masks'.format(
            self.mask_root, self.patch_size[0], self.patch_size[1], self.scale
        )

        fg_log_fn = '{}/{}_fg_mask.log'.format(fg_output_dir, wsi_id)
        fg_fn = '{}/{}_fg_mask.pth'.format(fg_output_dir, wsi_id)

        disc_output_dir = '{}/patch_size_{}_{}_scale_{}/disc_masks_round_0'.format(
            self.mask_root, self.patch_size[0], self.patch_size[1], self.scale
        )

        disc_fn = '{}/{}_disc_mask.pth'.format(disc_output_dir, wsi_id)

        if os.path.isfile(fg_fn) and os.path.isfile(disc_fn):
            return None
        
        wsi_tile_list = glob('{0}/{1}/*.png'.format(self.wsi_root, wsi_id))
        # sort tile names by x and then by y
        # wsi_tile_list = sorted(wsi_tile_list, key=lambda item: get_tile_xy_in_fn(item)) 

        # get wsi W and H using the last tile file name
        #  final_tile_basename = wsi_tile_list[-1].split('/')[-1].split('.')[0]
        start_x, start_y = get_max_xy_in_paths(wsi_tile_list)
        tw, th = self.tile_size
        wsi_W, wsi_H = int(start_x) + int(tw) - 1, int(start_y) + int(th) - 1
        # get number of rows and columns for patches (a patch is used to train a CNN)
        n_cols, n_rows = wsi_W // self.patch_size_d[0], wsi_H // self.patch_size_d[1]
        fg_mask = torch.zeros((n_rows, n_cols), dtype=torch.uint8)
        disc_wh = np.ones((n_rows, n_cols))
        disc_mask = torch.zeros((n_rows, n_cols), dtype=torch.uint8)

        cache_cross_status = None
        cache_txy_00 = None
        cache_txy_01 = None
        cache_txy_10 = None
        cache_txy_11 = None
        cache_tile_00 = None
        cache_tile_01 = None
        cache_tile_10 = None
        cache_tile_11 = None
        
        for col in range(1, n_cols + 1):
            for row in range(1, n_rows + 1):
                x_left, x_right = (col - 1) * self.patch_size_d[0] + 1, col * self.patch_size_d[0]  # start from 1
                y_top, y_bottom = (row - 1) * self.patch_size_d[1] + 1, row * self.patch_size_d[1]  # start from 1
                pxy = (x_left, x_right, y_top, y_bottom)
                txy_00, txy_01, txy_10, txy_11 = compute_tile_xys(pxy, self.tile_size)
                cross_status, pxy_00, pxy_01, pxy_10, pxy_11 = compute_patch_xys(pxy, self.patch_size_d, self.tile_size)

                use_cache = \
                    cache_cross_status == cross_status and \
                    cache_txy_00 == txy_00 and \
                    cache_txy_01 == txy_01 and \
                    cache_txy_10 == txy_10 and \
                    cache_txy_11 == txy_11

                if not use_cache:
                    cache_cross_status = cross_status
                    cache_txy_00 = txy_00
                    cache_txy_01 = txy_01
                    cache_txy_10 = txy_10
                    cache_txy_11 = txy_11
                    cache_tile_00 = None
                    cache_tile_01 = None
                    cache_tile_10 = None
                    cache_tile_11 = None

                if cross_status == "00":
                    if use_cache:
                        patch_00 = cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                    else:
                        patch_00, cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                    patch = patch_00
                elif cross_status == "01":
                    if use_cache:
                        patch_00 = cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                        patch_01 = cache_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                    else:
                        patch_00, cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                        patch_01, cache_tile_01 = self._load_rgb(wsi_tile_list, txy_01, pxy_01)
                    patch = np.concatenate((patch_00, patch_01), axis=1)
                elif cross_status == "10":
                    if use_cache:
                        patch_00 = cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                        patch_10 = cache_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                    else:
                        patch_00, cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                        patch_10, cache_tile_10 = self._load_rgb(wsi_tile_list, txy_10, pxy_10)
                    patch = np.concatenate((patch_00, patch_10), axis=0)
                elif cross_status == "11":
                    if use_cache:
                        patch_00 = cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                        patch_01 = cache_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                        patch_10 = cache_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                        patch_11 = cache_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                    else:
                        patch_00, cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                        patch_01, cache_tile_01 = self._load_rgb(wsi_tile_list, txy_01, pxy_01)
                        patch_10, cache_tile_10 = self._load_rgb(wsi_tile_list, txy_10, pxy_10)
                        patch_11, cache_tile_11 = self._load_rgb(wsi_tile_list, txy_11, pxy_11)
                    patch_0 = np.concatenate((patch_00, patch_01), axis=1)
                    patch_1 = np.concatenate((patch_10, patch_11), axis=1)
                    patch = np.concatenate((patch_0, patch_1), axis=0)
                else:
                    raise NotImplementedError

                patch0 = patch[::self.scale, ::self.scale, :]  # down-size retrieved patch
                patch = patch0 / 127.5 - 1.0
                wh = patch[..., 0].std() + patch[..., 1].std() + patch[..., 2].std()
                if wh >= 0.18:  # WSI foreground parameter
                    fg_mask[row - 1, col - 1] = 1
                    accept, mean_val = self._accept_patch(patch0)
                    if accept:
                        disc_wh[row - 1, col - 1] = mean_val
            
            fg_log_str = 'progress {}/{}\n'.format(col, n_cols)
            with open(fg_log_fn, 'a') as f:
                f.write(fg_log_str)

        n_cols_per_tile = self.tile_size[0] // self.patch_size_d[0]
        n_rows_per_tile = self.tile_size[1] // self.patch_size_d[1]

        n_cols_tile = int(np.ceil(n_cols / n_cols_per_tile))
        n_rows_tile = int(np.ceil(n_rows / n_rows_per_tile))

        MAX_PER_TILE = int(np.ceil(self.max_num_patches / (n_cols_tile * n_rows_tile)))

        for t_col_start in range(0, n_cols, n_cols_per_tile):
            t_col_end = min(t_col_start + n_cols_per_tile, n_cols)
            for t_row_start in range(0, n_rows, n_rows_per_tile):
                t_row_end = min(t_row_start + n_rows_per_tile, n_rows)
                disc_wh_t = disc_wh[t_row_start:t_row_end, t_col_start:t_col_end]
                # (disc_wh_t < 1).sum() too small means a lot of white patches
                MAX_NUM = min((disc_wh_t < 1).sum(), MAX_PER_TILE)
                if MAX_NUM < 1:
                    continue
                disc_wh_t_flat = disc_wh_t.flatten()
                if MAX_NUM >= disc_wh_t_flat.shape[0]:
                    wh_th = disc_wh_t_flat.max()
                else:
                    # partition disc_wh_t_flat, from 0th to MAX_NUM th values are less than from MAX_NUM th to end values
                    partitioned_ind = np.argpartition(disc_wh_t_flat, MAX_NUM)
                    wh_th = np.max(disc_wh_t_flat[partitioned_ind][:MAX_NUM])
                
                if wh_th == 1:
                    disc_mask_t = torch.from_numpy((disc_wh_t < wh_th).astype(np.uint8))
                else:
                    disc_mask_t = torch.from_numpy((disc_wh_t <= wh_th).astype(np.uint8))

                disc_mask[t_row_start:t_row_end, t_col_start:t_col_end] = disc_mask_t

        torch.save(fg_mask, fg_fn)
        torch.save(disc_mask, disc_fn)

        fg_log_str = 'done! total number of valid patches {}, fg patches {} \n'.format(disc_mask.sum().item(), fg_mask.sum().item())
        with open(fg_log_fn, 'a') as f:
            f.write(fg_log_str)

    # this function is deprecated
    def compute_fg(self):
        # compute foreground mask
        print('Computing foreground at scale {}'.format(self.scale))
        output_dir = '{}/patch_size_{}_{}_scale_{}/fg_masks'.format(
            self.mask_root, self.patch_size[0], self.patch_size[1], self.scale
        )
        ensure_dir(output_dir)
        
        self.wsi_labels = get_wsi_id_labels(self.label_file)
        wsi_id_with_labels = list(self.wsi_labels.keys())

        wsi_path_list = glob('{0}/*'.format(self.wsi_root))
        wsi_path_list.sort()

        for wsi_path in wsi_path_list:
            wsi_id = wsi_path.split('/')[-1]
            if len(find_str_in_list(wsi_id_with_labels, wsi_id)) == 0:
                continue
            
            wsi_tile_list = glob('{0}/{1}/*.png'.format(self.wsi_root, wsi_id))
            # sort tile names by x and then by y
            # wsi_tile_list = sorted(wsi_tile_list, key=lambda item: get_tile_xy_in_fn(item)) 

            # get wsi W and H using the last tile file name
            #  final_tile_basename = wsi_tile_list[-1].split('/')[-1].split('.')[0]
            start_x, start_y = get_max_xy_in_paths(wsi_tile_list)
            tw, th = self.tile_size
            wsi_W, wsi_H = int(start_x) + int(tw) - 1, int(start_y) + int(th) - 1
            # get number of rows and columns for patches (a patch is used to train a CNN)
            n_cols, n_rows = wsi_W // self.patch_size_d[0], wsi_H // self.patch_size_d[1]
            fg_mask = torch.zeros((n_rows, n_cols), dtype=torch.uint8)

            log_fn = '{}/{}_fg_mask.log'.format(output_dir, wsi_id)

            for col in range(1, n_cols + 1):
                for row in range(1, n_rows + 1):
                    x_left, x_right = (col - 1) * self.patch_size_d[0] + 1, col * self.patch_size_d[0]  # start from 1
                    y_top, y_bottom = (row - 1) * self.patch_size_d[1] + 1, row * self.patch_size_d[1]  # start from 1
                    pxy = (x_left, x_right, y_top, y_bottom)
                    txy_00, txy_01, txy_10, txy_11 = compute_tile_xys(pxy, self.tile_size)
                    cross_status, pxy_00, pxy_01, pxy_10, pxy_11 = compute_patch_xys(pxy, self.patch_size_d, self.tile_size)
                    
                    use_cache = self._reset_cache(
                        wsi_id, cross_status,
                        txy_00, txy_01, txy_10, txy_11,
                        pxy_00, pxy_01, pxy_10, pxy_11
                    )

                    if cross_status == "00":
                        if use_cache:
                            patch_00 = self.cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                        else:
                            patch_00, self.cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                        patch = patch_00
                    elif cross_status == "01":
                        if use_cache:
                            patch_00 = self.cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                            patch_01 = self.cache_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                        else:
                            patch_00, self.cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                            patch_01, self.cache_tile_01 = self._load_rgb(wsi_tile_list, txy_01, pxy_01)
                        patch = np.concatenate((patch_00, patch_01), axis=1)
                    elif cross_status == "10":
                        if use_cache:
                            patch_00 = self.cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                            patch_10 = self.cache_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                        else:
                            patch_00, self.cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                            patch_10, self.cache_tile_10 = self._load_rgb(wsi_tile_list, txy_10, pxy_10)
                        patch = np.concatenate((patch_00, patch_10), axis=0)
                    elif cross_status == "11":
                        if use_cache:
                            patch_00 = self.cache_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                            patch_01 = self.cache_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                            patch_10 = self.cache_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                            patch_11 = self.cache_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                        else:
                            patch_00, self.cache_tile_00 = self._load_rgb(wsi_tile_list, txy_00, pxy_00)
                            patch_01, self.cache_tile_01 = self._load_rgb(wsi_tile_list, txy_01, pxy_01)
                            patch_10, self.cache_tile_10 = self._load_rgb(wsi_tile_list, txy_10, pxy_10)
                            patch_11, self.cache_tile_11 = self._load_rgb(wsi_tile_list, txy_11, pxy_11)
                        patch_0 = np.concatenate((patch_00, patch_01), axis=1)
                        patch_1 = np.concatenate((patch_10, patch_11), axis=1)
                        patch = np.concatenate((patch_0, patch_1), axis=0)
                    else:
                        raise NotImplementedError

                    patch = patch[::self.scale, ::self.scale, :]  # down-size retrieved patch
                    patch = patch / 127.5 - 1.0
                    wh = patch[..., 0].std() + patch[..., 1].std() + patch[..., 2].std()
                    if wh >= 0.18:
                        fg_mask[row - 1, col - 1] = 1

                log_str = 'progress {}/{}\n'.format(col, n_cols)
                with open(log_fn, 'a') as f:
                    f.write(log_str)

            fn = '{}/{}_fg_mask.pth'.format(output_dir, wsi_id)
            torch.save(fg_mask, fn)

            log_str = 'done!\n'
            with open(log_fn, 'a') as f:
                f.write(log_str)

    def compute_fg_parallel(self, ncores):
        print('Computing foreground at scale {}'.format(self.scale))
        
        self.wsi_labels = get_wsi_id_labels(self.label_file)
        wsi_id_with_labels = list(self.wsi_labels.keys())

        wsi_path_list = glob('{0}/*'.format(self.wsi_root))
        wsi_path_list.sort()

        wsi_path_list_with_labels = []
        for wsi_path in wsi_path_list:
            wsi_id = wsi_path.split('/')[-1]
            if len(find_str_in_list(wsi_id_with_labels, wsi_id)) == 0:
                continue
            else:
                wsi_path_list_with_labels.append(wsi_path)

                
        fg_output_dir = '{}/patch_size_{}_{}_scale_{}/fg_masks'.format(
            self.mask_root, self.patch_size[0], self.patch_size[1], self.scale
        )
        ensure_dir(fg_output_dir)

        disc_output_dir = '{}/patch_size_{}_{}_scale_{}/disc_masks_round_0'.format(
            self.mask_root, self.patch_size[0], self.patch_size[1], self.scale
        )
        ensure_dir(disc_output_dir)

        if ncores > 1:
            p = Pool(ncores)
            print('Using {} cores'.format(ncores))
            for i, _ in enumerate(p.imap_unordered(self._compute_fg, wsi_path_list_with_labels), 1):
                sys.stderr.write('\rProgress {0:%} done! '.format(i / len(wsi_path_list_with_labels)))
        else:
            num_wsis = len(wsi_path_list_with_labels)
            for idx, wsi_path in enumerate(wsi_path_list_with_labels, 0):
                self._compute_fg(wsi_path)
                print('{}/{} done!'.format(idx, num_wsis))
        print('\nAll done! \n')
            
    def _accept_patch(self, patch):
        rgb_ratio = [0.299, 0.587, 0.114]
        patch = np.dot(patch[..., :3], rgb_ratio)
        H, W = patch.shape
        
        # parameters
        sigma = 15
        gray_th = 30
        white_th = 220
        white_rate_th = 0.3
        
        med = gaussian_filter(patch, sigma=sigma)
        gray_max = patch.max()

        white_rate = np.sum( (patch >= white_th).astype(np.float) ) / H / W

        return white_rate < white_rate_th and gray_max > gray_th, patch.mean() / 255.0

    
