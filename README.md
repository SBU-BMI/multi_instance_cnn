The code is to train classification for WSIs. The code is built on Python 3.6. PyTorch 1.0.

The training command is: <br>
python micnn_train.py --config ./configs/XXXX.json --gpu_ids GPU_NUM

The testing command is: <br>
python micnn_test.py --config ./configs/XXXX.json --gpu_ids GPU_NUM

All experimental settings are stored in ./config/XXXX.json. <br>
["tile_precess"]["WSIs"]["output_path"]  is the path of WSI images. WSI images are tile images. <br>
["tile_precess"]["WSIs"]["label_file"]  is the label file, check out the get_wsi_id_labels function in data_preprocess/data_preprocess.py to see how I get the labels for the WSIs according to WSI ids. <br>
If you use only the WSI images for training, then set ["use_rgb_only"] to true <br>
["intermediate"] is the output path 

Contact me at huidliu@cs.stonybrook.edu if you have any other questions. 
