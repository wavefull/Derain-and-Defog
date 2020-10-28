# Model-driven Multi-task  Network for Image Dehazing and Deraining

[Xia Li][1], [Jianlong Wu][2], [Zhouchen Lin][3], [Hong Liu][4], [Hongbin Zha][5]<br>

Key Laboratory of Machine Perception, Shenzhen Graduate School, Peking University<br>
Key Laboratory of Machine Perception (MOE), School of EECS, Peking University<br>
Cooperative Medianet Innovation Center, Shanghai Jiao Tong University<br>
{[ethanlee][6], [jlwu1992][7], [zlin][8], [hongliu][9]}@pku.edu.cn, zha@cis.pku.edu.cn

Images captured from rainy outdoors suffer from mani-fest degradation of scene visibility. Although a magnitude of rain removal methods for single image have been proposed and can acheive good deraining results on sythetic data, the failure to function on real-world scene make them struggling to apply in industry. To this issue, we propose a noval rain model consist of two components(rain streak and mist caused by rain accumulation), which render rainy scene by using image fusion technology. Based on this model, we develop a multi-task deep learning network that learns the appearance of rain streaks, the representation of mist and the transparency map in the common Encoder-Decoder framework. Specifically, the fused features of these components is extracted by a cascaded Contextualized Dilated Covolution structure and splited into specific features by a Feature Seperation Module (FSM) we proposed in this network. Extensive experiments on sythetic and real-world datasets demonstrate that the proposed deraining network outperforms the last state-of-the-art networks in terms of accuracy, robustness and runtime. In addtiton, an ablation analysis is conducted to gauge the improvments obtained by different modules in the proposed network.

## Prerequisite
- Python>=3.6
- Pytorch>=4.1.0
- Opencv>=3.1.0

## Dataset
Rain100H: [http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html][10]<br>
Rain800: [https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s][11]

We concatenate the two images(B and O) together as default inputs. If you want to change this setting, just modify config/dataset.py.
Moreover, there should be three folders 'train', 'val', 'test' in the dataset folder.
After download the datasets, don't forget to transform the format!

|         | PSNR  | SSIM  | inference time(Seconds) |
| :------:| :---: | :---: | :---------------------: | 
| testA   | 18.92 | 0.65  |          0.004          |

All PSNR and SSIM of results are computed by using skimage.measure. Please use this to evaluate your works.

## Train, Test and Show
    python train.py
    python eval.py
    python show.py


## Cite
If you use our code, please refer this repo.
If you publish your paper that refer to our paper, please cite:

    @inproceedings{li2018recurrent,  
        title={Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining},  
        author={Li, Xia and Wu, Jianlong and Lin, Zhouchen and Liu, Hong and Zha, Hongbin},  
        booktitle={European Conference on Computer Vision},  
        pages={262--277},  
        year={2018},  
        organization={Springer}  
    }


