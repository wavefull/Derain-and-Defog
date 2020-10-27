# RESCAN: [Recurrent Squeeze-and-Excitation Context Aggregation Net](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.pdf)

[Xia Li][1], [Jianlong Wu][2], [Zhouchen Lin][3], [Hong Liu][4], [Hongbin Zha][5]<br>

Key Laboratory of Machine Perception, Shenzhen Graduate School, Peking University<br>
Key Laboratory of Machine Perception (MOE), School of EECS, Peking University<br>
Cooperative Medianet Innovation Center, Shanghai Jiao Tong University<br>
{[ethanlee][6], [jlwu1992][7], [zlin][8], [hongliu][9]}@pku.edu.cn, zha@cis.pku.edu.cn

Rain streaks can severely degrade the visibility, which causes many current computer vision algorithms fail to work. So it is necessary to remove the rain from images. We propose a novel deep network architecture based on deep convolutional and recurrent neural networks for single image deraining. As contextual information is very important for rain removal, we first adopt the dilated convolutional neural network to acquire large receptive field. To better fit the rain removal task, we also modify the network. In heavy rain, rain streaks have various directions and shapes, which can be regarded as the accumulation of multiple rain streak layers. We assign different alpha-values to various rain streak layers according to the intensity and transparency by incorporating the squeeze-and-excitation block. Since rain streak layers overlap with each other, it is not easy to remove the rain in one stage. So we further decompose the rain removal into multiple stages. Recurrent neural network is incorporated to preserve the useful information in previous stages and benefit the rain removal in later stages. We conduct extensive experiments on both synthetic and real-world datasets. Our proposed method outperforms the state-of-the-art approaches under all evaluation metrics.

## Prerequisite
- Python>=3.6
- Pytorch>=4.1.0
- Opencv>=3.1.0

## Default Dataset settings
Rain100H: [http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html][10]<br>
Rain800: [https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s][11]

We concatenate the two images(B and O) together as default inputs. If you want to change this setting, just modify config/dataset.py.
Moreover, there should be three folders 'train', 'val', 'test' in the dataset folder.
After download the datasets, don't forget to transform the format!

Update: Rain100H has updated its testing set, from origin 100 images(test100) to 200(test200) images. We update the performance of RESCAN + GRU as follow:

|         | PSNR  | SSIM  | inference time(Seconds) |
| :------:| :---: | :---: | :---------------------: | 
| testA   | 18.92 | 0.65  |          0.004          |

## Train, Test and Show
    python train.py
    python eval.py
    python show.py

## Scripts
- explore.sh: Show the predicted images in browser
- config/tensorboard.sh: Open the tensorboard server
- config/clean.sh: Clear all the training records in the folder

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


  
## Recent Works
- [SDSEN](https://github.com/prismformore/SDSEN) from Hanrong Ye
- [SPANet](https://github.com/stevewongv/SPANet) from Tianyu Wang
- [HeavyRainRemoval](https://github.com/liruoteng/HeavyRainRemoval) from Ruoteng Li

