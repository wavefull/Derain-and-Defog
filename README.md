# MDMTN：Model-driven Multi-task  Network for Image Dehazing and Deraining

[Xia Li][1], [Jianlong Wu][2], [Zhouchen Lin][3], [Hong Liu][4], [Hongbin Zha][5]<br>

Key Laboratory of Machine Perception, Shenzhen Graduate School, Peking University<br>
Key Laboratory of Machine Perception (MOE), School of EECS, Peking University<br>
Cooperative Medianet Innovation Center, Shanghai Jiao Tong University<br>
{[ethanlee][6], [jlwu1992][7], [zlin][8], [hongliu][9]}@pku.edu.cn, zha@cis.pku.edu.cn



## Prerequisite
- Python>=3.6
- Pytorch>=4.1.0
- Opencv>=3.1.0

## Dataset
Rain100H: [http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html][10]<br>
Rain800: [https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s][11]

We concatenate the two images(B and L) together as default inputs for training. B is corresponding to background image,L is image whose three channels are corresponding to rain streak ,vapor and transparency respectively.
Moreover, there should be three folders 'train', 'test_real', 'test_syn' in the dataset folder.

|         | PSNR  | SSIM  | inference time(Seconds) |
| :------:| :---: | :---: | :---------------------: | 
| testA   | 18.92 | 0.65  |          0.004          |

All PSNR and SSIM of results are computed by using skimage.measure. Please use this to evaluate your works.

## Train, Show and Test
**Train:**
* Download the dataset(~7.8GB) and unpack it into code folder "../dataset/train". Then, run:

```bash
$ python train.py
```
**Show:**
* Download the test dataset(~455MB) and unpack it into code folder "../dataset/test". Then, run: 

```
$ python show.py
```
**Test:**
* Generate syn_test.txt into code folder. Then, run: 

```
$ python test.py
```


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


