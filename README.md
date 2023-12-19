# STSSNet-AAAI2024

[![arXiv](https://img.shields.io/badge/arXiv-2312.10890-b31b1b.svg)](https://arxiv.org/abs/2312.10890)

Official Implementation of "Low-latency Space-time Supersampling for Real-time Rendering" (AAAI 2024).


[![](https://markdown-videos-api.jorgenkh.no/youtube/8aPu2ECwVLk)](https://youtu.be/8aPu2ECwVLk)

## Environment

We use Torch-TensorRT 1.1.0, PyTorch 1.11, CUDA 11.4, cuDNN 8.2 and TensorRT 8.2.5.1. 

Please download the corresponding version of [CUDA](https://developer.nvidia.com/cuda-11-4-1-download-archive), [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive), and [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download). Then set the environment variables as follows:

```bash
export TRT_RELEASE=~/project/TensorRT-8.2.5.1
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-11.4"
export LD_LIBRARY_PATH="$TRT_RELEASE/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```

Next, create a conda environment and install PyTorch and Torch-TensorRT:

```bash
conda create -n tensorrt python=3.7 pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda activate tensorrt
pip3 install $TRT_RELEASE/python/tensorrt-8.2.5.1-cp37-none-linux_x86_64.whl
pip3 install torch-tensorrt==1.1.0 -f https://github.com/pytorch/TensorRT/releases/download/v1.1.0/torch_tensorrt-1.1.0-cp37-cp37m-linux_x86_64.whl
pip3 install opencv-python tqdm thop matplotlib scikit-image lpips visdom numpy pytorch_msssim
```

## Dataset

We release the dataset used in our paper at [ModelScope](https://www.modelscope.cn/datasets/ryanhe312/STSSNet-AAAI2024). The dataset contains four scenes, Lewis, SunTemple, Subway, and Arena, each with around 6000 frames for training and 1000 for testing. Every frame is a compressed numpy array with 16-bit float type. 

You can install ModelScope by running:

```bash
pip install modelscope
```

Then you can load the dataset by running the following code in Python:

```python
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('ryanhe312/STSSNet-AAAI2024', subset_name='Lewis', split='test')
# ds =  MsDataset.load('ryanhe312/STSSNet-AAAI2024', subset_name='SunTemple', split='test')
# ds =  MsDataset.load('ryanhe312/STSSNet-AAAI2024', subset_name='Subway', split='test')
# ds =  MsDataset.load('ryanhe312/STSSNet-AAAI2024', subset_name='Arena', split='test')
```

Note that the dataset is around 40GB per test scene. It may take a while to download the dataset.

Please modify the path in `dataloaders.py` to your own path before running the code.

## Evaluation

You can modify the `dataset` and `mode` in `eval.py` to evaluate different scenes and modes.

`all` mode means evaluating all the pixels, `edge` mode means evaluating the pixels on the canny edge of the HR frame, and `hole` mode means evaluating the pixels in warping holes in the LR frame.

Run the following command to evaluate the model for PSNR, SSIM and LPIPS:

```bash
python eval.py
```

To evaluate the VMAF, you need to:

1. Set `save_img` to `True` in `eval.py` and run it.
2. Run generate `utils/video.py` to generate `gt.avi` and `pred.avi`.
3. Install ffmpeg, and add its path to environment variable "PATH".
4. Follow the instructions of [VMAF](https://github.com/Netflix/vmaf) to use ffmpeg to compute VMAF metric between `gt.avi` and `pred.avi`.

## Benchmark

You can test model size, FLOPs, and the inference speed of our model by running:

```bash
python benchmark.py
```

You should get the following results:

```
Computational complexity:       31.502G 
Number of parameters:           417.241K
Time: 4.350 ms
```

Inference speed is tested on a single RTX 3090 GPU and may vary on different machines.

## Training

We will release the training dataset and generation scripts soon.

## Acknowledgement

We thank the authors of [ExtraNet](https://github.com/fuxihao66/ExtraNet) for their great work and data generation pipeline.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{he2023lowlatency,
      title={Low-latency Space-time Supersampling for Real-time Rendering}, 
      author={Ruian He and Shili Zhou and Yuqi Sun and Ri Cheng and Weimin Tan and Bo Yan},
      year={2023},
      eprint={2312.10890},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```