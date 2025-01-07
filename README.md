## InTraGen: Trajectory-controlled Video Generation for Object Interactions<br><sub>PyTorch Implementation</sub>
[![arXiv](https://img.shields.io/badge/arXiv-2401.03048-b31b1b.svg)](https://arxiv.org/abs/2411.16804)

<div align="center">
<img src=assets/logo.png width="80%"/>
</div>
<p align="center">


> [**InTraGen: Trajectory-controlled Video Generation for Object Interactions**](https://arxiv.org/abs/2411.16804)<br>
> [Zuhao Liu*](https://zuhaoliu.com/), [Aleksandar Yanev*](https://insait.ai/aleksandar-yanev/), [Ahmad Mahmood](https://ahmad-573.github.io/), [Ivan Nikolov](https://scholar.google.com/citations?user=K2m6cxcAAAAJ&hl=en&oi=sra), [Saman Motamed](https://sam-motamed.github.io/), [Wei-Shi Zheng](https://www.isee-ai.cn/~zhwshi/), [Xi Wang](https://xiwang1212.github.io/homepage/), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en&oi=ao), [Danda Pani Paudel](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en&oi=ao)
> (*Equal Contribution)

<!-- <video controls loop src="https://github.com/insait-institute/InTraGen.git/assets/video_demo.mp4" type="video/mp4"></video> -->
<video controls loop src="assets/video_demo.mp4" type="video/mp4"></video>


### How to Run

Dowload dataset

[MoVi-Extended](https://drive.google.com/file/d/10iqlXphUb_07ey-EACN6WFD7sUp9HAe2/view?usp=drive_link) | [Domino](https://drive.google.com/file/d/1O_fmHjUJhhDThovy8saYFf6_7HGn1cis/view?usp=drive_link) | [Football](https://drive.google.com/file/d/1TD67V1owYxsGXDJJ8Jvdd4CnLe4Ikfc6/view?usp=drive_link) | [Pool](https://drive.google.com/file/d/1Tu7csXSro4WmL55te9epxyTE13UuKj7Z/view?usp=drive_link)

1. Save and unzip the datasets into `datasets` folder


2. Install the Environment

```bash
bash install.sh
```

3. Download the [Pre-trained Checkpoints](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0) from [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), and save the `Open-Sora-Plan-v1.1.0` into `models` folder


4. Run the Training

```bash
bash train.sh
```


## Citation
If you find this work useful for your research, please consider citing it.
```bibtex
@article{liu2024intragen,
  title={InTraGen: Trajectory-controlled Video Generation for Object Interactions},
  author={Liu, Zuhao and Yanev, Aleksandar and Mahmood, Ahmad and Nikolov, Ivan and Motamed, Saman and Zheng, Wei-Shi and Wang, Xi and Van Gool, Luc and Paudel, Danda Pani},
  journal={arXiv preprint arXiv:2411.16804},
  year={2024}
}
```


### Acknowledgement
[Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)&nbsp;&nbsp;&nbsp;&nbsp; [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video)




