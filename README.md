## InTraGen: Trajectory-controlled Video Generation for Object Interactions<br><sub>PyTorch Implementation</sub>
### [Paper](https://arxiv.org/abs/2411.16804)

### How to Run

Dowload dataset

[MoVi-Extended](https://drive.google.com/file/d/10iqlXphUb_07ey-EACN6WFD7sUp9HAe2/view?usp=drive_link) | Domino | Football | Pool

1. Put and etract the dataset into `datasets` folder


2. Install the Environment

```bash
bash install.sh
```

3. Download the [Pre-trained Checkpoints](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0) from [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), and save the `Open-Sora-Plan-v1.1.0` into `models` folder


4. Run the Training

```bash
bash train.sh
```


### Acknowledgement
[Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)




