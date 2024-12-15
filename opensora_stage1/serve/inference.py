

import argparse
import sys
import os
import random
import shutil
import datetime
import imageio
import torch
from diffusers import PNDMScheduler
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from datetime import datetime
from typing import List, Union
import gradio as gr
import numpy as np
from gradio.components import Textbox, Video, Image
from transformers import T5Tokenizer, T5EncoderModel
from opensora_stage1.models.text_encoder import get_text_enc, get_text_warpper


from opensora_stage1.models.causalvideovae import ae_stride_config

from opensora_stage1.models import CausalVAEModelWrapper

from opensora_stage1.models.diffusion.latte.modeling_latte import LatteT2V
from opensora_stage1.sample.pipeline_videogen import VideoGenPipeline
from opensora_stage1.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples, DESCRIPTION

from datetime import datetime

import json

def read_json(json_file_path):
    # Initialize the lists
    paths = []
    anno_paths = []
    captions = []

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        for item in data:
            paths.append(item['path'])
            anno_paths.append(item['anno_path'])
            captions.append(item['cap'][0])  # Assuming each 'cap' list contains one string
    return paths, anno_paths, captions


@torch.inference_mode()
def generate_img(prompt, sample_steps, scale, seed=0, randomize_seed=False, force_images=False, trajectory_path = None, save_path = None):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)
    video_length = transformer_model.config.video_length if not force_images else 1
    height, width = int(args.version.split('x')[1]), int(args.version.split('x')[2])
    num_frames = 1 if video_length == 1 else int(args.version.split('x')[0])
    
    videos = videogen_pipeline(prompt,
                               num_frames=num_frames,
                               height=height,
                               width=width,
                               num_inference_steps=sample_steps,
                               guidance_scale=scale,
                               enable_temporal_attentions=not force_images,
                               num_images_per_prompt=1,
                               mask_feature=True,
                               trajectory_path=trajectory_path,
                               ).video

    torch.cuda.empty_cache()
    videos = videos[0]
    #tmp_save_path = 'tmp6.png'
    #tmp_save_path = 'tmp8.mp4'
    imageio.mimwrite(save_path, videos, fps=24, quality=6)  # highest quality is 10, lowest is 0
    display_model_info = f"Video size: {num_frames}×{height}×{width}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
    return save_path, prompt, display_model_info, seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/home/zuhao_liu/Open-Sora-Plan/V4_tora_v2_new_vae_correct_videos_old/checkpoint-20000/model')
    #parser.add_argument("--model_path", type=str, default='/home/zuhao_liu/Open-Sora-Plan/exp02/checkpoint-36000/model')
    parser.add_argument("--version", type=str, default='65x512x512', choices=['65x512x512', '221x512x512'])
    parser.add_argument("--ae", type=str, default='CausalVAEModel_D4_4x8x8')
    parser.add_argument("--vae_path", type=str, default="/home/zuhao_liu/Open-Sora-Plan/Open-Sora-Plan-v1.1.0/vae_1_2")
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--json_file_path", type=str, default='/home/zuhao_liu/ShareGPT4Video/sim_tora_v1_5K_change_name_v3_val.json')
    parser.add_argument('--force_images', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    
    print("Loading model ...")
    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, torch_dtype=torch.bfloat16, cache_dir='cache_dir').to(device)
    
    #vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device)
    # vae = getae_wrapper(args.ae)("/home/zuhao_liu/Open-Sora-Plan/Open-Sora-Plan-v1.1.0", subfolder="vae", cache_dir='cache_dir').to(device)
    vae = CausalVAEModelWrapper(args.vae_path, cache_dir='cache_dir').eval().to(device, dtype=torch.bfloat16)

    
    # print("\n\n\n vae.device", vae.device)
    # vae = vae.half()

    vae.vae.enable_tiling()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    transformer_model.force_images = args.force_images
    
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir="cache_dir")
    
    # text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir="cache_dir", enable_8bit_t5=True).to(device)
    # text_enc = get_text_warpper(args.text_encoder_name)(args, **kwargs).eval()
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir="cache_dir", torch_dtype=torch.bfloat16).to(device)
    
    # set eval mode
    transformer_model.eval()
    
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler()
    
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device)

    


    # prompts = ["The video showcases a sequence of frames that depict a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the arrangement of these balls changing across the frames. Initially, the table is set up with a green ball in the lower left quadrant, a white ball in the lower right quadrant, and a red ball in the upper left quadrant. As the video progresses, the balls are rearranged: the green ball moves to the upper left quadrant, the white ball shifts to the upper right quadrant, and the red ball moves to the lower left quadrant. Throughout the sequence, the background remains a solid dark color, and the camera's perspective does not change, maintaining a consistent view of the table and its evolving arrangement of balls.",
    #             "The video showcases a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the red ball positioned in the upper left corner and the white ball in the lower right corner. The table's surface is green, and the balls are white with black spots. Throughout the video, there is no movement or change in the arrangement of the balls, the environment, or the camera's perspective. The scene remains static, with the pool balls maintaining their positions on the table.",
    #             "The video showcases a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the arrangement of these balls suggesting a game in progress. The balls are of various colors, including red, white, and blue, and are scattered across the table, indicating a moment of pause in the game. The table's surface is green, a common color for pool tables, and is bordered by a purple line, likely indicating the edge of the table. Throughout the video, there is no change in the arrangement of the balls, the environment, or the camera's perspective, maintaining a consistent view of the game's current state."]
    # trajectory_paths = ["/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_1360_render0001-0120_56_120.npy",
    #                     "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_726_render0001-0120_56_120.npy",
    #                     "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_484_render0001-0120_0_64.npy"]
    # original_video_paths = [
    #                     "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_outputs/RandomLocations_1360_render0001-0120_56_120.mp4",
    #                     "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_outputs/RandomLocations_726_render0001-0120_56_120.mp4",
    #                     "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_outputs/RandomLocations_484_render0001-0120_0_64.mp4"
    # ]


    original_video_paths, trajectory_paths, prompts = read_json(args.json_file_path)


    #prompt = "The video showcases a sequence of frames that depict a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the arrangement of these balls changing across the frames. Initially, the table is set up with a green ball in the lower left quadrant, a white ball in the lower right quadrant, and a red ball in the upper left quadrant. As the video progresses, the balls are rearranged: the green ball moves to the upper left quadrant, the white ball shifts to the upper right quadrant, and the red ball moves to the lower left quadrant. Throughout the sequence, the background remains a solid dark color, and the camera's perspective does not change, maintaining a consistent view of the table and its evolving arrangement of balls."
    #prompt = "The video showcases a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the red ball positioned in the upper left corner and the white ball in the lower right corner. The table's surface is green, and the balls are white with black spots. Throughout the video, there is no movement or change in the arrangement of the balls, the environment, or the camera's perspective. The scene remains static, with the pool balls maintaining their positions on the table."
    #prompt = "The video showcases a digital representation of a pool table, viewed from a top-down perspective. The table is divided into quadrants by the pool balls, with the arrangement of these balls suggesting a game in progress. The balls are of various colors, including red, white, and blue, and are scattered across the table, indicating a moment of pause in the game. The table's surface is green, a common color for pool tables, and is bordered by a purple line, likely indicating the edge of the table. Throughout the video, there is no change in the arrangement of the balls, the environment, or the camera's perspective, maintaining a consistent view of the game's current state."
    #trajectory_path = "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_1360_render0001-0120_56_120.npy"
    #trajectory_path = "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_726_render0001-0120_56_120.npy"
    #trajectory_path = "/home/zuhao_liu/simulated_pool_videos_aleks/video_square_annos/RandomLocations_484_render0001-0120_0_64.npy"


    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_base = "./save_"+current_time
    os.mkdir(save_base)
    sample_steps = 50
    scale = 5.5
    seed = 99
    randomize_seed = False
    force_images = args.force_images
    for i in range(len(prompts)):
        prompt = prompts[i]
        trajectory_path = trajectory_paths[i]
        original_video_path = original_video_paths[i]
        save_path = os.path.join(save_base, f"video_{i}_output.mp4")
        shutil.copyfile(original_video_path, os.path.join(save_base, f"video_{i}_origin.mp4"))
        tmp_save_path, prompt, display_model_info, seed = generate_img(prompt, sample_steps, scale, seed, randomize_seed, force_images, trajectory_path, save_path)


    # print("\n\n\n $$$$$$$$$$$$$$$$$$$$$$$$$ Start configuring demo ... $$$$$$$$$$$$$$$$$$$$$$$$$ \n\n\n")
    # demo = gr.Interface(
    #     fn=generate_img,
    #     inputs=[Textbox(label="",
    #                     placeholder="Please enter your prompt. \n"),
    #             gr.Slider(
    #                 label='Sample Steps',
    #                 minimum=1,
    #                 maximum=500,
    #                 value=50,
    #                 step=10
    #             ),
    #             gr.Slider(
    #                 label='Guidance Scale',
    #                 minimum=0.1,
    #                 maximum=30.0,
    #                 value=10.0,
    #                 step=0.1
    #             ),
    #             gr.Slider(
    #                 label="Seed",
    #                 minimum=0,
    #                 maximum=203279,
    #                 step=1,
    #                 value=0,
    #             ),
    #             gr.Checkbox(label="Randomize seed", value=True),
    #             gr.Checkbox(label="Generate image (1 frame video)", value=False),
    #             ],
    #     outputs=[Video(label="Vid", width=512, height=512),
    #              Textbox(label="input prompt"),
    #              Textbox(label="model info"),
    #              gr.Slider(label='seed')],
    #     title=title_markdown, description=DESCRIPTION, theme=gr.themes.Default(), css=block_css,
    #     examples=examples,
    # )
    # print("\n\n\n $$$$$$$$$$$$$$$$$$$$$$$$$ demo configured ... $$$$$$$$$$$$$$$$$$$$$$$$$ \n\n\n")


    # demo.launch(server_name="gcp-eu-1", server_port=7861, share=True)

