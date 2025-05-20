# *E-Commerce Inpainting SDXL* &nbsp;

This workflow uses a base-SDXL + LoRA checkpoint to generate in-painted images of a model holding a physical product. This can be a suitcase, lotion bottle, packet of chips, phone, etc.

The end goal is to make the workflow as automated as possible, with the one and only input being the product image. By analysing this image, the ideal model persona and studio setting can be figured out by a robust VLM. Currently the workflow requires the product image to be put onto a blank canvas, and middle of the handle/strap erased to some extent to allow a hand to be in-painted. It should be possible to automate this as well. 

Flux is the obvious better choice for a base model, due to it's natural capability to generate better hands/fingers. However, the focus of this project is to push the capability of a low-VRAM model like SDXL and use specific ControlNet based strategies to produce high-quality outputs.

##### Current version : Limited to laptop bags, handbags, briefcases. The training data for the current LoRA model was specific to those. Other inputs might work, but the hand grip will likely have issues for every generation(fingers fusing, deformation, etc).

## Demo

<div style="
  display: flex;
  flex-direction: column;
  row-gap: 0px;
  align-items: center;        /* center each row */
">
<div style="display:flex;gap:0px">
  <img src=https://drive.google.com/thumbnail?id=1lj1XljwqTPoZsFd7hSeCHmHQxmXrMJtt width=192 height=192 alt="Image 1_1">
  <img src=https://drive.google.com/thumbnail?id=10YWLD0xQb3usZptoIelF2rOcA4LE5G6p width=192 height=192 alt="Image 1_2">
  <img src=https://drive.google.com/thumbnail?id=1UkpbJdWN53vB956uJt9KkwlnqMyuMbev width=192 height=192 alt="Image 1_3">
</div>

<div style="display:flex;gap:0px">
  <img src=https://drive.google.com/thumbnail?id=1uosO4TfTB-QypYFVq4g2x_HhDZy7hzWK width=192 height=192 alt="Image 2_1">
  <img src=https://drive.google.com/thumbnail?id=178YHATaxLjT0v70GHfkhE3WVNTHKrnkw width=192 height=192 alt="Image 2_2">
  <img src=https://drive.google.com/thumbnail?id=14OTe-b8nMHJuOElEV07N8A0PFkHRnxbo width=192 height=192 alt="Image 2_3">
</div>

<div style="display:flex;gap:0px">
  <img src=https://drive.google.com/thumbnail?id=1hQ-Ut2kczbRvvVPY2I0gZW1Epg2d7KSz width=192 height=192 alt="Image 3_1">
  <img src=https://drive.google.com/thumbnail?id=1oK4G9F3rIoxfyJPW6bHHqY7qaBbIQX8m width=192 height=192 alt="Image 3_2">
  <img src=https://drive.google.com/thumbnail?id=1haZBGnQJ_XzSAJ9LBrUYIXTZmq2N67Oq width=192 height=192 alt="Image 3_3">
</div>

<div style="display:flex;gap:0px">
  <img src=https://drive.google.com/thumbnail?id=19oltUf6_4HywnpK9cAeNWu0Y0yJXSOxO width=192 height=192 alt="Image 4_1">
  <img src=https://drive.google.com/thumbnail?id=12qF8sAAYF12jUAYlApJCXIFnsK9Syqtx width=192 height=192 alt="Image 4_2">
  <img src=https://drive.google.com/thumbnail?id=1lwS6E5DMjcgkOhNC8e-G0f7XF98p3WdX width=192 height=192 alt="Image 4_3">
</div>
</div>

## Models

- SDXL model currently used is [Juggernaut Ragnarok](https://civitai.com/models/133005?modelVersionId=1759168). Download location: `ComfyUI/models/checkpoints/sdxl`

- LoRA models are [here](https://drive.google.com/drive/folders/1ykqqjH8YGjjLKtxuz7fJxFr2dAxU1IEF). Download location: `ComfyUI/models/loras`. 

- BiRefNet models are [here](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM). It is useful to experiment with multiple options. Often for most segmentation models(including the popular SAM models), segmenting the straps accurately from the background is difficult. There seems to be a thin layer of white pixels in the gap between double straps or any other tiny gaps. I find the model `BiRefNet-general-epoch_244.pth` to be good for most cases though. Download location for the models: `ComfyUI/models/BiRefNet/pth`


## Workflow

### Stage 0: Input & Segmentation

<b>Input:</b> An image of the item placed on a plain background with part of the handle/strap erased. The placement <b>must</b> be in a natural position(as seen in the middle column above) roughly pertaining to the text prompt(input in next stage). In later versions of this workflow, this placement, resize and erase will be automated , and the only input will be the image of the item(as seen in the leftmost column above). 

The base SDXL model and segmentation of the bag is done in this stage.

### Stage 1: Prompt Adherence and Grip Generation

<b>Input:</b> The text prompt to guide the generation of the model holding the bag/briefcase.

Simple prompts could work, but the following format would be best. There are example prompts in the workflow as well.

```
<Describe model, his/her physical features, garments>, <lighting>, <background>. <Describe the nature of holding the bag(beside hips, in front of)>. <Describe nature of shot(close, half body, full body)>. <Describe nature of gaze and expression>. Only one person and only the specified item in her hand. Highly photorealistic image, tack-sharp focus, high definition.
```

Key points in this stage:

- The preprocessor for the ControlNet can be anything which helps to outline the edges of the bag clear enough. `lineart` works but `canny` seems to be the sharpest option.

- Two LoRA injections. One trained on images of various people holding different kinds of bags/briefcases. The other is focused on close-up shots of hands gripping handles/straps. 

- The controlnet guidance for the bag outline is at 0.4 strength for the first 20% of the denoising process. The strength is lower to encourage the generation to encourage the generation of the human model otherwise with a very high strength right from the beginning, the gap in the handle/strap would be inpainted quickly and generation of the human wouldn't happen. For the latter 80%, the strength is at 0.9 to avoid any further bulge around the edges of the bag.

- The whole idea of this stage, is to generate a correct grip of the model on the bag. The bag has the shape and color profile similar to the actual item. The guidance strength is set at 6 - 8 to encourage a correct bag holding and adherence to the text prompt.

### Stage 2: Realism and Grip Refine

Stage 1 is actual a common use of ControlNet and widely used in many workflows(especially in AI fahsion industry). However, Stage 2 has an uncommon use of ControlNet via [MeshGraphormer](https://github.com/microsoft/MeshGraphormer). 

The key idea is to correct any finger deformations/fusing at the end of Stage 1, refining the output further using a corrected mesh for the hands. The correction doesn't always work, but does help in most cases especially when there are finger issues in Stage 1. A scribble edge map is used to prevent any further growth of the fingers as indicated by the depth meshgraphormer guidance(can be thought of as a providing a sharp glove to the hands, using only depth maps sometimes leads to the denoising process elongating the fingers unrealistically).

Finally, the original segmented bag is pasted onto generated image. The pasting is done at 101% of the original scale in order to hide the thin segmentation line.


## LoRA training

The images for training the two loras were generated via Imagen 3 from Google Gemini API. One trained on images of various people holding different kinds of bags/briefcases. The other is focused on close-up shots of hands gripping handles/straps. 

I used [sd-scripts](https://github.com/kohya-ss/sd-scripts) from Kohya-SS to train the LoRAs. Preferred to use a terminal based method as compared to the more common Web GUI options to reserve as much GPU as possible for the training process. 

LoRA training takes a very long time given the limited resources I have access to currently. I feel the current LoRAs are both undertrained.

The following command works on a 8 GB VRAM.

```
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 "./sdxl_train_network.py" \
--enable_bucket --min_bucket_reso=256 \
--max_bucket_reso=2048 \
--pretrained_model_name_or_path=<path to SDXL model> \
--train_data_dir=<path to folder containing training images> \
--resolution="768,768" \
--output_dir=<path to ComfyUI loras folder> \
--network_alpha="8" \
--save_model_as=safetensors \
--network_module=networks.lora \
--text_encoder_lr=5e-05 \
--unet_lr=0.000125 \
--network_dim=16 \
--output_name="ecommerce_imagen3_sdxl_juggernautRagnarok" \
--lr_scheduler_num_cycles="1" \
--cache_text_encoder_outputs \
--no_half_vae \
--learning_rate="0.000125" \
--lr_scheduler="cosine" \
--train_batch_size="1" \
--max_train_steps="1400" \
--save_every_n_epochs="5" \
--mixed_precision="bf16" \
--save_precision="bf16" \
--caption_extension=".txt" \
--optimizer_type="Adafactor" \
--max_data_loader_n_workers="10" \
--gradient_accumulation_steps="40" \
--bucket_reso_steps=64 \
--gradient_checkpointing \
--xformers \
--bucket_no_upscale \
--noise_offset=0.0 \
--network_train_unet_only \
--max_train_epochs="100" \
--network_dropout 0.2 \
--cache_latents \
--fp8_base
```


## 🗺️ Workflow
![Workflow Diagram](https://drive.google.com/thumbnail?id=14sr5jhFWQCq6kxarKZRdNobbLFyGsfY2&sz=w1600)

Sample Data can be found [here](https://drive.google.com/drive/folders/1xs8_JuJ938vEFo89ajji7haVCqaBOQ_c).

## Runtime
This workflow takes about 90 seconds on a Laptop RTX 4070(8 GB VRAM) with 32GB RAM.

## Limitations and Further Improvements

This workflow is mainly to explore a concept and not meant to be productizable. The generation is not always successful and will generate unreal hand grips a lot of times. However, depending on the item and it's placement on canvas a successful generation usually happens between 30% to 80% of the time.


Improvements

- Use Flux instead of SDXL. Due to difference in architecture Flux is naturally better at generating realistic hands/fingers/grips. More recent Stable Diffusion models might be a good choice too.

- Automate the process of resizing, placing the bag on the canvas and erasing a section of the handle.

- The LoRA models must be trained for much longer with more data. The current models were trained on a 8GB VRAM with size of 768x768 for only 70 epochs.

- A segmentation model for this specific use-case of segmenting bags and briefcases are used. For a lot of cases, small areas of the background seep into the segmented item. This can be avoided after a lot of further fine-tuning an already good checkpoint.

