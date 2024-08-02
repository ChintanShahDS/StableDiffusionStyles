
from base64 import b64encode

import numpy
import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

# For video display:
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging, AutoProcessor, CLIPVisionModel
from transformers.tokenization_utils_base import BatchEncoding

import os, glob
from pathlib import Path

torch.manual_seed(1)

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# Prep Scheduler
def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32) # minor fix to ensure MPS compatibility, fixed in diffusers PR 3925

def get_output_embeds(input_embeddings):
    # CLIP's text model uses causal mask, so we prepare it here:
    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    # Getting the output embeddings involves calling the model with passing output_hidden_states=True
    # so that it doesn't just return the pooled final predictions:
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, # We aren't using an attention mask so that can be None
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True, # We want the output embs not the final output
        return_dict=None,
    )

    # We're interested in the output hidden state only
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = text_encoder.text_model.final_layer_norm(output)

    # And now they're ready!
    return output

# # Getting the Clip Embeddings for the image
# # Issues currently so not used but need to figure out and fix
# def getClipImageEmbed(image):
#     print("type of image:", type(image))
#     if(torch.is_tensor(image)):
#         imageData = {}
#         imageData['pixel_values'] = image
#         image_input = BatchEncoding(data=imageData)
#     else:
#         image_input = image_processor(images=image, return_tensors="pt").to(torch_device)
#     print("Type of image_input:", type(image_input))
#     # print("image_input:", image_input)
#     print("image_input['pixel_values'].shape:",image_input['pixel_values'].shape)

#     # Calculate features
#     with torch.no_grad():
#         image_features = image_encoder(**image_input)

#     # print("Type of image_features:", type(image_features))
#     pooler_output = image_features.pooler_output
#     # print("Type of pooler_output:", type(pooler_output))
#     # print("pooler_output shape:", pooler_output.shape)
#     # Pick the top 5 most similar labels for the image
#     # image_features /= image_features.norm(dim=-1, keepdim=True)
#     return pooler_output

# # Getting the Clip Embeddings for all the images in a folder and then averaging them to give a direction
# # Issues currently so not used but need to figure out and fix
# def getClipEmbeds(imageFolder):

#     meanImageEmbeds = []
#     for filename in glob.glob(os.path.join(imageFolder, '*.jpg')):
#         image = cv2.imread(filename)
#         # print("type of image:", type(image))
#         image = Image.fromarray(image.astype('uint8'), 'RGB')
#         image_features = getClipImageEmbed(image)
#         # print("image_features:", image_features)
#         if (meanImageEmbeds == None):
#             meanImageEmbeds.append(image_features)
#         else:
#             meanImageEmbeds.append(image_features)
#             # = torch.stack([meanImageEmbeds, image_features])
#             # print("meanImageEmbeds shape:", len(meanImageEmbeds))

#     stackedImageEmbeds = torch.stack(meanImageEmbeds)
#     print("stackedImageEmbeds shape:", stackedImageEmbeds.shape)
#     result = torch.mean(stackedImageEmbeds, dim=0)    
#     print("result shape:", result.shape)
#     # print("result:", result)
#     return result

# # The blue loss that is a direction used for other losses
# def blue_loss(images):
#     # How far are the blue channel values to 0.9:
#     error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
#     print("loss Error value:", error)
#     return error

# # Getting the loss for a set of iamges in this case Indian flag
# # This is to give a direction to have the images as per that
# # Not working due to some issues with gradient parts since I am doing some changes to the data
# # Issues currently so not used but need to figure out and fix
# def flag_loss(images, lossEmbeds):
#     # How far are the image embeds from lossembeds:

#     print("Type of images:", type(images))
#     print("Images shape:", images.shape)
#     # images = images.squeeze()
#     # print("After Images shape:", images.shape)
#     # imgs = tfms.functional.to_pil_image(images)
#     # print("Type of imgs:", type(imgs))
#     # print("imgs shape:", imgs.shape)
#     imageEmbeds = getClipImageEmbed(images)
#     print("Type of imageEmbeds:", imageEmbeds, "imageEmbeds Shape:", imageEmbeds.shape)
#     print("torch.sub(imageEmbeds, lossEmbeds):", torch.sub(imageEmbeds, lossEmbeds))
#     error = torch.abs(torch.sub(imageEmbeds, lossEmbeds)).mean()
#     print("loss Error value:", error)
#     return error


# Defined a latent loss that is purely based on one image instead of multiple images
# This is used and working
# Need to look at if I can have the latents of multiple images merged together to give a thought
# Rather than a image - So that the thought is more on the design rather than pure image

def latent_loss(latent, conditioning_image):
    # How far are the image embeds from lossembeds:
    image = Image.open(conditioning_image)
    r_image = image.resize((512,512))
    r_latent = pil_to_latent(r_image)
    error = F.mse_loss(0.5*latent,0.5*r_latent)
    return error
#Generating an image with these modified embeddings

def generate_with_embs(text_input, text_embeddings, conditioning_image, seed):
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 30            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    batch_size = 1
    loss_scale = 100 #@param
    imageCondSteps = 5

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #### ADDITIONAL GUIDANCE ###
        if conditioning_image:
            if i%imageCondSteps == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()

                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred

                # Decode to image space
                # denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

                # Calculate loss
                # loss = flag_loss(denoised_images, lossEmbeds) * blue_loss_scale
                loss = latent_loss(latents_x0, conditioning_image) * loss_scale
                # loss = blue_loss(denoised_images) * blue_loss_scale

                print(i, 'loss item:', loss.item())

                # Get gradient
                cond_grad = torch.autograd.grad(loss, latents)[0]
                # print("cond_grad:", cond_grad)

                # Modify the latents based on this gradient
                latents = latents.detach() - cond_grad * sigma**2

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents_to_pil(latents)[0]


def getImageWithStyle(prompt, style_path, conditioning_image, seed):
    prompt = prompt + ' in the style of puppy'

    style_loc = style_path + '/learned_embeds.bin'
    style_embed = torch.load(style_loc)
    # print(style_embed)
    # print(style_embed.keys(), style_embed[style_embed.keys()[0]].shape)
    new_style = list(style_embed.keys())[0]
    print("New style:", new_style)

    filename_style = new_style.replace("<", "")
    filename_style = filename_style.replace(">", "")
    filename = prompt.replace("puppy",  filename_style)
    if (conditioning_image):
        imagename = Path(conditioning_image).stem
        print("Conditioned imagename:", imagename)
        filename = filename + "_" + imagename + "_conditioned"
    print("filename:", filename)

    # Tokenize
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    # print("text_input:", text_input)
    input_ids = text_input.input_ids.to(torch_device)
    # print("Input Ids:", input_ids)
    print("Input Ids shape:", input_ids.shape)

    token_emb_layer = text_encoder.text_model.embeddings.token_embedding
    # Get token embeddings
    token_embeddings = token_emb_layer(input_ids)
    print("Token Embeddings shape:", token_embeddings.shape)

    # The new embedding - our special style word
    replacement_token_embedding = style_embed[new_style].to(torch_device)

    # Insert this into the token embeddings
    token_embeddings[0, torch.where(input_ids[0]==6829)] = replacement_token_embedding.to(torch_device)

    pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
    position_ids = text_encoder.text_model.embeddings.position_ids
    print("position_ids shape:", position_ids.shape)
    position_embeddings = pos_emb_layer(position_ids)
    print("Position Embeddings shape:", token_embeddings.shape)

    # Combine with pos embs
    input_embeddings = token_embeddings + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(input_embeddings)

    # And generate an image with this:
    image = generate_with_embs(text_input, modified_output_embeddings, conditioning_image, seed)
    name = "./Outputs/" + filename+".jpg"
    
    image.save(name)


# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)
image_encoder = image_encoder.to(torch_device)

# Used puppy as a placeholder here since the token is known
# Will replace with some other word that is better
prompt = 'A farm'
style_name = 'birb'
conditioning_image_folder = './conditioning_images/'
style_folder = './styles/'

seedlist = [*range(0, 10000, 500)]
print(seedlist)

i = 0
for style_path in glob.glob(os.path.join(style_folder, '*')):
    seed = seedlist[i]
    i = i + 1
    getImageWithStyle(prompt, style_path, None, seed)
    for conditioning_image in glob.glob(os.path.join(conditioning_image_folder, '*.jpg')):
        print("style_path:", style_path, "conditioning_image:", conditioning_image)
        getImageWithStyle(prompt, style_path, conditioning_image, seed)

# style_loc = './' + style_name + '/learned_embeds.bin'
# style_embed = torch.load(style_loc)
# # print(style_embed)
# # print(style_embed.keys(), style_embed[style_embed.keys()[0]].shape)
# new_style = list(style_embed.keys())[0]
# print("New style:", new_style)

# # filename_style = new_style.replace("<", "")
# # filename_style = filename_style.replace(">", "")
# filename = prompt.replace("puppy",  style_name)
# print("filename:", filename)

# # Tokenize
# text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
# # print("text_input:", text_input)
# input_ids = text_input.input_ids.to(torch_device)
# # print("Input Ids:", input_ids)
# print("Input Ids shape:", input_ids.shape)

# token_emb_layer = text_encoder.text_model.embeddings.token_embedding
# # Get token embeddings
# token_embeddings = token_emb_layer(input_ids)
# print("Token Embeddings shape:", token_embeddings.shape)

# # The new embedding - our special style word
# replacement_token_embedding = style_embed[new_style].to(torch_device)

# # Insert this into the token embeddings
# token_embeddings[0, torch.where(input_ids[0]==6829)] = replacement_token_embedding.to(torch_device)

# pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
# position_ids = text_encoder.text_model.embeddings.position_ids
# print("position_ids shape:", position_ids.shape)
# position_embeddings = pos_emb_layer(position_ids)
# print("Position Embeddings shape:", token_embeddings.shape)

# # Combine with pos embs
# input_embeddings = token_embeddings + position_embeddings

# #  Feed through to get final output embs
# modified_output_embeddings = get_output_embeds(input_embeddings)

# # And generate an image with this:
# image = generate_with_embs(modified_output_embeddings)
# name = filename+".jpg"
# image.save(name)
