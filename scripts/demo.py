from huggingface_hub import hf_hub_download
import torch
import os
from transformers import GenerationConfig
from open_flamingo.factory import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys
sys.path.append('..')
from src.utils import FlamingoProcessor
from demo_utils import image_paths, clean_generation,few_shot_prompt


def main():
    accelerator = Accelerator() #when using cpu: cpu=True

    device = accelerator.device
    
    print('Loading model..')

    # >>> add your local path to Llama-7B (v1) model here:
    llama_path = '/public_bme/data/llm/llama-7b'
    vision_path= '/public_bme/data/pretrain/clip/ViT-L-14.pt'
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        use_local_files=True,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    # checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    checkpoint_path="/public_bme/data/pretrain/med-flamingo/model.pt"
    print(f'Load Med-Flamingo checkpoint from {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    """
    Step 1: Load images
    """
    demo_images = [Image.open(path) for path in image_paths]

    """
    Step 2: Define multimodal few-shot prompt 
    """

    # example few-shot prompt:
    # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Write the finding of this Chest X-ray:"
    prompt=few_shot_prompt
    """
    Step 3: Preprocess data 
    """
    print('Preprocess data')
    pixels = processor.preprocess_images(demo_images)
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
    tokenized_data = processor.encode_text(prompt)

    """
    Step 4: Generate response 
    """

    # actually run few-shot prompt through model:
    print('Generate from multimodal few-shot prompt')
    generation_config = GenerationConfig(
            repetition_penalty=10., 
            max_new_tokens=20, 

        )
    generated_text = model.generate(
    vision_x=pixels.to(device),
    lang_x=tokenized_data["input_ids"].to(device),
    attention_mask=tokenized_data["attention_mask"].to(device),
    generation_config=generation_config
    )
    response = processor.tokenizer.decode(generated_text[0])
    response = clean_generation(response)

    print(f'{response=}')

if __name__ == "__main__":

    main()
