import os
from typing import List， Tuple, Dict 
image_prefix="/public_bme/data/iu_xray/images/"
# specify example image paths: 
# image_paths =  [
#     'synpic50962.jpg',
#     'synpic52767.jpg',
#     'synpic30324.jpg',
#     'synpic21044.jpg',
#     'synpic54802.jpg',
#     'synpic57813.jpg',
#     'synpic47964.jpg'
# ]
image_paths=[
    "CXR2384_IM-0942/0.png",
    "CXR2384_IM-0942/1.png",
    "CXR2926_IM-1328/0.png",
    "CXR2926_IM-1328/1.png",
    "CXR1451_IM-0291/0.png",
    "CXR1451_IM-0291/1.png",
    "CXR2887_IM-1289/0.png",
    "CXR2887_IM-1289/1.png",
]
report_paths=[
    "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.",
    "Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact.",
    "Left lower lobe calcified granuloma. Heart size normal. No pleural effusion or pneumothorax. Mild medial right atelectasis. Mild emphysema.",
]
image_paths = [os.path.join(image_prefix, p) for p in image_paths]
few_shot_prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question."
few_shot_prompt+=f"<image><image>Question: desribe this image. Answer: {report_paths[0]}<|endofchunk|>"
few_shot_prompt+=f"<image><image>Question: provide a diagnositc report. Answer: {report_paths[1]}<|endofchunk|>"
few_shot_prompt+=f"<image><image>Question: diagnose this medical image. Answer: {report_paths[2]}<|endofchunk|>"
few_shot_prompt+=f"<image><image>Question: Write the finding of this Chest X-ray:"
# few_shot_prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question.\
#  <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|>\
#  <image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|>\
#  <image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|>\
#  <image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|>\
#  <image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|>\
#  <image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|>\
#  <image>Question: Write the finding of this Chest X-ray:"

def ret_prompt(image_prefix:str, image_path:List[List[str]], reports_content:List[str])->str:
    # image_paths = [[os.path.join(image_prefix, elem) for elem in p] for p in image_paths]
    prompt_prefix = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question."
    prompt=""
    for i,report in enumerate(reports_content):
        subject=image_path[i]
    # for subject in image_paths:
        prompt+=len(subject)*"<image>"+f"Question: describe this image. Answer: {report}<|endofchunk|>"


def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    response=response.replace('<unk> ', '').strip()
    response=response.split("Answer: ")[-1]
    return response
    # return response.replace('<unk> ', '').strip()

