import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import pickle
import PIL.Image as Image
import json
import random
import sys
import clip
import PIL
import random
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import sys

# load CLIP model
# device = torch.device('cuda:0')

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:',device)
clip_model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
tokenizer = clip.tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()

        
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 512):
        super(DeCap, self).__init__()
        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,tokens):
        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out

def Decoding(model,clip_features):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    entry_length = 30
    temperature = 1
    tokens = None
    for i in range(entry_length):
        # print(location_token.shape)
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==49407:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _Tokenizer.decode(output_list)
    except:
        output = 'None'
    return output

# generate captions for a single image
def generate_caption(path_pic):
    image = Image.open(path_pic)
    display(image)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1,keepdim=True)
        sim = image_features@text_features.T.float()
        sim = (sim*100).softmax(dim=-1)
        prefix_embedding = sim@text_features.float()
        prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
        generated_text = Decoding(model,prefix_embedding)
        generated_text = generated_text.replace('<|startoftext|>','').replace('<|endoftext|>','')
        print(generated_text)

    return generated_text    


# generate captions for a video based on the mean pooling of the sampled frames in the video
def generate_caption_video(frame_path):
    
    # list all the frames *.jpg in the folder frame_path/
    frames_list = os.listdir(frame_path)
    frames_list.sort()
    print()
    print('frame_path:', frame_path)
    print('Total frames number:', len(frames_list))
    video_id = frame_path.split('/')[-1]
    print('video_id:', video_id)


    # uniformly sample 100 frames from the video
    sample_rate = max(int(len(frames_list)//100), 1) # for case with less than 100 frames

    # frame_sampled = frames_list[1::sample_rate] # skip the first second/frame as many are black frame initially
    frame_sampled = frames_list[::sample_rate]
    
    frame_list = [os.path.join(frame_path, frame) for frame in frame_sampled[:100]]
    print('Sampled frames number:', len(frame_list))

    image_features_list = []
    for frame in frame_list:
        image = Image.open(frame)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image).float()
            image_features /= image_features.norm(dim=-1,keepdim=True)
            image_features_list.append(image_features)
    
    # concate the image features to form a video feature, which is a 2D tensor 
    video_features = torch.cat(image_features_list, dim=0)
    print('video_features.shape:', video_features.shape)

    # save the video_features to a pickle file
    with open(f'ActivityNet/vid2seq/{video_id}.pkl', 'wb') as f:
        pickle.dump(video_features, f)
        print('video_features saved to', f'ActivityNet/vid2seq/{video_id}.pkl')
    stop 

    # # perform mean pooling on the image_features_list to get video features
    # video_features = torch.mean(torch.cat(image_features_list,dim=0),dim=0,keepdim=True)

    # video_features /= video_features.norm(dim=-1,keepdim=True)
    # sim = video_features@text_features.T.float()
    # sim = (sim*100).softmax(dim=-1)
    # prefix_embedding = sim@text_features.float()
    # prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
    # generated_text = Decoding(model, prefix_embedding)
    # generated_text = generated_text.replace('<|startoftext|>','').replace('<|endoftext|>','')
    # print("DeCap:")
    # print(generated_text)

    return generated_text    


# generate caption for each sampled frame in the video
def generate_caption_frames(frame_path, sample_rate=10):

    # list all the frames *.jpg in the folder frame_path/
    frame_list = os.listdir(frame_path)
    frame_list.sort()

    # sample a frame every 10 frames
    # frame_list = frame_list[1::sample_rate] # skip the first frame as many are black frame initially
    frame_list = frame_list[::sample_rate] 

    frame_list = [os.path.join(frame_path, frame) for frame in frame_list]
    print('Sampled frames number:', len(frame_list))

    # generate caption for each frame
    captions = []
    for frame in frame_list:
        caption = generate_caption(frame)
        captions.append(caption)

    return captions


if __name__ == '__main__':

    model = DeCap()
    weights_path = './coco_model/coco_prefix-009.pt'
    model.load_state_dict(torch.load(weights_path,map_location= torch.device('cpu')))
    model = model.to(device)
    model = model.eval()

    ## construct the support memory
    # global text_features

    # check if the support memory exists
    if not os.path.exists('./coco_model/coco_text_features.pt'):

        print("Constructing the support memory...")
        with open('./coco_train.json', 'r') as f:
            data = json.load(f)
        
        # random sample 500000 captions from the training set    
        data = random.sample(data,500000)

        text_features = []
        # captions = []
        batch_size = 1000
        clip_model.eval()
        for i in tqdm(range(0,len(data[:])//batch_size)):
            
            texts = data[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():
                texts_token = tokenizer(texts).to(device)
                text_feature = clip_model.encode_text(texts_token)
                text_features.append(text_feature)
                # captions.extend(texts)

        text_features = torch.cat(text_features,dim=0)
        text_features /= text_features.norm(dim=-1,keepdim=True).float()

        # save the support memory for later use
        torch.save(text_features, './coco_model/coco_text_features.pt')
        print("Support memory saved to ./coco_model/coco_text_features.pt")

    else:
        print("Loading the support memory...")
        text_features = torch.load('./coco_model/coco_text_features.pt', device)
        print("Support memory loaded from ./coco_model/coco_text_features.pt")



    ## load the video info file describing the video path and the annotated caption    
    video_file = sys.argv[1]
    print('video_file:', video_file)
    
    with open(video_file, 'r') as f:
        data = json.load(f)

    ## generate the DeCap caption for each video
    for vid, info in data.items():

        frame_path = info['slot2_framepath']
        if frame_path == '':            
            continue
        else:
            caption = generate_caption_video(frame_path)
            info['DeCap_caption'] = caption

    ## save the video info file with the DeCap caption
    out_file = video_file.replace('zerocap','decap')
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)
        print('\nDeCap caption saved to', out_file)