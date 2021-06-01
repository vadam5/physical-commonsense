import torch
import clip

import torch.nn as nn

from pc.models import mlp, init_weights

class ClipClassifier(nn.Module):
    
    def __init__(self, clip_model, d_in, input_dropout, h, activation, innner_dropout, d_out, train, text_only):
        super(ClipClassifier, self).__init__()
        self.text_only = text_only
        self.clip_model = clip_model
        self.mlp = mlp(d_in, input_dropout, h, activation, innner_dropout, d_out).half()

        #if train:
        #    self.mlp.half().apply(init_weights)

    def forward(self, text, image=None):
        text_feats = self.clip_model.encode_text(text)

        if not self.text_only:
            image_feats = self.clip_model.encode_image(image)
            mlp_input = torch.cat([text_feats, image_feats], dim=1)
        else:
            mlp_input = text_feats

        pred = self.mlp(mlp_input)

        return pred

def get_clip_classifier(d_in, input_dropout, h, activation, innner_dropout, d_out, train=True, text_only=False):
    clip_model, preprocess = clip.load("ViT-B/32")
    classifier = ClipClassifier(clip_model, d_in, input_dropout, h, activation, innner_dropout, d_out, train, text_only) 
    
    return classifier, preprocess
