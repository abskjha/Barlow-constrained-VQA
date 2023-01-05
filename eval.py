import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os,sys
import torch
import numpy as np
import json
import h5py
import copy
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import _pickle as cPickle
import re
from tqdm import tqdm
import pickle
from dataset import Dictionary, VQAFeatureDataset
from torch.utils.data import DataLoader
import base_model as base_model
from vqa_debias_loss_functions import *
import utils
from torch.autograd import Variable
import os

def invert_dict(d):
    return {v: k for k, v in d.items()}

def expand_batch(*args):
    return (t.unsqueeze(0) for t in args)

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
question_pt = '../GGE-main/data/vqacp_v2_test_questions.json' 
feature_h5_folder = '../GGE-main/data/detection_features' # path to trainval_feature.h5
image_dir = '../../../dataset/MS-COCO/coco/images/' # path to mscoco/val2014, containing all mscoco val images
ann_file = '../GGE-main/data/vqacp_v2_test_annotations.json' # path to mscoco/annotations/instances_val2014.json

##prepare data
seed=1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
name = 'val'  # train or val
answer_path = os.path.join('../GGE-main/data', 'cp-cache', '%s_target.pkl' % name)
name = "train" if name == "train" else "test"
question_path = os.path.join('../GGE-main/data', 'vqacp_v2_%s_questions.json' % name)

with open(question_path) as f:
    questions = json.load(f)
with open(answer_path, 'rb') as f:
    answers = cPickle.load(f)
    
questions.sort(key=lambda x: x['question_id'])
answers.sort(key=lambda x: x['question_id'])
    
dictionary = Dictionary.load_from_file('../GGE-main/data/dictionary.pkl')

dset = VQAFeatureDataset('val', dictionary, dataset='cpv2',
                                #   cache_image_features=args.cache_features)
                                cache_image_features=False)
eval_loader = DataLoader(dset, 256, shuffle=False, num_workers=2,pin_memory=True)


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 
 
    for v, q, a, b, qids, hint_score, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):

        ans=[]
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        b = Variable(b, requires_grad=False).cuda()
        q_mask=Variable(q_mask).cuda()
        pred, _, _, _ = model(v, q, None, None, None, None, q_mask, loss_type = None)

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results

ckpt_path = 'logs/gradient_cpv2/model.pth'
constructor = 'build_%s' % 'baseline0_newatt'
model = getattr(base_model, constructor)(dset, 1024).cuda()
model.w_emb.init_embedding('../GGE-main/data/glove6b_init_300d.npy')
model.debias_loss_fn = GreedyGradient()
model.eval()
ckpt = torch.load(ckpt_path)
dict_weights = ckpt['weights']
model.load_state_dict(dict_weights)
epoch=ckpt['epoch']
score=ckpt['eval_score']
print('Pre-trained accuracy and epoch', epoch, score)


with open('../GGE-main/util/qid2type_cpv2.json','r') as f:
    qid2type=json.load(f)

model.train(False)
results = evaluate(model, eval_loader, qid2type)
eval_score = results["score"]
bound = results["upper_bound"]
yn = results['score_yesno']
other = results['score_other']
num = results['score_number']
print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
print('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))




