import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy
import time
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def draw_hist(train_hist, valid_hist, constraint_hist = [0], lambd_hist = [0]):
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 5))

    ax[0].plot(train_hist['loss'], label = 'train')
    ax[0].plot(valid_hist['loss'], label = 'test')
    ax[0].legend()
    ax[0].set_title("Total Loss")

    ax[1].plot(train_hist['loss_ce'], label = 'train')
    ax[1].plot(valid_hist['loss_ce'], label = 'test')
    ax[1].legend()
    ax[1].set_title("Classification Loss")

    ax[2].plot(train_hist['loss_b'], label = 'train')
    ax[2].plot(valid_hist['loss_b'], label = 'test')
    ax[2].legend()
    ax[2].set_title("Barlow Twin loss")

    # ax[3].plot(train_hist['constraint'], label = 'train')
    # ax[3].plot(valid_hist['constraint'], label = 'test')
    # ax[3].legend()
    # ax[3].set_title("Constraint: BT loss -kappa")

    ax[3].plot(constraint_hist)
    ax[3].set_title("Constraint ma")
     
            
    ax[4].plot(lambd_hist)
    ax[4].set_title("Lambda")
    # plt.show()
    plt.savefig('logs/loss_plot.png')




def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader,train_dset,eval_dset,args,qid2type):
    dataset=args.dataset
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger_score = utils.Logger(os.path.join(output, 'log_score.txt'))

    total_step = 0
    best_eval_score = 0
    total_train_score=[]
    total_val_score=[]
    total_epoch=[]
#     Reference
    # https://github.com/denproc/Taming-VAEs/blob/master/train.py
    lambd_init = torch.FloatTensor([0.0001])

    lbd_step = 100

    alpha = 0.99 

    kappa = 0.93# its saturated barlaw value 

    pretrain = 1


    train_hist = {'loss':[], 'loss_ce':[], 'loss_b':[]}
    valid_hist = {'loss':[], 'loss_ce':[], 'loss_b':[]}
    lambd_hist = []
    constraint_hist=[]

    lambd = lambd_init.cuda()
    iter_num = 0


    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_b = 0
        train_score = 0
        total_loss_ce = 0


        # this is for loss plot
        model.train(True)
        train_hist['loss'].append(0)
        train_hist['loss_ce'].append(0)
        train_hist['loss_b'].append(0)
        # train_hist['constraint'].append(0)


        t = time.time()
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1
            ans=[]
            ans_tokens=[]
            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda()
            notype_mask=Variable(notype_mask).float().cuda()
            ans_index=torch.argmax(a, dim=1, keepdim=True).data.cpu()
            # print('ans_index',ans_index)
            for index in ans_index:
                ans.append(train_dset.label2ans[index])
            # print('ans',ans)
            for w in ans:
                # tokens.append(self.add_word(w))
                if w not in train_dset.dictionary.word2idx:
                    ans_tokens.append(18455)
                else:
                    ans_tokens.append(train_dset.dictionary.word2idx[w])
            # print('ans_tokens',ans_tokens)
            ans_tokens=torch.from_numpy(np.array(ans_tokens))
            ans_tokens=Variable(ans_tokens).cuda()
            # print('ans_tokens.shape',ans_tokens.shape)

            #########################################
            assert mode in ['base', 'gge_iter', 'gge_tog', 'gge_d_bias', 'gge_q_bias'], " %s not in modes. Please \'import train_ab as train\' in main.py" % mode
            if mode == 'gge_iter':
                pred, loss_ce, loss_b, _ = model(v, q, a, ans_tokens, b, None, q_mask, loss_type = 'q')
                if (loss_ce != loss_ce).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                pred, loss_ce,loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = 'joint')
                if (loss_ce != loss_ce).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                total_loss_b +=loss_b.item()* q.size(0)                
                total_loss_ce +=loss_ce.item()* q.size(0)                

                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

            elif mode =='gge_tog':
                pred, loss_ce,loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = 'tog')
                if (loss_ce != loss_ce).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                total_loss_b +=loss_b.item()* q.size(0)
                total_loss_ce +=loss_ce.item()* q.size(0)               

                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

            elif mode =='gge_d_bias':
                pred, loss_ce,loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = 'd_bias')
                if (loss_ce != loss_ce).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                total_loss_b +=loss_b.item()* q.size(0)
                total_loss_ce +=loss_ce.item()* q.size(0)                

                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

            elif mode =='gge_q_bias':
                pred, loss_ce,loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = 'q_bias')
                if (loss != loss).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step() 
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                total_loss_b +=loss_b.item()* q.size(0)
                total_loss_ce +=loss_ce.item()* q.size(0)                

                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score
            
            elif mode == 'base':
                pred, loss_ce,loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = None)
                if (loss_ce != loss_ce).any():
                    raise ValueError("NaN loss")
                if epoch>=args.barlow_epoch:
                    constraint= loss_b - kappa
                    loss=loss_ce + lambd * constraint
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                total_loss_b +=loss_b.item()* q.size(0)
                total_loss_ce +=loss_ce.item()* q.size(0)                
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score
            # exit()

            with torch.no_grad():
                if epoch == 0 and iter_num == 0:
                    constraint_ma = constraint
                else:
                    constraint_ma = alpha * constraint_ma.detach_() + (1 - alpha) * constraint
                if iter_num % lbd_step == 0 :
                    lambd *= torch.clamp(torch.exp(constraint_ma), 0.9, 1.05)
                                        
            train_hist['loss'][-1] += loss.item()/math.ceil(len(train_loader.dataset)/q.size(0))#N=856, no of training iteration
            train_hist['loss_ce'][-1] += loss_ce.item()/math.ceil(len(train_loader.dataset)/q.size(0))
            train_hist['loss_b'][-1] += loss_b.item()/math.ceil(len(train_loader.dataset)/q.size(0))
            # train_hist['constraint'][-1] += constraint.item()/math.ceil(len(train_loader.dataset)/q.size(0))

            iter_num += 1
        lambd_hist.append(lambd.item())
        constraint_hist.append(constraint_ma.item()) 



        total_loss /= len(train_loader.dataset)
        total_loss_b /= len(train_loader.dataset)
        total_loss_ce /= len(train_loader.dataset)

        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, train_loss_ce: %.2f,train_loss_b: %.2f, score: %.2f' % (total_loss,  total_loss_ce, total_loss_b, train_score))

        if run_eval:
            # if epoch % 2 == 0:
            model.train(False)

            results = evaluate(model, eval_loader, eval_dset, qid2type,valid_hist, lambd,kappa)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))


            if eval_score > best_eval_score:
                log_data = {
                'epoch': epoch,
                'weights': model.state_dict(),
                'eval_score': eval_score,
                }
                model_path = os.path.join(output, 'model.pth')
                torch.save(log_data, model_path)
                best_eval_score = eval_score
                
                
        total_train_score.append(train_score.item())
        total_val_score.append(100*eval_score.item())
        total_epoch.append(epoch)
        model_path = os.path.join(output, 'model_final.pth')
        torch.save(log_data, model_path)
        plt.figure()
        plt.plot(total_train_score,label='train')
        plt.plot(total_val_score,label='val')
        plt.savefig('logs/acc_plot.png')
        print('\n best_eval_score',best_eval_score)
        print('\n epoch', total_epoch,'\n total_train_score',total_train_score,'\n total_val_score',total_val_score)
        #to display the loss functions
        draw_hist(train_hist, valid_hist, constraint_hist, lambd_hist)
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]/856))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]/430))
        print("  lambd hist (in-iteration): \t{:.6f}".format(lambd_hist[-1]))
 

def evaluate(model, dataloader, eval_dset, qid2type,valid_hist,lambd,kappa):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 
    valid_hist['loss'].append(0)
    valid_hist['loss_ce'].append(0)
    valid_hist['loss_b'].append(0)
    # valid_hist['constraint'].append(0)


    for v, q, a, b, qids, _, q_mask in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        ans=[]
        ans_tokens=[]
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        b = Variable(b, requires_grad=False).cuda()
        q_mask=Variable(q_mask).cuda()
        ans_index=torch.argmax(a, dim=1, keepdim=True).data.cpu()
        # print('ans_index',ans_index)
        for index in ans_index:
            ans.append(eval_dset.label2ans[index])
        # print('ans',ans)
        for w in ans:
            # tokens.append(self.add_word(w))
            if w not in eval_dset.dictionary.word2idx:
                ans_tokens.append(18455)
            else:
                ans_tokens.append(eval_dset.dictionary.word2idx[w])
        # print('ans_tokens',ans_tokens)
        ans_tokens=torch.from_numpy(np.array(ans_tokens))
        ans_tokens=Variable(ans_tokens, requires_grad=False).cuda()
        pred, loss_ce, loss_b, _ = model(v, q, a,ans_tokens, b, None, q_mask, loss_type = None)
        # loss=loss_ce + lambd * loss_b
        constraint = loss_b - kappa
        loss=loss_ce + lambd * constraint


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

        valid_hist['loss'][-1] += loss.item()/math.ceil(len(dataloader.dataset)/q.size(0))# n=430
        valid_hist['loss_ce'][-1] += loss_ce.item()/math.ceil(len(dataloader.dataset)/q.size(0))
        valid_hist['loss_b'][-1] += loss_b.item()/math.ceil(len(dataloader.dataset)/q.size(0))
        # valid_hist['constraint'][-1] += constraint.item()/math.ceil(len(dataloader.dataset)/q.size(0))



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
