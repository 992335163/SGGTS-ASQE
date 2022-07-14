#coding utf-8
import json, os, math
import random
import argparse

import numpy
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np

from data import load_data_instances, DataIterator
from model import SGTA       # Sentence-guided Grid Tagging Approach for ASQE
import utils

def reset_params(args, model):
    for child in model.children():
        for p in child.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    args.initializer(p)
                else:    
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def train(args):
    # load word embedding
    word2index = json.load(open(args.prefix + 'utils/word_idx.json'))
    if args.dataset == "Laptop-ACOS":
        category2index = json.load(open(args.prefix + 'utils/cateLap_index.json'))
    elif args.dataset == "Restaurant-ACOS":
        category2index = json.load(open(args.prefix + 'utils/cateRes_index.json'))
    general_embedding = numpy.load(args.prefix +'utils/emb.vec.npy')
    general_embedding = torch.from_numpy(general_embedding)

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))

    instances_train = load_data_instances(train_sentence_packs, word2index, category2index, args)
    instances_dev = load_data_instances(dev_sentence_packs, word2index, category2index, args)

    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # build model
    model = SGTA(general_embedding, args).to(args.device)

    parameters = list(model.parameters())
    parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    reset_params(args, model)

    # training
    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, sentence_tokens, lengths, sentence_masks, masks, tags, tags_category = trainset.get_batch(j)
            predictions_logits, predictions_category = model(sentence_tokens, lengths, sentence_masks, masks)

            loss = 0.
            tags_flatten = tags[:, :lengths[0]+1, :lengths[0]+1].reshape([-1])
            predictions_logits_flatten = predictions_logits.reshape([-1, predictions_logits.shape[3]])

            tags_category_flatten = tags_category[:, :lengths[0]+1, :lengths[0]+1].reshape([-1])
            predictions_category_flatten = predictions_category.reshape([-1, predictions_category.shape[3]])

            loss = args.proportion * F.cross_entropy(predictions_logits_flatten, tags_flatten, ignore_index=-1) + (1 - args.proportion) * F.cross_entropy(predictions_category_flatten, tags_category_flatten, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))

def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        predictions=[]
        predictions_c=[]
        labels=[]
        labels_c=[]
        all_ids = []
        all_lengths = []
        for i in range(dataset.batch_count):
            sentence_ids, sentence_tokens, lengths, sentence_masks, masks, tags, tags_category = dataset.get_batch(i)
            predictions_logits, predictions_category = model.forward(sentence_tokens, lengths, sentence_masks, masks)

            prediction = torch.argmax(predictions_logits, dim=3)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
            predictions.append(prediction_padded)

            prediction = torch.argmax(predictions_category, dim=3)
            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
            predictions_c.append(prediction_padded)

            all_ids.extend(sentence_ids)
            labels.append(tags)
            labels_c.append(tags_category)
            all_lengths.append(lengths)

        predictions = torch.cat(predictions,dim=0).cpu().tolist()
        predictions_c = torch.cat(predictions_c,dim=0).cpu().tolist()
        labels = torch.cat(labels,dim=0).cpu().tolist()
        labels_c = torch.cat(labels_c,dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        results, ee_results, ei_results, ie_results, ii_results = utils.score_uniontags(args, predictions, predictions_c, labels, labels_c, all_lengths, ignore_index=-1)

        print('\n' + args.dataset + '-' + args.task + ": " + "Results:")
        print('Main\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(results[0], results[1], results[2]))
        print('EAEO\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(ee_results[0], ee_results[1], ee_results[2]))
        print('EAIO\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(ei_results[0], ei_results[1], ei_results[2]))
        print('IAEO\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(ie_results[0], ie_results[1], ie_results[2]))
        print('IAIO\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(ii_results[0], ii_results[1], ii_results[2]))

    model.train()
    return results[0], results[1], results[2]

def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + args.model + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    word2index = json.load(open(args.prefix + 'utils/word_idx.json'))
    if args.dataset == "Laptop-ACOS":
        category2index = json.load(open(args.prefix + 'utils/cateLap_index.json'))
    elif args.dataset == "Restaurant-ACOS":
        category2index = json.load(open(args.prefix + 'utils/cateRes_index.json'))
    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, word2index, category2index, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="quadruplet")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--model', type=str, default="SGTA", choices=["SGTA"],
                        help='option: SGTA')
    parser.add_argument('--dataset', type=str, default="Restaurant-ACOS", choices=["Laptop-ACOS, Restaurant-ACOS"],
                        help='Laptop-ACOS, Restaurant-ACOS')
    parser.add_argument('--max_sequence_len', type=int, default=128,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cpu",
                        help='gpu or cpu')

    parser.add_argument('--lstm_dim', type=int, default=50,
                        help='dimension of lstm cell')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='dimension of lstm cell')
    parser.add_argument('--cnn_dim', type=int, default=256,
                        help='dimension of cnn')

    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epoch number')
    parser.add_argument('--class_sentiment', type=int, default=6,
                        help='label number')
    parser.add_argument('--class_category', type=int, default=0,
                        help='label number')

    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--proportion', type=float, default=0.5)


    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    args.initializer = initializers[args.initializer]

    if args.dataset == "Laptop-ACOS":
        args.class_category = 121
    elif args.dataset == "Restaurant-ACOS":
        args.class_category = 13

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
