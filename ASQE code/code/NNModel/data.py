import math
import torch 

class2id = {'None': 0, 'A': 1, 'O': 2, 'Mixed': 3, 'Negative': 4, 'Neutral': 5, 'Positive': 6}

class Instance(object):
    def __init__(self, sentence_pack, word2index, category2index, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']

        '''generate sentence tokens'''
        self.sentence_tokens = torch.zeros(args.max_sequence_len).long()
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']
        
        self.tags = torch.zeros(1+args.max_sequence_len, 1+args.max_sequence_len).long()
        self.tags_category = torch.zeros(1+args.max_sequence_len, 1+args.max_sequence_len).long()
        self.tags[:, :] = -1
        self.tags_category[:, :] = -1

        for i in range(self.length+1):
            for j in range(i, self.length+1):
                self.tags[i][j] = 0
                self.tags_category[i][j] = 0

        self.quads = sentence_pack['quads']
        for quad in self.quads:
            a_l_span, a_r_span = quad[0]+1, quad[1]+1
            o_l_span, o_r_span = quad[4]+1, quad[5]+1
            category, sentiment = category2index[quad[2]], quad[3] + 3
 
            # EA & EO
            if a_l_span != 0 and a_r_span != 0 and o_l_span != 0 and o_r_span != 0:
                for i in range(a_l_span, a_r_span+1):
                    self.tags[i][i] = class2id['A']
                    if i > a_l_span: 
                        self.tags[i-1][i] = class2id['A']
                    for j in range(i, a_r_span+1):
                        self.tags[i][j] = class2id['A']

                for i in range(o_l_span, o_r_span+1):
                    self.tags[i][i] = class2id['O']
                    if i > o_l_span: 
                        self.tags[i-1][i] = class2id['O']
                    for j in range(i, o_r_span+1):
                        self.tags[i][j] = class2id['O']

                for i in range(a_l_span, a_r_span+1):
                    for j in range(o_l_span, o_r_span+1):
                        if i > j: 
                            self.tags[j][i] = sentiment
                            self.tags_category[j][i] = category
                        else: 
                            self.tags[i][j] = sentiment
                            self.tags_category[i][j] = category
            
            # IA & EO
            elif a_l_span == 0 and a_r_span == 0 and o_l_span != 0 and o_r_span != 0:
                if self.tags[0][0] < 4:
                    if self.tags[0][0] == class2id['None']:
                        self.tags[0][0] = class2id['A']
                    elif self.tags[0][0] == class2id['O']:
                        self.tags[0][0] = class2id['Mixed']

                for i in range(o_l_span, o_r_span+1):
                    self.tags[i][i] = class2id['O']
                    if i > o_l_span: 
                        self.tags[i-1][i] = class2id['O']
                    for j in range(i, o_r_span+1):
                        self.tags[i][j] = class2id['O']

                for i in range(a_l_span, a_r_span+1):
                    for j in range(o_l_span, o_r_span+1):
                        if i > j: 
                            self.tags[j][i] = sentiment
                            self.tags_category[j][i] = category
                        else: 
                            self.tags[i][j] = sentiment
                            self.tags_category[i][j] = category
            
            # EA & IO
            elif a_l_span != 0 and a_r_span != 0 and o_l_span == 0 and o_r_span == 0:
                for i in range(a_l_span, a_r_span+1):
                    self.tags[i][i] = class2id['A']
                    if i > a_l_span: 
                        self.tags[i-1][i] = class2id['A']
                    for j in range(i, a_r_span+1):
                        self.tags[i][j] = class2id['A']

                if self.tags[0][0] < 4:
                    if self.tags[0][0] == class2id['None']:
                        self.tags[0][0] = class2id['O']
                    elif self.tags[0][0] == class2id['A']:
                        self.tags[0][0] = class2id['Mixed']

                for i in range(a_l_span, a_r_span+1):
                    for j in range(o_l_span, o_r_span+1):
                        if i > j: 
                            self.tags[j][i] = sentiment
                            self.tags_category[j][i] = category
                        else: 
                            self.tags[i][j] = sentiment
                            self.tags_category[i][j] = category
           
            # IA & IO
            elif a_l_span == 0 and a_r_span == 0 and o_l_span == 0 and o_r_span == 0:
                self.tags[0][0] = sentiment
                self.tags_category[0][0] = category

        '''generate mask of the sentence'''
        self.sentence_mask = torch.zeros(args.max_sequence_len)
        self.sentence_mask[:self.length] = 1

        '''generate mask of the overall sentence'''
        self.mask = torch.zeros(args.max_sequence_len+1)
        self.mask[:self.length+1] = 1

def load_data_instances(sentence_packs, word2index, category2index, args):
    instances = list()
    for sentence_pack in sentence_packs:
        instances.append(Instance(sentence_pack, word2index, category2index, args))
    return instances

class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        lengths = []
        sentence_masks = []
        masks = []
        tags = []
        tags_category = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            sentence_masks.append(self.instances[i].sentence_mask)
            masks.append(self.instances[i].mask)
            tags.append(self.instances[i].tags)
            tags_category.append(self.instances[i].tags_category)

        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)

        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.args.device)[indexes]
        lengths = torch.tensor(lengths).to(self.args.device)[indexes]
        sentence_masks = torch.stack(sentence_masks).to(self.args.device)[indexes]
        masks = torch.stack(masks).to(self.args.device)[indexes]
        tags = torch.stack(tags).to(self.args.device)[indexes]
        tags_category = torch.stack(tags_category).to(self.args.device)[indexes]

        return sentence_ids, sentence_tokens, lengths, sentence_masks, masks, tags, tags_category
