import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from attention import SelfAttention

class SGTA(torch.nn.Module):
    def __init__(self, gen_emb, args):
        super(SGTA, self).__init__()
        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.dropout = torch.nn.Dropout(0.5)

        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1], args.hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1], args.hidden_dim, 2, padding=1)
        self.conv3 = torch.nn.Conv1d(gen_emb.shape[1], args.hidden_dim, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(gen_emb.shape[1], args.hidden_dim, 4, padding=2)
        self.conv5 = torch.nn.Conv1d(4*args.hidden_dim, 3*args.hidden_dim, 5, padding=2)
        self.conv6 = torch.nn.Conv1d(3*args.hidden_dim, 2*args.hidden_dim, 3, padding=1)

        self.bilstm = torch.nn.LSTM(2*args.hidden_dim, args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.sentence_capture = torch.nn.Parameter(torch.Tensor(2*args.hidden_dim, 2*args.hidden_dim))
        self.attention_st_layer = SelfAttention(args)
        self.attention_layer = SelfAttention(args)

        self.cls_linear = torch.nn.Linear(4*args.hidden_dim, args.class_sentiment)
        self.feature_linear = torch.nn.Linear(args.class_sentiment, args.hidden_dim)
        self.category_linear = torch.nn.Linear(args.hidden_dim, args.class_category)

    def _get_embedding(self, sentence_tokens, mask):
        embedding = self.gen_embedding(sentence_tokens)
        embedding = self.dropout(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _local_feature(self, embedding):
        word_emd = embedding.transpose(1, 2)
        word1_emd = self.conv1(word_emd)[:, :, :self.args.max_sequence_len]
        word2_emd = self.conv2(word_emd)[:, :, :self.args.max_sequence_len]
        word3_emd = self.conv3(word_emd)[:, :, :self.args.max_sequence_len]
        word4_emd = self.conv4(word_emd)[:, :, :self.args.max_sequence_len]
        x_emb = torch.cat((word1_emd, word2_emd), dim=1)
        x_emb = torch.cat((x_emb, word3_emd), dim=1)
        x_emb = torch.cat((x_emb, word4_emd), dim=1)
        x_conv = self.dropout(torch.nn.functional.relu(x_emb))

        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        # x_conv = x_conv[:, :lengths[0], :]
        # print(x_conv.size())
        return x_conv

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def forward(self, sentence_tokens, lengths, sentence_masks, masks):
        embedding = self._get_embedding(sentence_tokens, sentence_masks)
        local_feature = self._local_feature(embedding)
        contextual_feature = self._lstm_feature(local_feature, lengths)

        sentence_feature = self.attention_st_layer(contextual_feature, contextual_feature, sentence_masks[:, :lengths[0]])
        sentence_feature = torch.max(sentence_feature, dim=1)[0] @ self.sentence_capture
        # print(embedding.size())
        # print(local_feature.size())
        # print(contextual_feature.size())
        # print(sentence_feature.size())

        feature = torch.cat((sentence_feature.unsqueeze(1), contextual_feature), dim=1)
        feature_attention = self.attention_layer(feature, feature, masks[:, :lengths[0]+1])
        feature = feature + feature_attention
        feature = feature.unsqueeze(2).expand([-1, -1, lengths[0]+1, -1])
        feature_T = feature.transpose(1, 2)
        features = torch.cat([feature, feature_T], dim=3)
        # print(features.size())
        logits = self.cls_linear(features)
        # print(logits.size())

        category_feature = self.feature_linear(logits)
        category = self.category_linear(category_feature)
        # print(category.size())

        return logits, category

