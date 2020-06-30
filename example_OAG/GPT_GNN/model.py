from .conv import *
import numpy as np
from gensim.parsing.preprocessing import *


class GPT_GNN(nn.Module):
    def __init__(self, gnn, rem_edge_list, attr_decoder, types, neg_samp_num, device, neg_queue_size = 0):
        super(GPT_GNN, self).__init__()
        self.types = types
        self.gnn = gnn
        self.params = nn.ModuleList()
        self.neg_queue_size = neg_queue_size
        self.link_dec_dict = {}
        self.neg_queue = {}
        for source_type in rem_edge_list:
            self.link_dec_dict[source_type] = {}
            self.neg_queue[source_type] = {}
            for relation_type in rem_edge_list[source_type]:
                print(source_type, relation_type)
                matcher = Matcher(gnn.n_hid, gnn.n_hid)
                self.neg_queue[source_type][relation_type] = torch.FloatTensor([]).to(device)
                self.link_dec_dict[source_type][relation_type] = matcher
                self.params.append(matcher)
        self.attr_decoder = attr_decoder
        self.init_emb = nn.Parameter(torch.randn(gnn.in_dim))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.neg_samp_num = neg_samp_num
        
    def neg_sample(self, souce_node_list, pos_node_list):
        np.random.shuffle(souce_node_list)
        neg_nodes = []
        keys = {key : True for key in pos_node_list}
        tot       = 0
        for node_id in souce_node_list:
            if node_id not in keys:
                neg_nodes += [node_id]
                tot += 1
            if tot == self.neg_samp_num:
                break
        return neg_nodes
        
    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)
    def link_loss(self, node_emb, rem_edge_list, ori_edge_list, node_dict, target_type, use_queue = True, update_queue = False):
        losses = 0
        ress   = []
        for source_type in rem_edge_list:
            if source_type not in self.link_dec_dict:
                continue
            for relation_type in rem_edge_list[source_type]:
                if relation_type not in self.link_dec_dict[source_type]:
                    continue
                rem_edges = rem_edge_list[source_type][relation_type]
                if len(rem_edges) <= 8:
                    continue
                ori_edges = ori_edge_list[source_type][relation_type]
                matcher = self.link_dec_dict[source_type][relation_type]

                target_ids, positive_source_ids = rem_edges[:,0].reshape(-1, 1), rem_edges[:,1].reshape(-1, 1)
                n_nodes = len(target_ids)
                source_node_ids = np.unique(ori_edges[:, 1])

                negative_source_ids = [self.neg_sample(source_node_ids, \
                    ori_edges[ori_edges[:, 0] == t_id][:, 1].tolist()) for t_id in target_ids]
                sn = min([len(neg_ids) for neg_ids in negative_source_ids])

                negative_source_ids = [neg_ids[:sn] for neg_ids in negative_source_ids]

                source_ids = torch.LongTensor(np.concatenate((positive_source_ids, negative_source_ids), axis=-1) + node_dict[source_type][0])
                emb = node_emb[source_ids]
                
                if use_queue and len(self.neg_queue[source_type][relation_type]) // n_nodes > 0:
                    tmp = self.neg_queue[source_type][relation_type]
                    stx = len(tmp) // n_nodes
                    tmp = tmp[: stx * n_nodes].reshape(n_nodes, stx, -1)
                    rep_size = sn + 1 + stx
                    source_emb = torch.cat([emb, tmp], dim=1)
                    source_emb = source_emb.reshape(n_nodes * rep_size, -1)
                else:
                    rep_size = sn + 1
                    source_emb = emb.reshape(source_ids.shape[0] * rep_size, -1)
                    
                target_ids = target_ids.repeat(rep_size, 1) + node_dict[target_type][0]
                target_emb = node_emb[target_ids.reshape(-1)]
                res = matcher.forward(target_emb, source_emb)
                res = res.reshape(n_nodes, rep_size)
                ress += [res.detach()]
                losses += F.log_softmax(res, dim=-1)[:,0].mean()
                if update_queue and 'L1' not in relation_type and 'L2' not in relation_type:
                    tmp = self.neg_queue[source_type][relation_type]
                    self.neg_queue[source_type][relation_type] = \
                        torch.cat([node_emb[source_node_ids].detach(), tmp], dim=0)[:int(self.neg_queue_size * n_nodes)]
        return -losses / len(ress), ress

    
    def text_loss(self, reps, texts, w2v_model, device):
        def parse_text(texts, w2v_model, device):
            idxs = []
            pad  = w2v_model.wv.vocab['eos'].index
            for text in texts:
                idx = []
                for word in ['bos'] + preprocess_string(text) + ['eos']:
                    if word in w2v_model.wv.vocab:
                        idx += [w2v_model.wv.vocab[word].index]
                idxs += [idx]
            mxl = np.max([len(s) for s in idxs]) + 1
            inp_idxs = []
            out_idxs = []
            masks    = []
            for i, idx in enumerate(idxs):
                inp_idxs += [idx + [pad for _ in range(mxl - len(idx) - 1)]]
                out_idxs += [idx[1:] + [pad for _ in range(mxl - len(idx))]]
                masks    += [[1 for _ in range(len(idx))] + [0 for _ in range(mxl - len(idx) - 1)]]
            return torch.LongTensor(inp_idxs).transpose(0, 1).to(device), \
                   torch.LongTensor(out_idxs).transpose(0, 1).to(device), torch.BoolTensor(masks).transpose(0, 1).to(device)
        inp_idxs, out_idxs, masks = parse_text(texts, w2v_model, device)
        pred_prob = self.attr_decoder(inp_idxs, reps.repeat(inp_idxs.shape[0], 1, 1))      
        return self.ce(pred_prob[masks], out_idxs[masks]).mean()

    def feat_loss(self, reps, out):
        return -self.attr_decoder(reps, out).mean()


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

    
class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    
    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)

    
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs   

    
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, n_word, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(nhid, nhid, nlayers)
        self.encoder = nn.Embedding(n_word, nhid)
        self.decoder = nn.Linear(nhid, n_word)
        self.adp     = nn.Linear(ninp + nhid, nhid)
    def forward(self, inp, hidden = None):
        emb = self.encoder(inp)
        if hidden is not None:
            emb = torch.cat((emb, hidden), dim=-1)
            emb = F.gelu(self.adp(emb))
        output, _ = self.rnn(emb)
        decoded = self.decoder(self.drop(output))
        return decoded
    def from_w2v(self, w2v):
        initrange = 0.1
        self.encoder.weight.data = w2v
        self.decoder.weight = self.encoder.weight
        
        self.encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False
        