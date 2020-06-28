import sys
from data import *
from utils import *
from model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='Fine-Tuning on Reddit classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='/datadrive/dataset',
                    help='The address of preprocessed graph.')
parser.add_argument('--use_pretrain', help='Whether to use pre-trained model', action='store_true')
parser.add_argument('--pretrain_model_dir', type=str, default='/datadrive/models/gpt_all_cs',
                    help='The address for pretrained model.')
parser.add_argument('--model_dir', type=str, default='/datadrive/models',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='reddit',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=2,
                    help='Avaiable GPU ID')     
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=int, default=0.1,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=int, default=0.25,
                    help='Gradient Norm Clipping') 


args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

graph = dill.load(open(os.path.join(args.data_dir, 'reddit.pk' % args.domain), 'rb'))

idx = np.arange(len(graph.node_feature[target_type]))
np.random.seed(43)
np.random.shuffle(idx)
train_target_nodes = idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)]
valid_target_nodes = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
test_target_nodes  = idx[int(len(idx) * 0.9) : ]

types = graph.get_types()
criterion = nn.NLLLoss()

def node_classification_sample(seed, nodes, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    samp_nodes = np.random.choice(nodes, args.batch_size, replace = False)
    feature, times, edge_list, _, texts = sample_subgraph(graph_reddit, time_range, \
                inp = {'def': np.concatenate([samp_nodes, np.ones(args.batch_size)]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, feature_extractor = feature_reddit)
    
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph_reddit)

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph_reddit.y[samp_nodes]
    
    
def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_nodes, {1: True}))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), valid_nodes, {1: True}))
    jobs.append(p)
    return jobs


'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph_reddit.node_feature['def']['emb'].values[0]), \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(types), num_relations = len(graph.get_meta_graph()) + 1)
if args.use_pretrain:
    gnn.load_state_dict(torch.load(args.pretrain_model_dir).gnn.state_dict())
classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)


optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4)

stats = []
res = []
best_val   = 0
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)


for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    model.train()
    train_losses = []
    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.to(device))

        optimizer.zero_grad() 
        torch.cuda.empty_cache()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        del res, loss
    '''
        Valid
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res  = classifier.forward(node_rep[x_ids])
        loss = criterion(res, ylabel.to(device))
        
        '''
            Calculate Valid F1. Update the best model based on highest F1 score.
        '''
        valid_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
        
        if valid_f1 > best_val:
            best_val = valid_f1
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid F1: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist(), valid_f1))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    del train_data, valid_data
    
    

best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name)).to(device)
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_nodes, {1: True})
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        test_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
        test_res += [test_f1]
    print('Best Test F1: %.4f' % np.average(test_res))