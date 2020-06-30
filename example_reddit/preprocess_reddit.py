from torch_geometric.datasets import Reddit
from GPT_GNN.data import *

dataset = Reddit(root='/datadrive/dataset')
graph_reddit = Graph()
el = defaultdict(  #target_id
                    lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))
for i, j in tqdm(dataset.data.edge_index.t()):
    el[i.item()][j.item()] = 1

target_type = 'def'
graph_reddit.edge_list['def']['def']['def'] = el
n = list(el.keys())
degree = np.zeros(np.max(n)+1)
for i in n:
    degree[i] = len(el[i])
x = np.concatenate((dataset.data.x.numpy(), np.log(degree).reshape(-1, 1)), axis=-1)
graph_reddit.node_feature['def'] = pd.DataFrame({'emb': list(x)})

idx = np.arange(len(graph_reddit.node_feature[target_type]))
np.random.seed(43)
np.random.shuffle(idx)

graph_reddit.pre_target_nodes   = idx[ : int(len(idx) * 0.7)]
graph_reddit.train_target_nodes = idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)]
graph_reddit.valid_target_nodes = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
graph_reddit.test_target_nodes  = idx[int(len(idx) * 0.9) : ]

graph_reddit.y = dataset.data.y
dill.dump(graph_reddit, open('/datadrive/dataset/graph_reddit.pk', 'wb'))
