import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import classification_report
import sklearn

def load_data(ratio_generated, path="../dataset/citeseer/", dataset="citeseer"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}features.{}".format(path, dataset),
                                        dtype=np.float32)
    features = sp.csr_matrix(idx_features_labels[:, 0:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    idx_train = np.genfromtxt("{}train.{}".format(path, dataset),
                              dtype=np.int32).squeeze()

    idx_test = np.genfromtxt("{}test.{}".format(path, dataset),
                             dtype=np.int32).squeeze()

    majority = np.array([x for x in idx_train if labels[x] == 0])
    minority = np.array([x for x in idx_train if labels[x] == 1])

    num_minority = minority.shape[0]
    num_majority = majority.shape[0]
    print("Number of majority: ", num_majority)
    print("Number of minority: ", num_minority)

    generate_node = []
    generate_label=[]
    for i in range(labels.shape[0], labels.shape[0]+int(ratio_generated*num_majority)-num_minority):
        generate_node.append(i)
        generate_label.append(1)
    idx_train= np.hstack((idx_train, np.array(generate_node)))
    print(idx_train.shape)

    minority_test = np.array([x for x in idx_test if labels[x] == 1])
    minority_all = np.hstack((minority, minority_test))


    labels= np.hstack((labels, np.array(generate_label)))


    edges = np.genfromtxt("{}edges.{}".format(path, dataset),
                                    dtype=np.int32)

    adj_real = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj_real + adj_real.T.multiply(adj_real.T > adj_real) - adj_real.multiply(adj_real.T > adj_real)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    generate_node=torch.LongTensor(np.array(generate_node))
    minority = torch.LongTensor(minority)
    majority = torch.LongTensor(majority)
    minority_all = torch.LongTensor(minority_all)

    return adj, adj_real,features, labels, idx_train, idx_test, generate_node, minority, majority, minority_all#, generate_node_test, minority_test




def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, output_AUC):
    preds = output.max(1)[1].type_as(labels)


    recall = sklearn.metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1_score = sklearn.metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    AUC = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), output_AUC.detach().cpu().numpy())
    acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = sklearn.metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    return recall, f1_score, AUC, acc, precision


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_edges(adj_real, adj_new):
    adj = adj_real+adj_new
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj
