from predictor.Base_Predictor import Predictor
from predictor.module.TEST import MPNN
from predictor.module.GNNs import GCN, GIN, SAGE
import torch
import torch.nn.functional as F
import time
import nni
from copy import deepcopy
from torch_geometric.utils import subgraph, dropout_edge


class tfr_gcn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)


    def method_init(self, conf, data):

        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.decoder = MPNN(in_channels=conf.model['n_classes'], hidden_channels=conf.model['n_hidden'],
                            out_channels=conf.model['n_feat'], decoder_dropout=conf.model['decoder_dropout'],
                            att_softmax=conf.model['att_softmax']).to(self.device)
        self.encoder = torch.nn.Linear(in_features=conf.model['n_feat'], out_features=conf.model['n_hidden'], bias=True).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        self.pseudo_coef = 0.0
        self.pseudo_ratio = conf.model['pseudo_ratio']
        self.beta = conf.model['beta']
        self.beta_dropout_ratio = conf.model['beta_dropout_ratio']
        self.beta_dropout_type = conf.model['beta_dropout_type']
        self.use_recon_softmax = conf.model['use_recon_softmax']
        self.high_confidence_clean_acc_list = []
        self.high_naive_clean_acc_list = []


    def get_prediction(self, features, adj, label=None, mask=None, training=True):
        loss, acc = None, None
        output = self.model(features, adj)
        if (label is not None) and (mask is not None):
            pseudo_label = torch.softmax(output, dim=1)
            if self.beta_dropout_type == 'edge':
                dropout_adj = dropout_edge(adj.indices(), p=self.beta_dropout_ratio)[0]
            else:
                subset = torch.randperm(self.n_nodes)[:int(self.n_nodes * self.beta_dropout_ratio)].to(self.device)
                dropout_adj = subgraph(subset=subset, edge_index=adj.indices())[0]

            recon_feats, confidence = self.decoder(pseudo_label, dropout_adj)

            high_confidence_indices = torch.argsort(confidence.squeeze(), descending=True)[
                                      :int(self.n_nodes * self.pseudo_ratio)]
            high_confidence_clean_acc = self.metric(self.clean_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            high_confidence_noisy_acc = self.metric(self.noisy_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            naive_confidencce = pseudo_label.max(1)[0]
            high_naive_confidence_indices = torch.argsort(naive_confidencce.squeeze(), descending=True)[
                                            :int(self.n_nodes * self.pseudo_ratio)]
            high_naive_clean_acc = self.metric(self.clean_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            high_naive_noisy_acc = self.metric(self.noisy_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            if self.conf.training['debug'] and training:
                print("high confidence clean accuracy: {}, high confidence noisy accuracy: {}".format(
                    high_confidence_clean_acc, high_confidence_noisy_acc))
                print("high naive clean accuracy: {}, high naive noisy accuracy: {}".format(high_naive_clean_acc,
                                                                                            high_naive_noisy_acc))
                # Record accuracies
                self.high_confidence_clean_acc_list.append(high_confidence_clean_acc)
                self.high_naive_clean_acc_list.append(high_naive_clean_acc)

            if self.use_recon_softmax:
                recon_feats = F.softmax(recon_feats, dim=1)
            loss_recon = self.loss_fn(recon_feats, features)
            loss_pseudo = self.loss_fn(output[high_confidence_indices],
                                       pseudo_label[high_confidence_indices].detach())
            # recon_feats = F.softmax(recon_feats, dim=1)
            # proj_feats = self.encoder(F.dropout(features, p=0.5, training=True))
            # proj_feats = F.softmax(proj_feats, dim=1)
            # proj_feats = features

            # loss_pseudo = self.loss_fn(output[high_naive_confidence_indices], pseudo_label[high_naive_confidence_indices].detach())
            loss_sup = self.loss_fn(output[mask], label[mask])
            loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon + self.pseudo_coef * loss_pseudo
            # loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            if training:
                self.pseudo_coef = acc
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            self.epoch = epoch
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            features, adj = self.feats, self.adj
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask, training=True)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            loss_test, acc_test = self.evaluate(self.clean_label, self.test_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | Loss(test) {:.4f} | Acc(test) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, loss_test,
                        acc_test, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.decoder.eval()
        self.encoder.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask, training=False)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)


class tfr_gin_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)


    def method_init(self, conf, data):

        self.model = GIN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         mlp_layers=conf.model['mlp_layers'], train_eps=conf.model['train_eps']).to(self.device)
        self.decoder = MPNN(in_channels=conf.model['n_classes'], hidden_channels=conf.model['n_hidden'],
                            out_channels=conf.model['n_feat'], decoder_dropout=conf.model['decoder_dropout'],
                            att_softmax=conf.model['att_softmax']).to(self.device)
        self.encoder = torch.nn.Linear(in_features=conf.model['n_feat'], out_features=conf.model['n_hidden'], bias=True).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        self.pseudo_coef = 0.0
        self.pseudo_ratio = conf.model['pseudo_ratio']
        self.beta = conf.model['beta']
        self.beta_dropout_ratio = conf.model['beta_dropout_ratio']
        self.beta_dropout_type = conf.model['beta_dropout_type']
        self.use_recon_softmax = conf.model['use_recon_softmax']
        self.high_confidence_clean_acc_list = []
        self.high_naive_clean_acc_list = []

    def get_prediction(self, features, adj, label=None, mask=None, training=True):
        loss, acc = None, None
        output = self.model(features, adj)
        if (label is not None) and (mask is not None):
            pseudo_label = torch.softmax(output, dim=1)
            if self.beta_dropout_type == 'edge':
                dropout_adj = dropout_edge(adj.indices(), p=self.beta_dropout_ratio)[0]
            else:
                subset = torch.randperm(self.n_nodes)[:int(self.n_nodes * self.beta_dropout_ratio)].to(self.device)
                dropout_adj = subgraph(subset=subset, edge_index=adj.indices())[0]

            recon_feats, confidence = self.decoder(pseudo_label, dropout_adj)

            high_confidence_indices = torch.argsort(confidence.squeeze(), descending=True)[
                                      :int(self.n_nodes * self.pseudo_ratio)]
            high_confidence_clean_acc = self.metric(self.clean_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            high_confidence_noisy_acc = self.metric(self.noisy_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            naive_confidencce = pseudo_label.max(1)[0]
            high_naive_confidence_indices = torch.argsort(naive_confidencce.squeeze(), descending=True)[
                                            :int(self.n_nodes * self.pseudo_ratio)]
            high_naive_clean_acc = self.metric(self.clean_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            high_naive_noisy_acc = self.metric(self.noisy_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            if self.conf.training['debug'] and training:
                print("high confidence clean accuracy: {}, high confidence noisy accuracy: {}".format(
                    high_confidence_clean_acc, high_confidence_noisy_acc))
                print("high naive clean accuracy: {}, high naive noisy accuracy: {}".format(high_naive_clean_acc,
                                                                                            high_naive_noisy_acc))
                # Record accuracies
                self.high_confidence_clean_acc_list.append(high_confidence_clean_acc)
                self.high_naive_clean_acc_list.append(high_naive_clean_acc)

            if self.use_recon_softmax:
                recon_feats = F.softmax(recon_feats, dim=1)
            loss_recon = self.loss_fn(recon_feats, features)
            loss_pseudo = self.loss_fn(output[high_confidence_indices],
                                       pseudo_label[high_confidence_indices].detach())
            # recon_feats = F.softmax(recon_feats, dim=1)
            # proj_feats = self.encoder(F.dropout(features, p=0.5, training=True))
            # proj_feats = F.softmax(proj_feats, dim=1)
            # proj_feats = features

            # loss_pseudo = self.loss_fn(output[high_naive_confidence_indices], pseudo_label[high_naive_confidence_indices].detach())
            loss_sup = self.loss_fn(output[mask], label[mask])
            loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon + self.pseudo_coef * loss_pseudo
            # loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            if training:
                self.pseudo_coef = acc
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            self.epoch = epoch
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            features, adj = self.feats, self.adj
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask,
                                                                training=True)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            loss_test, acc_test = self.evaluate(self.clean_label, self.test_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | Loss(test) {:.4f} | Acc(test) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, loss_test,
                        acc_test, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.decoder.eval()
        self.encoder.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask, training=False)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)


class tfr_sage_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)


    def method_init(self, conf, data):

        self.model = SAGE(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'], out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout']).to(self.device)
        self.decoder = MPNN(in_channels=conf.model['n_classes'], hidden_channels=conf.model['n_hidden'],
                            out_channels=conf.model['n_feat'], decoder_dropout=conf.model['decoder_dropout'],
                            att_softmax=conf.model['att_softmax']).to(self.device)
        self.encoder = torch.nn.Linear(in_features=conf.model['n_feat'], out_features=conf.model['n_hidden'], bias=True).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()),
                                      lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])

        self.pseudo_coef = 0.0
        self.pseudo_ratio = conf.model['pseudo_ratio']
        self.beta = conf.model['beta']
        self.beta_dropout_ratio = conf.model['beta_dropout_ratio']
        self.beta_dropout_type = conf.model['beta_dropout_type']
        self.use_recon_softmax = conf.model['use_recon_softmax']
        self.high_confidence_clean_acc_list = []
        self.high_naive_clean_acc_list = []

    def get_prediction(self, features, adj, label=None, mask=None, training=True):
        loss, acc = None, None
        output = self.model(features, adj)
        if (label is not None) and (mask is not None):
            pseudo_label = torch.softmax(output, dim=1)
            if self.beta_dropout_type == 'edge':
                dropout_adj = dropout_edge(adj.indices(), p=self.beta_dropout_ratio)[0]
            else:
                subset = torch.randperm(self.n_nodes)[:int(self.n_nodes * self.beta_dropout_ratio)].to(self.device)
                dropout_adj = subgraph(subset=subset, edge_index=adj.indices())[0]

            recon_feats, confidence = self.decoder(pseudo_label, dropout_adj)

            high_confidence_indices = torch.argsort(confidence.squeeze(), descending=True)[
                                      :int(self.n_nodes * self.pseudo_ratio)]
            high_confidence_clean_acc = self.metric(self.clean_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            high_confidence_noisy_acc = self.metric(self.noisy_label[high_confidence_indices].cpu().numpy(),
                                                    pseudo_label[high_confidence_indices].detach().cpu().numpy())
            naive_confidencce = pseudo_label.max(1)[0]
            high_naive_confidence_indices = torch.argsort(naive_confidencce.squeeze(), descending=True)[
                                            :int(self.n_nodes * self.pseudo_ratio)]
            high_naive_clean_acc = self.metric(self.clean_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            high_naive_noisy_acc = self.metric(self.noisy_label[high_naive_confidence_indices].cpu().numpy(),
                                               pseudo_label[high_naive_confidence_indices].detach().cpu().numpy())
            if self.conf.training['debug'] and training:
                print("high confidence clean accuracy: {}, high confidence noisy accuracy: {}".format(
                    high_confidence_clean_acc, high_confidence_noisy_acc))
                print("high naive clean accuracy: {}, high naive noisy accuracy: {}".format(high_naive_clean_acc,
                                                                                            high_naive_noisy_acc))
                # Record accuracies
                self.high_confidence_clean_acc_list.append(high_confidence_clean_acc)
                self.high_naive_clean_acc_list.append(high_naive_clean_acc)

            if self.use_recon_softmax:
                recon_feats = F.softmax(recon_feats, dim=1)
            loss_recon = self.loss_fn(recon_feats, features)
            loss_pseudo = self.loss_fn(output[high_confidence_indices],
                                       pseudo_label[high_confidence_indices].detach())
            # recon_feats = F.softmax(recon_feats, dim=1)
            # proj_feats = self.encoder(F.dropout(features, p=0.5, training=True))
            # proj_feats = F.softmax(proj_feats, dim=1)
            # proj_feats = features

            # loss_pseudo = self.loss_fn(output[high_naive_confidence_indices], pseudo_label[high_naive_confidence_indices].detach())
            loss_sup = self.loss_fn(output[mask], label[mask])
            loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon + self.pseudo_coef * loss_pseudo
            # loss = (1 - self.pseudo_coef) * loss_sup + self.beta * loss_recon
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            if training:
                self.pseudo_coef = acc
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            self.epoch = epoch
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            features, adj = self.feats, self.adj
            output, loss_train, acc_train = self.get_prediction(features, adj, self.noisy_label, self.train_mask,
                                                                training=True)
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            loss_test, acc_test = self.evaluate(self.clean_label, self.test_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | Loss(test) {:.4f} | Acc(test) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, loss_test,
                        acc_test, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))

        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.decoder.eval()
        self.encoder.eval()
        features, adj = self.feats, self.adj
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, adj, label, mask, training=False)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)
