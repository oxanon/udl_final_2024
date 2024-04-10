import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Subset
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VCL(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        num_tasks,
        learning_rate=0.001,
        coreset_size=40,
        num_samples=10,
        multihead=False,
    ):
        super(VCL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.coreset_size = coreset_size
        self.num_samples = num_samples
        self.multihead = multihead

        self.prior_mean = 0.0
        self.prior_var = 1.0
        self.initial_var = 1e-6

        if multihead:
            shared_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims[:-1]:
                shared_layers.append(self.create_bayesian_layer(prev_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.shared_layers = nn.Sequential(*shared_layers).to(device)

            self.head_layers = nn.ModuleList()
            for _ in range(num_tasks):
                head_layer = nn.Sequential(
                    self.create_bayesian_layer(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    self.create_bayesian_layer(hidden_dims[-1], output_dim),
                )
                self.head_layers.append(head_layer)
            self.head_layers.to(device)
        else:
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(self.create_bayesian_layer(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(self.create_bayesian_layer(prev_dim, output_dim))
            self.layers = nn.Sequential(*layers).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def create_bayesian_layer(self, in_features, out_features, bias=True):
        layer = nn.Linear(in_features, out_features, bias=bias)
        layer.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        layer.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            layer.bias_mu = nn.Parameter(torch.Tensor(out_features))
            layer.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            layer.register_parameter("bias_mu", None)
            layer.register_parameter("bias_rho", None)

        nn.init.normal_(layer.W_mu, 0, 0.1)
        nn.init.normal_(layer.W_rho, -3, 0.1)
        if bias:
            nn.init.normal_(layer.bias_mu, 0, 0.1)
            nn.init.normal_(layer.bias_rho, -3, 0.1)

        return layer

    def forward(self, x, head_idx=None):
        x = x.view(x.size(0), -1)
        if self.multihead:
            x = self.shared_layers(x)
            x = self.head_layers[head_idx](x)
        else:
            x = self.layers(x)
        return x

    def vcl_loss(self, x, y, head_idx=None):
        loss = 0
        for _ in range(self.num_samples):
            if self.multihead:
                y_pred = self.forward(x, head_idx)
            else:
                y_pred = self.forward(x)

            nll_loss = F.cross_entropy(y_pred, y)
            kl_loss = self.get_kl(head_idx)
            loss += nll_loss + kl_loss / self.num_samples
        return loss

    def train_task(self, task_loader, coreset, num_epochs, head_idx=None):
        if head_idx == 0:
            for layer in self.children():
                if isinstance(layer, nn.Linear):
                    layer.W_mu.data.normal_(0, 0.1)
                    layer.W_rho.data.fill_(np.log(self.initial_var))
                    layer.bias_mu.data.normal_(0, 0.1)
                    layer.bias_rho.data.fill_(np.log(self.initial_var))
                elif isinstance(layer, nn.ModuleList):
                    for head_layer in layer:
                        head_layer[0].W_mu.data.normal_(0, 0.1)
                        head_layer[0].W_rho.data.fill_(np.log(self.initial_var))
                        head_layer[0].bias_mu.data.normal_(0, 0.1)
                        head_layer[0].bias_rho.data.fill_(np.log(self.initial_var))
                        head_layer[2].W_mu.data.normal_(0, 0.1)
                        head_layer[2].W_rho.data.fill_(np.log(self.initial_var))
                        head_layer[2].bias_mu.data.normal_(0, 0.1)
                        head_layer[2].bias_rho.data.fill_(np.log(self.initial_var))

        for epoch in range(num_epochs):
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = self.vcl_loss(x, y, head_idx)
                if coreset is not None:
                    coreset_x, coreset_y = coreset
                    loss += self.vcl_loss(coreset_x, coreset_y, head_idx)
                loss.backward()
                self.optimizer.step()
        self.update_prior()

    def get_kl(self, head_idx):
        kl = 0
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                kl += self.kl_divergence(layer)
            elif isinstance(layer, nn.ModuleList):
                kl += self.kl_divergence(self.head_layers[head_idx][0])
                kl += self.kl_divergence(self.head_layers[head_idx][2])
        return kl

    def kl_divergence(self, layer):
        prior_mean = self.prior_mean
        prior_var = torch.tensor(self.prior_var).to(device)

        W_mu, W_rho = layer.W_mu, layer.W_rho
        b_mu, b_rho = layer.bias_mu, layer.bias_rho

        W_sigma = torch.log1p(torch.exp(W_rho))
        b_sigma = torch.log1p(torch.exp(b_rho))

        kl_W = torch.mean(
            W_sigma**2
            + W_mu**2
            - torch.log(W_sigma**2)
            - 1
            + torch.log(prior_var)
            - prior_var
            - (W_mu - prior_mean) ** 2 / prior_var
        )
        kl_b = torch.mean(
            b_sigma**2
            + b_mu**2
            - torch.log(b_sigma**2)
            - 1
            + torch.log(prior_var)
            - prior_var
            - (b_mu - prior_mean) ** 2 / prior_var
        )

        return (kl_W + kl_b) / 2

    def update_prior(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                layer.W_mu.data = layer.W_mu.data.clone().detach()
                layer.W_rho.data = layer.W_rho.data.clone().detach()
                if layer.bias_mu is not None:
                    layer.bias_mu.data = layer.bias_mu.data.clone().detach()
                    layer.bias_rho.data = layer.bias_rho.data.clone().detach()
            elif isinstance(layer, nn.ModuleList):
                for head_layer in layer:
                    head_layer[0].W_mu.data = head_layer[0].W_mu.data.clone().detach()
                    head_layer[0].W_rho.data = head_layer[0].W_rho.data.clone().detach()
                    if head_layer[0].bias_mu is not None:
                        head_layer[0].bias_mu.data = (
                            head_layer[0].bias_mu.data.clone().detach()
                        )
                        head_layer[0].bias_rho.data = (
                            head_layer[0].bias_rho.data.clone().detach()
                        )
                    head_layer[2].W_mu.data = head_layer[2].W_mu.data.clone().detach()
                    head_layer[2].W_rho.data = head_layer[2].W_rho.data.clone().detach()
                    if head_layer[2].bias_mu is not None:
                        head_layer[2].bias_mu.data = (
                            head_layer[2].bias_mu.data.clone().detach()
                        )
                        head_layer[2].bias_rho.data = (
                            head_layer[2].bias_rho.data.clone().detach()
                        )

    def select_coreset(self, task_loader):
        if self.coreset_size is None:
            return None

        coreset_indices = np.random.choice(
            len(task_loader.dataset), self.coreset_size, replace=False
        )
        coreset_x = []
        coreset_y = []
        for i in coreset_indices:
            x, y = task_loader.dataset[i]
            x, y = x.to(device), y.to(device)
            coreset_x.append(x)
            coreset_y.append(y)
        coreset_x = torch.stack(coreset_x).to(device)
        coreset_y = torch.tensor(coreset_y).to(device)
        return coreset_x, coreset_y

    def select_coreset_with_uncertainty(self, task_loader, head_idx=None, num_models=5):
        if self.coreset_size is None:
            return None

        ensemble_preds = []
        for _ in range(num_models):
            preds = []
            for batch in task_loader:
                x, _ = batch
                x = x.to(device)
                if self.multihead:
                    pred = F.softmax(self.forward(x, head_idx), dim=1)
                else:
                    pred = F.softmax(self.forward(x), dim=1)
                preds.append(pred)
            preds = torch.cat(preds, dim=0)
            ensemble_preds.append(preds)
        ensemble_preds = torch.stack(ensemble_preds)

        mean_preds = torch.mean(ensemble_preds, dim=0)
        uncertainty_scores = torch.sum(torch.var(ensemble_preds, dim=0), dim=1)

        coreset_indices = torch.argsort(uncertainty_scores, descending=True)[
            : self.coreset_size
        ]
        coreset_x = []
        coreset_y = []
        for i in coreset_indices:
            x, y = task_loader.dataset[i]
            x, y = x.to(device), y.to(device)
            coreset_x.append(x)
            coreset_y.append(y)
        coreset_x = torch.stack(coreset_x).to(device)
        coreset_y = torch.tensor(coreset_y).to(device)
        return coreset_x, coreset_y


class EWC(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        num_tasks,
        learning_rate=0.001,
        ewc_lambda=1,
        fisher_samples=200,
        multihead=False,
    ):
        super(EWC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.multihead = multihead

        self.fisher = []
        self.prev_params = []

        if multihead:
            shared_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims[:-1]:
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.shared_layers = nn.Sequential(*shared_layers).to(device)

            self.head_layers = nn.ModuleList()
            for _ in range(num_tasks):
                head_layer = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], output_dim),
                )
                self.head_layers.append(head_layer)
            self.head_layers.to(device)
        else:

            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.layers = nn.Sequential(*layers).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, head_idx=None):
        x = x.view(x.size(0), -1)
        if self.multihead:
            x = self.shared_layers(x)
            x = self.head_layers[head_idx](x)
        else:
            x = self.layers(x)
        return x

    def ewc_loss(self, x, y, head_idx=None):
        if self.multihead:
            output = self.forward(x, head_idx)
        else:
            output = self.forward(x)

        log_likelihood = F.log_softmax(output, dim=1)[range(len(y)), y]
        ewc_loss = 0
        if len(self.fisher) > 0:
            for fisher, prev_param in zip(self.fisher, self.prev_params):
                for name, param in self.named_parameters():
                    if name in fisher:
                        fisher_diagonal = fisher[name]
                        param_diff = (param - prev_param[name]) ** 2
                        ewc_loss += (fisher_diagonal * param_diff).sum()

        return -log_likelihood.mean() + self.ewc_lambda * ewc_loss / len(x)

    def compute_fisher(self, task_loader, head_idx=None):
        self.fisher.append({})
        self.prev_params.append({})
        for name, param in self.named_parameters():
            self.fisher[-1][name] = torch.zeros_like(param)
            self.prev_params[-1][name] = param.data.clone()

        self.eval()
        for x, y in task_loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            if self.multihead:
                output = self.forward(x, head_idx)
            else:
                output = self.forward(x)
            log_likelihood = F.log_softmax(output, dim=1)
            log_likelihood = log_likelihood[range(len(y)), y]
            fisher = torch.autograd.grad(
                log_likelihood.mean(),
                self.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            for name, param in zip(self.named_parameters(), fisher):
                name = name[0]
                if param is not None:
                    if self.multihead:
                        if name.startswith(
                            f"head_layers.{head_idx}"
                        ) or name.startswith("shared_layers"):
                            self.fisher[-1][name] += (param**2) / len(
                                task_loader.dataset
                            )
                    else:
                        self.fisher[-1][name] += (param**2) / len(task_loader.dataset)
        self.train()

    def train_task(self, task_loader, num_epochs, head_idx=None):
        for epoch in range(num_epochs):
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = self.ewc_loss(x, y, head_idx)
                loss.backward()
                self.optimizer.step()

        self.compute_fisher(task_loader, head_idx)


class SI(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        num_tasks,
        learning_rate=0.001,
        si_lambda=1,
        multihead=False,
    ):
        super(SI, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.si_lambda = si_lambda
        self.multihead = multihead

        if multihead:

            shared_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims[:-1]:
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.shared_layers = nn.Sequential(*shared_layers).to(device)

            self.head_layers = nn.ModuleList()
            for _ in range(num_tasks):
                head_layer = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], output_dim),
                )
                self.head_layers.append(head_layer)
            self.head_layers = self.head_layers.to(device)

        else:

            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.layers = nn.Sequential(*layers).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.importance = []
        self.prev_params = []

    def forward(self, x, head_idx=None):
        x = x.view(x.size(0), -1)
        if self.multihead:
            x = self.shared_layers(x)
            x = self.head_layers[head_idx](x)
        else:
            x = self.layers(x)
        return x

    def si_loss(self, x, y, head_idx=None):
        if self.multihead:
            loss = F.cross_entropy(self.forward(x, head_idx), y)
        else:
            loss = F.cross_entropy(self.forward(x), y)

        if len(self.importance) > 0:
            for importance, prev_param in zip(self.importance, self.prev_params):
                for name, param in self.named_parameters():
                    if self.multihead:
                        if name.startswith(
                            f"head_layers.{head_idx}"
                        ) or name.startswith("shared_layers"):
                            if name in importance:
                                importance_for_param = importance[name].view_as(param)
                                param_diff = (param - prev_param[name]).pow(2)
                                si_term = (
                                    self.si_lambda * importance_for_param * param_diff
                                )
                                loss += si_term.sum()
                    else:
                        if name in importance:
                            importance_for_param = importance[name].view_as(param)
                            param_diff = (param - prev_param[name]).pow(2)
                            si_term = self.si_lambda * importance_for_param * param_diff
                            loss += si_term.sum()
        return loss

    def update_importance(self, task_loader, head_idx=None):
        self.importance.append({})
        self.prev_params.append({})
        for name, param in self.named_parameters():
            if self.multihead:
                if name.startswith(f"head_layers.{head_idx}") or name.startswith(
                    "shared_layers"
                ):
                    self.importance[-1][name] = torch.zeros_like(param)
                    self.prev_params[-1][name] = param.data.clone()
            else:
                self.importance[-1][name] = torch.zeros_like(param)
                self.prev_params[-1][name] = param.data.clone()

        self.eval()
        for x, y in task_loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            if self.multihead:
                output = self.forward(x, head_idx)
            else:
                output = self.forward(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            for name, param in self.named_parameters():
                if self.multihead:
                    if name.startswith(f"head_layers.{head_idx}") or name.startswith(
                        "shared_layers"
                    ):
                        if name in self.importance[-1]:
                            self.importance[-1][name] += param.grad.data.clone().abs()
                else:
                    if name in self.importance[-1]:
                        self.importance[-1][name] += param.grad.data.clone().abs()

        self.train()

    def train_task(self, task_loader, num_epochs, head_idx=None):
        for epoch in range(num_epochs):
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = self.si_loss(x, y, head_idx)
                loss.backward()
                self.optimizer.step()
            self.update_importance(task_loader, head_idx)


def run_experiment(
    model,
    train_loaders,
    test_loaders,
    num_epochs_per_task,
    uncertainty_coreset=False,
):
    accuracies = []

    coreset = None

    for task, task_loader in tqdm(enumerate(train_loaders), total=len(train_loaders)):
        tqdm.write(f"Processing Task: {task} out of {len(train_loaders)}")

        if isinstance(model, VCL):
            if uncertainty_coreset:
                coreset = model.select_coreset_with_uncertainty(
                    task_loader, head_idx=task
                )
            else:
                coreset = model.select_coreset(task_loader)

            if coreset is not None:
                coreset = (
                    coreset[0].view(coreset[0].size(0), -1),
                    coreset[1],
                )
            if model.multihead:
                model.train_task(task_loader, coreset, num_epochs_per_task, task)
            else:
                model.train_task(task_loader, coreset, num_epochs_per_task)
        else:
            if model.multihead:
                model.train_task(task_loader, num_epochs_per_task, task)
            else:
                model.train_task(task_loader, num_epochs_per_task)

        model.eval()
        model = model.to(device)

        task_accuracies = []
        with torch.no_grad():
            for test, test_loader in enumerate(test_loaders[: task + 1]):

                correct = 0
                total = 0

                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    x = x.view(x.size(0), -1)
                    if model.multihead:
                        y_pred = model(x, test)
                    else:
                        y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                accuracy = 100 * correct / total
                task_accuracies.append(accuracy)
        accuracies.append(task_accuracies)

    return accuracies


def plot_results(
    vcl_accuracies,
    ewc_accuracies,
    si_accuracies,
    title,
    vcl_accuracies_with_uncertainty=None,
):
    num_tasks = len(vcl_accuracies)

    if vcl_accuracies_with_uncertainty is not None:
        vcl_mean_accuracies_wu = [
            np.mean(task_acc[: i + 1])
            for i, task_acc in enumerate(vcl_accuracies_with_uncertainty)
        ]

    vcl_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(vcl_accuracies)
    ]
    ewc_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(ewc_accuracies)
    ]
    si_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(si_accuracies)
    ]

    plt.figure(figsize=(8, 6))
    if vcl_accuracies_with_uncertainty is not None:
        plt.plot(
            range(1, num_tasks + 1),
            vcl_mean_accuracies_wu,
            marker="o",
            label="VCL with Uncertainty Coreset",
        )
    plt.plot(
        range(1, num_tasks + 1),
        vcl_mean_accuracies,
        marker="o",
        label="VCL",
    )
    plt.plot(range(1, num_tasks + 1), ewc_mean_accuracies, marker="o", label="EWC")
    plt.plot(range(1, num_tasks + 1), si_mean_accuracies, marker="o", label="SI")

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    plt.title("VCL vs EWC vs SI all with Multi-Head")
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


class PermutedMNIST(MNIST):
    def __init__(
        self, root, train=True, transform=None, download=True, permute_idx=None
    ):
        super(PermutedMNIST, self).__init__(
            root, train=train, transform=transform, download=download
        )
        if permute_idx is None:
            permute_idx = torch.randperm(784)
        self.permute_idx = permute_idx

        if self.train:
            self.data = self.data.view(-1, 784)[:, self.permute_idx].view(-1, 28, 28)
        else:
            self.data = self.data.view(-1, 784)[:, self.permute_idx].view(-1, 28, 28)


def permuted_mnist_dataloaders(
    train_batch_size, test_batch_size, num_tasks, train_size=1000, test_size=500
):
    train_loaders = []
    test_loaders = []

    for task in range(num_tasks):
        if task == 0:
            permute_idx = None
        else:
            permute_idx = torch.randperm(784)

        train_dataset = PermutedMNIST(
            root="./data", train=True, download=True, permute_idx=permute_idx
        )
        test_dataset = PermutedMNIST(
            root="./data", train=False, download=True, permute_idx=permute_idx
        )

        train_indices = torch.randperm(len(train_dataset))[:train_size]
        test_indices = torch.randperm(len(test_dataset))[:test_size]

        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

        train_tensor_dataset = data.TensorDataset(
            train_subset.dataset.data[train_indices].float().view(train_size, -1)
            / 255.0,
            train_subset.dataset.targets[train_indices],
        )
        test_tensor_dataset = data.TensorDataset(
            test_subset.dataset.data[test_indices].float().view(test_size, -1) / 255.0,
            test_subset.dataset.targets[test_indices],
        )

        train_loader = DataLoader(
            train_tensor_dataset, batch_size=train_batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_tensor_dataset, batch_size=test_batch_size, shuffle=False
        )

        print(f"Task {task} - Number of train samples: {len(train_indices)}")
        print(f"Task {task} - Number of test samples: {len(test_indices)}")

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def fmnist_dataloaders(
    train_batch_size, test_batch_size, num_tasks, train_size=1000, test_size=500
):
    train_loaders = []
    test_loaders = []

    for train, loaders, bs, size in zip(
        (True, False),
        (train_loaders, test_loaders),
        (train_batch_size, test_batch_size),
        (train_size, test_size),
    ):
        fmnist = FashionMNIST("./data", train=train, download=True)
        indices = torch.randperm(len(fmnist))[:size]
        x = fmnist.data[indices].float().view(size, -1) / 255.0
        y = fmnist.targets[indices]

        for i in range(num_tasks):
            index = y.ge(i * 2) & y.lt((i + 1) * 2)
            if index.sum() > 0:
                loaders.append(
                    data.DataLoader(
                        data.TensorDataset(x[index], y[index].sub(2 * i).long()),
                        bs,
                        shuffle=True,
                    )
                )
            else:

                print(
                    f"Warning: No samples found for task {i} (train={train}). Skipping DataLoader creation."
                )

    return train_loaders, test_loaders


def split_kmnist_dataloaders(
    train_batch_size, test_batch_size, num_tasks, train_size=1000, test_size=500
):
    train_loaders = []
    test_loaders = []

    for train, loaders, bs, size in zip(
        (True, False),
        (train_loaders, test_loaders),
        (train_batch_size, test_batch_size),
        (train_size, test_size),
    ):
        kmnist = KMNIST("./data", train=train, download=True)
        indices = torch.randperm(len(kmnist))[:size]
        x = kmnist.data[indices].float().view(size, -1) / 255.0
        y = kmnist.targets[indices]

        for i in range(num_tasks):
            index = y.ge(i * 2) & y.lt((i + 1) * 2)
            if index.sum() > 0:
                loaders.append(
                    data.DataLoader(
                        data.TensorDataset(x[index], y[index].sub(2 * i).long()),
                        bs,
                        shuffle=True,
                    )
                )
            else:
                print(
                    f"Warning: No samples found for task {i} (train={train}). Skipping DataLoader creation."
                )

    return train_loaders, test_loaders


def split_mnist_dataloaders(
    train_batch_size, test_batch_size, num_tasks, train_size=1000, test_size=500
):
    train_loaders = []
    test_loaders = []

    for train, loaders, bs, size in zip(
        (True, False),
        (train_loaders, test_loaders),
        (train_batch_size, test_batch_size),
        (train_size, test_size),
    ):
        mnist = MNIST("./data", train=train, download=True)
        indices = torch.randperm(len(mnist))[:size]
        x = mnist.data[indices].float().view(size, -1) / 255.0
        y = mnist.targets[indices]

        for task in range(num_tasks):
            digit1, digit2 = task * 2, task * 2 + 1
            task_indices = (y == digit1) | (y == digit2)

            if task_indices.sum() > 0:
                task_labels = (y[task_indices] == digit2).long()
                loaders.append(
                    data.DataLoader(
                        data.TensorDataset(x[task_indices], task_labels),
                        bs,
                        shuffle=True,
                    )
                )
                print(f"Task {task} - Number of samples: {task_indices.sum()}")

            else:
                print(
                    f"Warning: No samples found for task {task} (train={train}). Skipping DataLoader creation."
                )

    return train_loaders, test_loaders


def run_and_plot_permuted_mnist_experiment(subset_size=None, force=False):
    task = "permuted_mnist"

    vcl_accuracies_path = f"vcl_{task}.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        os.path.exists(vcl_accuracies_path)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path, "rb") as f:
            vcl_accuracies = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = permuted_mnist_dataloaders(
            train_batch_size=256,
            test_batch_size=256,
            num_tasks=10,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [100, 100]
        output_dim = 10
        learning_rate = 0.001
        num_tasks = 10

        vcl_model = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=200,
            num_samples=10,
            multihead=False,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=600,
            multihead=False,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=0.5,
            multihead=False,
        ).to(device)

        vcl_accuracies = run_experiment(
            vcl_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=100,
        )
        ewc_accuracies = run_experiment(
            ewc_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )
        si_accuracies = run_experiment(
            si_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        with open(vcl_accuracies_path, "wb") as f:
            pickle.dump(vcl_accuracies, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results(vcl_accuracies, ewc_accuracies, si_accuracies, "Permuted MNIST")


def run_and_plot_permuted_mnist_coreset_experiment(subset_size=None, force=False):
    input_dim = 28 * 28
    hidden_dims = [100, 100]
    output_dim = 10
    learning_rate = 0.001
    num_tasks = 10

    coreset_sizes = [200, 400, 1000, 2500, 5000]
    plt.figure(figsize=(8, 6))
    for coreset_size in coreset_sizes:
        accuracies_path = f"vcl_coreset_{coreset_size}_permuted_mnist.pkl"

        if os.path.exists(accuracies_path) and not force:
            with open(accuracies_path, "rb") as f:
                vcl_accuracies_coreset = pickle.load(f)
        else:

            train_loaders, test_loaders = permuted_mnist_dataloaders(
                train_batch_size=coreset_size,
                test_batch_size=coreset_size,
                num_tasks=10,
                train_size=60000,
                test_size=10000,
            )
            vcl_model = VCL(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_tasks=num_tasks,
                learning_rate=learning_rate,
                coreset_size=coreset_size,
                num_samples=10,
                multihead=False,
            ).to(device)

            vcl_accuracies_coreset = run_experiment(
                vcl_model,
                train_loaders,
                test_loaders,
                num_epochs_per_task=100,
            )
            with open(accuracies_path, "wb") as f:
                pickle.dump(vcl_accuracies_coreset, f)

        vcl_mean_accuracies_coreset = [
            np.mean(task_acc[: i + 1])
            for i, task_acc in enumerate(vcl_accuracies_coreset)
        ]

        plt.plot(
            range(1, num_tasks + 1),
            vcl_mean_accuracies_coreset,
            marker="o",
            label=f"VCL + Coreset ({coreset_size})",
        )

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    title = "Permuted MNIST - Coreset Size Comparison"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


def run_and_plot_split_mnist_experiment(subset_size=None, force=False):

    task = "split_mnist"

    vcl_accuracies_path = f"vcl_{task}.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"
    vcl_accuracies_path_wu = f"vcl_{task}_wu.pkl"

    if (
        os.path.exists(vcl_accuracies_path)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path, "rb") as f:
            vcl_accuracies = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = split_mnist_dataloaders(
            train_batch_size=60000,
            test_batch_size=10000,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = 40

        vcl_model = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=True,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=True,
        ).to(device)

        vcl_accuracies = run_experiment(
            vcl_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=120,
        )

        ewc_accuracies = run_experiment(
            ewc_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        si_accuracies = run_experiment(
            si_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        vcl_accuracies_with_uncertainty = None

        if vcl_accuracies_with_uncertainty:
            with open(vcl_accuracies_path_wu, "wb") as f:
                pickle.dump(vcl_accuracies_with_uncertainty, f)
        else:
            vcl_accuracies_with_uncertainty = None

        with open(vcl_accuracies_path, "wb") as f:
            pickle.dump(vcl_accuracies, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results(
        vcl_accuracies,
        ewc_accuracies,
        si_accuracies,
        "Split MNIST",
    )


def run_and_plot_split_fmnist_experiment(subset_size=None, force=False):
    task = "fashion_mnist"

    vcl_accuracies_path = f"vcl_{task}.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        os.path.exists(vcl_accuracies_path)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path, "rb") as f:
            vcl_accuracies = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = fmnist_dataloaders(
            train_batch_size=60000,
            test_batch_size=10000,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = None

        vcl_model = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=False,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=True,
        ).to(device)

        vcl_accuracies = run_experiment(
            vcl_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=120,
        )

        ewc_accuracies = run_experiment(
            ewc_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        si_accuracies = run_experiment(
            si_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        with open(vcl_accuracies_path, "wb") as f:
            pickle.dump(vcl_accuracies, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results(vcl_accuracies, ewc_accuracies, si_accuracies, "Fashion MNIST")


def run_and_plot_split_kmnist_experiment(subset_size=None, force=False):
    task = "kanji_mnist"

    vcl_accuracies_path = f"vcl_{task}.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        os.path.exists(vcl_accuracies_path)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path, "rb") as f:
            vcl_accuracies = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:

        train_loaders, test_loaders = split_kmnist_dataloaders(
            train_batch_size=256,
            test_batch_size=256,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = None

        vcl_model = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=False,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=True,
        ).to(device)

        vcl_accuracies = run_experiment(
            vcl_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        ewc_accuracies = run_experiment(
            ewc_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        si_accuracies = run_experiment(
            si_model,
            train_loaders,
            test_loaders,
            num_epochs_per_task=20,
        )

        with open(vcl_accuracies_path, "wb") as f:
            pickle.dump(vcl_accuracies, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results(vcl_accuracies, ewc_accuracies, si_accuracies, "Kuzushiji-MNIST")


def run_and_plot_split_mnist_batch_size_experiment(subset_size=None, force=False):
    task = "split_mnist_batch_size"

    batch_sizes = [60000, 10000, 1024, 256]
    accuracies_paths = [f"vcl_{task}_{bs}.pkl" for bs in batch_sizes]

    if all(os.path.exists(path) for path in accuracies_paths) and not force:
        accuracies_list = []
        for path in accuracies_paths:
            with open(path, "rb") as f:
                accuracies_list.append(pickle.load(f))
    else:
        accuracies_list = []
        for batch_size in batch_sizes:
            train_loaders, test_loaders = split_mnist_dataloaders(
                train_batch_size=batch_size,
                test_batch_size=10000,
                num_tasks=5,
                train_size=60000,
                test_size=10000,
            )

            input_dim = 28 * 28
            hidden_dims = [256, 256]
            output_dim = 2
            learning_rate = 0.001
            num_tasks = 5
            coreset_size = 40

            vcl_model = VCL(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_tasks=num_tasks,
                learning_rate=learning_rate,
                coreset_size=coreset_size,
                num_samples=10,
                multihead=True,
            ).to(device)

            accuracies = run_experiment(
                vcl_model, train_loaders, test_loaders, num_epochs_per_task=120
            )
            accuracies_list.append(accuracies)

            accuracies_path = f"vcl_{task}_{batch_size}.pkl"
            with open(accuracies_path, "wb") as f:
                pickle.dump(accuracies, f)

    num_tasks = len(accuracies_list[0])

    plt.figure(figsize=(8, 6))

    for batch_size, accuracies in zip(batch_sizes, accuracies_list):
        mean_accuracies = [
            np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(accuracies)
        ]
        plt.plot(
            range(1, num_tasks + 1),
            mean_accuracies,
            marker="o",
            label=f"VCL - Batch Size: {batch_size}",
        )

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    title = "Split MNIST - VCL Batch Size Comparison"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def run_and_plot_split_mnist_experiment_comparison(subset_size=None, force=False):
    task = "split_mnist"

    vcl_accuracies_path_120 = f"vcl_{task}_120.pkl"
    vcl_accuracies_path_20 = f"vcl_{task}_20.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        os.path.exists(vcl_accuracies_path_120)
        and os.path.exists(vcl_accuracies_path_20)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path_120, "rb") as f:
            vcl_accuracies_120 = pickle.load(f)
        with open(vcl_accuracies_path_20, "rb") as f:
            vcl_accuracies_20 = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = split_mnist_dataloaders(
            train_batch_size=60000,
            test_batch_size=10000,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = 40

        vcl_model_120 = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        vcl_model_20 = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=True,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=True,
        ).to(device)

        vcl_model_no_coreset = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=None,
            num_samples=10,
            multihead=True,
        ).to(device)

        vcl_model_random_coreset = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        vcl_accuracies_120 = run_experiment(
            vcl_model_120, train_loaders, test_loaders, num_epochs_per_task=120
        )
        vcl_accuracies_20 = run_experiment(
            vcl_model_20, train_loaders, test_loaders, num_epochs_per_task=20
        )
        ewc_accuracies = run_experiment(
            ewc_model, train_loaders, test_loaders, num_epochs_per_task=20
        )
        si_accuracies = run_experiment(
            si_model, train_loaders, test_loaders, num_epochs_per_task=20
        )

        with open(vcl_accuracies_path_120, "wb") as f:
            pickle.dump(vcl_accuracies_120, f)
        with open(vcl_accuracies_path_20, "wb") as f:
            pickle.dump(vcl_accuracies_20, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results_extra(
        vcl_accuracies_120,
        vcl_accuracies_20,
        ewc_accuracies,
        si_accuracies,
        "Split MNIST - Epoch Comparison",
    )


def plot_results_extra(
    vcl_accuracies_120,
    vcl_accuracies_20,
    ewc_accuracies,
    si_accuracies,
    title,
    labels=None,
):
    num_tasks = len(vcl_accuracies_120)

    vcl_mean_accuracies_120 = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(vcl_accuracies_120)
    ]
    vcl_mean_accuracies_20 = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(vcl_accuracies_20)
    ]
    ewc_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(ewc_accuracies)
    ]
    si_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(si_accuracies)
    ]

    plt.figure(figsize=(8, 6))

    if labels is None:
        labels = [
            "VCL (120 epochs)",
            "VCL (20 epochs)",
            "EWC (20 epochs)",
            "SI (20 epochs)",
        ]

    plt.plot(
        range(1, num_tasks + 1),
        vcl_mean_accuracies_120,
        marker="o",
        label=labels[0],
    )
    plt.plot(
        range(1, num_tasks + 1),
        vcl_mean_accuracies_20,
        marker="o",
        label=labels[1],
    )
    plt.plot(range(1, num_tasks + 1), ewc_mean_accuracies, marker="o", label=labels[2])
    plt.plot(range(1, num_tasks + 1), si_mean_accuracies, marker="o", label=labels[3])

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))

    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def run_and_plot_split_mnist_epochs_experiment(subset_size=None, force=False):
    task = "split_mnist_epochs"

    vcl_epochs = [120, 80, 20]
    ewc_epochs = 20
    si_epochs = 20

    vcl_accuracies_paths = [f"vcl_{task}_{epochs}.pkl" for epochs in vcl_epochs]
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        all(os.path.exists(path) for path in vcl_accuracies_paths)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
        and not force
    ):
        vcl_accuracies_list = []
        for path in vcl_accuracies_paths:
            with open(path, "rb") as f:
                vcl_accuracies_list.append(pickle.load(f))
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = split_mnist_dataloaders(
            train_batch_size=60000,
            test_batch_size=10000,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = 40

        vcl_accuracies_list = []
        for epochs in vcl_epochs:
            vcl_model = VCL(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_tasks=num_tasks,
                learning_rate=learning_rate,
                coreset_size=coreset_size,
                num_samples=10,
                multihead=True,
            ).to(device)

            vcl_accuracies = run_experiment(
                vcl_model, train_loaders, test_loaders, num_epochs_per_task=epochs
            )
            vcl_accuracies_list.append(vcl_accuracies)

            accuracies_path = f"vcl_{task}_{epochs}.pkl"
            with open(accuracies_path, "wb") as f:
                pickle.dump(vcl_accuracies, f)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=False,
        ).to(device)

        ewc_accuracies = run_experiment(
            ewc_model, train_loaders, test_loaders, num_epochs_per_task=ewc_epochs
        )

        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=False,
        ).to(device)

        si_accuracies = run_experiment(
            si_model, train_loaders, test_loaders, num_epochs_per_task=si_epochs
        )

        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    num_tasks = len(vcl_accuracies_list[0])

    plt.figure(figsize=(8, 6))

    for epochs, accuracies in zip(vcl_epochs, vcl_accuracies_list):
        mean_accuracies = [
            np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(accuracies)
        ]
        plt.plot(
            range(1, num_tasks + 1),
            mean_accuracies,
            marker="o",
            label=f"VCL - Epochs: {epochs}",
        )

    ewc_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(ewc_accuracies)
    ]
    plt.plot(
        range(1, num_tasks + 1),
        ewc_mean_accuracies,
        marker="o",
        label=f"EWC - Epochs: {ewc_epochs}",
    )

    si_mean_accuracies = [
        np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(si_accuracies)
    ]
    plt.plot(
        range(1, num_tasks + 1),
        si_mean_accuracies,
        marker="o",
        label=f"SI - Epochs: {si_epochs}",
    )

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    title = "Split MNIST - Epochs Comparison"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def run_and_plot_kmnist_fmnist_experiment(subset_size=None, force=False):
    tasks = ["Kuzushiji-MNIST", "FashionMNIST"]
    num_tasks = 5
    models = ["VCL", "EWC", "SI"]

    plt.figure(figsize=(10, 6))

    for task in tasks:
        for model_name in models:
            accuracies_path = f"{model_name.lower()}_{task}.pkl"

            if os.path.exists(accuracies_path) and not force:
                with open(accuracies_path, "rb") as f:
                    accuracies = pickle.load(f)
            else:
                if task == "Kuzushiji-MNIST":
                    train_loaders, test_loaders = split_kmnist_dataloaders(
                        train_batch_size=60000,
                        test_batch_size=10000,
                        num_tasks=num_tasks,
                        train_size=60000,
                        test_size=10000,
                    )
                else:
                    train_loaders, test_loaders = fmnist_dataloaders(
                        train_batch_size=60000,
                        test_batch_size=10000,
                        num_tasks=num_tasks,
                        train_size=60000,
                        test_size=10000,
                    )

                input_dim = 28 * 28
                hidden_dims = [256, 256]
                output_dim = 2
                learning_rate = 0.001
                coreset_size = None if model_name != "VCL" else 40

                if model_name == "VCL":
                    model = VCL(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        num_tasks=num_tasks,
                        learning_rate=learning_rate,
                        coreset_size=coreset_size,
                        num_samples=10,
                        multihead=True,
                    ).to(device)
                    num_epochs_per_task = 120

                elif model_name == "EWC":
                    print("here!!")
                    model = EWC(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        num_tasks=num_tasks,
                        learning_rate=learning_rate,
                        ewc_lambda=1,
                        fisher_samples=200,
                        multihead=False,
                    ).to(device)
                    num_epochs_per_task = 20
                else:
                    model = SI(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        num_tasks=num_tasks,
                        learning_rate=learning_rate,
                        si_lambda=1,
                        multihead=True,
                    ).to(device)
                    num_epochs_per_task = 20

                accuracies = run_experiment(
                    model,
                    train_loaders,
                    test_loaders,
                    num_epochs_per_task=num_epochs_per_task,
                )

                with open(accuracies_path, "wb") as f:
                    pickle.dump(accuracies, f)

            mean_accuracies = [
                np.mean(task_acc[: i + 1]) for i, task_acc in enumerate(accuracies)
            ]

            colors = {"VCL": "C0", "EWC": "C1", "SI": "C2"}

            linestyle = "--" if task == "FashionMNIST" else "-"
            plt.plot(
                range(1, num_tasks + 1),
                mean_accuracies,
                marker="o",
                linestyle=linestyle,
                color=colors[model_name],
                label=f"{model_name} - {task}",
            )

    plt.xlabel("Number of tasks")
    plt.ylabel("Average accuracy")
    title = "Kuzushiji-MNIST and FashionMNIST - Model Comparison"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_tasks + 1))
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


def run_and_plot_split_mnist_coreset_comparison(subset_size=None, force=False):
    task = "split_mnist"

    vcl_accuracies_path_uncertainty = f"vcl_{task}_uncertainty_coreset.pkl"
    vcl_accuracies_path_random = f"vcl_{task}_random_coreset.pkl"
    ewc_accuracies_path = f"ewc_{task}.pkl"
    si_accuracies_path = f"si_{task}.pkl"

    if (
        os.path.exists(vcl_accuracies_path_uncertainty)
        and os.path.exists(vcl_accuracies_path_random)
        and os.path.exists(ewc_accuracies_path)
        and os.path.exists(si_accuracies_path)
    ) and not force:
        with open(vcl_accuracies_path_uncertainty, "rb") as f:
            vcl_accuracies_uncertainty = pickle.load(f)
        with open(vcl_accuracies_path_random, "rb") as f:
            vcl_accuracies_random = pickle.load(f)
        with open(ewc_accuracies_path, "rb") as f:
            ewc_accuracies = pickle.load(f)
        with open(si_accuracies_path, "rb") as f:
            si_accuracies = pickle.load(f)
    else:
        train_loaders, test_loaders = split_mnist_dataloaders(
            train_batch_size=60000,
            test_batch_size=10000,
            num_tasks=5,
            train_size=60000,
            test_size=10000,
        )

        input_dim = 28 * 28
        hidden_dims = [256, 256]
        output_dim = 2
        learning_rate = 0.001
        num_tasks = 5
        coreset_size = 40

        vcl_model_uncertainty = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        vcl_model_random = VCL(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            coreset_size=coreset_size,
            num_samples=10,
            multihead=True,
        ).to(device)

        ewc_model = EWC(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            ewc_lambda=1,
            fisher_samples=200,
            multihead=True,
        ).to(device)

        si_model = SI(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            si_lambda=1,
            multihead=True,
        ).to(device)

        vcl_accuracies_uncertainty = run_experiment(
            vcl_model_uncertainty,
            train_loaders,
            test_loaders,
            num_epochs_per_task=120,
            uncertainty_coreset=True,
        )
        vcl_accuracies_random = run_experiment(
            vcl_model_random,
            train_loaders,
            test_loaders,
            num_epochs_per_task=120,
        )
        ewc_accuracies = run_experiment(
            ewc_model, train_loaders, test_loaders, num_epochs_per_task=20
        )
        si_accuracies = run_experiment(
            si_model, train_loaders, test_loaders, num_epochs_per_task=20
        )

        with open(vcl_accuracies_path_uncertainty, "wb") as f:
            pickle.dump(vcl_accuracies_uncertainty, f)
        with open(vcl_accuracies_path_random, "wb") as f:
            pickle.dump(vcl_accuracies_random, f)
        with open(ewc_accuracies_path, "wb") as f:
            pickle.dump(ewc_accuracies, f)
        with open(si_accuracies_path, "wb") as f:
            pickle.dump(si_accuracies, f)

    plot_results_extra(
        vcl_accuracies_uncertainty,
        vcl_accuracies_random,
        ewc_accuracies,
        si_accuracies,
        "Split MNIST - Coreset Comparison",
        labels=[
            "VCL (Uncertainty Coreset)",
            "VCL (Random Coreset)",
            "EWC",
            "SI",
        ],
    )


def main():
    subset_size = None
    run_and_plot_split_mnist_experiment(subset_size=subset_size, force=True)


if __name__ == "__main__":
    main()
