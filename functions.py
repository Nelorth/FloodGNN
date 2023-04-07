import copy
import matplotlib.pyplot as plt
import numpy as np
import poptorch
import random
import torch
import torch.nn as nn

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from models import FloodMLP, FloodGCN, FloodGRAFFNN
from torch_geometric.loader import DataLoader
from torchinfo import summary
from tqdm import tqdm


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_edge_weights(weight_type, edge_attr):
    if weight_type == "disabled":
        return torch.zeros(edge_attr.size(0))
    elif weight_type == "binary":
        return torch.ones(edge_attr.size(0))
    elif weight_type == "dist_hdn":
        return edge_attr[:, 0]
    elif weight_type == "elev_diff":
        return edge_attr[:, 1]
    elif weight_type == "strm_slope":
        return edge_attr[:, 2]
    elif weight_type == "learned":
        return nn.Parameter(torch.ones(edge_attr.size(0)))
    else:
        raise ValueError("Invalid weight type given!")


def construct_model(hparams, edge_weights):
    model_arch = hparams["model"]["architecture"]
    if model_arch == "MLP":
        return FloodMLP(in_channels=hparams["data"]["window_size"],
                        hidden_channels=hparams["model"]["hidden_size"],
                        num_hidden=hparams["model"]["propagation_dist"],
                        residual=hparams["model"]["residual"])
    elif model_arch == "GCN":
        return FloodGCN(in_channels=hparams["data"]["window_size"],
                        hidden_channels=hparams["model"]["hidden_size"],
                        num_hidden=hparams["model"]["propagation_dist"],
                        residual=hparams["model"]["residual"],
                        edge_weights=edge_weights)
    elif model_arch == "GRAFFNN":
        return FloodGRAFFNN(in_channels=hparams["data"]["window_size"],
                            hidden_channels=hparams["model"]["hidden_size"],
                            num_hidden=hparams["model"]["propagation_dist"],
                            edge_weights=edge_weights)
    raise ValueError("Unknown model architecture", hparams["model"]["architecture"])


def train_step(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        if device != "IPU":
            batch = batch.to(device)
            optimizer.zero_grad()
        out, loss = model(batch.x, batch.edge_index, batch.y)
        if device != "IPU":
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
    return train_loss


def val_step(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if device != "IPU":
                batch = batch.to(device)
            out, loss = model(batch.x, batch.edge_index, batch.y)
            val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
    return val_loss


def train(model, train_dataset, val_dataset, hparams, on_ipu=False):
    print(summary(model, depth=2))
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True,
                              drop_last=on_ipu)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False, drop_last=on_ipu)

    if on_ipu:
        optimizer = poptorch.optim.Adam(model.parameters(),
                                        lr=hparams["training"]["learning_rate"],
                                        weight_decay=hparams["training"]["weight_decay"])
        model = poptorch.trainingModel(model, optimizer=optimizer)
        compile_ipu_model(model, train_loader)
        device = "IPU"
        print("Training on IPU")
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hparams["training"]["learning_rate"],
                                     weight_decay=hparams["training"]["weight_decay"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "model_params": [], "optim_params": []}

    for epoch in range(hparams["training"]["num_epochs"]):
        train_loss = train_step(model, train_loader, optimizer, device)
        val_loss = val_step(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["model_params"].append(copy.deepcopy(model.state_dict()))
        history["optim_params"].append(copy.deepcopy(optimizer.state_dict()))

        print("[Epoch {0}/{1}] Train: {2:.4f} | Val {3:.4f}".format(
            epoch + 1, hparams['training']['num_epochs'], train_loss, val_loss
        ))

    torch.save({
        "history": history,
        "hparams": hparams
    }, datetime.now().strftime("runs/%Y-%m-%d_%H-%M-%S.run"))
    return history


def evaluate(model, dataset, hparams, on_ipu=False):
    loader = DataLoader(dataset, batch_size=hparams["training"]["batch_size"], shuffle=False, drop_last=on_ipu)
    if on_ipu:
        device = "IPU"
        model = poptorch.inferenceModel(model)
        compile_ipu_model(model, loader)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if device != "IPU":
                batch = batch.to(device)
            pred, loss = model(batch.x, batch.edge_index, batch.y)
            test_loss += loss.item() * batch.num_graphs / len(dataset)
    return test_loss


def compile_ipu_model(model, loader):
    data = loader.dataset[0]
    fake_x = data.x.repeat(loader.batch_size, 1)
    fake_y = data.y.repeat(loader.batch_size, 1)
    fake_idx = data.edge_index.repeat(1, loader.batch_size)
    model.compile(fake_x, fake_idx, fake_y)


def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("normalized MSE")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
