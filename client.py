from collections import OrderedDict

from tqdm import tqdm
import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

NUM_PARTITIONS = 5

def load_data(node_id, data_cfg: DictConfig):
    """Load partition CIFAR100 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_PARTITIONS})
    partition = fds.load_partition(node_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=data_cfg.val_ratio)
    pytorch_transforms = Compose(
        [ToTensor(), Resize(224), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=data_cfg.train_batch, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=data_cfg.val_batch)
    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, evalloader, model: nn.Module):

        self.trainloader = trainloader
        self.evalloader = evalloader
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.evalloader, self.device)
        return loss, len(self.evalloader.dataset), {"accuracy": accuracy}


@hydra.main(config_path="conf", config_name="client", version_base=None)
def main(cfg: DictConfig) -> None:

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # Prepare dataset
    trainloader, testloader = load_data(node_id=cfg.node_id, data_cfg=cfg.dataset)

    # Setup model
    # The config system was designed for running a benchmark so we need to extract
    # the model node in an ugly way
    # Note that if you don't care about re-using the configs provided, you can directly
    # define your model in this file and pass it to your client.
    # See an example in: https://github.com/adap/flower/tree/main/examples/quickstart-pytorch
    model_name, model_cfg  = list(cfg.model.keys())[0], list(cfg.model.values())[0]
    print(f"Instantiating: {model_name}")

    # Let's set the model to be pretrained (not the cleanest usage of Hydra)
    build = call(model_cfg.model.build, pretrained=True)
    model = instantiate(model_cfg.model, build=build)
    if cfg.finetune:
        model.set_for_finetuning()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(trainloader, testloader, model),
    )

if __name__ == "__main__":
    main()