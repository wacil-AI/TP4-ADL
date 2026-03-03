import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    model.eval()
    model = model.to(device)

    backbone = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten(1))
    backbone = backbone.to(device)

    loader = DataLoader(dataset , batch_size=64, shuffle=False)


    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = backbone(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    return TensorDataset(features_tensor, labels_tensor)
