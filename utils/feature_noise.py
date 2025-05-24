import torch

def feature_process(feature, noise_rate):
    """
    Process the features of the dataset by adding noise to them.

    Parameters:
        feature (torch.Tensor): The original feature tensor.
        noise_rate (float): The rate of noise to be added to the features.

    Returns:
        torch.Tensor: The processed feature tensor with noise added.
    """
    if noise_rate > 0:
        x_max_mean = feature.max(1)[0].mean()
        feature_noise = torch.randn(feature.shape).to(feature.device) * x_max_mean * noise_rate
        processed_feature = feature + feature_noise
    else:
        processed_feature = feature
    return processed_feature
