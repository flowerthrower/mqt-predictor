from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from stable_baselines3.common.preprocessing import get_flattened_obs_dim

if TYPE_CHECKING:
    from gymnasium import spaces
    from gymnasium.spaces import Dict
    from stable_baselines3.common.type_aliases import TensorDict

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import functional as F

logger = logging.getLogger("mqt-predictor")
PATH_LENGTH = 260


class CustomCombinedExtractor(BaseFeaturesExtractor):  # type: ignore[misc]
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        out_dim: int = 512,
        normalized_image: bool = False,
        hidden_dim: int = 256,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "circuit":
                extractors[key] = CustomCNN(
                    subspace,
                    out_dim,
                    normalized_image,
                    hidden_dim,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    num_layers,
                )
                total_concat_size += out_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class CustomCNN(BaseFeaturesExtractor):  # type: ignore[misc]
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        out_dim: int = 512,
        normalized_image: bool = False,
        hidden_dim: int = 256,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(observation_space, out_dim)

        if normalized_image:
            print("Normalized image is not supported yet.")

        qubit_num, n_input_channels = 25, 1
        self.cnn = None
        if n_input_channels != out_channels:
            self.cnn = nn.Conv2d(n_input_channels, out_channels, kernel_size, stride, padding)
        self.lstm = nn.LSTM(
            input_size=out_channels * qubit_num * qubit_num,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: list[th.Tensor] | th.Tensor) -> th.Tensor:
        cnn_outs, lengths = [], []

        for sample in x:  # sample in batch
            seq_len, C, H, W = sample.shape
            # batch, channel, height, width
            cnn_out = self.cnn(sample.float()) if self.cnn else sample.float()
            # seq_len, out_channels * height * width
            cnn_out = F.relu(cnn_out.view(seq_len, -1)) if self.cnn else cnn_out.view(seq_len, -1)

            cnn_outs.append(cnn_out)
            lengths.append(seq_len)

        # Sort sequences by length in descending order
        lengths, perm_idx = th.tensor(lengths).sort(0, descending=True)
        cnn_outs = [cnn_outs[i] for i in perm_idx]

        # Pack sequences
        packed_input = nn.utils.rnn.pack_sequence(cnn_outs, enforce_sorted=True)

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        return self.linear(attn_out[:, -1, :])
