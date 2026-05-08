from __future__ import annotations

import torch

from paper2601_splitmae_client import Paper2601SplitMAEClient, SplitMAEClientConfig
from paper2601_splitmae_server import Paper2601SplitMAEServer, SplitMAEServerConfig
from paper2601_splitmae_utils import labels_for_bce


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_cfg = SplitMAEClientConfig(
        input_fdim=128,
        input_tdim=64,
        model_size="tiny",
        n_client_blocks=1,
        mask_ratio=0.75,
    )
    server_cfg = SplitMAEServerConfig(
        input_fdim=128,
        input_tdim=64,
        model_size="tiny",
        n_client_blocks=1,
        decoder_embed_dim=256,
        decoder_depth=1,
        decoder_num_heads=4,
        num_labels=1,
        static_feature_dim=131,
    )

    client = Paper2601SplitMAEClient(client_cfg).to(device)
    server = Paper2601SplitMAEServer(server_cfg).to(device)

    x = torch.randn(2, 64, 128, device=device)
    static = torch.randn(2, 131, device=device)
    labels = torch.tensor([0, 1], device=device)

    smashed_pre = client(x, mode="pretrain")
    pre_out = server(smashed_pre.to(device))
    pre_out["loss"].backward()

    client.zero_grad(set_to_none=True)
    server.zero_grad(set_to_none=True)

    smashed_ft = client(x, mode="finetune", static_features=static)
    ft_out = server(smashed_ft.to(device))
    target = labels_for_bce(labels, ft_out["logits"].shape[-1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(ft_out["logits"], target)
    loss.backward()

    print(
        {
            "pretrain_loss": float(pre_out["loss"].detach().cpu()),
            "finetune_logits_shape": tuple(ft_out["logits"].shape),
            "smashed_pre_tokens": tuple(smashed_pre.tokens.shape),
            "smashed_ft_tokens": tuple(smashed_ft.tokens.shape),
        }
    )


if __name__ == "__main__":
    main()
