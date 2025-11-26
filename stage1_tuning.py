import os

import numpy as np
import pandas as pd
import torch
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm

# ----- Force working directory to script location -----
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# -------------------------------------------------------

ID = 2

# sampler
folder = "Forward_Models"
sampler_path = os.path.join(folder, f"stage1_sampler_{ID}.pth")
globals()[f"stage1_sampler_{ID}"] = torch.load(sampler_path)


# final stage 1
def NF_sample_generator_torch(num_samples, device, model):
    z, _ = model.sample(num_samples)
    return z.to(device, dtype=torch.float32)


def train_final_stage1(
    id,
    max_iter,
    num_samples,
    I=5,
    K=10,
    hidden_layers=2,
    hidden_units=128,
    tail_bound=0.3,
    num_bins=8,
    lr=1e-3,
    weight_decay=1e-6,
    show=False,
    save=True,
):
    base_dir = f"ID_{id}_Final_Stage1"
    loss_dir = os.path.join(base_dir, "Loss")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    globals().pop("nfm", None)
    globals().pop("loss_hist", None)

    sampler_model = globals()[f"stage1_sampler_{id}"]
    sampler_model.eval()

    torch.manual_seed(0)
    latent_size = I

    flows = []
    for _ in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_input_channels=latent_size,
                num_blocks=hidden_layers,
                num_hidden_channels=hidden_units,
                num_bins=num_bins,
                tail_bound=tail_bound,
                init_identity=True,
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(I)

    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfm = nfm.to(device).float()

    optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=weight_decay)
    loss_hist = np.array([])

    for _it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        x_tensor = NF_sample_generator_torch(
            num_samples=num_samples, device=device, model=sampler_model
        )

        loss = nfm.forward_kld(x_tensor)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save:
        # Build base model file name (without directory)
        model_file_name = (
            f"id{id}_model_"
            f"maxIter{max_iter}_"
            f"numSamples{num_samples}_"
            f"K{K}_"
            f"Layers{hidden_layers}_Units{hidden_units}_"
            f"tail{tail_bound}_bins{num_bins}_"
            f"lr{lr}_wd{weight_decay}.pth"
        )
        model_path_full = os.path.join(model_dir, model_file_name)
        model_state_file = model_file_name.replace(".pth", "_dict.pth")
        model_state_path = os.path.join(model_dir, model_state_file)

        torch.save(nfm, model_path_full)
        torch.save(nfm.state_dict(), model_state_path)


    plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, label="loss")
    plt.xlabel("Iteration")
    plt.ylabel("Forward KL Loss")
    plt.title(f"ID = {id}, Training Loss")
    plt.legend()

    if save:
        loss_filename = (
            f"id{id}_loss_"
            f"maxIter{max_iter}_"
            f"numSamples{num_samples}_"
            f"K{K}_"
            f"Layers{hidden_layers}_Units{hidden_units}_"
            f"tail{tail_bound}_bins{num_bins}_"
            f"lr{lr}_wd{weight_decay}.png"
        )
        plot_path = os.path.join(loss_dir, loss_filename)
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    return nfm


max_iter = 15000
num_samples = 300
K_val = 5
hidden_layers_val = 2
hidden_units_val = 4
tail_bound_val = 0.3
num_bins_val = 4
lr_val = 1e-3
weight_decay_val = 1e-5

model_name = f"stage1_final_{ID}"
globals()[model_name] = train_final_stage1(
    id=ID,
    max_iter=max_iter,
    num_samples=num_samples,
    I=5,
    K=K_val,
    hidden_layers=hidden_layers_val,
    hidden_units=hidden_units_val,
    tail_bound=tail_bound_val,
    num_bins=num_bins_val,
    lr=lr_val,
    weight_decay=weight_decay_val,
    show=False,
    save=True,
)

torch.manual_seed(0)
np.random.seed(0)
model = globals()[model_name]
model.eval()
nf_samples, _ = model.sample(5000)


nf_samples_np = nf_samples.detach().cpu().numpy()
columns = ['logAge', 'FeH', 'parallax', 'absorption', 'mass']
df_nf = pd.DataFrame(nf_samples_np, columns=columns)


base_dir = f"ID_{ID}_Final_Stage1"
nf_samples_dir = os.path.join(base_dir, "nf_samples")
os.makedirs(nf_samples_dir, exist_ok=True)

nf_samples_filename = (
    f"id{ID}_nfSamples_"
    f"maxIter{max_iter}_"
    f"numSamples{num_samples}_"
    f"K{K_val}_"
    f"HL{hidden_layers_val}_HU{hidden_units_val}_"
    f"tail{tail_bound_val}_bins{num_bins_val}_"
    f"lr{lr_val}_wd{weight_decay_val}.csv"
)
nf_samples_path = os.path.join(nf_samples_dir, nf_samples_filename)

df_nf.to_csv(nf_samples_path, index=False)