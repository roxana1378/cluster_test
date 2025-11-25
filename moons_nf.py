import os
import time

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (good for cluster)

import torch
import numpy as np
import normflows as nf
import pandas as pd

from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------
# Setup: folders and timing
# ----------------------------------------------------------------------
start_time = time.time()

fig_dir = "moons_Figures"
data_dir = "data"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Set up model
# ----------------------------------------------------------------------
K = 16
torch.manual_seed(0)

latent_size = 2
hidden_units = 128
hidden_layers = 2

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(2, trainable=False)

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)

# ----------------------------------------------------------------------
# Plot target distribution
# ----------------------------------------------------------------------
x_np, _ = make_moons(2 ** 20, noise=0.1)
plt.figure(figsize=(7, 7))
plt.hist2d(x_np[:, 0], x_np[:, 1], bins=200)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Target distribution (moons)")
plt.savefig(os.path.join(fig_dir, "target_distribution.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Plot initial flow distribution
# ----------------------------------------------------------------------
grid_size = 100
xx, yy = torch.meshgrid(
    torch.linspace(-1.5, 2.5, grid_size),
    torch.linspace(-2, 2, grid_size),
    indexing="ij"  # explicit indexing for clarity
)
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

nfm.eval()
log_prob = nfm.log_prob(zz).to("cpu").view(*xx.shape)
nfm.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(7, 7))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.gca().set_aspect("equal", "box")
plt.title("Initial flow density")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig(os.path.join(fig_dir, "initial_flow.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Train model
# ----------------------------------------------------------------------
max_iter = 10000
num_samples = 2 ** 9
show_iter = 1000

loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x_np, _ = make_moons(num_samples, noise=0.1)
    x = torch.tensor(x_np).float().to(device)
    
    # Compute loss
    loss = nfm.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())
    
    # Plot learned distribution occasionally
    if (it + 1) % show_iter == 0:
        nfm.eval()
        log_prob = nfm.log_prob(zz)
        nfm.train()
        prob = torch.exp(log_prob.to("cpu").view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(5, 5))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect("equal", "box")
        plt.title(f"Flow density at iteration {it + 1}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig(os.path.join(fig_dir, f"flow_iter_{it + 1}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

# ----------------------------------------------------------------------
# Plot loss
# ----------------------------------------------------------------------
plt.figure(figsize=(5, 5))
plt.plot(loss_hist, label="loss")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss history")
plt.savefig(os.path.join(fig_dir, "loss_history.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Plot final learned distribution
# ----------------------------------------------------------------------
nfm.eval()
log_prob = nfm.log_prob(zz).to("cpu").view(*xx.shape)
nfm.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(7, 7))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.gca().set_aspect("equal", "box")
plt.title("Final learned flow density")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig(os.path.join(fig_dir, "final_flow.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Sample from the trained NF and save to CSV
# ----------------------------------------------------------------------
n_samples_nf = 2 ** 15  # you can change this if you want more/less samples

with torch.no_grad():
    # Depending on normflows version, sample() might return just samples
    # or (samples, log_q). Adjust if needed.
    nf_samples = nfm.sample(n_samples_nf)
    if isinstance(nf_samples, tuple):
        nf_samples = nf_samples[0]

nf_samples = nf_samples.to("cpu").numpy()
dim = nf_samples.shape[1]

columns = [f"x{i+1}" for i in range(dim)]
df_nf = pd.DataFrame(nf_samples, columns=columns)

csv_path = os.path.join(data_dir, "nf_samples.csv")
df_nf.to_csv(csv_path, index=False)

# ----------------------------------------------------------------------
# Print execution time
# ----------------------------------------------------------------------
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
print(f"NF samples saved to: {csv_path}")
print(f"Figures saved in folder: {fig_dir}")