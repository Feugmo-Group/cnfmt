#!/usr/bin/env python3
"""Compare Optimized (fixed A,B) vs CNN (learned A(η), B(η))."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def Z_CS(eta): return (1 + eta + eta**2 - eta**3) / (1 - eta)**3
def Z_PY(eta): return (1 + eta + eta**2) / (1 - eta)**3
def Z_lutsko(eta, A, B):
    C = 8*A + 2*B - 9
    return Z_PY(eta) + C * eta**2 / (3 * (1 - eta)**3)
def mu_ex_CS(eta): return eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3
def mu_ex_RF(eta): return -jnp.log(1 - eta) + eta * (14 - 13*eta + 5*eta**2) / (2 * (1 - eta)**3)
def mu_ex_lutsko(eta, A, B):
    C = 8*A + 2*B - 9
    return mu_ex_RF(eta) + C * eta**2 * (3 - eta) / (6 * (1 - eta)**3)

class ABNetwork(eqx.Module):
    layers: list
    def __init__(self, hidden=[32, 32], key=None):
        if key is None: key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, len(hidden) + 1)
        dims = [1] + hidden + [2]
        self.layers = [eqx.nn.Linear(d_in, d_out, key=k) for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)]
    def __call__(self, eta):
        x = jnp.atleast_1d(eta)
        for layer in self.layers[:-1]: x = jnp.tanh(layer(x))
        out = self.layers[-1](x)
        return 0.8 + 1.0*jax.nn.sigmoid(out[0]), -1.5 + 2.0*jax.nn.sigmoid(out[1])

def train_cnn(n_epochs=500):
    print("Training CNN...")
    eta_train = jnp.linspace(0.05, 0.45, 20)
    network = ABNetwork(key=jax.random.PRNGKey(42))
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    def loss_fn(net, eta):
        A, B = net(eta)
        return ((Z_lutsko(eta, A, B) - Z_CS(eta)) / Z_CS(eta))**2 + ((mu_ex_lutsko(eta, A, B) - mu_ex_CS(eta)) / (jnp.abs(mu_ex_CS(eta)) + 0.1))**2
    @eqx.filter_value_and_grad
    def batch_loss(net): return jnp.mean(jnp.array([loss_fn(net, eta) for eta in eta_train]))
    for epoch in range(n_epochs):
        loss_val, grads = batch_loss(network)
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), opt_state)
        network = eqx.apply_updates(network, updates)
        if epoch % 100 == 0: print(f"  Epoch {epoch}: loss = {loss_val:.4e}")
    return network

def plot_comparison(network):
    print("Generating comparison plot...")
    eta = np.linspace(0.01, 0.5, 100)
    A_OPT, B_OPT = 1.3, -1.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    A_cnn = [float(network(e)[0]) for e in eta]
    B_cnn = [float(network(e)[1]) for e in eta]
    axes[0,0].plot(eta, A_cnn, 'b-', lw=2, label='A(η) CNN')
    axes[0,0].plot(eta, B_cnn, 'r-', lw=2, label='B(η) CNN')
    axes[0,0].axhline(A_OPT, ls='--', color='b', alpha=0.7, label='A Opt')
    axes[0,0].axhline(B_OPT, ls='--', color='r', alpha=0.7, label='B Opt')
    axes[0,0].legend(); axes[0,0].set_title('(a) Parameters'); axes[0,0].grid(True, alpha=0.3)
    
    C_cnn = [8*A_cnn[i] + 2*B_cnn[i] - 9 for i in range(len(eta))]
    axes[0,1].plot(eta, C_cnn, 'g-', lw=2, label='C(η) CNN')
    axes[0,1].axhline(-0.6, ls='--', color='g', label='C Opt')
    axes[0,1].axhline(-1.0, ls=':', color='orange', label='C Lutsko')
    axes[0,1].legend(); axes[0,1].set_title('(b) C = 8A+2B-9'); axes[0,1].grid(True, alpha=0.3)
    
    Z_cs = [float(Z_CS(e)) for e in eta]
    Z_cnn = [float(Z_lutsko(e, *network(e))) for e in eta]
    Z_opt = [float(Z_lutsko(e, A_OPT, B_OPT)) for e in eta]
    axes[0,2].plot(eta, Z_cs, 'k-', lw=2, label='CS')
    axes[0,2].plot(eta, Z_cnn, 'b--', lw=2, label='CNN')
    axes[0,2].plot(eta, Z_opt, 'r:', lw=2, label='Opt')
    axes[0,2].legend(); axes[0,2].set_title('(c) Z'); axes[0,2].grid(True, alpha=0.3)
    
    mu_cs = [float(mu_ex_CS(e)) for e in eta]
    mu_cnn = [float(mu_ex_lutsko(e, *network(e))) for e in eta]
    mu_opt = [float(mu_ex_lutsko(e, A_OPT, B_OPT)) for e in eta]
    axes[1,0].plot(eta, mu_cs, 'k-', lw=2, label='CS')
    axes[1,0].plot(eta, mu_cnn, 'b--', lw=2, label='CNN')
    axes[1,0].plot(eta, mu_opt, 'r:', lw=2, label='Opt')
    axes[1,0].legend(); axes[1,0].set_title('(d) μ_ex'); axes[1,0].grid(True, alpha=0.3)
    
    err_Z_cnn = [abs(Z_cnn[i]-Z_cs[i])/Z_cs[i]*100 for i in range(len(eta))]
    err_Z_opt = [abs(Z_opt[i]-Z_cs[i])/Z_cs[i]*100 for i in range(len(eta))]
    axes[1,1].semilogy(eta, err_Z_cnn, 'b-', lw=2, label='CNN')
    axes[1,1].semilogy(eta, err_Z_opt, 'r--', lw=2, label='Opt')
    axes[1,1].legend(); axes[1,1].set_title('(e) Z Error %'); axes[1,1].grid(True, alpha=0.3)
    
    err_mu_cnn = [abs(mu_cnn[i]-mu_cs[i])/(abs(mu_cs[i])+0.1)*100 for i in range(len(eta))]
    err_mu_opt = [abs(mu_opt[i]-mu_cs[i])/(abs(mu_cs[i])+0.1)*100 for i in range(len(eta))]
    axes[1,2].semilogy(eta, err_mu_cnn, 'b-', lw=2, label='CNN')
    axes[1,2].semilogy(eta, err_mu_opt, 'r--', lw=2, label='Opt')
    axes[1,2].legend(); axes[1,2].set_title('(f) μ Error %'); axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle('Optimized (A=1.3, B=-1.0) vs CNN (A(η), B(η))', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimized_vs_cnn.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'optimized_vs_cnn.png'}")
    plt.close()

def main():
    print("="*60 + "\nCompare: Optimized vs CNN\n" + "="*60)
    network = train_cnn(500)
    plot_comparison(network)
    print("COMPLETE")

if __name__ == "__main__": main()
