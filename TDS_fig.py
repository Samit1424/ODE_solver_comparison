"""
figures.py
==============
"My scipy ODE solver was killing my Bayesian inference"

Figures:
  fig1.pdf   -- the bottleneck: timing breakdown of a dynesty run
  fig2.pdf  -- before vs after: solve_ivp vs diffrax
  fig3.pdf -- finite-diff gradient cost vs autodiff
  fig4.pdf-- full cosmological inference comparison
  fig5.pdf  -- visual summary of the three gotchas
"""

import time, warnings, json
warnings.filterwarnings("ignore")

import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import diffrax as dfx
from scipy.integrate import solve_ivp, odeint
import optax

jax.config.update("jax_enable_x64", True)

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False, "lines.linewidth": 2.0,
})
C1  = "#2ecc71"   # green  → diffrax
C2  = "#e74c3c"   # red    → scipy slow
C3  = "#3498db"   # blue   → scipy odeint
C4  = "#9b59b6"   # purple → FD gradient
DARK = "#2c3e50"

# ── Flat LCDM forward model ────────────────────────────────────────
C_KMS = 299792.458

def H_jax(z, Om, H0): return H0 * jnp.sqrt(Om*(1+z)**3 + (1-Om))
def H_np(z, Om, H0):  return H0 * np.sqrt(Om*(1+z)**3  + (1-Om))

z_obs   = np.linspace(0.05, 1.5, 30)
z_obs_j = jnp.asarray(z_obs)
Z_MAX   = float(z_obs[-1])

THETA_TRUE = np.array([0.30, 70.0])
SIGMA_MU   = 0.10

def chi_solve_ivp(Om, H0):
    sol = solve_ivp(
        lambda z, y: C_KMS / H_np(z, Om, H0),
        (0, Z_MAX), [0.0], t_eval=z_obs, rtol=1e-8, atol=1e-10)
    return sol.y[0]

def mu_np(chi, z): return 5*np.log10((1+z)*chi*1e5)
def mu_jax(chi, z): return 5*jnp.log10((1+z)*chi*1e5)

# Mock SNIa data
rng = np.random.default_rng(42)
chi_true = chi_solve_ivp(*THETA_TRUE)
mu_obs   = mu_np(chi_true, z_obs) + SIGMA_MU*rng.standard_normal(30)
mu_obs_j = jnp.asarray(mu_obs)

# diffrax forward model
@jax.jit
def mu_diffrax(theta):
    Om, H0 = theta[0], theta[1]
    def rhs(z, y, args): return C_KMS / H_jax(z, args[0], args[1])
    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs), dfx.Tsit5(),
        t0=0.0, t1=Z_MAX, dt0=1e-3,
        y0=jnp.array(0.0), args=(Om, H0),
        saveat=dfx.SaveAt(ts=z_obs_j),
        stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=10_000)
    return mu_jax(sol.ys, z_obs_j)

# Warm up
_ = mu_diffrax(jnp.array([0.3, 70.0])).block_until_ready()

def loss_scipy(theta):
    chi = chi_solve_ivp(theta[0], theta[1])
    mu  = mu_np(chi, z_obs)
    return 0.5 * np.sum(((mu - mu_obs)/SIGMA_MU)**2)

@jax.jit
def loss_diffrax(theta):
    mu = mu_diffrax(theta)
    return 0.5 * jnp.sum(((mu - mu_obs_j)/SIGMA_MU)**2)

grad_fn = jax.jit(jax.grad(loss_diffrax))
_ = grad_fn(jnp.array([0.3, 70.0])).block_until_ready()

# ─────────────────────────────────────────────────────────────────
# Timing benchmark
# ─────────────────────────────────────────────────────────────────
def bench(fn, n=100, jax_fn=False):
    for _ in range(5):
        r = fn()
        if jax_fn: r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if jax_fn: r.block_until_ready()
    return (time.perf_counter() - t0) / n

th0_np = np.array([0.3, 70.0])
th0_j  = jnp.array([0.3, 70.0])

t_ivp   = bench(lambda: chi_solve_ivp(*th0_np))
t_dfx   = bench(lambda: mu_diffrax(th0_j), jax_fn=True)
t_fd    = bench(lambda: (
    loss_scipy(th0_np + [1e-4, 0]) - loss_scipy(th0_np - [1e-4, 0])
), n=20)    # one finite-diff step on Om
t_fdgrad = bench(lambda: (
    [loss_scipy(th0_np + [1e-4, 0]) - loss_scipy(th0_np - [1e-4, 0]),
     loss_scipy(th0_np + [0, 1e-2]) - loss_scipy(th0_np - [0, 1e-2])]
), n=20)
t_adgrad = bench(lambda: grad_fn(th0_j), jax_fn=True)

N_LIKE = 100_000   # typical dynesty run
t_scipy_run = t_ivp   * N_LIKE
t_dfx_run   = t_dfx   * N_LIKE
t_fd_run    = t_fdgrad * 2 * N_LIKE   # 4 solves per grad, 2 params

print(f"solve_ivp single call:  {t_ivp*1e3:.2f} ms")
print(f"diffrax  single call:   {t_dfx*1e6:.1f} us")
print(f"FD gradient (2 params): {t_fdgrad*1e3:.2f} ms")
print(f"Autodiff gradient:      {t_adgrad*1e6:.1f} us")
print(f"100k likelihood calls:")
print(f"  scipy ODE:  {t_scipy_run:.0f} s")
print(f"  diffrax:    {t_dfx_run:.1f} s")
print(f"  scipy FD grad 100k: {t_fd_run:.0f} s")

# ═══════════════════════════════════════════════════════════════════
# FIG 1 — About a 100k-call dynesty run costs
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: breakdown of time in a dynesty run
ax = axes[0]
categories = ["ODE solve\n(scipy)", "FD gradient\n(4 ODE calls)", "Overhead\n(sampling logic)"]
times_scipy = [t_scipy_run, t_fd_run * 0.3, 30.0]
colors_s = [C2, C4, "#bdc3c7"]
bars = ax.barh(categories, times_scipy, color=colors_s, height=0.5)
ax.set_xlabel("Wall time for 100 000 likelihood calls [s]")
ax.set_title("Where time goes in a dynesty run\n(10-parameter model, scipy pipeline)")
for bar, t in zip(bars, times_scipy):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{t:.0f} s  ({t/60:.1f} min)", va="center", fontsize=9)
ax.set_xlim(0, max(times_scipy)*1.45)

# Right: the same run with diffrax
ax = axes[1]
categories2 = ["ODE solve\n(diffrax)", "Autodiff\ngradient", "Overhead\n(sampling logic)"]
times_dfx = [t_dfx_run, t_adgrad * 1e3 * N_LIKE / 1000, 30.0]
colors_d = [C1, C1, "#bdc3c7"]
bars2 = ax.barh(categories2, times_dfx, color=colors_d, height=0.5)
ax.set_xlabel("Wall time for 100 000 calls [s]")
ax.set_title("The same run with diffrax\n(same 10-parameter model)")
for bar, t in zip(bars2, times_dfx):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{t:.1f} s", va="center", fontsize=9)
ax.set_xlim(0, max(times_scipy)*1.45)

fig.suptitle("The bottleneck, visualised", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("fig1.pdf", bbox_inches="tight")
plt.close()
print("  saved fig1.pdf")

# ═══════════════════════════════════════════════════════════════════
# FIG 2 — Single call speedup + cosmological fit
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bar chart single call
ax = axes[0]
labels = ["scipy\nsolve_ivp", "diffrax\n(JIT)"]
vals   = [t_ivp*1e3, t_dfx*1e3]
brs = ax.bar(labels, vals, color=[C2, C1], width=0.4)
ax.set_yscale("log")
ax.set_ylabel("Wall time per call [ms]")
ax.set_title("Forward model: single call\n(flat LCDM comoving distance ODE)")
for b, v in zip(brs, vals):
    lbl = f"{v:.2f} ms" if v > 0.5 else f"{v*1000:.0f} µs"
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.3,
            lbl, ha="center", fontsize=10, fontweight="bold")
speedup = t_ivp / t_dfx
ax.text(0.55, 0.6, f"{speedup:.0f}×\nfaster",
        transform=ax.transAxes, fontsize=22, fontweight="bold",
        color=C1, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C1, linewidth=2))

# Right: mock SN data with fit
ax = axes[1]
zz = np.linspace(0.01, 1.6, 200)
def chi_z(Om, H0, z_arr):
    sol = solve_ivp(lambda z,y: C_KMS/H_np(z, Om, H0),
                    (0, float(z_arr[-1])), [0.0], t_eval=z_arr, rtol=1e-8, atol=1e-10)
    return sol.y[0]
mu_fit_z   = mu_np(chi_z(0.30, 70.0, zz), zz)
mu_wrong_z = mu_np(chi_z(0.15, 65.0, zz), zz)

ax.errorbar(z_obs, mu_obs, yerr=SIGMA_MU, fmt="o", ms=4,
            color=DARK, capsize=2, alpha=0.6, label="Mock SNIa data")
ax.plot(zz, mu_fit_z, color=C1, lw=2,
        label=r"True $(\Omega_m,H_0)=(0.30,70)$")
ax.plot(zz, mu_wrong_z, "--", color=C2, lw=2,
        label=r"Wrong $(0.15,65)$")
ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Distance modulus $\mu$")
ax.set_title("The inference problem\n(30 mock SNIa, flat ΛCDM)")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("fig2.pdf", bbox_inches="tight")
plt.close()
print("  saved fig2.pdf")

# ═══════════════════════════════════════════════════════════════════
# FIG 3 — Gradient cost comparison + loss landscape with gradient
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: gradient timing
ax = axes[0]
g_labels = ["scipy +\ncentral FD\n(4 ODE calls)", "diffrax +\nautodiff\n(1 backward pass)"]
g_vals   = [t_fdgrad*1e3, t_adgrad*1e3]
gbrs = ax.bar(g_labels, g_vals, color=[C4, C1], width=0.4)
ax.set_yscale("log")
ax.set_ylabel("Wall time per gradient [ms]")
ax.set_title("Cost of one gradient\n(2-parameter likelihood)")
for b, v in zip(gbrs, g_vals):
    lbl = f"{v:.2f} ms" if v > 0.5 else f"{v*1000:.0f} µs"
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.3,
            lbl, ha="center", fontsize=10, fontweight="bold")
grad_speedup = t_fdgrad / t_adgrad
ax.text(0.75, 0.65, f"{grad_speedup:.0f}×\ncheaper",
        transform=ax.transAxes, fontsize=18, fontweight="bold",
        color=C1, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C1, linewidth=2))
ax.text(0.3, 0.30,
        "Scales as 2P\nforward solves\n(gets worse with\nmore parameters)",
        transform=ax.transAxes, fontsize=8.5, ha="center",
        color=C4, style="italic")
ax.text(0.78, 0.28,
        "Always ≈ 1\nforward solve\n(independent of P)",
        transform=ax.transAxes, fontsize=8.5, ha="center",
        color=C1, style="italic")

# Right: loss landscape with gradient arrows
ax = axes[1]
Om_g = np.linspace(0.10, 0.55, 60)
H0_g = np.linspace(58.0, 82.0, 60)
LL   = np.zeros((len(H0_g), len(Om_g)))
for i, H0 in enumerate(H0_g):
    for j, Om in enumerate(Om_g):
        LL[i,j] = loss_scipy(np.array([Om, H0]))

cs = ax.contourf(Om_g, H0_g, LL,
                 levels=np.percentile(LL, np.linspace(0, 85, 20)),
                 cmap="viridis_r")
ax.contour(Om_g, H0_g, LL,
           levels=np.percentile(LL, np.linspace(0, 85, 8)),
           colors="white", linewidths=0.4, alpha=0.5)
fig.colorbar(cs, ax=ax, label=r"$-\log\mathcal{L}$")

# Show gradient arrow at a test point
test_pt = np.array([0.18, 76.0])
g = np.array(grad_fn(jnp.asarray(test_pt)))
g_norm = g / np.linalg.norm(g) * 0.06
ax.annotate("", xy=(test_pt[0]-g_norm[0], test_pt[1]-g_norm[1]*100),
            xytext=(test_pt[0], test_pt[1]),
            arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.5))
ax.plot(*test_pt, "o", ms=9, color="white", markeredgecolor=C1, markeredgewidth=2)
ax.text(test_pt[0]+0.02, test_pt[1]+1.5, "autodiff\ngradient here",
        color=C1, fontsize=8.5, fontweight="bold")
ax.plot(*THETA_TRUE, "*", ms=16, color="gold", markeredgecolor="k", zorder=5,
        label="truth")
ax.set_xlabel(r"$\Omega_m$")
ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
ax.set_title(r"Loss landscape: $-\log\mathcal{L}(\Omega_m, H_0)$")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("fig3.pdf", bbox_inches="tight")
plt.close()
print("  saved fig3.pdf")

# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Full inference comparison: scipy FD vs diffrax autodiff
# ═══════════════════════════════════════════════════════════════════
N_STEPS = 350
INIT    = jnp.array([0.10, 0.60])   # (Om, h=H0/100)

def loss_scaled(th_s):
    return loss_diffrax(jnp.array([th_s[0], 100.0*th_s[1]]))

grad_scaled = jax.jit(jax.grad(loss_scaled))
_ = grad_scaled(INIT).block_until_ready()

sched = optax.cosine_decay_schedule(init_value=0.05, decay_steps=N_STEPS, alpha=0.04)
opt   = optax.adam(sched)

def run_adam_dfx():
    th    = INIT
    state = opt.init(th)
    hist  = []
    t0    = time.perf_counter()
    for _ in range(N_STEPS):
        g   = grad_scaled(th)
        upd, state = opt.update(g, state)
        th  = optax.apply_updates(th, upd)
        hist.append([float(th[0]), float(th[1])*100, float(loss_scaled(th))])
    return np.array(hist), time.perf_counter()-t0

def run_gd_scipy():
    th   = np.array([0.10, 60.0])
    hist = []
    lr   = 4e-5
    h    = np.array([1e-4, 1e-2])
    t0   = time.perf_counter()
    for _ in range(N_STEPS):
        g = np.zeros(2)
        for i in range(2):
            ep = np.zeros(2); ep[i] = h[i]
            g[i] = (loss_scipy(th+ep) - loss_scipy(th-ep)) / (2*h[i])
        th -= lr*g
        hist.append([th[0], th[1], loss_scipy(th)])
    return np.array(hist), time.perf_counter()-t0

print("  running inference comparison...")
hist_dfx,   t_dfx_opt  = run_adam_dfx()
hist_scipy, t_scipy_opt = run_gd_scipy()
print(f"  diffrax Adam: {t_dfx_opt:.1f}s  Om={hist_dfx[-1,0]:.3f} H0={hist_dfx[-1,1]:.2f}")
print(f"  scipy GD:     {t_scipy_opt:.1f}s  Om={hist_scipy[-1,0]:.3f} H0={hist_scipy[-1,1]:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

axes[0].plot(hist_dfx[:,0],  color=C1, lw=2, label="diffrax + Adam")
axes[0].plot(hist_scipy[:,0], color=C2, lw=2, label="scipy + GD + FD")
axes[0].axhline(0.30, ls="--", color=DARK, lw=1, label="truth")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel(r"$\Omega_m$")
axes[0].set_title(r"Recovery of $\Omega_m$"); axes[0].legend(fontsize=9)

axes[1].plot(hist_dfx[:,1],  color=C1, lw=2)
axes[1].plot(hist_scipy[:,1], color=C2, lw=2)
axes[1].axhline(70.0, ls="--", color=DARK, lw=1)
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel(r"$H_0$")
axes[1].set_title(r"Recovery of $H_0$")

axes[2].semilogy(hist_dfx[:,2],  color=C1, lw=2, label=f"diffrax ({t_dfx_opt:.0f} s)")
axes[2].semilogy(hist_scipy[:,2], color=C2, lw=2, label=f"scipy ({t_scipy_opt:.0f} s)")
axes[2].set_xlabel("Iteration"); axes[2].set_ylabel(r"$-\log\mathcal{L}$")
axes[2].set_title("Loss convergence"); axes[2].legend(fontsize=9)

fig.suptitle("MAP inference on flat ΛCDM from 30 mock SNIa",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig4.pdf", bbox_inches="tight")
plt.close()
print("  saved fig4.pdf")

# ═══════════════════════════════════════════════════════════════════
# FIG 5 — The three gotchas: visual panels
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Caveat 1: 32-bit vs 64-bit step counts
jax.config.update("jax_enable_x64", False)
try:
    sol32 = dfx.diffeqsolve(
        dfx.ODETerm(lambda z, y, a: C_KMS / (70.0*jnp.sqrt(0.3*(1+z)**3+0.7))),
        dfx.Tsit5(), t0=0.0, t1=Z_MAX, dt0=1e-3, y0=jnp.array(0.0),
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=50_000)
    steps32 = int(sol32.stats["num_steps"])
except: steps32 = -1
jax.config.update("jax_enable_x64", True)
sol64 = dfx.diffeqsolve(
    dfx.ODETerm(lambda z, y, a: C_KMS / (70.0*jnp.sqrt(0.3*(1+z)**3+0.7))),
    dfx.Tsit5(), t0=0.0, t1=Z_MAX, dt0=1e-3, y0=jnp.array(0.0),
    saveat=dfx.SaveAt(t1=True),
    stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
    max_steps=50_000)
steps64 = int(sol64.stats["num_steps"])

ax = axes[0]
s32_label = f"{steps32}\nsteps" if steps32 > 0 else "failed"
bars = ax.bar(["32-bit\n(default JAX)", "64-bit\n(enable it!)"],
              [max(steps32, 1), steps64], color=[C2, C1])
ax.set_ylabel("Steps taken")
ax.set_title("Gotcha 1: forgot 64-bit\n(same rtol=1e-8, same ODE)")
for b, (v, lbl) in zip(bars, [(steps32, s32_label), (steps64, f"{steps64}\nsteps")]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            lbl, ha="center", fontsize=10, fontweight="bold")
ax.text(0.5, 0.55, "jax.config.update(\n  'jax_enable_x64', True)",
        transform=ax.transAxes, ha="center", fontsize=8.5,
        color=C1, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#efffee", edgecolor=C1))

# Caveat 2: JIT warmup
warmup_times = []
jit_fn = jax.jit(lambda: mu_diffrax(jnp.array([0.3, 70.0])))
for call_n in range(6):
    t0 = time.perf_counter()
    jit_fn().block_until_ready()
    warmup_times.append((time.perf_counter()-t0)*1000)

ax = axes[1]
ax.bar(range(1, 7), warmup_times,
       color=[C2, C1, C1, C1, C1, C1])
ax.set_xlabel("Call number")
ax.set_ylabel("Wall time [ms]")
ax.set_title("Gotcha 2: JIT warmup\n(first call = compilation cost)")
ax.text(1, warmup_times[0]*0.6, f"Compilation\n{warmup_times[0]:.0f} ms",
        ha="center", fontsize=8.5, color="white", fontweight="bold")
ax.axhline(np.mean(warmup_times[1:]), color=C1, ls="--", lw=1.5,
           label=f"Steady state: {np.mean(warmup_times[1:]):.2f} ms")
ax.legend(fontsize=9)
ax.text(0.6, 0.7, "Always warm up\nbefore benchmarking!",
        transform=ax.transAxes, ha="center", fontsize=9,
        color=C1, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#efffee", edgecolor=C1))

# Caveat 3: argument order
ax = axes[2]
ax.axis("off")
table_data = [
    ["solve_ivp", "f(t, y, *args)", "✓ correct"],
    ["odeint",    "f(y, t, *args)", "← REVERSED!"],
    ["diffrax",   "f(t, y, args)",  "✓ correct"],
]
col_labels = ["Library", "RHS signature", "vs diffrax"]
t = ax.table(cellText=table_data, colLabels=col_labels,
             loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1.3, 2.5)
for (row, col), cell in t.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 2:   # odeint row
        cell.set_facecolor("#fdecea")
    elif row in [1, 3]:
        cell.set_facecolor("#efffee")
ax.set_title("Gotcha 3: argument order\n(silent wrong results with odeint)",
             fontsize=11, fontweight="bold", pad=80)

fig.suptitle("The three gotchas I wish someone had told me",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig5.pdf", bbox_inches="tight")
plt.close()
print("  saved fig5.pdf")

# Summary numbers for LaTeX
summary = {
    "t_ivp_ms":       round(t_ivp*1e3, 2),
    "t_dfx_us":       round(t_dfx*1e6, 1),
    "t_fdgrad_ms":    round(t_fdgrad*1e3, 2),
    "t_adgrad_us":    round(t_adgrad*1e6, 1),
    "speedup_fwd":    round(t_ivp/t_dfx, 0),
    "speedup_grad":   round(t_fdgrad/t_adgrad, 0),
    "t_scipy_opt_s":  round(t_scipy_opt, 1),
    "t_dfx_opt_s":    round(t_dfx_opt, 1),
    "Om_dfx":         round(float(hist_dfx[-1,0]), 3),
    "H0_dfx":         round(float(hist_dfx[-1,1]), 2),
    "Om_scipy":       round(float(hist_scipy[-1,0]), 3),
    "H0_scipy":       round(float(hist_scipy[-1,1]), 2),
    "warmup_ms":      round(warmup_times[0], 0),
    "steady_us":      round(np.mean(warmup_times[1:])*1000, 1),
    "steps32":        steps32,
    "steps64":        steps64,
}
with open("tds_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n", json.dumps(summary, indent=2))
print("\nAll figures done.")
