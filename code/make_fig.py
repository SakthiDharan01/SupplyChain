import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("results.json"))
diag = json.load(open("diag.json"))
extra = json.load(open("extra.json"))

BLUE, ORANGE, GRAY, GREEN, INK, GRID = "#1f5fa8", "#c8620a", "#6b7280", "#2f7d4f", "#1f2937", "#e5e7eb"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


W, H = 3.45, 2.4
FIGS = {}


def newfig(name):
    fig, ax = plt.subplots(figsize=(W, H))
    FIGS[name] = fig
    return ax


def style(ax, title):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)



# (a) saving vs approval-only as a function of approval latency
ax = newfig("a")
ds = [0, 1, 2, 3]
sg, sn = [], []
for d in ds:
    v = res["delta"][str(d)]
    h, g, n = v["h"][0], v["g"][0], v["n"][0]
    sg.append(100 * (h - g) / h)
    sn.append(100 * (h - n) / h)
ax.plot(ds, sn, color=ORANGE, marker="s", ms=4.5, lw=1.4, ls="--", label="Unrestricted autonomy")
ax.plot(ds, sg, color=BLUE, marker="o", ms=4.5, lw=1.6, label="Governed autonomy")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xticks(ds)
ax.set_xlabel("Human approval latency $\\delta$ (days)")
ax.set_ylabel("Saving vs. approval-only (%)")
style(ax, "(a) Value of autonomy grows with latency")
ax.legend(frameon=False, loc="upper left", fontsize=6.8)

# (b) precision of autonomous action selection at matched coverage
ax = newfig("b")
covs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
pro = diag["pro"]
xs = [100 * c for c in covs]
ax.plot(xs, [pro["tail"][str(c)] for c in covs], color=BLUE, marker="o", ms=4.5, lw=1.6,
        label="Posterior tail-probability gate")
ax.plot(xs, [pro["ratio"][str(c)] for c in covs], color=ORANGE, marker="s", ms=4.5, lw=1.4, ls="--",
        label="Point-estimate margin gate")
ax.plot(xs, [pro["rand"][str(c)] for c in covs], color=GRAY, marker="^", ms=4, lw=1.1, ls=":",
        label="Random selection")
ax.set_xlabel("Coverage of eligible actions (%)")
ax.set_ylabel("Precision")
ax.set_ylim(0, 0.75)
style(ax, "(b) Selection precision at matched coverage")
ax.legend(frameon=False, loc="upper right", fontsize=6.8)

# (c) Brier gain of trust graph vs independent beliefs as shock amplitude grows
ax = newfig("c")
scs = ["0.0", "0.5", "1.0", "1.5", "2.0"]
rel, err = [], []
for sc in scs:
    v = res["corr"][sc]
    base = v["ind"][0]
    rel.append(-100 * v["diff"][0] / base)       # positive = improvement
    err.append(100 * v["diff"][1] / base)
xv = [float(s) for s in scs]
ax.errorbar(xv, rel, yerr=err, color=BLUE, marker="o", ms=4.5, lw=1.6, capsize=2.5, elinewidth=0.9)
ax.set_xticks(xv)
ax.set_xlabel("Regional shock amplitude (× default)")
ax.set_ylabel("Brier reduction (%)")
ax.set_ylim(0, 4.5)
style(ax, "(c) Gain grows with failure correlation")

# (d) cold start: supervised models trained on in-network history
ax = newfig("d")
Hs = [20, 40, 60, 90]
cold = extra["cold"]
gb = [cold["H"][str(h)]["gbm"][0] for h in Hs]
lr = [cold["H"][str(h)]["lr"][0] for h in Hs]
gbe = [cold["H"][str(h)]["gbm"][1] for h in Hs]
lre = [cold["H"][str(h)]["lr"][1] for h in Hs]
ax.errorbar(Hs, gb, yerr=gbe, color=ORANGE, marker="s", ms=4.5, lw=1.4, ls="--", capsize=2.5,
            elinewidth=0.9, label="Gradient boosting")
ax.errorbar(Hs, lr, yerr=lre, color=GREEN, marker="^", ms=4.5, lw=1.4, ls="-.", capsize=2.5,
            elinewidth=0.9, label="Logistic regression")
ax.axhline(cold["ref"]["pro"][0], color=BLUE, lw=1.6, label="Trust graph")
ax.axhline(cold["ref"]["ind"][0], color=GRAY, lw=1.1, ls=":", label="Independent Beta")
ax.set_xticks(Hs)
ax.set_xlabel("In-network history $H$ (days)")
ax.set_ylabel("Brier score")
ax.set_ylim(0.12, 0.168)
style(ax, "(d) Cold start: when supervised models catch up")
ax.legend(frameon=False, loc="upper right", fontsize=6.4, ncol=2, columnspacing=1.0, handlelength=2.2)

for k, f in FIGS.items():
    f.tight_layout(pad=0.4)
    f.savefig(f"fig_{k}.pdf")
    f.savefig(f"fig_{k}.png", dpi=200)
print("rel", [round(x, 2) for x in rel], "err", [round(x, 2) for x in err])
print("cold ref", cold["ref"]["pro"][0], cold["ref"]["ind"][0])
