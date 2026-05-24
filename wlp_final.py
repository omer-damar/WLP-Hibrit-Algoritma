"""
============================================================
Depo Yeri Secimi Problemi (Warehouse Location Problem - WLP)
Hibrit Yaklasim: K-Means Kumeleme + Greedy Algoritma
Karsilastirma: Greedy-Only | Random Search | Integer Programming (IP)

Yazarlar : Omer Faruk DAMAR, Mehmet Ali AVCI
Universite: Manisa Celal Bayar Universitesi
Fakulte  : Hasan Ferdi Turgutlu Teknoloji Fakultesi
Bolum    : Yazilim Muhendisligi
Ders     : Algoritma Analizi ve Tasarimi - 2025/2026 Bahar
============================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pulp import (LpProblem, LpMinimize, LpVariable, lpSum,
                  value, PULP_CBC_CMD, LpStatus)
import random
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# RENK & STİL TANIMLARI
# ─────────────────────────────────────────────────────────
C_GREEDY = "#388E3C"
C_HYBRID = "#1976D2"
C_RANDOM = "#E64A19"
C_IP     = "#6A1B9A"

ALG_LABELS = {
    "greedy": "Greedy-Only",
    "hybrid": "K-Means+Greedy (Hibrit)",
    "random": "Random Search",
    "ip":     "Integer Programming (IP)",
}

# ─────────────────────────────────────────────────────────
# 1. VERİ OLUŞTURMA
# ─────────────────────────────────────────────────────────

def generate_dataset(n_customers, n_warehouses, seed=42):
    """
    [0,100]x[0,100] duzleminde sentetik WLP veri seti uretir.
    Musteriler 5 dogal kume etrafinda Gaussian dagilir.
    Kapasite toplam talebin en az 3 katı olacak sekilde ayarlanir,
    boylece K=2 bile feasible olur.
    """
    rng = np.random.default_rng(seed)
    n_centers = 5
    centers = rng.uniform(15, 85, (n_centers, 2))

    customer_list = []
    per = n_customers // n_centers
    added = 0
    for i, c in enumerate(centers):
        cnt = per if i < n_centers - 1 else n_customers - added
        pts = rng.normal(loc=c, scale=10, size=(cnt, 2))
        pts = np.clip(pts, 0, 100)
        customer_list.append(pts)
        added += cnt
    customers = np.vstack(customer_list)

    warehouses = rng.uniform(5, 95, (n_warehouses, 2))
    demand     = rng.integers(5, 20, n_customers).astype(float)

    # Toplam talep
    total_demand = demand.sum()
    # Her deponun kapasitesi: toplam talebin 3 katını n_warehouses'a böl
    # Boylece her K icin feasible cozum garantili
    base_cap = (total_demand * 3.0) / n_warehouses
    capacity = rng.uniform(base_cap * 0.8, base_cap * 1.2, n_warehouses)

    open_cost = rng.uniform(500, 2000, n_warehouses)

    dist = np.linalg.norm(
        customers[:, None, :] - warehouses[None, :, :], axis=2
    )

    return dict(
        customers=customers, warehouses=warehouses,
        demand=demand, capacity=capacity,
        open_cost=open_cost, dist=dist,
        n_c=n_customers, n_w=n_warehouses,
    )


# ─────────────────────────────────────────────────────────
# 2. MALİYET FONKSİYONU
# ─────────────────────────────────────────────────────────

def compute_cost(selected, data, penalty_coef=1e5):
    """
    Toplam maliyet = sabit acilis maliyeti + agirlikli tasima maliyeti.
    Kapasite ihlali buyuk ceza ile engellenir.
    """
    if not selected:
        return float("inf"), {}

    idx  = list(selected)
    sub  = data["dist"][:, idx]
    near = np.argmin(sub, axis=1)
    n_c  = sub.shape[0]

    transport = float(np.sum(data["demand"][:n_c] * sub[np.arange(n_c), near]))
    fixed     = float(np.sum(data["open_cost"][idx]))

    penalty = 0.0
    for local_j, global_j in enumerate(idx):
        load   = float(data["demand"][near == local_j].sum())
        excess = max(0.0, load - data["capacity"][global_j])
        penalty += excess * penalty_coef

    cost   = transport + fixed + penalty
    detail = dict(transport=round(transport, 1),
                  fixed=round(fixed, 1),
                  penalty=round(penalty, 1))
    return cost, detail


# ─────────────────────────────────────────────────────────
# 3. ALGORİTMALAR
# ─────────────────────────────────────────────────────────

def greedy_only(k, data):
    """
    Tum aday depolar (acilis_maliyeti / kapasite) oranina gore siralenir;
    en dusuk orana sahip K depo secilir.
    Zaman karmasikligi: O(n_w log n_w)
    """
    scores = data["open_cost"] / (data["capacity"] + 1e-9)
    idx    = np.argsort(scores)[:k]
    sel    = set(int(j) for j in idx)
    cost, detail = compute_cost(sel, data)
    return sorted(sel), round(cost, 1), detail


def kmeans_greedy(k, data, n_init=15):
    """
    Asama 1: K-Means ile musterileri K kumeye ayir.
    Asama 2: Her kume merkezine en yakin henuz secilmemis depoyu sec.
    Asama 3: Kapasite ihlali varsa swap-based yerel iyilestirme uygula.
    Zaman karmasikligi: O(K*n_c*t + n_w^2)
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    km.fit(data["customers"])
    centers = km.cluster_centers_

    selected = set()
    for center in centers:
        dists = np.linalg.norm(data["warehouses"] - center, axis=1)
        for j in np.argsort(dists):
            if int(j) not in selected:
                selected.add(int(j))
                break

    cost, detail = compute_cost(selected, data)
    if detail.get("penalty", 0) > 0:
        selected, cost, detail = _swap_improve(selected, data)

    return sorted(selected), round(cost, 1), detail


def _swap_improve(selected, data, max_iter=100):
    """
    Tek depo takas (1-opt swap) ile yerel arama.
    Maliyet dusurucU her takas kabul edilir.
    """
    best      = set(selected)
    best_cost, best_detail = compute_cost(best, data)
    outside   = [j for j in range(data["n_w"]) if j not in best]

    for _ in range(max_iter):
        improved = False
        for out_j in list(best):
            for in_j in outside:
                trial = (best - {out_j}) | {in_j}
                c, d  = compute_cost(trial, data)
                if c < best_cost - 1e-6:
                    best        = trial
                    best_cost   = c
                    best_detail = d
                    outside     = [j for j in range(data["n_w"]) if j not in best]
                    improved    = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, round(best_cost, 1), best_detail


def random_search(k, data, n_iter=1000, seed=0):
    """
    n_iter kez rastgele K depo sec; en dusuk maliyetli cozumu dondur.
    Zaman karmasikligi: O(n_iter * n_c * K)
    """
    rng_r     = random.Random(seed)
    best_cost = float("inf")
    best_sel  = None
    best_det  = {}

    for _ in range(n_iter):
        sel  = set(rng_r.sample(range(data["n_w"]), k))
        c, d = compute_cost(sel, data)
        if c < best_cost:
            best_cost = c
            best_sel  = sel
            best_det  = d

    return sorted(best_sel), round(best_cost, 1), best_det


def integer_programming(k, data, time_limit=60):
    """
    WLP'yi Tam Sayili Program (MIP) olarak PuLP/CBC ile cozer.

    Karar degiskenleri:
        y_j in {0,1}  -> depo j acilsin mi?
        x_ij in {0,1} -> musteri i, depo j'ye atansin mi?

    Amac: min sum(f_j*y_j) + sum(d_i*c_ij*x_ij)
    Kisitlar:
        - Her musteri tam olarak bir depoya atanmali
        - Atama ancak acik depoya yapilabilir
        - Kapasite kisiti
        - Tam olarak K depo acilacak
    """
    n_c = data["n_c"]
    n_w = data["n_w"]

    prob = LpProblem("WLP_IP", LpMinimize)

    y = [LpVariable(f"y_{j}", cat="Binary") for j in range(n_w)]
    x = [[LpVariable(f"x_{i}_{j}", cat="Binary")
          for j in range(n_w)] for i in range(n_c)]

    # Amac fonksiyonu
    prob += (lpSum(data["open_cost"][j] * y[j] for j in range(n_w)) +
             lpSum(data["demand"][i] * data["dist"][i][j] * x[i][j]
                   for i in range(n_c) for j in range(n_w)))

    # Kisit 1: Her musteri tam olarak bir depoya atanmali
    for i in range(n_c):
        prob += lpSum(x[i][j] for j in range(n_w)) == 1

    # Kisit 2: Atama ancak acik depoya yapilabilir
    for i in range(n_c):
        for j in range(n_w):
            prob += x[i][j] <= y[j]

    # Kisit 3: Kapasite kisiti
    for j in range(n_w):
        prob += lpSum(data["demand"][i] * x[i][j]
                      for i in range(n_c)) <= data["capacity"][j] * y[j]

    # Kisit 4: Tam olarak K depo acilacak
    prob += lpSum(y[j] for j in range(n_w)) == k

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    status = LpStatus[prob.status]
    if prob.objective is None or value(prob.objective) is None:
        return None, None, {"status": status}

    obj_val = value(prob.objective)
    if obj_val is None:
        return None, None, {"status": status}

    selected = [j for j in range(n_w)
                if y[j].varValue is not None and y[j].varValue > 0.5]
    cost     = round(obj_val, 1)
    detail   = {"status": status}
    return sorted(selected), cost, detail


# ─────────────────────────────────────────────────────────
# 4. DENEYSEL ANALİZ
# ─────────────────────────────────────────────────────────

DATASETS = [
    dict(label="Kucuk",  n_c=50,  n_w=10, seed=1),
    dict(label="Orta",   n_c=100, n_w=20, seed=2),
    dict(label="Buyuk",  n_c=200, n_w=30, seed=3),
]

K_VALUES = [2, 3, 4, 5, 6, 7, 8]


def run_all_experiments():
    all_results = {}

    for ds_cfg in DATASETS:
        label = ds_cfg["label"]
        data  = generate_dataset(ds_cfg["n_c"], ds_cfg["n_w"], ds_cfg["seed"])

        # IP buyuk veri setinde K<=5 ile sinirla (sure nedeniyle)
        ip_k_limit = 5 if ds_cfg["n_c"] >= 200 else 8

        print(f"\n{'='*70}")
        print(f"  Veri Seti: {label}  |  "
              f"Musteri: {ds_cfg['n_c']}  |  Aday Depo: {ds_cfg['n_w']}")
        print(f"  Toplam Talep: {data['demand'].sum():.0f}  |  "
              f"Toplam Kapasite: {data['capacity'].sum():.0f}")
        print(f"{'='*70}")
        print(f"  {'K':>2}  {'Greedy':>12}  {'Hibrit':>12}  "
              f"{'Random':>12}  {'IP':>12}  "
              f"{'H(ms)':>7}  {'R(ms)':>7}  {'IP(ms)':>8}  "
              f"{'GapG%':>8}  {'GapH%':>8}")
        print("  " + "-"*100)

        rows = []
        for k in K_VALUES:
            # Greedy-Only
            t0 = time.perf_counter()
            sg, cg, dg = greedy_only(k, data)
            tg = (time.perf_counter() - t0) * 1000

            # Hibrit
            t0 = time.perf_counter()
            sh, ch, dh = kmeans_greedy(k, data)
            th = (time.perf_counter() - t0) * 1000

            # Random Search
            t0 = time.perf_counter()
            sr, cr, dr = random_search(k, data)
            tr = (time.perf_counter() - t0) * 1000

            # Integer Programming
            if k <= ip_k_limit:
                t0 = time.perf_counter()
                si, ci, di = integer_programming(k, data, time_limit=60)
                ti = (time.perf_counter() - t0) * 1000
            else:
                si, ci, di, ti = None, None, {}, 0.0

            # En iyi referans maliyeti (gap hesabi icin)
            valid = [c for c in [cg, ch, cr, ci]
                     if c is not None and c < 1e12]
            c_ref = min(valid) if valid else 1.0

            gap_g = round((cg - c_ref) / c_ref * 100, 2) if c_ref > 0 else 0
            gap_h = round((ch - c_ref) / c_ref * 100, 2) if c_ref > 0 else 0

            ci_str = f"{ci:>12,.1f}" if ci is not None else f"{'N/A':>12}"
            print(f"  {k:>2}  {cg:>12,.1f}  {ch:>12,.1f}  "
                  f"{cr:>12,.1f}  {ci_str}  "
                  f"{th:>7.1f}  {tr:>7.1f}  {ti:>8.1f}  "
                  f"{gap_g:>7.1f}%  {gap_h:>7.1f}%")

            rows.append(dict(
                k=k,
                greedy_cost=cg, greedy_time=round(tg,2), greedy_sel=sg,
                hybrid_cost=ch, hybrid_time=round(th,2), hybrid_sel=sh,
                random_cost=cr, random_time=round(tr,2), random_sel=sr,
                ip_cost=ci,     ip_time=round(ti,2),     ip_sel=si,
                gap_greedy=gap_g, gap_hybrid=gap_h,
                data=data,
            ))

        all_results[label] = rows

    return all_results


# ─────────────────────────────────────────────────────────
# 5. GRAFİKLER
# ─────────────────────────────────────────────────────────

def _legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([0],[0], color=C_GREEDY, marker="s", lw=2, ms=7,
               ls="--",  label=ALG_LABELS["greedy"]),
        Line2D([0],[0], color=C_HYBRID, marker="o", lw=2, ms=7,
               ls="-",   label=ALG_LABELS["hybrid"]),
        Line2D([0],[0], color=C_RANDOM, marker="^", lw=2, ms=7,
               ls=":",   label=ALG_LABELS["random"]),
        Line2D([0],[0], color=C_IP,     marker="D", lw=2, ms=7,
               ls="-.",  label=ALG_LABELS["ip"]),
    ]


def plot_cost_curves(all_results, out="fig_cost_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ds_labels = ["Küçük", "Orta", "Büyük"]
    for ax, (label, rows), ds_lbl in zip(axes, all_results.items(), ds_labels):
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["greedy_cost"] for r in rows],
                "s--", color=C_GREEDY, lw=2, ms=7)
        ax.plot(ks, [r["hybrid_cost"] for r in rows],
                "o-",  color=C_HYBRID, lw=2, ms=7)
        ax.plot(ks, [r["random_cost"] for r in rows],
                "^:",  color=C_RANDOM, lw=2, ms=7)
        ip_ks   = [r["k"]       for r in rows if r["ip_cost"] is not None]
        ip_vals = [r["ip_cost"] for r in rows if r["ip_cost"] is not None]
        if ip_ks:
            ax.plot(ip_ks, ip_vals, "D-.", color=C_IP, lw=2, ms=7)
        ax.set_title(f"{ds_lbl} Veri Seti", fontsize=12, fontweight="bold")
        ax.set_xlabel("Depo Sayisi (K)", fontsize=11)
        ax.set_ylabel("Toplam Maliyet", fontsize=11)
        ax.legend(handles=_legend_handles(), fontsize=8)
        ax.grid(True, alpha=0.3, ls="--")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    plt.suptitle("Farkli Veri Setleri — Algoritma Maliyet Karsilastirmasi",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"/home/claude/{out}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_time_bars(all_results, out="fig_time_bars.png"):
    """
    4 algoritmanin calisma suresi — grouped bar chart.
    Greedy(yesil) | Hibrit(mavi) | Random(kirmizi) | IP(mor)
    """
    ds_labels = ["Küçük", "Orta", "Büyük"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    w = 0.18

    for ax, (label, rows), ds_lbl in zip(axes, all_results.items(), ds_labels):
        ks = [r["k"] for r in rows]
        x  = np.arange(len(ks))

        g_t = [r["greedy_time"] for r in rows]
        h_t = [r["hybrid_time"] for r in rows]
        r_t = [r["random_time"] for r in rows]
        i_t = [r["ip_time"]     for r in rows]

        ax.bar(x - 1.5*w, g_t, w, color=C_GREEDY,
               label=ALG_LABELS["greedy"], alpha=0.92)
        ax.bar(x - 0.5*w, h_t, w, color=C_HYBRID,
               label=ALG_LABELS["hybrid"], alpha=0.92)
        ax.bar(x + 0.5*w, r_t, w, color=C_RANDOM,
               label=ALG_LABELS["random"], alpha=0.92)
        ax.bar(x + 1.5*w, i_t, w, color=C_IP,
               label=ALG_LABELS["ip"],    alpha=0.92)

        ax.set_title(f"{ds_lbl} Veri Seti", fontsize=12, fontweight="bold")
        ax.set_xlabel("Depo Sayisi (K)", fontsize=11)
        ax.set_ylabel("Calisma Suresi (ms)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ks], rotation=40, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3, ls="--")

    plt.suptitle("Algoritmalarin Calisma Suresi Karsilastirmasi (ms)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"/home/claude/{out}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_gap_heatmap(all_results, out="fig_gap_heatmap.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    configs = [
        ("Greedy-Only — Optimality Gap (%)", "gap_greedy"),
        ("Hibrit — Optimality Gap (%)",       "gap_hybrid"),
    ]
    ylabels = ["Küçük", "Orta", "Büyük"]

    for ax, (title, key) in zip(axes, configs):
        matrix = np.array([[r[key] for r in rows]
                            for rows in all_results.values()])
        vmax = max(abs(matrix).max(), 1)
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=9)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(len(ylabels)):
            for j in range(len(K_VALUES)):
                ax.text(j, i, f"{matrix[i,j]:.1f}%",
                        ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Gap (%)")

    plt.suptitle("En Iyi Referansa Gore Optimality Gap Isi Haritasi",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"/home/claude/{out}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_solution_map(all_results, out="fig_solution_map.png"):
    rows = all_results["Orta"]
    row  = next(r for r in rows if r["k"] == 5)
    data = row["data"]
    k    = 5
    cmap = plt.cm.get_cmap("tab10", k)

    configs = [
        ("Greedy-Only",             row["greedy_sel"], C_GREEDY),
        ("K-Means+Greedy (Hibrit)", row["hybrid_sel"], C_HYBRID),
        ("Random Search",            row["random_sel"], C_RANDOM),
    ]
    if row.get("ip_sel"):
        configs.append(("Integer Programming", row["ip_sel"], C_IP))

    ncols = len(configs)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, (title, sel, color) in zip(axes, configs):
        sub  = data["dist"][:, sel]
        near = np.argmin(sub, axis=1)
        for ci in range(len(sel)):
            mask = near == ci
            ax.scatter(data["customers"][mask, 0],
                       data["customers"][mask, 1],
                       color=cmap(ci), alpha=0.5, s=22, zorder=2)
        ax.scatter(data["warehouses"][:, 0], data["warehouses"][:, 1],
                   c="#B0BEC5", marker="s", s=55, zorder=3,
                   label="Aday Depo")
        ax.scatter(data["warehouses"][sel, 0], data["warehouses"][sel, 1],
                   c=color, marker="*", s=260, zorder=4,
                   edgecolors="black", lw=0.7, label="Secilen Depo")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("X Koordinati", fontsize=10)
        ax.set_ylabel("Y Koordinati", fontsize=10)
        ax.grid(True, alpha=0.2, ls="--")

    plt.suptitle("Orta Veri Seti — K=5 Depo Secim Haritalari",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"/home/claude/{out}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_scalability(all_results, out="fig_scalability.png"):
    labels_x = ["Küçük\n(50 müş.)", "Orta\n(100 müş.)", "Büyük\n(200 müş.)"]
    k_ref    = 5
    x        = np.arange(len(labels_x))
    w        = 0.18

    g_t, h_t, r_t, i_t = [], [], [], []
    g_c, h_c, r_c, i_c = [], [], [], []

    for rows in all_results.values():
        row = next(r for r in rows if r["k"] == k_ref)
        g_t.append(row["greedy_time"]); h_t.append(row["hybrid_time"])
        r_t.append(row["random_time"]); i_t.append(row["ip_time"])
        g_c.append(row["greedy_cost"]); h_c.append(row["hybrid_cost"])
        r_c.append(row["random_cost"])
        ic = row["ip_cost"] if row["ip_cost"] is not None else np.nan
        i_c.append(ic)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x - 1.5*w, g_t, w, color=C_GREEDY,
            label=ALG_LABELS["greedy"], alpha=0.92)
    ax1.bar(x - 0.5*w, h_t, w, color=C_HYBRID,
            label=ALG_LABELS["hybrid"], alpha=0.92)
    ax1.bar(x + 0.5*w, r_t, w, color=C_RANDOM,
            label=ALG_LABELS["random"], alpha=0.92)
    ax1.bar(x + 1.5*w, i_t, w, color=C_IP,
            label=ALG_LABELS["ip"],    alpha=0.92)
    ax1.set_xticks(x); ax1.set_xticklabels(labels_x)
    ax1.set_ylabel("Calisma Suresi (ms)", fontsize=11)
    ax1.set_title(f"K={k_ref} — Olceklenebilirlik (Sure)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3, ls="--")

    ax2.plot(labels_x, g_c, "s--", color=C_GREEDY, lw=2, ms=9,
             label=ALG_LABELS["greedy"])
    ax2.plot(labels_x, h_c, "o-",  color=C_HYBRID, lw=2, ms=9,
             label=ALG_LABELS["hybrid"])
    ax2.plot(labels_x, r_c, "^:",  color=C_RANDOM, lw=2, ms=9,
             label=ALG_LABELS["random"])
    vx = [labels_x[i] for i,v in enumerate(i_c) if not np.isnan(v)]
    vv = [v for v in i_c if not np.isnan(v)]
    if vx:
        ax2.plot(vx, vv, "D-.", color=C_IP, lw=2, ms=9,
                 label=ALG_LABELS["ip"])
    ax2.set_ylabel("Toplam Maliyet", fontsize=11)
    ax2.set_title(f"K={k_ref} — Olceklenebilirlik (Maliyet)",
                  fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig(f"/home/claude/{out}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


# ─────────────────────────────────────────────────────────
# 6. ÖZET
# ─────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*60)
    print("  GENEL OZET")
    print("="*60)
    for label, rows in all_results.items():
        avg_g = np.mean([r["gap_greedy"] for r in rows])
        avg_h = np.mean([r["gap_hybrid"] for r in rows])
        print(f"  {label:<8} | Greedy gap: {avg_g:7.2f}%  "
              f"| Hibrit gap: {avg_h:7.2f}%")


# ─────────────────────────────────────────────────────────
# 7. ANA PROGRAM
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("WLP — Hibrit Algoritma Tam Deneysel Analiz")
    print("Manisa CBU Yazilim Muhendisligi | 2025-2026 Bahar\n")

    results = run_all_experiments()

    print("\nGrafikler olusturuluyor...")
    plot_cost_curves(results)
    plot_time_bars(results)
    plot_gap_heatmap(results)
    plot_solution_map(results)
    plot_scalability(results)

    print_summary(results)
    print("\n[TAMAMLANDI]")
