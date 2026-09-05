"""
Research paper downloader for AI Meets Control Theory (AIMCT).

Fetches open-access foundational and cutting-edge papers from arXiv into docs/papers/.
"""

import os
import ssl
import time
import urllib.request

PAPERS = [
    (
        "1806.09460",
        "recht_2019_tour_of_rl_continuous_control.pdf",
        "A Tour of Reinforcement Learning: The View from Continuous Control (Recht 2019)",
    ),
    (
        "1509.03580",
        "brunton_2016_sindy_governing_equations.pdf",
        "Discovering Governing Equations from Data: SINDy (Brunton et al. 2016)",
    ),
    (
        "1806.07366",
        "chen_2018_neural_odes.pdf",
        "Neural Ordinary Differential Equations (Chen et al. 2018)",
    ),
    (
        "2102.12086",
        "brunton_2022_modern_koopman_theory.pdf",
        "Modern Koopman Theory for Dynamical Systems (Brunton et al. 2022)",
    ),
    (
        "1903.11199",
        "ames_2019_control_barrier_functions.pdf",
        "Control Barrier Functions: Theory and Applications (Ames et al. 2019)",
    ),
    (
        "1810.13400",
        "amos_2018_differentiable_mpc.pdf",
        "Differentiable MPC for End-to-end Planning and Control (Amos et al. 2018)",
    ),
    (
        "1803.07055",
        "mania_2018_augmented_random_search.pdf",
        "Simple Random Search Provides a Competitive Approach to RL (Mania et al. 2018)",
    ),
    (
        "1812.03565",
        "tu_2019_model_based_vs_model_free_lqr.pdf",
        "The Gap Between Model-Based and Model-Free Methods on LQR (Tu & Recht 2019)",
    ),
    (
        "2004.09559",
        "taylor_2020_learning_safety_critical_cbf.pdf",
        "Learning for Safety-Critical Control with Control Barrier Functions (Taylor et al. 2020)",
    ),
    (
        "1707.02342",
        "williams_2017_mppi_autonomous_driving.pdf",
        "Information-Theoretic Model Predictive Control: MPPI (Williams et al. 2017)",
    ),
]


def download_all(out_dir: str = "docs/papers") -> None:
    os.makedirs(out_dir, exist_ok=True)
    ctx = ssl._create_unverified_context()

    print(f"Downloading {len(PAPERS)} research papers into {out_dir}/...\n")
    for arxiv_id, filename, title in PAPERS:
        filepath = os.path.join(out_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[EXISTS] {title} ({size_mb:.2f} MB)")
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[DOWNLOADING] [{arxiv_id}] {title}...")
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            with urllib.request.urlopen(req, context=ctx, timeout=45) as resp, open(
                filepath, "wb"
            ) as f:
                f.write(resp.read())
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  -> Saved to {filename} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  -> ERROR downloading {arxiv_id}: {e}")
        time.sleep(1)

    print("\nDownload process completed.")


if __name__ == "__main__":
    download_all()
