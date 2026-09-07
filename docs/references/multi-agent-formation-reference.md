# Multi-Agent Formation Control & Graph Topology Reference Specification

## 1. Executive Summary & Distributed Consensus Foundations

Multi-agent coordination enables a collective of $N$ autonomous vehicles (drones, mobile robots, surface vessels) to achieve a desired geometric shape (polygon, line, V-wedge, grid) and track navigational references using only **local distributed communication**.

Rather than relying on a vulnerable centralized coordinator, distributed formation control couples local agent dynamics through the **Algebraic Graph Laplacian** $L$ of the inter-agent communication network.

```
   Target Formation Geometry                     Distributed Communication Graph
          [ Agent 1 ]                                      ( 1 )
          /         \                                     /     \
         /           \                                   /       \
   [ Agent 2 ] --- [ Agent 3 ]                         ( 2 ) --- ( 3 )
   d_12 = p1* - p2*, d_23 = p2* - p3*              Adjacency A, Laplacian L
```

---

## 2. Algebraic Graph Theory

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be an undirected communication graph with $N$ vertices $\mathcal{V} = \{1, \dots, N\}$ and edges $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$.

### 2.1 Adjacency, Degree, and Laplacian Matrices
- **Adjacency Matrix** $A = [a_{ij}] \in \mathbb{R}^{N \times N}$: $a_{ij} = 1$ if $(i, j) \in \mathcal{E}$, else $0$ (with $a_{ii} = 0$).
- **Degree Matrix** $D = \text{diag}(d_1, \dots, d_N)$ where $d_i = \sum_{j=1}^N a_{ij}$.
- **Graph Laplacian Matrix**:
  $$L = D - A$$

### 2.2 Spectrum & Fiedler Eigenvalue
The eigenvalues of the symmetric positive semi-definite Laplacian satisfy:

$$0 = \lambda_1(L) \le \lambda_2(L) \le \dots \le \lambda_N(L)$$

- **Nullspace Property**: $L \mathbf{1}_N = 0$, so $\mathbf{1}_N = [1, \dots, 1]^T$ is the eigenvector associated with $\lambda_1 = 0$.
- **Algebraic Connectivity (Fiedler Eigenvalue $\lambda_2(L)$)**:
  $$\lambda_2(L) > 0 \iff \text{The communication graph } \mathcal{G} \text{ is connected.}$$
  The convergence rate of consensus is directly bounded by $\lambda_2(L)$: $\rho(t) \le e^{-\lambda_2(L) t}$.

---

## 3. Distributed Formation Control Protocols

Let $p_i \in \mathbb{R}^2$ and $v_i \in \mathbb{R}^2$ be the 2D Cartesian position and velocity of agent $i$.
Let $d_i^* \in \mathbb{R}^2$ denote the desired nominal offset of agent $i$ with respect to the formation centroid, defining relative target offsets $d_{ij} = d_i^* - d_j^*$.

### 3.1 Double-Integrator Consensus Protocol
For agents governed by $\ddot{p}_i = u_i$:

$$u_i = -\underbrace{k_p \sum_{j \in \mathcal{N}_i} a_{ij} (p_i - p_j - d_{ij})}_{\text{Relative Position Alignment}} - \underbrace{k_v \sum_{j \in \mathcal{N}_i} a_{ij} (v_i - v_j)}_{\text{Velocity Damping}} - \underbrace{k_0 (p_i - d_i^* - r_0(t)) - k_{v0} (v_i - v_0(t))}_{\text{Navigational Leader Pinning}}$$

where $\mathcal{N}_i = \{j \mid a_{ij} > 0\}$ is the neighborhood of agent $i$, and $r_0(t), v_0(t)$ is the virtual leader reference trajectory.

### 3.2 Lyapunov Asymptotic Stability Proof
Define the formation tracking error state $\tilde{p}_i = p_i - d_i^* - r_0(t)$ and $\tilde{v}_i = v_i - v_0(t)$.
Stacking all agents: $\tilde{p} = [\tilde{p}_1^T, \dots, \tilde{p}_N^T]^T \in \mathbb{R}^{2N}$.

Consider the candidate Lyapunov function:

$$V(\tilde{p}, \tilde{v}) = \frac{1}{2} k_p \tilde{p}^T ( (L + B_0) \otimes I_2 ) \tilde{p} + \frac{1}{2} \tilde{v}^T \tilde{v}$$

where $B_0 = \text{diag}(b_1, \dots, b_N)$ denotes leader pinning gains ($b_i > 0$ for at least one pinned agent).
Differentiating along closed-loop trajectories yields:

$$\dot{V} = -\tilde{v}^T ( (k_v L + k_{v0} B_0) \otimes I_2 ) \tilde{v} \le 0$$

By LaSalle's Invariance Principle, as long as the graph remains connected ($\lambda_2(L) > 0$), the error asymptotically converges to the zero manifold: $\tilde{p}(t) \to 0$, $\tilde{v}(t) \to 0$ as $t \to \infty$.

---

## 4. Switching Graph Topologies & Fault Tolerance

In practical deployment, wireless communication links undergo packet drops, distance-dependent fading, or directional shadowing:

$$A(t) \in \{ A_{\text{mesh}}, A_{\text{ring}}, A_{\text{star}}, A_{\text{line}}, A_{\text{disconnected}} \}$$

- **Common Lyapunov Function (CLF)**: If every switched topology $\mathcal{G}_{\sigma(t)}$ is connected ($\lambda_2(L_{\sigma(t)}) > 0$), the shared quadratic energy $V$ strictly decreases across arbitrary switching sequences.
- **Average Dwell-Time Stability**: If the graph is temporarily disconnected, asymptotic stability is preserved provided the connected topologies are active for a sufficient fraction of the total time (Hespanha & Morse, 1999).

---

## 5. Collision Avoidance Artificial Potential Fields

To prevent inter-agent collisions during formation transitions or switching transients, a repulsive potential barrier is activated when inter-agent distance $r_{ij} = \|p_i - p_j\|$ violates safety threshold $d_{\text{safe}}$:

$$u_{\text{coll}, i} = k_{\text{coll}} \sum_{j \neq i, \, r_{ij} < d_{\text{safe}}} \left( \frac{1}{r_{ij}} - \frac{1}{d_{\text{safe}}} \right) \frac{1}{r_{ij}^2} \frac{p_i - p_j}{r_{ij}}$$
