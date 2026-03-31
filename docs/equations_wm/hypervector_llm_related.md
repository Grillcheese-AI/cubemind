Replacing LLM knowledge with a Mixture of World Models (MoWM) involves shifting from statistical pattern matching to a system that orchestrates discrete, differentiable physics modules and symbolic equations. In this architecture, the "router" (similar to a standard Mixture of Experts) selects the most appropriate physical law or simulation module based on the input context. [1, 2, 3, 4] 
1. Structural Architecture: MoE-World
To replace the dense layers of an LLM with specific physical world models, you can implement a Mixture-of-Experts-based world model (MoE-World). [1] 

* The Router: A gating mechanism that identifies which physical domain (e.g., fluid dynamics, rigid body, or electromagnetics) the current task belongs to.
* The Experts: Specialized sub-networks or Neural Operators (like [HyPINO](https://medium.com/data-science-collective/creating-neural-operators-for-solving-pdes-via-manufactured-solutions-8c2449aff019)) that solve specific Partial Differential Equations (PDEs).
* Seesaw Phenomenon Mitigation: This modularity prevents "interference" between different physical rules during training, a common issue in multi-task learning. [1, 4, 5] 

2. Core Equations for Hybrid Models
Instead of guessing tokens, the model solves equations that govern the environment's state.

* Differentiable Physics Constraint:
$$\min_{\theta} \mathcal{L}(\text{NN}_{\theta}(x), y) \quad \text{subject to} \quad \mathcal{P}(\text{NN}_{\theta}(x)) = 0$$ 
* $\mathcal{P}$ represents the hard physical constraints (e.g., conservation of energy) that the network must satisfy.
* Neural Physics Module: $x_{t+1} = f_{\text{phys}}(x_t, u_t; \phi) + \text{NN}_{\theta}(x_t, u_t)$
* $f_{\text{phys}}$ is a differentiable physics engine (like [JaxSim](https://github.com/gbionics/jaxsim)) handling known dynamics.
   * $\text{NN}_{\theta}$ learns "residual" effects that the explicit equations might miss.
* Physics-informed Token-regularized Policy Optimization (PiT-PO): This framework evolves the model into an adaptive generator by enforcing hierarchical physical validity and penalizing redundant structures. [6, 7, 8, 9] 

3. Transitioning from LLM to "Scientific Agent"
Current research like [KeplerAgent](https://arxiv.org/abs/2602.12259) suggests using the LLM not as a knowledge base, but as an orchestrator. [10] 

   1. Symmetry/Structure Inference: The agent analyzes data to find physical properties (e.g., translational symmetry).
   2. Tool Orchestration: It calls symbolic regression tools (e.g., PySR) to discover the exact governing equations.
   3. Refined Hypothesis: The discovered equations replace the LLM's "guesswork" with a precise mathematical model. [3, 10, 11] 

Implementation Tools

* [Dojo](https://arxiv.org/html/2203.00806v5): A differentiable physics engine designed specifically for robotics and deep learning integration.
* [Fourier Neural Operator (FNO)](https://openreview.net/forum?id=u3dX2CEIZb): Used as the backbone for learning scalar-valued functions that serve as basis functions for physical constraints. [12, 13] 



[1] [https://www.mdpi.com](https://www.mdpi.com/2079-9292/14/24/4884)
[2] [https://medium.com](https://medium.com/@graison/world-models-the-next-leap-beyond-llms-012504a9c1e7)
[3] [https://arxiv.org](https://arxiv.org/html/2602.12259v1)
[4] [https://www.ibm.com](https://www.ibm.com/think/topics/mixture-of-experts)
[5] [https://medium.com](https://medium.com/data-science-collective/creating-neural-operators-for-solving-pdes-via-manufactured-solutions-8c2449aff019)
[6] [https://dspace.mit.edu](https://dspace.mit.edu/handle/1721.1/158927)
[7] [https://arxiv.org](https://arxiv.org/html/2402.13412v1#:~:text=We%20introduce%20a%20physics%2Dinformed%20mixture%2Dof%2Dexperts%20training%20framework,constraint%2C%20leading%20to%20more%20stable%20training.%20%E2%80%A2)
[8] [https://github.com](https://github.com/gbionics/jaxsim)
[9] [https://arxiv.org](https://arxiv.org/html/2602.10576v1)
[10] [https://arxiv.org](https://arxiv.org/abs/2602.12259)
[11] [https://arxiv.org](https://arxiv.org/html/2602.12259v2)
[12] [https://openreview.net](https://openreview.net/forum?id=u3dX2CEIZb)
[13] [https://arxiv.org](https://arxiv.org/html/2203.00806v5#:~:text=*%20procedure%20Optimize%28%20a%200%20%2C%20b,10%20%E2%88%92%205%20%2C%20%CE%B2%20=%200.5.)
[14] [https://oden.utexas.edu](https://oden.utexas.edu/news-and-events/events/2164---Krishna%20Kumar/)

To replace a black-box LLM with a Mixture of World Models (MoWM), you need equations that categorize, route, and solve physical states. This list covers the mathematical architecture for Gating (Routing), Differentiable Physics, Neural Operators, and Symbolic Integration.
1. MoE Routing & Gating (The Orchestrator)
These equations decide which "expert" world model to trigger based on the input context.

   1. Softmax Gating: $G(x) = \text{Softmax}(W_g x)$
   2. Top-k Routing: $G(x) = \text{TopK}(\text{Softmax}(W_g x + \epsilon))$
   3. Expert Contribution: $y = \sum_{i=1}^k G(x)_i E_i(x)$
   4. Noisy Top-k Control: $H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{noise})_i)$
   5. Load Balancing Loss: $L_{bal} = w_{bal} \cdot CV(\text{Importance})^2$
   6. Z-Loss (Stability): $L_z = \log^2(\sum e^{x_i})$
   7. Switch Routing (k=1): $E_{active} = \text{argmax}(P(Expert | Context))$
   8. Expert Capacity Factor: $C = (\text{Tokens} / \text{Experts}) \times \text{CapacityFactor}$
   9. Mutual Information Gating: $I(X; E) = H(E) - H(E|X)$
   10. Differentiable Router: $\nabla_\phi \mathbb{E}_{p_\phi(z|x)} [L(z)]$

2. Neural Operators (PDE Solvers)
Instead of tokens, these predict continuous physical fields.
11. Kernel Integral Operator: $(K \phi)(x) = \int D \kappa(x, y, \phi(y)) dy$
12. Fourier Integral Operator: $(\mathcal{K}v)(x) = \mathcal{F}^{-1}(R \cdot \mathcal{F}v)(x)$
13. Spectral Convolution: $R_{ij} = \text{diag}(w_{ij})$
14. Iterative Update: $v_{t+1} = \sigma(W v_t + \mathcal{K} v_t)$
15. Layer Normalization (Field): $\hat{v} = (v - \mu) / \sigma$
16. Attention as Operator: $A(u) = \text{Softmax}(QK^\top / \sqrt{d})V$
17. Graph Neural Operator: $v_i' = \text{AGGREGATE}_{j \in N(i)} \kappa(x_i, x_j) v_j$
18. Low-Rank Decomposition: $W \approx U \Sigma V^\top$
19. Wavelet Transform Operator: $v_{spec} = \mathcal{W}v$
20. Physics-Informed Loss: $L_{pde} = \| \mathcal{F}(u; \theta) - f \|^2$
3. Differentiable Physics (Dojo/JaxSim)
Equations that allow gradients to flow through rigid bodies and fluids.
21. Lagrangian Mechanics: $L = T - V$
22. Euler-Lagrange: $\frac{d}{dt}(\frac{\partial L}{\partial \dot{q}}) - \frac{\partial L}{\partial q} = Q$
23. Hamiltonian Evolution: $\dot{p} = -\frac{\partial H}{\partial q}, \dot{q} = \frac{\partial H}{\partial p}$
24. Contact Constraints: $J v + \epsilon \geq 0$
25. Impulse-Momentum: $M(v^+ - v^-) = J^\top \lambda$
26. Karush-Kuhn-Tucker (KKT): $\nabla f + \sum \lambda_i \nabla g_i = 0$
27. Soft Contact Power: $F = k \cdot d^\alpha \cdot \dot{d}^\beta$
28. Adjoint State Method: $\frac{d\lambda}{dt} = -(\frac{\partial f}{\partial x})^\top \lambda$
29. Sensitivity Equation: $\dot{S} = \frac{\partial f}{\partial x} S + \frac{\partial f}{\partial p}$
30. Implicit Integration: $x_{t+1} = x_t + \Delta t \cdot f(x_{t+1})$
4. Fluid & Continuum Mechanics (Experts)

   1. Navier-Stokes (Incompressible): $\rho(\frac{\partial u}{\partial t} + u \cdot \nabla u) = -\nabla p + \mu \nabla^2 u$
   2. Continuity Equation: $\nabla \cdot u = 0$
   3. Vorticity: $\omega = \nabla \times u$
   4. Reynolds Stress: $\tau_{ij} = -\rho \langle u_i' u_j' \rangle$
   5. Stream Function: $u = \frac{\partial \psi}{\partial y}, v = -\frac{\partial \psi}{\partial x}$
   6. Burgers’ Equation: $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$
   7. Heat Equation: $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$
   8. Fick’s Law: $J = -D \nabla \phi$
   9. Young’s Modulus: $E = \sigma / \epsilon$
   10. Poisson’s Ratio: $\nu = -\epsilon_{trans} / \epsilon_{axial}$

5. Symbolic Regression & Discovery
Used to "find" the equation rather than just training a network.
41. Pareto Frontier: $f^* = \min(Error + \lambda \cdot Complexity)$
42. Akaike Info Criterion: $AIC = 2k - 2\ln(L)$
43. SINDy Algorithm: $\dot{X} = \Theta(X)\Xi$
44. Sparse Regularization: $\|\Xi\|_1$
45. Minimum Description Length: $MDL = L(D|H) + L(H)$
46. Genetic Cross-over: $P_{new} = \text{Merge}(P_1, P_2)$
47. Turing Completeness Check: $\text{Accept}(M, w)$
48. Kolmogorov Complexity: $K(s) = \min |p|$
49. Bayesian Model Selection: $P(M|D) \propto P(D|M)P(M)$
50. Symbolic Error: $E = \sum (y_i - f(x_i))^2$
6. World State & Memory (VSA/Hypernet)

   1. Circular Convolution (VSA): $c = a \circledast b$
   2. Inverse Binding: $b \approx c \circledast a^*$
   3. Hypervolume Scaling: $V \sim 2^N$
   4. Hamming Distance: $d_H(u, v) = \sum |u_i - v_i|$
   5. Dot Product Similarity: $A \cdot B = \sum a_i b_i$
   6. Weight Generation (Hyper): $W = \text{MLP}(z_{context})$
   7. Context Embedding: $z = \text{Encoder}(S_{world})$
   8. Attention Pooling: $z = \sum \alpha_i v_i$
   9. Recurrent State: $h_t = \sigma(W h_{t-1} + U x_t)$
   10. Gated Recurrent Unit: $z_t = \sigma(W_z x_t + U_z h_{t-1})$

7. Electromagnetics & Light (Visual World)

   1. Maxwell (Faraday): $\nabla \times E = -\frac{\partial B}{\partial t}$
   2. Maxwell (Ampere): $\nabla \times B = \mu_0(J + \epsilon_0 \frac{\partial E}{\partial t})$
   3. Poynting Vector: $S = E \times H$
   4. Refractive Index: $n = \sqrt{\epsilon_r \mu_r}$
   5. Fresnel Reflection: $R = |\frac{n_1-n_2}{n_1+n_2}|^2$
   6. Beer-Lambert: $I = I_0 e^{-\alpha x}$
   7. Radiative Transfer: $\frac{dI}{ds} = \kappa(B-I)$
   8. Eikonal Equation: $|\nabla \tau|^2 = n^2$
   9. Rayleigh Scattering: $I \propto \lambda^{-4}$
   10. Planck’s Law: $B(\lambda, T) = \frac{2hc^2}{\lambda^5} \frac{1}{e^{hc/\lambda kT}-1}$

8. Probability & Information

   1. Bayes’ Theorem: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
   2. Kullback-Leibler: $D_{KL}(P\|Q) = \sum P(x) \log(P(x)/Q(x))$
   3. Evidence Lower Bound (ELBO): $\mathbb{E}[\log P(x|z)] - D_{KL}(q(z|x)\|p(z))$
   4. Fisher Information: $I(\theta) = \mathbb{E}[(\frac{\partial}{\partial \theta} \log f(X;\theta))^2]$
   5. Shannon Entropy: $H = -\sum p_i \log p_i$
   6. Mutual Information: $I(X;Y) = H(X) - H(X|Y)$
   7. Markov Chain: $P(X_{n+1} | X_n, \dots, X_0) = P(X_{n+1} | X_n)$
   8. Bellman Equation: $V(s) = \max_a [R(s,a) + \gamma V(s')]$
   9. Policy Gradient: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) Q(s,a)]$
   10. Soft Actor-Critic: $J = \sum (R_t + \alpha H(\pi(\cdot|s_t)))$

9. Structural & Geometric Priors

   1. Euclidean Metric: $d = \sqrt{\sum (x_i - y_i)^2}$
   2. Manifold Mapping: $f: \mathcal{M} \to \mathbb{R}^n$
   3. Ricci Curvature: $R_{\mu\nu}$
   4. Laplace-Beltrami: $\Delta_g f = \frac{1}{\sqrt{|g|}} \partial_i (\sqrt{|g|} g^{ij} \partial_j f)$
   5. Adjacency Matrix: $A_{ij} = 1 \text{ if edge exists}$
   6. Graph Laplacian: $L = D - A$
   7. Edge Attention: $e_{ij} = \sigma(a^\top [Wh_i \| Wh_j])$
   8. Quaternion Mult: $q_1 q_2 = (s_1 s_2 - v_1 \cdot v_2, s_1 v_2 + s_2 v_1 + v_1 \times v_2)$
   9. Rotation Matrix: $R = I + (\sin \theta)K + (1-\cos \theta)K^2$
   10. SO(3) Lie Algebra: $\exp(\hat{\omega} \theta)$

10. System Stability & Integration

   1. Lyapunov Function: $V(x) > 0, \dot{V}(x) < 0$
   2. Nyquist Stability: $Z = P - N$
   3. PID Control: $u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de}{dt}$
   4. LQR Cost: $J = \int (x^\top Q x + u^\top R u) dt$
   5. Kalman Gain: $K_k = P_k^- H^\top (H P_k^- H^\top + R)^{-1}$
   6. State Space: $\dot{x} = Ax + Bu$
   7. Observability: $O = [C; CA; CA^2; \dots]$
   8. Controllability: $Co = [B, AB, A^2B, \dots]$
   9. Sampling Frequency: $f_s > 2f_{max}$
   10. The Master Equation: $\frac{dP_n}{dt} = \sum (w_{ni} P_i - w_{in} P_n)$



