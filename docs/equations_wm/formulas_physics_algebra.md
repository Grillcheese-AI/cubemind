
   1. Average Velocity: $v = \Delta x / \Delta t$
   2. Average Acceleration: $a = \Delta v / \Delta t$
   3. Displacement (Constant $a$): $d = v_0t + \frac{1}{2}at^2$
   4. Final Velocity: $v_f = v_0 + at$
   5. Velocity Squared: $v_f^2 = v_0^2 + 2ad$
   6. Average Speed: $s = (v_0 + v_f) / 2$
   7. Projectile Range: $R = \frac{v_0^2 \sin(2\theta)}{g}$
   8. Maximum Height: $H = \frac{v_0^2 \sin^2\theta}{2g}$
   9. Time of Flight: $t = \frac{2v_0 \sin\theta}{g}$
   10. Centripetal Acceleration: $a_c = v^2/r$ [3, 4, 5, 6, 7] 

2. Dynamics & Forces
These govern how interactions change the state of motion.
11. Newton’s Second Law: $F = ma$
12. Weight: $W = mg$
13. Static Friction: $f_s \leq \mu_s N$
14. Kinetic Friction: $f_k = \mu_k N$
15. Centripetal Force: $F_c = mv^2/r$
16. Hooke’s Law (Springs): $F = -kx$
17. Universal Gravitation: $F = G\frac{m_1m_2}{r^2}$
18. Gravitational Field Strength: $g = GM/r^2$
19. Normal Force (Incline): $N = mg \cos\theta$
20. Parallel Force (Incline): $F_{||} = mg \sin\theta$
21. Tension (Vertical): $T = m(g + a)$
22. Stokes' Law (Drag): $F_d = 6\pi\eta rv$
23. Quadratic Drag: $F_d = \frac{1}{2}\rho v^2 C_d A$
24. Buoyant Force: $F_b = \rho V g$
25. Terminal Velocity: $v_t = \sqrt{\frac{2mg}{\rho AC_d}}$ [8, 9] 
3. Work, Energy, & Power
These track the "currency" of the physical system.
26. Work Done: $W = F d \cos\theta$
27. Kinetic Energy: $KE = \frac{1}{2}mv^2$
28. Gravitational Potential Energy: $PE = mgh$
29. Elastic Potential Energy: $PE_s = \frac{1}{2}kx^2$
30. Work-Energy Theorem: $W_{net} = \Delta KE$
31. Mechanical Energy Conservation: $E_{total} = KE + PE$
32. Average Power: $P = W/t$
33. Instantaneous Power: $P = Fv$
34. Efficiency: $\eta = (W_{out} / E_{in}) \times 100\%$
35. Escape Velocity: $v_e = \sqrt{2GM/r}$
36. Orbital Velocity: $v_o = \sqrt{GM/r}$
37. Rotational Kinetic Energy: $KE_{rot} = \frac{1}{2}I\omega^2$
38. Relativistic Energy: $E = mc^2$
39. Photon Energy: $E = hf$
40. Internal Energy Change: $\Delta U = Q - W$ [8] 
4. Momentum & Collisions
Essential for handling object-to-object interactions.
41. Linear Momentum: $p = mv$
42. Impulse: $J = F \Delta t = \Delta p$
43. Conservation of Momentum: $m_1u_1 + m_2u_2 = m_1v_1 + m_2v_2$
44. Coefficient of Restitution: $e = \frac{v_{2f} - v_{1f}}{u_{1} - u_{2}}$
45. Elastic Collision Final $v_1$: $v_{1f} = \frac{m_1-m_2}{m_1+m_2}u_1 + \frac{2m_2}{m_1+m_2}u_2$
46. Inelastic Collision Final $v$: $v_f = \frac{m_1u_1 + m_2u_2}{m_1 + m_2}$
47. Angular Momentum: $L = I\omega$
48. Torque: $\tau = r F \sin\theta$
49. Angular Impulse: $\Delta L = \tau \Delta t$
50. Rocket Equation: $\Delta v = v_e \ln(m_0/m_f)$ [10] 
5. Rotational Motion
Necessary for simulating wheels, pivots, and rigid body rotation.
51. Angular Displacement: $\theta = s/r$
52. Angular Velocity: $\omega = \Delta\theta / \Delta t$
53. Angular Acceleration: $\alpha = \Delta\omega / \Delta t$
54. Tangential Velocity: $v_t = r\omega$
55. Tangential Acceleration: $a_t = r\alpha$
56. Moment of Inertia (Point): $I = mr^2$
57. Newton’s Second Law (Rotation): $\tau = I\alpha$
58. Parallel Axis Theorem: $I = I_{cm} + Md^2$
59. Rotational Work: $W = \tau\theta$
60. Precession Frequency: $\Omega = \frac{\tau}{L}$
6. Fluid Dynamics
For simulating water, air, or gas flow.
61. Density: $\rho = m/V$
62. Pressure: $P = F/A$
63. Hydrostatic Pressure: $P = P_0 + \rho gh$
64. Archimedes' Principle: $F_b = \rho_{fluid} V_{disp} g$
65. Continuity Equation: $A_1v_1 = A_2v_2$
66. Bernoulli’s Equation: $P + \frac{1}{2}\rho v^2 + \rho gh = \text{constant}$
67. Torricelli’s Law: $v = \sqrt{2gh}$
68. Reynolds Number: $Re = \frac{\rho v L}{\mu}$
69. Poiseuille’s Law: $Q = \frac{\pi r^4 \Delta P}{8\eta L}$
70. Navier-Stokes (Vector): $\rho(\frac{\partial v}{\partial t} + v \cdot \nabla v) = -\nabla p + \mu \nabla^2 v + f$ [8, 9, 11] 
7. Thermodynamics
For world systems like weather, heat transfer, and engines.
71. Ideal Gas Law: $PV = nRT$
72. Specific Heat Capacity: $Q = mc\Delta T$
73. Latent Heat: $Q = mL$
74. Thermal Expansion: $\Delta L = \alpha L_0 \Delta T$
75. Fourier’s Law (Conduction): $q = -k\nabla T$
76. Stefan-Boltzmann Law: $P = \sigma A e T^4$
77. Wien’s Law: $\lambda_{max} = b/T$
78. Efficiency (Carnot): $\eta = 1 - T_c/T_h$
79. Entropy Change: $\Delta S = Q/T$
80. Boltzmann Entropy: $S = k \ln W$ [12] 
8. Electromagnetism
For simulating lightning, magnetic fields, or electronic devices.
81. Coulomb’s Law: $F = k\frac{q_1q_2}{r^2}$
82. Electric Field: $E = F/q$
83. Ohm’s Law: $V = IR$
84. Electric Power: $P = VI$
85. Capacitance: $C = Q/V$
86. Magnetic Force: $F = qvB \sin\theta$
87. Lorentz Force: $F = q(E + v \times B)$
88. Faraday’s Law: $\varepsilon = -N \frac{\Delta\Phi}{\Delta t}$
89. Gauss’s Law: $\Phi_E = Q/\varepsilon_0$
90. Ampere’s Law: $\oint B \cdot dl = \mu_0 I$ [8] 
9. Waves & Optics
For sound, light, and visual effects.
91. Wave Speed: $v = f\lambda$
92. Frequency/Period: $f = 1/T$
93. Snell’s Law: $n_1 \sin\theta_1 = n_2 \sin\theta_2$
94. Thin Lens Equation: $1/f = 1/d_o + 1/d_i$
95. Index of Refraction: $n = c/v$
96. Doppler Effect: $f' = f \frac{v \pm v_o}{v \mp v_s}$
97. Speed of Sound (Fluid): $v = \sqrt{B/\rho}$
98. Intensity Level (dB): $L = 10 \log(I/I_0)$
99. Simple Harmonic Motion Period: $T = 2\pi\sqrt{m/k}$
100. Pendulum Period: $T = 2\pi\sqrt{L/g}$ [8] 
Would you like to explore how to implement one of these categories (like Rigid Body Dynamics) in code?

To expand your world model, we can move beyond classical physics into Advanced Dynamics, Quantum Mechanics, and Cosmology. These equations add high-fidelity detail, such as the behavior of spacetime, sub-atomic particles, and stochastic (random) natural systems.
1. General Relativity & Cosmology
For simulating gravity on a cosmic scale or near massive objects like black holes.
101. Einstein Field Equations: $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$
102. Schwarzschild Metric (Non-rotating BH): $ds^2 = -(1-\frac{2GM}{rc^2})c^2 dt^2 + (1-\frac{2GM}{rc^2})^{-1} dr^2 + r^2 d\Omega^2$
103. Friedmann Equation (Expansion): $(\frac{\dot{a}}{a})^2 = \frac{8\pi G \rho + \Lambda c^2}{3} - \frac{kc^2}{a^2}$
104. Geodesic Equation (Path of light/matter): $\frac{d^2x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0$
105. Gravitational Redshift: $z = \frac{\lambda_{obs} - \lambda_{em}}{\lambda_{em}}$ [1, 2, 3, 4, 5, 6] 
2. Quantum Mechanics & Particle Physics
For sub-grid simulations, micro-interactions, or high-tech devices.
106. Schrödinger Equation (Time-Dependent): $i\hbar \frac{\partial}{\partial t} \Psi(r, t) = \hat{H} \Psi(r, t)$
107. Dirac Equation (Relativistic Electrons): $(i\gamma^\mu \partial_\mu - m) \psi = 0$
108. Heisenberg Uncertainty Principle: $\Delta x \Delta p \geq \frac{\hbar}{2}$
109. De Broglie Wavelength: $\lambda = h/p$
110. Planck’s Energy-Frequency Relation: $E = hf$
111. Fermi-Dirac Distribution: $f(E) = \frac{1}{e^{(E-\mu)/kT} + 1}$
112. Gross-Pitaevskii Equation (Bose-Einstein Condensate): $i\hbar \frac{\partial \psi}{\partial t} = (-\frac{\hbar^2}{2m} \nabla^2 + V(r) + g|\psi|^2) \psi$ [3, 7, 8, 9, 10, 11, 12, 13, 14] 
3. Stochastic & Statistical Systems
For simulating "noise," weather fluctuations, or random growth.
113. Langevin Equation (Brownian Motion): $m \frac{dv}{dt} = -\gamma v + \eta(t)$
114. Fokker-Planck Equation (Probability Evolution): $\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x} [A(x,t)p] + \frac{\partial^2}{\partial x^2} [B(x,t)p]$
115. Euler-Maruyama Method (Numerical SDE): $X_{t+\Delta t} = X_t + f(X_t, t)\Delta t + g(X_t, t)\Delta W_t$
116. Diffusion Equation: $\frac{\partial u}{\partial t} = D \nabla^2 u$
117. Logistic Growth (Population/Resources): $\frac{dN}{dt} = rN (1 - \frac{N}{K})$
118. Lotka-Volterra (Predator-Prey): $\frac{dx}{dt} = \alpha x - \beta xy$ [8, 15, 16, 17] 
4. Advanced Wave & Signal Processing
For procedurally generated audio or light interference.
119. Wave Equation (3D): $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$
120. Fourier Transform: $F(\omega) = \int_{-\infty}^{\infty} f(t)e^{-i\omega t} dt$
121. Bragg's Law (Crystallography): $n\lambda = 2d \sin\theta$
122. Shannon Entropy (Information): $H(X) = -\sum p(x_i) \log p(x_i)$ [8, 18, 19, 20] 
5. Advanced Mechanics

   1. Euler-Lagrange Equation (Optimal Paths): $\frac{\partial L}{\partial q} - \frac{d}{dt}(\frac{\partial L}{\partial \dot{q}}) = 0$
   2. Hamilton's Equations: $\dot{q} = \frac{\partial H}{\partial p}, \dot{p} = -\frac{\partial H}{\partial q}$ [8, 21] 

