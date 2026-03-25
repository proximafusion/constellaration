# Point Cloud Fitting to SurfaceRZFourier

## Problem Statement

Given a set of 3D points $\{\mathbf{x}_i\} \subset \mathbb{R}^3$ sampled from an unknown toroidal surface, reconstruct a `SurfaceRZFourier` representation. We assume:

- The **toroidal angle** $\phi_i$ of each point is known (e.g., from $\phi = \mathrm{atan2}(y, x)$).
- The **poloidal angle** $\theta_i$ is **unknown** — this is the core difficulty.

A `SurfaceRZFourier` represents a surface in cylindrical coordinates $(R, \phi, Z)$ via truncated Fourier series:

$$
R(\theta, \phi) = \sum_{m=0}^{M} \sum_{n=-N}^{N} R^c_{mn} \cos(m\theta - n N_{\mathrm{fp}} \phi)
$$

$$
Z(\theta, \phi) = \sum_{m=0}^{M} \sum_{n=-N}^{N} Z^s_{mn} \sin(m\theta - n N_{\mathrm{fp}} \phi)
$$

where $M$ is the maximum poloidal mode, $N$ the maximum toroidal mode, and $N_{\mathrm{fp}}$ the number of field periods (assuming stellarator symmetry).

If $\theta_i$ were known for each point, the coefficients $R^c_{mn}$ and $Z^s_{mn}$ could be found by **linear least squares** (the `from_points` function). The challenge is that both $\theta$ and the coefficients are unknowns.

## Approach: Alternating Optimization

We use an EM-style alternating optimization that iterates between solving for $\theta$ and solving for the Fourier coefficients.

### Step 1 — Initialize $\theta$ via arc-length parameterization

At each toroidal slice $\phi = \phi_j$, the data points form a closed curve in the $(R, Z)$ plane. We:

1. Compute the geometric angle of each point from the cross-section centroid: $\alpha_i = \mathrm{atan2}(Z_i - \bar{Z},\; R_i - \bar{R})$.
2. Sort points by $\alpha_i$ to recover the polygon ordering.
3. Assign $\theta$ **proportional to cumulative arc length** along the sorted polygon:

$$
\theta_i = 2\pi \cdot \frac{\sum_{k=1}^{i} \|\mathbf{x}_{k} - \mathbf{x}_{k-1}\|}{\text{total perimeter}}
$$

Arc-length parameterization is more accurate than equal spacing because it accounts for non-uniform point density around the cross-section.

### Step 2 — Alternating optimization loop

Repeat until convergence:

**(a) Fit Fourier coefficients** (linear least squares):

Given current $\theta_i$, solve for $R^c_{mn}$ and $Z^s_{mn}$ by minimizing $\sum_i \|\mathbf{x}_i - \mathbf{S}(\theta_i, \phi_i)\|^2$ via `from_points`. This is a standard linear least-squares problem.

**(b) Update $\theta$ by R-Z projection**:

For each point, find the $\theta$ on the fitted curve that minimizes the distance **in the $(R, Z)$ plane**:

$$
\theta_i^{\mathrm{new}} = \arg\min_{\theta} \left\| \begin{pmatrix} R_i \\ Z_i \end{pmatrix} - \begin{pmatrix} R(\theta, \phi_i) \\ Z(\theta, \phi_i) \end{pmatrix} \right\|^2
$$

We solve this by evaluating the curve on a fine grid ($N_{\mathrm{fine}} = 2048$ points) and performing vectorized line-segment projection onto each segment. Working in $(R, Z)$ rather than 3D $(x,y,z)$ is essential — it avoids artifacts from the toroidal embedding where nearby points in 3D may be far apart in $\theta$.

**(c) Damped update with re-sorting**:

To stabilize convergence, we apply a **relaxation factor** $\lambda = 0.5$:

$$
\theta_i \leftarrow \theta_i + \lambda \cdot \Delta\theta_i, \quad \Delta\theta_i = \theta_i^{\mathrm{new}} - \theta_i^{\mathrm{old}} \;\;(\mathrm{mod}\; 2\pi, \text{ wrapped to } [-\pi, \pi])
$$

Points are **re-sorted** by their projected $\theta$ at each step, since the projection may change the ordering (important for concave cross-sections).

**(d) Best-tracking**:

We track the solution with the lowest residual sum-of-squares (RSS) across all iterations, since the optimization can oscillate for non-convex cross-sections.

### Step 3 — Outlier correction

After the main loop, a small number of points may have their $\theta$ misassigned — typically at concavities where the R-Z projection maps a point to the wrong branch of the curve. These create localized Fourier ringing (spikes).

We detect outliers as points whose R-Z residual exceeds $8 \times$ the median residual. For each outlier:

1. Estimate the expected $\theta$ by interpolating from the two nearest non-outlier neighbors on the circle.
2. Re-project the point onto the curve, but **restricted to a local window** around the expected $\theta$ (preventing it from snapping to the wrong branch).
3. Re-fit the surface with corrected $\theta$, accepting the fix only if RSS improves.

### Step 4 — Coarse-to-fine refinement

For complex surfaces (e.g., non-convex stellarator boundaries), the alternating optimization can oscillate when using the target resolution directly. We use a **coarse-to-fine strategy**:

1. **Phase 1**: Fit with low resolution ($M=3, N=3$). Fewer Fourier modes act as natural regularization — the surface can't overfit $\theta$ errors, so the optimization converges more robustly.
2. **Phase 2**: Re-fit at the target resolution ($M=4, N=4$), initializing $\theta$ by projecting the data points onto the coarse fit. The coarse surface provides a good starting $\theta$ that is already close to the correct assignment.

### Step 5 — Spectral condensation

The final fitted surface is passed through **spectral condensation** (Hirshman, 1985), which minimizes the spectral width:

$$
W_p = \sum_{m,n} (m^2 + n^2)^p \left[(R^c_{mn})^2 + (Z^s_{mn})^2\right]
$$

subject to a constraint on the maximum normal displacement from the original surface. This compresses the energy into low-order modes without significantly changing the surface shape.

## Results

Tested on three surfaces of increasing complexity, with shuffled point ordering to verify robustness:

| Surface | Max normal displacement | RMS normal displacement |
|---------|------------------------|------------------------|
| D-Shape tokamak (Hirshman 1985) | 1.1e-2 | 4.3e-3 |
| 3D stellarator (test boundary) | 7.1e-3 | 8.8e-4 |
| HuggingFace stellarator (coarse-to-fine) | 3.7e-2 | 6.4e-3 |

## Key Design Decisions

- **R-Z projection instead of 3D**: The toroidal embedding maps nearby $\theta$ values to distant 3D points at different $\phi$. Projecting in the 2D $(R, Z)$ cross-section plane avoids this and is the natural space for the problem.
- **Damped updates**: Without relaxation, the theta updates overshoot and oscillate. $\lambda = 0.5$ provides a good balance between convergence speed and stability.
- **Re-sorting after projection**: The R-Z projection can change point ordering, especially at concavities. Allowing re-sorting is essential for complex cross-sections.
- **Coarse-to-fine**: Low-resolution fits are naturally regularized and converge more robustly. They provide good $\theta$ initialization for the fine fit, which is much more sensitive to the starting point.

## References

- Hirshman, S. P. "Optimized Fourier representations for three-dimensional magnetic surfaces." *The Physics of Fluids* 28.5 (1985): 1387-1391.
