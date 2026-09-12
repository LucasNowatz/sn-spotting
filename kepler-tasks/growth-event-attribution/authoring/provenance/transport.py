"""Horizontal transport operators for the air-mass-history benchmark.

Unsplit flux-form advection with a monotonic (Koren) limiter, plus explicit
horizontal diffusion.  Everything operates on the trailing two axes, so a
stack of size bins or tagged tracers is advanced in one pass.

Two properties matter for fairness and are verified in pilot_checks.py:

  * face winds are the discrete curl of the published corner streamfunction,
    so the flow is non-divergent on any grid and a uniform tracer stays
    uniform to machine precision;
  * the limiter keeps the scheme second order where the field is smooth, which
    holds numerical diffusion near 400 m2 s-1 on the 5 km grid.  First-order
    upwind there carries thousands of m2 s-1 and would bury any realistic
    horizontal eddy diffusivity.
"""

import numpy as np


def _koren(r):
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r,
                                                 (1.0 + 2.0 * r) / 3.0), 2.0))


def _stencil(q, axis):
    """The four-cell stencil for every interior face along `axis`.

    Returns (c_{k-1}, c_k, c_{k+1}, c_{k+2}) for faces k+1/2, k = 0..n-2, with
    the outermost cell replicated where the stencil runs off the grid.  Built
    from slices rather than np.roll, which copies the whole array.
    """
    n = q.shape[axis]
    if axis == -1:
        cm1 = np.concatenate((q[..., 0:1], q[..., 0:n - 2]), axis=-1)
        c0 = q[..., 0:n - 1]
        cp1 = q[..., 1:n]
        cp2 = np.concatenate((q[..., 2:n], q[..., n - 1:n]), axis=-1)
    else:
        cm1 = np.concatenate((q[..., 0:1, :], q[..., 0:n - 2, :]), axis=-2)
        c0 = q[..., 0:n - 1, :]
        cp1 = q[..., 1:n, :]
        cp2 = np.concatenate((q[..., 2:n, :], q[..., n - 1:n, :]), axis=-2)
    return cm1, c0, cp1, cp2


def _face_values(q, axis, vel_pos):
    """Limited face value of q on the interior faces of `axis` (-1 or -2)."""
    eps = 1e-300
    cm1, c0, cp1, cp2 = _stencil(q, axis)

    dn = cp1 - c0
    r = (c0 - cm1) / np.where(np.abs(dn) < eps, eps, dn)
    f_pos = c0 + 0.5 * _koren(r) * dn

    dn2 = -dn
    r2 = (cp1 - cp2) / np.where(np.abs(dn2) < eps, eps, dn2)
    f_neg = cp1 + 0.5 * _koren(r2) * dn2

    return np.where(vel_pos, f_pos, f_neg)


def advect(q, uf, vf, dx, dy, dt, q_in):
    """One unsplit advection sub-step on the trailing two axes of q.

    uf has shape (..., ny, nx+1), vf (..., ny+1, nx); both broadcast against
    q's leading axes.  q_in is the background carried in through inflow faces
    and broadcasts against q.
    """
    bg = np.broadcast_to(np.asarray(q_in, dtype=q.dtype), q.shape)

    fx = np.empty(q.shape[:-1] + (q.shape[-1] + 1,), dtype=q.dtype)
    fx[..., 1:-1] = uf[..., 1:-1] * _face_values(q, -1, uf[..., 1:-1] >= 0.0)
    inflow_w = uf[..., 0] >= 0.0
    fx[..., 0] = uf[..., 0] * np.where(inflow_w, bg[..., 0], q[..., 0])
    outflow_e = uf[..., -1] >= 0.0
    fx[..., -1] = uf[..., -1] * np.where(outflow_e, q[..., -1], bg[..., -1])

    fy = np.empty(q.shape[:-2] + (q.shape[-2] + 1, q.shape[-1]), dtype=q.dtype)
    fy[..., 1:-1, :] = vf[..., 1:-1, :] * _face_values(
        q, -2, vf[..., 1:-1, :] >= 0.0)
    inflow_s = vf[..., 0, :] >= 0.0
    fy[..., 0, :] = vf[..., 0, :] * np.where(inflow_s, bg[..., 0, :],
                                             q[..., 0, :])
    outflow_n = vf[..., -1, :] >= 0.0
    fy[..., -1, :] = vf[..., -1, :] * np.where(outflow_n, q[..., -1, :],
                                               bg[..., -1, :])

    return (q - dt / dx * (fx[..., 1:] - fx[..., :-1])
              - dt / dy * (fy[..., 1:, :] - fy[..., :-1, :]))


def diffuse(q, kh, dx, dy, dt, q_in):
    """Explicit horizontal diffusion, background value outside the domain."""
    bg = np.broadcast_to(np.asarray(q_in, dtype=q.dtype), q.shape)
    qe = np.empty(q.shape[:-2] + (q.shape[-2] + 2, q.shape[-1] + 2),
                  dtype=q.dtype)
    qe[..., 1:-1, 1:-1] = q
    qe[..., 0, 1:-1] = bg[..., 0, :]
    qe[..., -1, 1:-1] = bg[..., -1, :]
    qe[..., 1:-1, 0] = bg[..., :, 0]
    qe[..., 1:-1, -1] = bg[..., :, -1]
    qe[..., 0, 0] = bg[..., 0, 0]
    qe[..., 0, -1] = bg[..., 0, -1]
    qe[..., -1, 0] = bg[..., -1, 0]
    qe[..., -1, -1] = bg[..., -1, -1]
    lap = ((qe[..., 1:-1, 2:] - 2.0 * q + qe[..., 1:-1, :-2]) / (dx * dx)
           + (qe[..., 2:, 1:-1] - 2.0 * q + qe[..., :-2, 1:-1]) / (dy * dy))
    return q + dt * kh * lap
