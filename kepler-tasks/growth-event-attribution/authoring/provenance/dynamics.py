"""Horizontal dynamics for the event-attribution generator.

Flux-form advection on face-normal winds with a monotone (Koren) limiter,
applied unsplit in the two horizontal directions, followed by an explicit
diffusion step.  Every function acts on the trailing two axes, so a stack of
diameter bins or tagged tracers advances in one call.

The face winds come from the discrete curl of the corner streamfunction, so
the flow is non-divergent on the solver's grid and a uniform field stays
uniform to rounding.  The limiter keeps the scheme second order where the
field is smooth, which holds numerical diffusion well below the eddy
diffusivity that has to be inferred.  The generator runs with a 30 s step;
the reference solution is a separate implementation of the same class of
scheme at 60 s, and the alternative (monotonized-central) limiter is used to
measure how far a different correct scheme lands from the truth.
"""

import numpy as np


LIMITER = "koren"          # the generator's choice; "mc" is the alternative scheme


def limiter(r):
    """Second-order TVD limiter: Koren by default, monotonized-central when
    LIMITER is set to "mc" (used to measure the alternative-scheme floor)."""
    if LIMITER == "mc":
        return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, (1.0 + 2.0 * r) / 3.0), 2.0))


def _stencil(q, axis):
    n = q.shape[axis]
    if axis == -1:
        a = np.concatenate((q[..., 0:1], q[..., 0:n - 2]), axis=-1)
        b = q[..., 0:n - 1]
        c = q[..., 1:n]
        d = np.concatenate((q[..., 2:n], q[..., n - 1:n]), axis=-1)
    else:
        a = np.concatenate((q[..., 0:1, :], q[..., 0:n - 2, :]), axis=-2)
        b = q[..., 0:n - 1, :]
        c = q[..., 1:n, :]
        d = np.concatenate((q[..., 2:n, :], q[..., n - 1:n, :]), axis=-2)
    return a, b, c, d


def face_values(q, axis, upwind_positive):
    """Limited face value on the interior faces along `axis`."""
    tiny = 1e-300
    a, b, c, d = _stencil(q, axis)
    fwd = c - b
    r_pos = (b - a) / np.where(np.abs(fwd) < tiny, tiny, fwd)
    v_pos = b + 0.5 * limiter(r_pos) * fwd
    bwd = -fwd
    r_neg = (c - d) / np.where(np.abs(bwd) < tiny, tiny, bwd)
    v_neg = c + 0.5 * limiter(r_neg) * bwd
    return np.where(upwind_positive, v_pos, v_neg)


def advection_step(q, uf, vf, dx, dy, dt, inflow):
    """One unsplit flux-form advection update.

    uf is (..., ny, nx+1), vf is (..., ny+1, nx).  `inflow` is the value
    carried in through inflow faces and broadcasts against q.
    """
    bg = np.broadcast_to(np.asarray(inflow, dtype=q.dtype), q.shape)
    fx = np.empty(q.shape[:-1] + (q.shape[-1] + 1,), dtype=q.dtype)
    fx[..., 1:-1] = uf[..., 1:-1] * face_values(q, -1, uf[..., 1:-1] >= 0.0)
    fx[..., 0] = uf[..., 0] * np.where(uf[..., 0] >= 0.0, bg[..., 0], q[..., 0])
    fx[..., -1] = uf[..., -1] * np.where(uf[..., -1] >= 0.0, q[..., -1], bg[..., -1])
    fy = np.empty(q.shape[:-2] + (q.shape[-2] + 1, q.shape[-1]), dtype=q.dtype)
    fy[..., 1:-1, :] = vf[..., 1:-1, :] * face_values(q, -2, vf[..., 1:-1, :] >= 0.0)
    fy[..., 0, :] = vf[..., 0, :] * np.where(vf[..., 0, :] >= 0.0, bg[..., 0, :], q[..., 0, :])
    fy[..., -1, :] = vf[..., -1, :] * np.where(vf[..., -1, :] >= 0.0, q[..., -1, :], bg[..., -1, :])
    return (q - dt / dx * (fx[..., 1:] - fx[..., :-1])
              - dt / dy * (fy[..., 1:, :] - fy[..., :-1, :]))


def diffusion_step(q, kh, dx, dy, dt, outside):
    """Explicit Laplacian diffusion with the background value outside."""
    bg = np.broadcast_to(np.asarray(outside, dtype=q.dtype), q.shape)
    qe = np.empty(q.shape[:-2] + (q.shape[-2] + 2, q.shape[-1] + 2), dtype=q.dtype)
    qe[..., 1:-1, 1:-1] = q
    qe[..., 0, 1:-1] = bg[..., 0, :]
    qe[..., -1, 1:-1] = bg[..., -1, :]
    qe[..., 1:-1, 0] = bg[..., :, 0]
    qe[..., 1:-1, -1] = bg[..., :, -1]
    qe[..., 0, 0] = bg[..., 0, 0]; qe[..., 0, -1] = bg[..., 0, -1]
    qe[..., -1, 0] = bg[..., -1, 0]; qe[..., -1, -1] = bg[..., -1, -1]
    lap = ((qe[..., 1:-1, 2:] - 2.0 * q + qe[..., 1:-1, :-2]) / (dx * dx)
           + (qe[..., 2:, 1:-1] - 2.0 * q + qe[..., :-2, 1:-1]) / (dy * dy))
    return q + dt * kh * lap


def transport_step(q, uf, vf, dx, dy, dt, kh, background):
    return diffusion_step(advection_step(q, uf, vf, dx, dy, dt, background),
                          kh, dx, dy, dt, background)
