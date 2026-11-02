import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve
from cadkernel.geometry.frames import (
    angle_about_tangent,
    curvature_torsion,
    frenet_frames,
    rotation_minimizing_frames,
)

S_CURVE = NurbsCurve.bezier([[0, 0, 0], [1, 1, 0], [2, -1, 0], [3, 0, 0]])  # inflection at 1/2
SPACE_PATH = NurbsCurve.bezier([[0, 0, 0], [2, 0, 0.5], [2, 2, 1], [0, 2, 2]])


def helix(t, a=1.0, b=0.5):
    """C(t) = (a cos t, a sin t, b t): curvature a / (a^2 + b^2), torsion b / (a^2 + b^2),
    speed sqrt(a^2 + b^2), all constant."""
    zero = np.zeros_like(t)
    return (np.c_[a * np.cos(t), a * np.sin(t), b * t],
            np.c_[-a * np.sin(t), a * np.cos(t), b + zero],
            np.c_[-a * np.cos(t), -a * np.sin(t), zero])


def trefoil(t):
    """The (2, 3) torus knot: C(t) = ((2 + cos 3t) cos 2t, (2 + cos 3t) sin 2t, sin 3t), closed on
    [0, 2 pi], with its first three derivatives."""
    c3, s3, c2, s2 = np.cos(3 * t), np.sin(3 * t), np.cos(2 * t), np.sin(2 * t)
    r, r1, r2, r3 = 2 + c3, -3 * s3, -9 * c3, 27 * s3  # r(t) = 2 + cos 3t and its derivatives
    return (np.c_[r * c2, r * s2, s3],
            np.c_[r1 * c2 - 2 * r * s2, r1 * s2 + 2 * r * c2, 3 * c3],
            np.c_[r2 * c2 - 4 * r1 * s2 - 4 * r * c2, r2 * s2 + 4 * r1 * c2 - 4 * r * s2, -9 * s3],
            np.c_[r3 * c2 - 6 * r2 * s2 - 12 * r1 * c2 + 8 * r * s2,
                  r3 * s2 + 6 * r2 * c2 - 12 * r1 * s2 - 8 * r * c2, -27 * c3])


def test_frenet_is_undefined_on_a_line_and_flips_at_an_inflection():
    """On a line C'' = 0, so N and B do not exist. On a planar curve B = +-(the plane's normal),
    and its sign is the sign of the curvature's turn: through the inflection N jumps to -N."""
    _, d1, d2 = NurbsCurve.line([0, 0, 0], [1, 2, 3]).derivatives(np.linspace(0, 1, 5), 2)
    assert np.isnan(frenet_frames(d1, d2)[1]).all()
    _, d1, d2 = S_CURVE.derivatives([0.5 - 1e-6, 0.5, 0.5 + 1e-6], 2)
    _, normal, binormal = frenet_frames(d1, d2)
    assert np.isnan(normal[1]).all()
    assert normal[0] @ normal[2] < -0.999999
    assert np.allclose(binormal[[0, 2]], [[0, 0, -1], [0, 0, 1]])


@pytest.mark.parametrize("method", ["double_reflection", "projection"])
def test_rmf_of_a_planar_curve_is_the_plane_normal(method):
    """Started at the plane's normal e, the RMF stays e: e is normal to T and e' = 0 is parallel
    to T (05_Sweep.md). Both schemes keep it to rounding: each step's reflections are in planes
    that contain e, and projection removes a component that is zero."""
    points, d1 = S_CURVE.derivatives(np.linspace(0, 1, 50), 1)
    tangent = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    _, r, _ = rotation_minimizing_frames(points, tangent, [0, 0, 1], method)
    assert np.abs(r - [0, 0, 1]).max() < 1e-15


def test_double_reflection_keeps_an_orthonormal_frame():
    """R_2 R_1 is a rotation taking T_i to T_{i+1}, so it takes a unit normal of T_i to a unit
    normal of T_{i+1}: orthonormality holds to rounding at every step, with no normalization."""
    points, d1, _ = helix(np.linspace(0, 20, 400))
    tangent = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    _, r, s = rotation_minimizing_frames(points, tangent, [1, 0, 0])
    assert np.abs(np.linalg.norm(r, axis=1) - 1).max() < 1e-14
    assert np.abs(np.einsum("ij,ij->i", r, tangent)).max() < 1e-14
    assert np.abs(np.linalg.norm(s, axis=1) - 1).max() < 1e-14


def test_rmf_convergence_on_a_helix():
    """On the helix (a, b) = (1, 0.5) over t in [0, 4 pi], the exact RMF turns against the Frenet
    frame by theta(t) = -tau s(t) = -b t / sqrt(a^2 + b^2) (05_Sweep.md). The largest error of the
    computed angle, with m steps (double reflection, projection):
        m = 8:    7.86e-3   1.52
        m = 16:   4.79e-4   0.345
        m = 32:   2.98e-5   0.171
        m = 64:   1.86e-6   0.0863
        m = 128:  1.16e-7   0.0435
    Doubling m divides the double-reflection error by 16 (fourth order, as Wang et al. state) and
    the projection error by 2 (first order)."""
    expected = {"double_reflection": [7.86e-3, 4.79e-4, 2.98e-5, 1.86e-6, 1.16e-7],
                "projection": [1.52, 0.345, 0.171, 0.0863, 0.0435]}
    for method, errors in expected.items():
        for m, error in zip((8, 16, 32, 64, 128), errors):
            t = np.linspace(0, 4 * np.pi, m + 1)
            points, d1, d2 = helix(t)
            tangent, normal, binormal = frenet_frames(d1, d2)
            _, r, _ = rotation_minimizing_frames(points, tangent, normal[0], method)
            theta = np.unwrap(angle_about_tangent(normal, binormal, r))
            assert np.isclose(np.abs(theta + 0.5 * t / np.sqrt(1.25)).max(), error, rtol=0.01)


def test_frenet_turns_against_rmf_by_the_integrated_torsion():
    """The same relation on a NURBS path (a cubic Bezier with torsion between 0.095 and 0.36):
    the angle of the RMF normal, measured from N towards B, is -\\int_0^t tau ds. The integral
    by the trapezoid rule on 20001 points; the RMF by double reflection on 2049. They agree to
    1e-8, and by t = 1 the two frames are 1.004 rad (57.5 deg) apart."""
    tt = np.linspace(0, 1, 20001)
    _, d1, d2, d3 = SPACE_PATH.derivatives(tt, 3)
    kappa, tau = curvature_torsion(d1, d2, d3)
    assert np.isclose(tau.min(), 0.0952, atol=1e-4) and np.isclose(tau.max(), 0.3565, atol=1e-4)
    integrand = tau * np.linalg.norm(d1, axis=1)
    total = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(tt))])
    t = np.linspace(0, 1, 2049)
    points, d1, d2 = SPACE_PATH.derivatives(t, 2)
    tangent, normal, binormal = frenet_frames(d1, d2)
    _, r, _ = rotation_minimizing_frames(points, tangent, normal[0])
    theta = angle_about_tangent(normal, binormal, r)
    assert np.abs(theta + np.interp(t, tt, total)).max() < 1e-8
    assert np.isclose(theta[-1], -1.00401, atol=1e-5)


def test_rmf_does_not_close_on_a_trefoil():
    """Carried once round the closed trefoil, the RMF comes back turned by -\\oint tau ds mod 2 pi
    = 3.0270 rad (173.4 deg) about the tangent: Frenet's frame closes up (it is periodic), and the
    RMF differs from it by that angle. Error of the measured mismatch with m steps (double
    reflection, projection): m = 64: 3.1e-4, 0.077; m = 512: 7.7e-8, 9.5e-4."""
    tt = np.linspace(0, 2 * np.pi, 200001)
    _, d1, d2, d3 = trefoil(tt)
    _, tau = curvature_torsion(d1, d2, d3)
    total = np.trapezoid(tau * np.linalg.norm(d1, axis=1), tt)
    expected = np.angle(np.exp(-1j * total))
    assert np.isclose(expected, 3.0270, atol=1e-4)
    for m, errors in {64: (3.1e-4, 0.077), 512: (7.7e-8, 9.5e-4)}.items():
        points, d1, d2, _ = trefoil(np.linspace(0, 2 * np.pi, m + 1))
        tangent, normal, _ = frenet_frames(d1, d2)
        for method, error in zip(("double_reflection", "projection"), errors):
            _, r, s = rotation_minimizing_frames(points, tangent, normal[0], method)
            mismatch = angle_about_tangent(r[:1], s[:1], r[-1:])[0]
            assert np.isclose(abs(mismatch - expected), error, rtol=0.02)
