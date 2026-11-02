import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs, polygon_segments
from cadkernel.geometry.frames import curvature_torsion
from cadkernel.ops.extrude import extrude
from cadkernel.ops.measure import loop_vector_area, surface_vector_area, volume
from cadkernel.ops.sweep import arc_length, path_frames, place, sweep
from cadkernel.topology.halfedge import check

RECT = polygon_segments([[0.2, -0.1], [0.2, 0.1], [-0.2, 0.1], [-0.2, -0.1]])  # area 0.08
CIRCLE = circle_arcs([0, 0], 0.2)
SPACE_PATH = NurbsCurve.bezier([[0, 0, 0], [2, 0, 0.5], [2, 2, 1], [0, 2, 2]])
S_PATH = NurbsCurve.bezier([[0, 0, 0], [0, 1.5, 1], [0, -1.5, 2], [0, 0, 3]])  # planar, inflected


def sides(solid):
    return [f.surface for f in solid.faces if not isinstance(f.surface, Plane)]


@pytest.mark.parametrize("profile", [RECT, CIRCLE])
@pytest.mark.parametrize("frame", ["rmf", "frenet"])
def test_every_step_is_valid_and_the_result_is_consistent(profile, frame):
    """Valid after every operator, every rail on both of its skins (check), an extrusion's
    counts, and the Stokes orientation identity of 03_Revolve.md on every side face."""
    steps = []
    solid = sweep(profile, SPACE_PATH, frame=frame, steps=steps)
    assert all(step["problems"] == [] for step in steps)
    assert check(solid) == []
    c, n = solid.counts(), len(profile)
    assert (c["V"], c["E"], c["F"]) == (2 * n, 3 * n, n + 2)
    for face in solid.faces:
        if not isinstance(face.surface, Plane):
            boundary = sum(loop_vector_area(loop) for loop in face.loops)
            assert np.allclose(boundary, face.sense * surface_vector_area(face.surface), atol=1e-12)


def test_every_section_lies_on_the_solid():
    """Exact at the sections: they are interpolated (04_Loft.md)."""
    solid = sweep(RECT, SPACE_PATH, sections=6)
    C, T, r, s = path_frames(SPACE_PATH, np.linspace(0, 1, 6))
    for frame in zip(C, T, r, s):
        for curve in place(RECT, *frame):
            assert min(f.distance(curve(np.linspace(0, 1, 5))).max() for f in sides(solid)) < 1e-10


def test_a_straight_path_gives_the_extrusion():
    """Along a line every frame is constant and the sections are translates; cubic interpolation
    reproduces a linear function exactly, so the skin is the extruded surface, and the volume is
    A L to rounding. Frenet has no frame on a line, and is rejected."""
    line = NurbsCurve.line([0, 0, 0], [0, 0, 2])
    assert np.isclose(volume(sweep(RECT, line)), volume(extrude(RECT, (0, 0, 2))), atol=1e-15)
    assert np.isclose(volume(sweep(CIRCLE, line)), 2 * np.pi * 0.04, atol=1e-14)
    with pytest.raises(ValueError, match="Frenet frame is undefined"):
        sweep(RECT, line, frame="frenet")


def test_sweep_converges_to_the_exact_sweep():
    """The rectangle along SPACE_PATH (L = 4.4967) with K sections. err: largest distance from
    the exact swept surface (each profile edge placed by path_frames at 21 values of t, 5 points
    per edge) to the skin that edge sweeps; dV: volume - A L, which is 0 for the exact sweep
    (05_Sweep.md). Measured:
        K   err (rmf)   dV (rmf)    err (frenet)  dV (frenet)
        5   9.6e-3      3.1e-3      9.5e-3        4.3e-3
        9   1.7e-4      -6.6e-6     1.6e-4        -1.2e-5
        17  4.7e-6      -1.5e-6     4.5e-6        -1.9e-6
        33  2.7e-7      -8.0e-8     2.5e-7        -1.0e-7
    """
    length = arc_length(SPACE_PATH)
    assert np.isclose(length, 4.496718565, atol=1e-9)
    expected = {"rmf": {5: (9.55e-3, 3.10e-3), 9: (1.66e-4, -6.61e-6), 17: (4.66e-6, -1.46e-6),
                        33: (2.65e-7, -8.03e-8)},
                "frenet": {5: (9.47e-3, 4.32e-3), 9: (1.61e-4, -1.16e-5), 17: (4.50e-6, -1.88e-6),
                           33: (2.51e-7, -1.01e-7)}}
    t = np.linspace(0, 1, 21)
    for frame, rows in expected.items():
        exact = [place(RECT, *f) for f in zip(*path_frames(SPACE_PATH, t, frame))]
        for k, (err, dv) in rows.items():
            solid = sweep(RECT, SPACE_PATH, frame=frame, sections=k)
            skins = sides(solid)  # skin j is swept by profile edge j
            measured = max(skins[j].distance(section[j](np.linspace(0, 1, 5))).max()
                           for section in exact for j in range(len(RECT)))
            assert np.isclose(measured, err, rtol=0.02)
            assert np.isclose(volume(solid) - 0.08 * length, dv, rtol=0.02)


def test_frenet_flips_the_profile_at_an_inflection():
    """On the planar S_PATH, Frenet's N reverses at the inflection (t = 1/2), so between the two
    sections around it the profile turns half a turn and the skin twists through the axis. The
    RMF never turns: started at N(0), in the path's plane, its s = T x r is the plane's normal
    (1, 0, 0) at every section."""
    t = np.linspace(0, 1, 8)
    _, _, r_frenet, _ = path_frames(S_PATH, t, "frenet")
    _, _, _, s_rmf = path_frames(S_PATH, t, "rmf")
    turns = np.einsum("ij,ij->i", r_frenet[:-1], r_frenet[1:])
    assert turns[3] < -0.99 and (np.delete(turns, 3) > 0.5).all()
    assert np.abs(np.abs(s_rmf) - [1, 0, 0]).max() < 1e-15
    assert np.allclose(s_rmf, s_rmf[0])


def test_sweep_spacing_comparison():
    """Relative volume error, volume / (A L) - 1, of the rectangle swept along two paths, for
    three ways of spacing the sections in v (the sections are equally spaced in t, so "path",
    v_s = t_s, is also "uniform"). Measured, K = 5, 9, 17, 33:
        SPACE_PATH     path         8.6e-3   -1.8e-5   -4.1e-6   -2.2e-7
                       chord        2.9e-2   -6.4e-4   -2.1e-5   -6.0e-7
                       centripetal  1.8e-2   -2.0e-4   -1.0e-5   -4.0e-7
        uneven         path         7.9e-6    2.5e-5    1.9e-6    1.3e-8
                       chord        5.0e-2   -9.3e-5    3.6e-6    1.1e-7
                       centripetal  4.3e-3    9.1e-6    3.2e-6    5.1e-8
    "path" is the most accurate in 7 of the 8 columns; centripetal wins once (uneven, K = 9).
    `uneven` crowds three control points near its start, so |C'| varies strongly along it."""
    uneven = NurbsCurve.bezier([[0, 0, 0], [0.2, 0, 0.1], [0.4, 0.1, 0.2], [3, 2, 2]])
    cases = [
        (SPACE_PATH, {"path": (8.60e-3, -1.84e-5, -4.07e-6, -2.23e-7),
                      "chord": (2.94e-2, -6.38e-4, -2.07e-5, -5.97e-7),
                      "centripetal": (1.76e-2, -1.98e-4, -1.00e-5, -3.95e-7)}),
        (uneven, {"path": (7.87e-6, 2.48e-5, 1.87e-6, 1.30e-8),
                  "chord": (4.95e-2, -9.28e-5, 3.60e-6, 1.12e-7),
                  "centripetal": (4.26e-3, 9.06e-6, 3.22e-6, 5.13e-8)}),
    ]
    for path, rows in cases:
        exact = 0.08 * arc_length(path)
        for rule, errors in rows.items():
            for k, error in zip((5, 9, 17, 33), errors):
                solid = sweep(RECT, path, sections=k, parametrization=rule)
                assert np.isclose(volume(solid) / exact - 1, error, rtol=0.02)


def test_frenet_needs_far_more_sections_where_it_turns_fast():
    """The path of scripts/04_modeling_ops/04_sweep.py: a cubic Bezier through (0, 0), (1.4, 1.8),
    (2.2, -1.8), (3.6, 0) seen from above, lifted to heights lift * (0, 2, 1, 2). Its curvature
    dips near t = 1/2 (to 0.078 at lift 0.5, 0.025 at lift 0.15), and there Frenet's frame turns
    by nearly half a turn against the RMF (int tau ds = 3.005 and 3.094 rad). Relative volume
    error, volume / (A L) - 1, with K sections:
        lift 0.5    K = 12: rmf 4.8e-4, frenet -7.8e-3;  24: -1.3e-5, -1.5e-3;  48: 7.0e-7, -1.05e-4
        lift 0.15   K = 12: rmf 6.8e-4, frenet -1.2e-2;  24: -1.4e-5, -5.2e-3;  48: 1.2e-6, -1.5e-3
    At 48 sections Frenet is 150 and 1200 times worse: the skin has to follow its fast turn."""
    xy = np.array([[0.0, 0.0], [1.4, 1.8], [2.2, -1.8], [3.6, 0.0]])
    rect = polygon_segments([[0.25, -0.08], [0.25, 0.08], [-0.25, 0.08], [-0.25, -0.08]])
    expected = {0.5: {12: (4.79e-4, -7.82e-3), 24: (-1.30e-5, -1.49e-3), 48: (6.96e-7, -1.05e-4)},
                0.15: {12: (6.77e-4, -1.18e-2), 24: (-1.37e-5, -5.15e-3), 48: (1.22e-6, -1.46e-3)}}
    shape = {0.5: (0.078, 3.005), 0.15: (0.025, 3.094)}  # smallest kappa, int tau ds
    for lift, rows in expected.items():
        path = NurbsCurve.bezier(np.c_[xy, lift * np.array([0, 2, 1, 2])])
        tt = np.linspace(0, 1, 4001)
        _, d1, d2, d3 = path.derivatives(tt, 3)
        kappa, tau = curvature_torsion(d1, d2, d3)
        total = np.trapezoid(tau * np.linalg.norm(d1, axis=1), tt)
        assert np.allclose((kappa.min(), total), shape[lift], atol=1e-3)
        exact = 0.08 * arc_length(path)
        for k, errors in rows.items():
            for frame, error in zip(("rmf", "frenet"), errors):
                measured = volume(sweep(rect, path, frame=frame, sections=k)) / exact - 1
                assert np.isclose(measured, error, rtol=0.02)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="xy-plane"):
        sweep([c.translated((0, 0, 1)) for c in RECT], SPACE_PATH)
    with pytest.raises(ValueError, match="regular"):
        sweep(RECT, NurbsCurve.bezier([[0, 0, 0], [0, 0, 0], [1, 1, 0], [2, 0, 1]]))
    with pytest.raises(ValueError, match="two sections"):
        sweep(RECT, SPACE_PATH, sections=1)
