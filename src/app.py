import copy
import tkinter as tk
from functools import lru_cache
import time
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
from scipy.optimize import brentq
from scipy.special import fresnel

ROAD_CLEARANCE_MARGIN = 0.5
LAKE_BYPASS_MARGIN = 2.0
DEFAULT_BUFFER_KM = 28.0

DEFAULT_A = 1.67
DEFAULT_R = 2.0
DEFAULT_OPTIMIZE = False

EPS = 1e-10
MIN_CIRCULAR_ARC_LEN = 0.05
MAX_OBSTACLE_INSERTIONS_FACTOR = 4
MAX_OBSTACLE_BUILD_SECONDS = 4.0

DEFAULT_CITIES = [
    {
        "name": "M1",
        "center": np.array([0.0, 0.0]),
        "radius": 12.0,
        "road_angle": 0.0,
        "road_side": "auto",
        "_bypass_manual": False,
    },
    {
        "name": "M2",
        "center": np.array([280.0, 250.0]),
        "radius": 16.0,
        "road_angle": 30.0,
        "road_side": "auto",
        "_bypass_manual": False,
    },
    {
        "name": "M3",
        "center": np.array([600.0, -95.0]),
        "radius": 18.0,
        "road_angle": -25.0,
        "road_side": 1,
        "_bypass_manual": True,
    },
    {
        "name": "M4",
        "center": np.array([950.0, -365.0]),
        "radius": 16.0,
        "road_angle": -20.0,
        "road_side": 1,
        "_bypass_manual": True,
    },
    {
        "name": "M5",
        "center": np.array([1330.0, 400.0]),
        "radius": 20.0,
        "road_angle": 70.0,
        "road_side": 1,
        "_bypass_manual": True,
    },
]

DEFAULT_LAKES = []


def road_contact_radius(city):
    if city.get("is_lake"):
        return city["radius"]
    return city["radius"] + ROAD_CLEARANCE_MARGIN


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def unit(a):
    return np.array([np.cos(a), np.sin(a)])


def normal_left(t):
    return np.array([-t[1], t[0]])


def normalize_road_side(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return None
    text = str(value).strip().lower()
    if text in ("+", "+1", "1", "left", "l", "plus"):
        return 1
    if text in ("-", "-1", "right", "r", "minus"):
        return -1
    return None


def road_side_label(value):
    side = normalize_road_side(value)
    if side == 1:
        return "+"
    if side == -1:
        return "-"
    return "auto"


def signed_angle(a_from, a_to):
    diff = a_to - a_from
    return (diff + np.pi) % (2 * np.pi) - np.pi


def lines_intersect(P1, t1, P2, t2):
    cross = t1[0] * t2[1] - t1[1] * t2[0]
    if abs(cross) < EPS:
        return None, None, None
    dP = P2 - P1
    s1 = (dP[0] * t2[1] - dP[1] * t2[0]) / cross
    s2 = (dP[0] * t1[1] - dP[1] * t1[0]) / cross
    return P1 + s1 * t1, s1, s2


def road_line_exit(center, radius, road_origin, road_dir, margin=ROAD_CLEARANCE_MARGIN):
    f = road_origin - center
    b = 2.0 * np.dot(f, road_dir)
    c = np.dot(f, f) - (radius + margin) ** 2
    disc = b * b - 4.0 * c
    if disc < 0:
        return road_origin
    sq = np.sqrt(disc)
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    candidates = [t for t in (t1, t2) if t >= -1e-6]
    if not candidates:
        return road_origin
    return road_origin + max(candidates) * road_dir


def _segment_intersects_circle(p_start, p_end, center, radius, tol=0.1):
    d = p_end - p_start
    f = p_start - center
    a = np.dot(d, d)
    if a < 1e-12:
        return np.linalg.norm(f) < radius - tol
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - (radius - tol) ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)


def check_collision(segments, city_center, radius):
    for seg in segments:
        if seg["type"] == "straight":
            if _segment_intersects_circle(seg["p_start"], seg["p_end"], city_center, radius):
                return True
        else:
            for key in ("pts_in_g", "pts_arc_g", "pts_exit_g"):
                pts = seg[key]
                if np.any(np.linalg.norm(pts - city_center, axis=1) < radius - 0.1):
                    return True
    return False


def segment_collides_with_circle(seg, center, radius):
    if seg["type"] == "straight":
        return _segment_intersects_circle(seg["p_start"], seg["p_end"], center, radius)
    for key in ("pts_in_g", "pts_arc_g", "pts_exit_g"):
        pts = seg[key]
        if np.any(np.linalg.norm(pts - center, axis=1) < radius - 0.1):
            return True
    return False


def segment_length(segments):
    total = 0.0
    for seg in segments:
        if seg["type"] == "straight":
            total += np.linalg.norm(seg["p_end"] - seg["p_start"])
        else:
            for key in ("pts_in_g", "pts_arc_g", "pts_exit_g"):
                total += np.sum(np.linalg.norm(np.diff(seg[key], axis=0), axis=1))
    return total


def road_length(segments):
    return segment_length(segments)


def figure_count(segments):
    return sum(1 for seg in segments if seg["type"] == "figure")


def euclid_dist(c1, c2):
    return np.linalg.norm(c1["center"] - c2["center"])


def total_euclid(order, cities):
    return sum(euclid_dist(cities[order[i]], cities[order[i + 1]]) for i in range(len(order) - 1))


def nearest_neighbor(cities, start_idx=0):
    n = len(cities)
    order = [start_idx]
    visited = {start_idx}
    current = start_idx
    while len(order) < n:
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if j in visited:
                continue
            d = euclid_dist(cities[current], cities[j])
            if d < best_d:
                best_d, best_j = d, j
        if best_j == -1:
            break
        order.append(best_j)
        visited.add(best_j)
        current = best_j
    return order


def two_opt(order, cities):    
    best = order[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                candidate = best[:]
                candidate[i:j] = reversed(candidate[i:j])
                if total_euclid(candidate, cities) + 1e-9 < total_euclid(best, cities):
                    best = candidate
                    improved = True
        order = best
    return best


def first_blocked_with_margin(p_start, p_end, obstacles, margin=5.0):
    best_obs = None
    best_t = None
    d = p_end - p_start
    a = np.dot(d, d)
    if a < 1e-12:
        return None
    for obs in obstacles:
        center = obs["center"]
        radius = obs["radius"] + margin
        f = p_start - center
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue
        sq = np.sqrt(disc)
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
        if not candidates:
            if t1 < 0.0 and t2 > 1.0:
                t_hit = 0.0
            else:
                continue
        else:
            t_hit = min(candidates)
        if best_t is None or t_hit < best_t:
            best_t = t_hit
            best_obs = obs
    return best_obs


def blocked_with_margin_sorted(p_start, p_end, obstacles, margin=5.0):
    hits = []
    d = p_end - p_start
    a = np.dot(d, d)
    if a < 1e-12:
        return hits
    for obs in obstacles:
        center = obs["center"]
        radius = obs["radius"] + margin
        f = p_start - center
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue
        sq = np.sqrt(disc)
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
        if not candidates:
            if t1 < 0.0 and t2 > 1.0:
                t_hit = 0.0
            else:
                continue
        else:
            t_hit = min(candidates)
        hits.append((t_hit, obs))
    hits.sort(key=lambda item: item[0])
    return hits


def _obstacle_key(obs):
    c = obs["center"]
    return (round(float(c[0]), 6), round(float(c[1]), 6), round(float(obs["radius"]), 6))


def _adaptive_plot_samples(total_turn, base_count=600):
    extra = int(np.ceil(120 * max(0.0, total_turn)))
    return max(base_count, 240 + extra)


def _phase_integral(a, b, c, s_values):
    s_values = np.asarray(s_values, dtype=float)
    if abs(a) < EPS:
        if abs(b) < EPS:
            return s_values * np.cos(c), s_values * np.sin(c)
        phase = c + b * s_values
        x = (np.sin(phase) - np.sin(c)) / b
        y = (-np.cos(phase) + np.cos(c)) / b
        return x, y

    sigma = 1.0 if a > 0 else -1.0
    abs_a = abs(a)
    shift = b / (2.0 * a)
    scale = np.sqrt(2.0 * abs_a / np.pi)
    w0 = scale * shift
    w = scale * (s_values + shift)
    d = c - b * b / (4.0 * a)

    s0, c0 = fresnel(w0)
    s1, c1 = fresnel(w)
    fw0 = c0 + 1j * sigma * s0
    fw1 = c1 + 1j * sigma * s1
    pref = np.sqrt(np.pi / (2.0 * abs_a)) * np.exp(1j * d)
    vals = pref * (fw1 - fw0)
    return vals.real, vals.imag


def _clothoid_xy_exact(A_c, s_values, sign):
    return _phase_integral(sign * A_c / 2.0, 0.0, 0.0, s_values)


def _city_label(city, fallback_index):
    name = city.get("name")
    return str(name) if name else f"point {fallback_index + 1}"


def _min_arc_ok(phi_cir, R_c):
    return R_c * max(phi_cir, 0.0) >= MIN_CIRCULAR_ARC_LEN - 1e-9


@lru_cache(maxsize=20000)
def _figure_core_cached(alpha_in, alpha_out, R_c, A_c):
    theta = signed_angle(alpha_in, alpha_out)
    sign = np.sign(theta) if abs(theta) > EPS else 1.0
    lam = 1.0 / R_c
    S0 = lam / A_c
    phi_cl = A_c * S0**2 / 2.0
    phi_cir = abs(theta) - 2.0 * phi_cl

    if phi_cir < -1e-9:
        raise ValueError(
            "For the chosen R and A the classical clothoid-circle-clothoid figure does not exist."
        )

    phi_cir = max(phi_cir, 0.0)
    if not _min_arc_ok(phi_cir, R_c):
        raise ValueError(
            "The circular arc part of the classical figure is too short "
            f"(L_arc={R_c * phi_cir:.4g} < {MIN_CIRCULAR_ARC_LEN:.4g})."
        )

    Xin_end, Yin_end = _clothoid_xy_exact(A_c, np.array([S0]), sign)
    P_in_end = np.array([Xin_end[0], Yin_end[0]])
    angle_in_end = sign * phi_cl
    n_to_center = normal_left(unit(angle_in_end)) * sign
    C_arc = P_in_end + R_c * n_to_center

    v0 = P_in_end - C_arc
    a_start = np.arctan2(v0[1], v0[0])
    a_end = a_start + sign * phi_cir
    P_arc_end = C_arc + R_c * np.array([np.cos(a_end), np.sin(a_end)])
    angle_arc_end = sign * (phi_cl + phi_cir)

    Xex_end, Yex_end = _phase_integral(
        -sign * A_c / 2.0,
        sign * lam,
        angle_arc_end,
        np.array([S0]),
    )
    P_exit_end = P_arc_end + np.array([Xex_end[0], Yex_end[0]])

    phi_mid = a_start + sign * phi_cir / 2.0
    P_mid = C_arc + R_c * unit(phi_mid)
    bisect = sign * abs(theta) / 2.0
    n_mid = np.array([-np.sin(bisect), np.cos(bisect)]) * sign
    if abs(n_mid[1]) > EPS:
        lam_c = -P_mid[1] / n_mid[1]
        W_loc_x = P_mid[0] + lam_c * n_mid[0]
    else:
        W_loc_x = P_mid[0]

    return {
        "theta": theta,
        "sign": sign,
        "S0": S0,
        "phi_cl": phi_cl,
        "phi_cir": phi_cir,
        "W_loc_x": W_loc_x,
        "P_in_end": P_in_end,
        "C_arc": C_arc,
        "P_arc_end": P_arc_end,
        "P_exit_end": P_exit_end,
        "angle_arc_end": angle_arc_end,
    }


def build_figure(alpha_in, alpha_out, R_c, A_c, n=600):
    core = _figure_core_cached(alpha_in, alpha_out, R_c, A_c)
    theta = core["theta"]
    sign = core["sign"]
    S0 = core["S0"]
    phi_cl = core["phi_cl"]
    phi_cir = core["phi_cir"]
    lam = 1.0 / R_c
    n = _adaptive_plot_samples(abs(theta), base_count=n)
    s = np.linspace(0.0, S0, n)

    Xin, Yin = _clothoid_xy_exact(A_c, s, sign)
    pts_in = np.column_stack([Xin, Yin])

    P_in_end = core["P_in_end"].copy()
    C_arc = core["C_arc"].copy()
    P_arc_end = core["P_arc_end"].copy()
    angle_arc_end = core["angle_arc_end"]
    v0 = P_in_end - C_arc
    a_start = np.arctan2(v0[1], v0[0])
    a_end = np.arctan2(P_arc_end[1] - C_arc[1], P_arc_end[0] - C_arc[0])
    angs = np.linspace(a_start, a_end, n)
    pts_arc = C_arc + R_c * np.column_stack([np.cos(angs), np.sin(angs)])

    Xex, Yex = _phase_integral(-sign * A_c / 2.0, sign * lam, angle_arc_end, s)
    pts_exit = P_arc_end + np.column_stack([Xex, Yex])

    return {
        **core,
        "pts_in": pts_in,
        "pts_arc": pts_arc,
        "pts_exit": pts_exit,
    }


def place_figure(floc, W_global, t_in, alpha_in):
    Phi_start = W_global - floc["W_loc_x"] * t_in
    R = rot2(alpha_in)

    def tg(pts):
        return Phi_start + (R @ pts.T).T

    def tg1(p):
        return Phi_start + R @ p

    return {
        **floc,
        "Phi_start": Phi_start,
        "Phi_end": tg(floc["pts_exit"])[-1],
        "pts_in_g": tg(floc["pts_in"]),
        "pts_arc_g": tg(floc["pts_arc"]),
        "pts_exit_g": tg(floc["pts_exit"]),
        "P_in_end_g": tg1(floc["P_in_end"]),
        "C_arc_g": tg1(floc["C_arc"]),
        "P_arc_end_g": tg1(floc["P_arc_end"]),
    }


def place_figure_right(floc, T_end, alpha_in):
    R = rot2(alpha_in)
    Phi_start = T_end - R @ floc["P_exit_end"]

    def tg(pts):
        return Phi_start + (R @ pts.T).T

    def tg1(p):
        return Phi_start + R @ p

    return {
        **floc,
        "Phi_start": Phi_start,
        "Phi_end": T_end.copy(),
        "pts_in_g": tg(floc["pts_in"]),
        "pts_arc_g": tg(floc["pts_arc"]),
        "pts_exit_g": tg(floc["pts_exit"]),
        "P_in_end_g": tg1(floc["P_in_end"]),
        "C_arc_g": tg1(floc["C_arc"]),
        "P_arc_end_g": tg1(floc["P_arc_end"]),
    }


def _place_figure_light(floc, W_global, t_in, alpha_in):
    Phi_start = W_global - floc["W_loc_x"] * t_in
    R = rot2(alpha_in)
    return {"Phi_start": Phi_start, "Phi_end": Phi_start + R @ floc["P_exit_end"]}


def _place_figure_right_light(floc, T_end, alpha_in):
    Phi_start = T_end - rot2(alpha_in) @ floc["P_exit_end"]
    return {"Phi_start": Phi_start, "Phi_end": T_end.copy()}


def can_use_one_figure(T_from, alpha_from, T_to, alpha_to, R_c, A_c):
    t1 = unit(alpha_from)
    t2 = unit(alpha_to)
    W, s1, s2 = lines_intersect(T_from, t1, T_to, t2)
    if W is None:
        return False, None
    lam = 1.0 / R_c
    S0 = lam / A_c
    phi_cl = A_c * S0**2 / 2.0
    phi_cir = abs(signed_angle(alpha_from, alpha_to)) - 2.0 * phi_cl
    ok_geom = (s1 > -1e-6) and (s2 < 1e-6)
    ok_arc = phi_cir >= -1e-9 and _min_arc_ok(max(phi_cir, 0.0), R_c)
    return ok_geom and ok_arc, W


def _two_figure_trial(T_from, alpha_from, T_to, alpha_to, R_c, A_c, alpha_B):
    t_from = unit(alpha_from)
    floc2 = _figure_core_cached(alpha_from, alpha_B, R_c, A_c)
    W2 = T_from + floc2["W_loc_x"] * t_from
    fig2 = _place_figure_light(floc2, W2, t_from, alpha_from)

    floc3 = _figure_core_cached(alpha_B, alpha_to, R_c, A_c)
    fig3 = _place_figure_right_light(floc3, T_to, alpha_B)

    bridge = fig3["Phi_start"] - fig2["Phi_end"]
    t_B = unit(alpha_B)
    cross = bridge[0] * t_B[1] - bridge[1] * t_B[0]
    dot = np.dot(bridge, t_B)
    return cross, dot


def _solve_alpha_B(T_from, alpha_from, T_to, alpha_to, R_c, A_c):
    alpha_direct = np.arctan2(*(T_to - T_from)[::-1])
    
    def residual(angle):
        cross, _ = _two_figure_trial(T_from, alpha_from, T_to, alpha_to, R_c, A_c, angle)
        return cross

    def scan_grid(span, count):
        grid = alpha_direct + np.linspace(-span, span, count)
        samples = []
        for angle in grid:
            try:
                samples.append((angle, *_two_figure_trial(T_from, alpha_from, T_to, alpha_to, R_c, A_c, angle)))
            except ValueError:
                samples.append(None)

        roots = []
        for left, right in zip(samples, samples[1:]):
            if left is None or right is None:
                continue
            a0, f0, _ = left
            a1, f1, _ = right
            if abs(f0) < 1e-8:
                roots.append(a0)
                continue
            if f0 * f1 > 0:
                continue
            try:
                roots.append(brentq(residual, a0, a1, xtol=1e-10, rtol=1e-10, maxiter=100))
            except ValueError:
                pass
        return roots

    candidates = []
    for span, count in (
        (np.pi / 3.0, 241),
        (2.0 * np.pi / 3.0, 361),
        (np.pi - 1e-4, 721),
    ):
        candidates = scan_grid(span, count)
        if candidates:
            break

    best = None
    for alpha_B in candidates:
        try:
            cross, dot = _two_figure_trial(T_from, alpha_from, T_to, alpha_to, R_c, A_c, alpha_B)
        except ValueError:
            continue
        if abs(cross) > 1e-7 or dot < -1e-7:
            continue
        item = (
            np.linalg.norm(T_to - T_from) + max(dot, 0.0),
            abs(signed_angle(alpha_B, alpha_direct)),
            alpha_B,
            dot,
        )
        if best is None or item < best:
            best = item

    if best is None:
        raise ValueError(
            "The classical clothoid-straight-clothoid transition does not exist "
            f"for the chosen geometry and fixed parameters R={R_c:.4g}, A={A_c:.4g}."
        )
    return best[2], max(best[3], 0.0)


def build_segment(T_from, alpha_from, T_to, alpha_to, R_c, A_c):
    if np.linalg.norm(T_to - T_from) < 1e-6:
        return []

    alpha_direct = np.arctan2(*(T_to - T_from)[::-1])
    if abs(signed_angle(alpha_from, alpha_to)) < 1e-6 and abs(signed_angle(alpha_from, alpha_direct)) < 1e-6:
        return [{"type": "straight", "p_start": T_from, "p_end": T_to}]

    use_one, W = can_use_one_figure(T_from, alpha_from, T_to, alpha_to, R_c, A_c)
    if use_one:
        fig = place_figure(build_figure(alpha_from, alpha_to, R_c, A_c), W, unit(alpha_from), alpha_from)
        return [
            {"type": "straight", "p_start": T_from, "p_end": fig["Phi_start"]},
            {"type": "figure", **fig, "alpha_in": alpha_from, "alpha_out": alpha_to},
            {"type": "straight", "p_start": fig["Phi_end"], "p_end": T_to},
        ]

    alpha_B, bridge_len = _solve_alpha_B(T_from, alpha_from, T_to, alpha_to, R_c, A_c)
    fig2 = place_figure(
        build_figure(alpha_from, alpha_B, R_c, A_c),
        T_from + _figure_core_cached(alpha_from, alpha_B, R_c, A_c)["W_loc_x"] * unit(alpha_from),
        unit(alpha_from),
        alpha_from,
    )
    fig3 = place_figure_right(build_figure(alpha_B, alpha_to, R_c, A_c), T_to, alpha_B)
    bridge_end = fig2["Phi_end"] + bridge_len * unit(alpha_B)
    return [
        {"type": "figure", **fig2, "alpha_in": alpha_from, "alpha_out": alpha_B},
        {"type": "straight", "p_start": fig2["Phi_end"], "p_end": bridge_end, "is_B": True},
        {"type": "figure", **fig3, "alpha_in": alpha_B, "alpha_out": alpha_to},
    ]


def _resolve_tangent_point(cities, idx):
    city = cities[idx]
    r_contact = road_contact_radius(city)
    road_side = normalize_road_side(city.get("road_side"))
    road_a = np.radians(city["road_angle"])
    n_road = normal_left(unit(road_a))
    T_plus = city["center"] + r_contact * n_road
    T_minus = city["center"] - r_contact * n_road

    if city.get("_bypass_manual") and road_side is not None:
        chosen = T_plus if road_side == 1 else T_minus
        city["bypass_angle"] = np.degrees(
            np.arctan2(chosen[1] - city["center"][1], chosen[0] - city["center"][0])
        ) % 360.0
        return chosen

    if city.get("bypass_angle") is not None:
        rad = np.radians(city["bypass_angle"])
        return city["center"] + r_contact * np.array([np.cos(rad), np.sin(rad)])

    if road_side == 1:
        chosen = T_plus
    elif road_side == -1:
        chosen = T_minus
    else:
        prev_pt = cities[idx - 1]["center"] if idx > 0 else None
        next_pt = cities[idx + 1]["center"] if idx < len(cities) - 1 else None

        def score(point):
            total = 0.0
            if prev_pt is not None:
                total += np.linalg.norm(point - prev_pt)
            if next_pt is not None:
                total += np.linalg.norm(point - next_pt)
            return total

        chosen = T_plus if score(T_plus) <= score(T_minus) else T_minus

    city["bypass_angle"] = np.degrees(
        np.arctan2(chosen[1] - city["center"][1], chosen[0] - city["center"][0])
    ) % 360.0
    return chosen


def _build_pair(from_city, from_T, to_city, to_T, R_c, A_c, L_buffer):
    alpha_from = np.radians(from_city["road_angle"])
    alpha_to = np.radians(to_city["road_angle"])
    dist_TT = np.linalg.norm(to_T - from_T)
    buf = min(L_buffer, dist_TT * 0.45)
    exit_base = road_line_exit(
        from_city["center"],
        road_contact_radius(from_city),
        from_T,
        unit(alpha_from),
        margin=0.0,
    )
    P_exit = exit_base + buf * unit(alpha_from)
    entry_base = road_line_exit(
        to_city["center"],
        road_contact_radius(to_city),
        to_T,
        -unit(alpha_to),
        margin=0.0,
    )
    P_entry = entry_base - buf * unit(alpha_to)
    return (
        [{"type": "straight", "p_start": from_T, "p_end": P_exit}]
        + build_segment(P_exit, alpha_from, P_entry, alpha_to, R_c, A_c)
        + [{"type": "straight", "p_start": P_entry, "p_end": to_T}]
    )


def _build_pair_safe(from_city, from_T, to_city, to_T, R_c, A_c, L_buffer):
    alpha_from = np.radians(from_city["road_angle"])
    alpha_to = np.radians(to_city["road_angle"])
    dist_TT = np.linalg.norm(to_T - from_T)
    if dist_TT < 1e-9:
        return []

    exit_base = road_line_exit(
        from_city["center"],
        road_contact_radius(from_city),
        from_T,
        unit(alpha_from),
        margin=0.0,
    )
    entry_base = road_line_exit(
        to_city["center"],
        road_contact_radius(to_city),
        to_T,
        -unit(alpha_to),
        margin=0.0,
    )

    base_buf = min(L_buffer, dist_TT * 0.45)
    buf_candidates = []
    for factor in (1.0, 1.25, 1.5, 1.75, 2.0):
        cand = min(base_buf * factor, dist_TT * 0.49)
        if cand > 0 and all(abs(cand - old) > 1e-9 for old in buf_candidates):
            buf_candidates.append(cand)
    if not buf_candidates:
        buf_candidates = [base_buf]

    fallback = None
    for buf in buf_candidates:
        P_exit = exit_base + buf * unit(alpha_from)
        P_entry = entry_base - buf * unit(alpha_to)
        segs = (
            [{"type": "straight", "p_start": from_T, "p_end": P_exit}]
            + build_segment(P_exit, alpha_from, P_entry, alpha_to, R_c, A_c)
            + [{"type": "straight", "p_start": P_entry, "p_end": to_T}]
        )
        fallback = segs

        progress_ok = True
        for seg in segs:
            if seg["type"] != "straight":
                continue
            if np.linalg.norm(seg["p_start"] - from_T) < 1e-9:
                if np.dot(seg["p_end"] - seg["p_start"], unit(alpha_from)) < -1e-8:
                    progress_ok = False
                    break
            if np.linalg.norm(seg["p_end"] - to_T) < 1e-9:
                if np.dot(seg["p_end"] - seg["p_start"], unit(alpha_to)) < -1e-8:
                    progress_ok = False
                    break
        if not progress_ok:
            continue

        if check_collision(segs, from_city["center"], road_contact_radius(from_city)):
            continue
        if check_collision(segs, to_city["center"], road_contact_radius(to_city)):
            continue
        return segs

    return fallback


def _choose_bypass(city, prev_city, prev_T, next_city, next_T, R_c, A_c, L_buffer):
    center = city["center"]
    road_a = np.radians(city["road_angle"])
    n_road = normal_left(unit(road_a))
    r_contact = road_contact_radius(city)
    candidates = [(1, center + r_contact * n_road), (-1, center - r_contact * n_road)]

    def evaluate(T):
        segs = []
        try:
            if prev_city is not None and prev_T is not None:
                segs.extend(_build_pair_safe(prev_city, prev_T, city, T, R_c, A_c, L_buffer))
            if next_city is not None and next_T is not None:
                segs.extend(_build_pair_safe(city, T, next_city, next_T, R_c, A_c, L_buffer))
        except Exception:
            return False, True, float("inf"), float("inf")
        if not segs:
            return True, False, 0.0, 0
        return (
            True,
            check_collision(segs, center, city["radius"]),
            segment_length(segs),
            figure_count(segs),
        )

    results = [evaluate(T) for _, T in candidates]
    preferred_side = normalize_road_side(city.get("road_side"))
    score = lambda r: (not r[0], r[1], round(r[2], 3), r[3])
    if preferred_side is None:
        best_idx = min(range(len(candidates)), key=lambda i: score(results[i]))
    else:
        preferred_idx = next(i for i, (side, _) in enumerate(candidates) if side == preferred_side)
        other_idx = 1 - preferred_idx
        preferred_valid, preferred_collides = results[preferred_idx][:2]
        other_valid, other_collides = results[other_idx][:2]
        if (not preferred_valid or preferred_collides) and other_valid and not other_collides:
            best_idx = other_idx
        else:
            best_idx = preferred_idx

    side_chosen, T_chosen = candidates[best_idx]
    city["road_side"] = side_chosen
    angle = np.arctan2(T_chosen[1] - center[1], T_chosen[0] - center[0])
    return np.degrees(angle) % 360.0, T_chosen


def _city_collision_local(cities, T_points, idx, R_c, A_c, L_buffer):
    city = cities[idx]
    segs = []
    try:
        if idx > 0:
            segs.extend(_build_pair_safe(cities[idx - 1], T_points[idx - 1], city, T_points[idx], R_c, A_c, L_buffer))
        if idx < len(cities) - 1:
            segs.extend(_build_pair_safe(city, T_points[idx], cities[idx + 1], T_points[idx + 1], R_c, A_c, L_buffer))
    except Exception:
        return True
    return check_collision(segs, city["center"], city["radius"])


def build_road(cities, R_c, A_c, L_buffer=DEFAULT_BUFFER_KM):
    cities = [copy.copy(c) for c in cities]
    T_points = [_resolve_tangent_point(cities, i) for i in range(len(cities))]
    all_segments = []

    for _ in range(5):
        changed = False
        for i, city in enumerate(cities):
            if city.get("_bypass_manual"):
                continue
            prev_city = cities[i - 1] if i > 0 else None
            next_city = cities[i + 1] if i < len(cities) - 1 else None
            prev_T = T_points[i - 1] if i > 0 else None
            next_T = T_points[i + 1] if i < len(cities) - 1 else None
            new_angle, new_T = _choose_bypass(
                city, prev_city, prev_T, next_city, next_T, R_c, A_c, L_buffer
            )
            if np.linalg.norm(new_T - T_points[i]) > 1e-6:
                changed = True
            city["bypass_angle"] = new_angle
            T_points[i] = new_T
        if not changed:
            break

    for i in range(len(cities) - 1):
        c_curr = cities[i]
        c_next = cities[i + 1]
        alpha_curr = np.radians(c_curr["road_angle"])
        alpha_next = np.radians(c_next["road_angle"])
        T_curr = T_points[i]
        T_next = T_points[i + 1]

        if i == 0:
            all_segments.append(
                {
                    "type": "straight",
                    "p_start": T_curr - L_buffer * unit(alpha_curr),
                    "p_end": T_curr,
                    "pair_idx": i,
                }
            )

        try:
            pair_segments = _build_pair_safe(c_curr, T_curr, c_next, T_next, R_c, A_c, L_buffer)
        except ValueError as exc:
            raise ValueError(
                "Classical transition is impossible in theoretical mode.\n"
                f"Segment: {_city_label(c_curr, i)} -> {_city_label(c_next, i + 1)}\n"
                f"Parameters: R={R_c:.4g}, A={A_c:.4g}, "
                f"turn angle={abs(np.degrees(signed_angle(alpha_curr, alpha_next))):.2f} deg\n"
                "This mode keeps R and A fixed and does not use heuristic re-fitting.\n"
                "Try one of these:\n"
                "- increase A;\n"
                "- decrease R;\n"
                "- change road side or bypass angle for one of the cities;\n"
                "- remove obstacles for the theoretical test.\n"
                f"Details: {exc}"
            ) from exc
        for seg in pair_segments:
            seg["pair_idx"] = i
        all_segments.extend(pair_segments)

        if i == len(cities) - 2:
            all_segments.append(
                {
                    "type": "straight",
                    "p_start": T_next,
                    "p_end": T_next + L_buffer * unit(alpha_next),
                    "pair_idx": i,
                }
            )

    return all_segments, T_points


def inject_obstacles(ordered_cities, obstacles, R_c=None, A_c=None):
    if R_c is None:
        R_c = DEFAULT_R
    if A_c is None:
        A_c = DEFAULT_A

    final_path = [copy.deepcopy(city) for city in ordered_cities]
    inserted_keys = {_obstacle_key(c) for c in final_path if c.get("is_lake")}
    max_insertions = max(8, len(obstacles) * MAX_OBSTACLE_INSERTIONS_FACTOR + len(ordered_cities))
    insertions = len(inserted_keys)
    started = time.perf_counter()

    while True:
        if insertions > max_insertions:
            raise ValueError(
                "Обхід перешкод зупинено: надто багато допоміжних вузлів для озер. "
                "Ймовірно, конфігурація перешкод викликає зациклення або є занадто складною."
            )
        if time.perf_counter() - started > MAX_OBSTACLE_BUILD_SECONDS:
            raise ValueError(
                "Обхід перешкод зупинено через перевищення часу. "
                "Спробуй прибрати вкладені озера або спростити конфігурацію."
            )

        inserted = False

        for i in range(len(final_path) - 1):
            current_city = final_path[i]
            next_city = final_path[i + 1]
            blocked_list = blocked_with_margin_sorted(current_city["center"], next_city["center"], obstacles)
            blocked_list = [
                obs
                for _, obs in blocked_list
                if not (
                    _obstacle_key(obs) in inserted_keys
                    or
                    (current_city.get("is_lake") and np.allclose(current_city["center"], obs["center"]))
                    or (next_city.get("is_lake") and np.allclose(next_city["center"], obs["center"]))
                )
            ]
            if not blocked_list:
                continue

            diff = next_city["center"] - current_city["center"]
            opt_a = np.degrees(np.arctan2(diff[1], diff[0]))
            for blocked_by in reversed(blocked_list):
                key = _obstacle_key(blocked_by)
                final_path.insert(
                    i + 1,
                    {
                        "name": blocked_by.get("name", "Lake"),
                        "center": blocked_by["center"],
                        "radius": blocked_by["radius"] + LAKE_BYPASS_MARGIN,
                        "road_angle": opt_a,
                        "is_lake": True,
                    },
                )
                inserted_keys.add(key)
                insertions += 1
            inserted = True
            break

        if inserted:
            continue

        try:
            segments, _ = build_road(copy.deepcopy(final_path), R_c, A_c)
        except Exception:
            break

        collision_found = None
        for obs in obstacles:
            if _obstacle_key(obs) in inserted_keys:
                continue
            for seg in segments:
                if segment_collides_with_circle(seg, obs["center"], obs["radius"]):
                    pair_idx = seg.get("pair_idx")
                    if pair_idx is None or pair_idx >= len(final_path) - 1:
                        continue
                    collision_found = (pair_idx, obs)
                    break
            if collision_found is not None:
                break

        if collision_found is None:
            break

        pair_idx, obs = collision_found
        current_city = final_path[pair_idx]
        next_city = final_path[pair_idx + 1]
        diff = next_city["center"] - current_city["center"]
        opt_a = np.degrees(np.arctan2(diff[1], diff[0]))
        key = _obstacle_key(obs)
        final_path.insert(
            pair_idx + 1,
            {
                "name": obs.get("name", "Lake"),
                "center": obs["center"],
                "radius": obs["radius"] + LAKE_BYPASS_MARGIN,
                "road_angle": opt_a,
                "is_lake": True,
            },
        )
        inserted_keys.add(key)
        insertions += 1
    return final_path


def optimize_route(cities, start_idx=0, R_c=DEFAULT_R, A_c=DEFAULT_A, obstacles=None, exact_top_k=2):
    n = len(cities)
    free = [i for i in range(n) if i != start_idx]
    candidates = []

    nn_order = nearest_neighbor(cities, start_idx)
    candidates.append(two_opt(nn_order, cities))

    for second in free:
        rest = [x for x in free if x != second]
        visited = {start_idx, second}
        order = [start_idx, second]
        current = second
        for _ in range(len(rest)):
            best_d, best_j = float("inf"), -1
            for j in rest:
                if j not in visited:
                    d = euclid_dist(cities[current], cities[j])
                    if d < best_d:
                        best_d, best_j = d, j
            if best_j == -1:
                break
            order.append(best_j)
            visited.add(best_j)
            current = best_j
        candidate = two_opt(order, cities)
        if candidate not in candidates:
            candidates.append(candidate)

    for last in free:
        rest = [x for x in free if x != last]
        nn_sub = nearest_neighbor([cities[i] for i in [start_idx] + rest], 0)
        order = [([start_idx] + rest)[i] for i in nn_sub] + [last]
        candidate = two_opt(order, cities)
        if candidate not in candidates:
            candidates.append(candidate)

    candidates.sort(key=lambda o: total_euclid(o, cities))
    max_candidates = 12 if n <= 6 else 8
    max_candidates = max(max_candidates, exact_top_k * 4)
    top_candidates = candidates[: min(len(candidates), max_candidates)]

    best_order = None
    best_length = float("inf")
    for order in top_candidates:
        ordered = [copy.deepcopy(cities[i]) for i in order]
        if obstacles:
            ordered = inject_obstacles(ordered, obstacles, R_c=R_c, A_c=A_c)
        try:
            segs, _ = build_road(copy.deepcopy(ordered), R_c=R_c, A_c=A_c)
            length = road_length(segs)
            if length < best_length:
                best_length = length
                best_order = order
        except Exception:
            pass
    return best_order, best_length


class EditDialog(tk.Toplevel):
    def __init__(self, parent, title, fields, values=None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self.grab_set()

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.vars = {}
        for row, (label, key, default) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            value = values.get(key, default) if values else default
            var = tk.StringVar(value=str(value))
            ttk.Entry(frame, textvariable=var, width=14).grid(row=row, column=1, sticky="ew", pady=3)
            self.vars[key] = var

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=len(fields), column=0, columnspan=2, pady=(12, 0))
        ttk.Button(btn_row, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Cancel", command=self.destroy).pack(side="left", padx=4)
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())

    def _ok(self):
        try:
            text_fields = {"name", "road_side"}
            self.result = {}
            for key, var in self.vars.items():
                self.result[key] = var.get() if key in text_fields else float(var.get())
            self.destroy()
        except ValueError:
            messagebox.showerror("Input error", "All numeric fields must be numbers.", parent=self)


def _plot_onto_ax(ax, segments, T_points, cities, lakes=None):
    ROAD = "royalblue"
    CARC = "crimson"

    lake_labeled = False
    lake_margin_labeled = False
    if lakes:
        for lake in lakes:
            c, r = lake["center"], lake["radius"]
            circ = Circle(
                c,
                r,
                facecolor="#a8d8ea",
                edgecolor="#2271b3",
                lw=2,
                zorder=1,
                alpha=0.75,
                label="Озеро" if not lake_labeled else "_",
            )
            lake_labeled = True
            ax.add_patch(circ)
            margin_circ = Circle(
                c,
                r + LAKE_BYPASS_MARGIN,
                facecolor="none",
                edgecolor="#2271b3",
                lw=1.4,
                ls="--",
                zorder=1,
                alpha=0.9,
                label="Запас обходу озера" if not lake_margin_labeled else "_",
            )
            lake_margin_labeled = True
            ax.add_patch(margin_circ)
            ax.text(c[0], c[1], lake.get("name", "Lake"), ha="center", va="center", fontsize=8, color="#2271b3", fontweight="bold")

    city_num = 0
    city_margin_labeled = False
    tangent_labeled = False
    city_labeled = False
    for city, T in zip(cities, T_points):
        if city.get("is_lake"):
            continue
        city_num += 1
        c, r = city["center"], city["radius"]
        ax.add_patch(
            Circle(
                c,
                r,
                facecolor="#8575b8",
                edgecolor="#654db5",
                lw=2,
                zorder=2,
                alpha=0.7,
                label="Місто" if not city_labeled else "_",
            )
        )
        city_labeled = True
        margin_circ = Circle(
            c,
            r + ROAD_CLEARANCE_MARGIN,
            facecolor="none",
            edgecolor="#654db5",
            lw=1.2,
            ls="--",
            zorder=3,
            alpha=0.9,
            label="Відступ" if not city_margin_labeled else "_",
        )
        city_margin_labeled = True
        ax.add_patch(margin_circ)
        lbl = city.get("name", f"M{city_num}")
        ax.text(c[0], c[1], lbl, ha="center", va="center", fontsize=8, color="#ffffff", fontweight="bold")
        ax.plot(
            *T,
            marker="o",
            color="red",
            ms=6,
            mec="white",
            mew=0.9,
            zorder=6,
            label="Точка дотику" if not tangent_labeled else "_",
        )
        tangent_labeled = True

    road_start = None
    road_end = None
    if segments:
        road_start = segments[0]["p_start"] if segments[0]["type"] == "straight" else segments[0]["pts_in_g"][0]
        last_seg = segments[-1]
        road_end = last_seg["p_end"] if last_seg["type"] == "straight" else last_seg["pts_exit_g"][-1]

    if road_start is not None and road_end is not None:
        ax.plot(*road_start, "k^", ms=11, zorder=7, label="Початок / Кінець")
        ax.plot(*road_end, "k^", ms=11, zorder=7)
        L_arr = 12
        for Tpt, city in [(road_start, cities[0]), (road_end, cities[-1])]:
            a = np.radians(city["road_angle"])
            ax.annotate(
                "",
                xy=(Tpt[0] + L_arr * np.cos(a), Tpt[1] + L_arr * np.sin(a)),
                xytext=Tpt,
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=2),
            )

    for seg in segments:
        if seg["type"] == "straight":
            p1, p2 = seg["p_start"], seg["p_end"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ROAD, lw=2.5, label="Пряма", zorder=3)
        else:
            ax.plot(*seg["pts_in_g"].T, color="orange", lw=3, label="Вхідна клотоїда", zorder=4)
            ax.plot(*seg["pts_arc_g"].T, color=CARC, lw=3, label="Дуга кола", zorder=4)
            ax.plot(*seg["pts_exit_g"].T, color="green", lw=3, label="Вихідна клотоїда", zorder=4)

    all_x, all_y = [], []
    for seg in segments:
        if seg["type"] == "straight":
            all_x += [seg["p_start"][0], seg["p_end"][0]]
            all_y += [seg["p_start"][1], seg["p_end"][1]]
        else:
            for key in ("pts_in_g", "pts_arc_g", "pts_exit_g"):
                all_x.extend(seg[key][:, 0])
                all_y.extend(seg[key][:, 1])
    for city in cities:
        c, r = city["center"], road_contact_radius(city)
        all_x += [c[0] - r, c[0] + r]
        all_y += [c[1] - r, c[1] + r]
    if lakes:
        for lake in lakes:
            c, r = lake["center"], lake["radius"] + LAKE_BYPASS_MARGIN
            all_x += [c[0] - r, c[0] + r]
            all_y += [c[1] - r, c[1] + r]
    if all_x:
        pad_x = max((max(all_x) - min(all_x)) * 0.05, 10)
        pad_y = max((max(all_y) - min(all_y)) * 0.08, 10)
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="best", fontsize=9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Road Builder — Single Academic File")
        self.geometry("1920x1080")
        self.configure(bg="#ecf0f1")
        self.cities = copy.deepcopy(DEFAULT_CITIES)
        self.lakes = copy.deepcopy(DEFAULT_LAKES)
        self.A_var = tk.DoubleVar(value=DEFAULT_A)
        self.R_var = tk.DoubleVar(value=DEFAULT_R)
        self.start_var = tk.IntVar(value=0)
        self.optimize_var = tk.BooleanVar(value=DEFAULT_OPTIMIZE)
        self.status_var = tk.StringVar(value="Single-file theoretical version.")
        self._build_ui()
        self._refresh_tables()
        self.after(200, self._build)

    def _build_ui(self):
        paned = tk.PanedWindow(self, orient="horizontal", sashwidth=5, sashrelief="raised", bg="#bdc3c7")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, minsize=1200, stretch="always")
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.fig.patch.set_facecolor("#f8f9fa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(left)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        right = ttk.Frame(paned, width=540)
        paned.add(right, minsize=500, stretch="never")
        self._build_controls(right)

    def _build_controls(self, parent):
        scroll_canvas = tk.Canvas(parent, bg="#ecf0f1", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(scroll_canvas, padding=(8, 6))
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(_event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            scroll_canvas.itemconfig(win_id, width=scroll_canvas.winfo_width())

        inner.bind("<Configure>", on_configure)
        scroll_canvas.bind("<Configure>", on_configure)

        self._section_header(inner, "Cities")
        self.city_tree = self._make_tree(inner, ("Name", "X", "Y", "R", "Angle°", "Side"))
        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Add", command=self._add_city).pack(side="left", padx=2)
        ttk.Button(row, text="Edit", command=self._edit_city).pack(side="left", padx=2)
        ttk.Button(row, text="Delete", command=self._del_city).pack(side="left", padx=2)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=6)

        self._section_header(inner, "Lakes / Obstacles")
        self.lake_tree = self._make_tree(inner, ("Name", "X", "Y", "R"))
        row = ttk.Frame(inner)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Add", command=self._add_lake).pack(side="left", padx=2)
        ttk.Button(row, text="Edit", command=self._edit_lake).pack(side="left", padx=2)
        ttk.Button(row, text="Delete", command=self._del_lake).pack(side="left", padx=2)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=6)

        self._section_header(inner, "Parameters")
        grid = ttk.Frame(inner, padding=(4, 4))
        grid.pack(fill="x")
        ttk.Label(grid, text="A (clothoid) [km⁻²]:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(grid, textvariable=self.A_var, width=10).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(grid, text="R (arc radius) [km]:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(grid, textvariable=self.R_var, width=10).grid(row=1, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(grid, text="Start city index:").grid(row=2, column=0, sticky="w", pady=3)
        self.start_spin = ttk.Spinbox(grid, from_=0, to=99, textvariable=self.start_var, width=8)
        self.start_spin.grid(row=2, column=1, sticky="ew", padx=(4, 0))
        ttk.Checkbutton(grid, text="Enumerate all route orders (slow)", variable=self.optimize_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=3
        )
        grid.columnconfigure(1, weight=1)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=6)
        tk.Button(
            inner,
            text="▶  BUILD ROAD",
            font=("Segoe UI", 13, "bold"),
            bg="#f39c12",
            fg="black",
            padx=24,
            pady=16,
            cursor="hand2",
            command=self._build,
        ).pack(fill="x", pady=(4, 10))
        ttk.Button(inner, text="↺  Reset to defaults", command=self._reset).pack(fill="x")
        ttk.Label(inner, textvariable=self.status_var, font=("Segoe UI", 8), foreground="#555").pack(fill="x", pady=(6, 0))

    def _section_header(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"), bg="#2c3e50", fg="white", padx=8, pady=4).pack(fill="x", pady=(0, 4))

    def _make_tree(self, parent, cols):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=15, selectmode="browse")
        col_widths = {"Name": 52, "X": 52, "Y": 52, "R": 44, "Angle°": 52, "Side": 52}
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=col_widths.get(col, 52), anchor="center", stretch=True)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(side="left", fill="x", expand=True)
        return tree

    def _refresh_tree(self, tree, data, row_fn):
        sel = tree.selection()
        sel_idx = tree.index(sel[0]) if sel else None
        for item in tree.get_children():
            tree.delete(item)
        for item in data:
            tree.insert("", "end", values=row_fn(item))
        items = tree.get_children()
        if items and sel_idx is not None:
            tree.selection_set(items[min(sel_idx, len(items) - 1)])

    def _refresh_tables(self):
        self._refresh_tree(
            self.city_tree,
            self.cities,
            lambda c: (
                c.get("name", "?"),
                f"{c['center'][0]:.1f}",
                f"{c['center'][1]:.1f}",
                f"{c['radius']:.1f}",
                f"{c['road_angle']:.1f}",
                road_side_label(c.get("road_side")),
            ),
        )
        self._refresh_tree(
            self.lake_tree,
            self.lakes,
            lambda l: (l.get("name", "?"), f"{l['center'][0]:.1f}", f"{l['center'][1]:.1f}", f"{l['radius']:.1f}"),
        )
        self.start_spin.configure(to=max(0, len(self.cities) - 1))

    def _city_fields(self):
        return [
            ("Name", "name", "City"),
            ("X [km]", "x", "0"),
            ("Y [km]", "y", "0"),
            ("Radius [km]", "radius", "15"),
            ("Road angle°", "road_angle", "0"),
            ("Road side (+/-/auto)", "road_side", "auto"),
        ]

    def _lake_fields(self):
        return [
            ("Name", "name", "Lake"),
            ("X [km]", "x", "0"),
            ("Y [km]", "y", "0"),
            ("Radius [km]", "radius", "12"),
        ]

    def _add_city(self):
        dlg = EditDialog(self, "Add City", self._city_fields())
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self.cities.append(
                {
                    "name": r.get("name", f"M{len(self.cities)+1}"),
                    "center": np.array([r["x"], r["y"]]),
                    "radius": r["radius"],
                    "road_angle": r["road_angle"],
                    "road_side": road_side_label(r.get("road_side")),
                    "_bypass_manual": normalize_road_side(r.get("road_side")) is not None,
                }
            )
            self._refresh_tables()

    def _edit_city(self):
        sel = self.city_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a city to edit.", parent=self)
            return
        idx = self.city_tree.index(sel[0])
        city = self.cities[idx]
        dlg = EditDialog(
            self,
            f"Edit City {idx+1}",
            self._city_fields(),
            {
                "name": city.get("name", ""),
                "x": city["center"][0],
                "y": city["center"][1],
                "radius": city["radius"],
                "road_angle": city["road_angle"],
                "road_side": road_side_label(city.get("road_side")),
            },
        )
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self.cities[idx] = {
                "name": r.get("name", city.get("name", "")),
                "center": np.array([r["x"], r["y"]]),
                "radius": r["radius"],
                "road_angle": r["road_angle"],
                "road_side": road_side_label(r.get("road_side")),
                "_bypass_manual": normalize_road_side(r.get("road_side")) is not None,
            }
            self._refresh_tables()

    def _del_city(self):
        sel = self.city_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a city to delete.", parent=self)
            return
        idx = self.city_tree.index(sel[0])
        if len(self.cities) <= 2:
            messagebox.showwarning("Min cities", "At least 2 cities required.", parent=self)
            return
        if messagebox.askyesno("Delete", f"Delete city {self.cities[idx].get('name', '?')}?", parent=self):
            self.cities.pop(idx)
            self._refresh_tables()

    def _add_lake(self):
        dlg = EditDialog(self, "Add Lake", self._lake_fields())
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self.lakes.append(
                {"name": r.get("name", f"L{len(self.lakes)+1}"), "center": np.array([r["x"], r["y"]]), "radius": r["radius"]}
            )
            self._refresh_tables()

    def _edit_lake(self):
        sel = self.lake_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a lake to edit.", parent=self)
            return
        idx = self.lake_tree.index(sel[0])
        lake = self.lakes[idx]
        dlg = EditDialog(
            self,
            f"Edit Lake {idx+1}",
            self._lake_fields(),
            {"name": lake.get("name", ""), "x": lake["center"][0], "y": lake["center"][1], "radius": lake["radius"]},
        )
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self.lakes[idx] = {
                "name": r.get("name", lake.get("name", "")),
                "center": np.array([r["x"], r["y"]]),
                "radius": r["radius"],
            }
            self._refresh_tables()

    def _del_lake(self):
        sel = self.lake_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a lake to delete.", parent=self)
            return
        idx = self.lake_tree.index(sel[0])
        if messagebox.askyesno("Delete", f"Delete lake {self.lakes[idx].get('name', '?')}?", parent=self):
            self.lakes.pop(idx)
            self._refresh_tables()

    def _reset(self):
        if messagebox.askyesno("Reset", "Reset data to defaults for the single-file theory version?", parent=self):
            self.cities = copy.deepcopy(DEFAULT_CITIES)
            self.lakes = copy.deepcopy(DEFAULT_LAKES)
            self.A_var.set(DEFAULT_A)
            self.R_var.set(DEFAULT_R)
            self.start_var.set(0)
            self.optimize_var.set(DEFAULT_OPTIMIZE)
            self._refresh_tables()

    def _build(self):
        if len(self.cities) < 2:
            messagebox.showwarning("Too few", "Add at least 2 cities.", parent=self)
            return

        self.status_var.set("Building in single-file theoretical mode…")
        self.update_idletasks()

        try:
            A_c = float(self.A_var.get())
            R_c = float(self.R_var.get())
            if A_c <= 0 or R_c <= 0:
                raise ValueError("A and R must be positive.")
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc), parent=self)
            self.status_var.set("Error in parameters.")
            return

        start_idx = max(0, min(int(self.start_var.get()), len(self.cities) - 1))

        try:
            cities_copy = copy.deepcopy(self.cities)
            ordered = cities_copy
            route_prefix = "Маршрут"
            if self.optimize_var.get() and len(cities_copy) > 2:
                best_order, _ = optimize_route(
                    cities_copy, start_idx=start_idx, R_c=R_c, A_c=A_c, obstacles=self.lakes
                )
                if best_order is not None:
                    ordered = [copy.deepcopy(cities_copy[i]) for i in best_order]
                    route_prefix = "Оптимальний маршрут"
            if self.lakes:
                ordered = inject_obstacles(ordered, self.lakes, R_c=R_c, A_c=A_c)
            segments, T_points = build_road(copy.deepcopy(ordered), R_c=R_c, A_c=A_c)
            length = road_length(segments)
        except Exception as exc:
            messagebox.showerror("Build error", str(exc), parent=self)
            self.status_var.set(f"Build failed: {exc}")
            return

        city_names = " → ".join(c.get("name", f"M{i+1}") for i, c in enumerate(ordered) if not c.get("is_lake"))
        title = (
            f"{route_prefix}: {city_names}\n"
            f"Довжина: {length:.1f} км  |  A={A_c}  R={R_c} км"
        )

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.2, linestyle="--")
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel("X [km]")
        self.ax.set_ylabel("Y [km]")
        _plot_onto_ax(self.ax, segments, T_points, ordered, lakes=self.lakes)
        self.canvas.draw()
        self.status_var.set(f"Done. Road length: {length:.1f} km")

    def on_close(self):
        plt.close("all")
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
