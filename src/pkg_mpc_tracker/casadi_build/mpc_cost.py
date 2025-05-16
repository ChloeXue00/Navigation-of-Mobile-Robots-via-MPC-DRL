from typing import Union

import casadi.casadi as cs # type: ignore

from .mpc_helper import *


def cost_inside_cvx_polygon(state: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX, weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for being inside a convex polygon defined by `b - [a0,a1]*[x,y]' > 0`.
        
    Args:
        state: The (n*1)-dim target point.
        b:  Shape (1*m) with m half-space offsets.
        a0: Shape (1*m) with m half-space weight vectors.
        a1: Shape (1*m) with m half-space weight vectors.
        
    Returns:
        cost: The (1*1)-dim weighted square cost. If inside, return positive value, else return 0.

    Notes:
        Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
        If prod(|max(0,all)|)>0, then the point is inside; Otherwise not.
    """
    indicator = inside_cvx_polygon(state, b, a0, a1)
    cost = weight * indicator**2
    return cost

def cost_inside_ellipses(state: cs.SX, ellipse_param: list[cs.SX], weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for being inside a set of ellipses defined by `(cx, cy, sx, sy, angle, alpha)`.
    
    Args:
        state: The (n*1)-dim target point.
        ellipse_param: Shape (5 or 6 * m) with m ellipses. 
                       Each ellipse is defined by (cx, cy, rx, ry, angle, alpha).
                       
    Returns:
        cost: The (1*m)-dim cost. If inside, return positive value, else return negative value.
    """
    if len(ellipse_param) > 5:
        alpha = ellipse_param[5]
    else:
        alpha = 1
    indicator = inside_ellipses(state, ellipse_param) # indicator<0, if outside ellipse
    indicator = weight * alpha * cs.fmax(0.0, indicator)**2
    cost = cs.sum2(indicator)
    return cost

def cost_fleet_collision(state: cs.SX, points: cs.SX, safe_distance: float, weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for colliding with other robots.
    
    Args:
        state: The (n*1)-dim target point.
        points: The (n*m)-dim points of other robots.
        
    Notes:
        Only have cost when the distance is smaller than `safe_distance`.
    """
    cost = weight * cs.sum2(cs.fmax(0.0, safe_distance**2 - dist_to_points_square(state, points)))
    return cost

def cost_refpath_deviation(state: cs.SX, line_segments: cs.SX, weight:Union[cs.SX, float]=1.0):
    """Reference deviation cost (weighted squared) penalizes on the deviation from the reference path.

    Args:
        state: The (n*1)-dim point.
        line_segments: The (n*m)-dim var with m n-dim points.

    Returns:
        The weighted squared distance to the reference path.
    """
    distances_sqrt = cs.SX.ones(1)
    for i in range(line_segments.shape[0]-1):
        distance = dist_to_lineseg(state[:2], line_segments[i:i+2,:2])
        distances_sqrt = cs.horzcat(distances_sqrt, distance**2)
    cost = cs.mmin(distances_sqrt[1:]) * weight
    return cost




