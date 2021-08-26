import autograd.numpy as np
from typing import Callable
import scipy.optimize as opt
# unfortunately this method did not work for our required simulation
# I suspect that assuming skeleton are stiff springs does not really fit here


gravity = np.array([0., -9.8, 0.])
def t_particle(qdot: np.ndarray, mass: float):
    return np.linalg.norm(qdot) ** 2 * mass * 0.5


def v_gravity_particle(q: np.ndarray, mass: float):
    return q.dot(gravity) * mass


def v_spring_particle_particle(q0: np.ndarray, q1: np.ndarray, l0: float, stiffness: float):
    q = np.concatenate((q0, q1))
    B = np.array([-1, 0, 0, 1, 0, 0,
                 0, -1, 0, 0, 1, 0,
                 0, 0, -1, 0, 0, 1]).reshape(3, 6)
    l = np.sqrt(q.T @ B.T @ B @ q)
    return 0.5 * stiffness * ((l-l0) ** 2)


def dv_gravity_particle_dq(mass: float):
    return mass * gravity


def dv_spring_particle_particle_dq(q0: np.ndarray, q1: np.ndarray, l0: float, stiffness: float):
    q1_ = q0[0]
    q2 = q0[1]
    q3 = q0[2]
    q4 = q1[0]
    q5 = q1[1]
    q6 = q1[2]
    f = np.zeros(shape=6)
    f[0] = stiffness * (q1_ * 2.0 - q4 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) * (
                   -1.0 / 2.0)
    f[1] = stiffness * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) * (
                   -1.0 / 2.0)
    f[2] = stiffness * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) * (
                   -1.0 / 2.0)
    f[3] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6))) / 2.0
    f[4] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6))) / 2.0
    f[5] = (stiffness * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6))) / 2.0
    return f


def d2v_spring_particle_particle_dq2(q0: np.ndarray, q1: np.ndarray, l0: float, stiffness: float):
    q1_ = q0[0]
    q2 = q0[1]
    q3 = q0[2]
    q4 = q1[0]
    q5 = q1[1]
    q6 = q1[2]
    H = np.zeros(shape=(6, 6))
    H[0, 0] = (stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[0, 1] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[0, 2] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[0, 3] = (stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[0, 4] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[0, 5] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[1, 0] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[1, 1] = (stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[1, 2] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[1, 3] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[1, 4] = (stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[1, 5] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[2, 0] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[2, 1] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[2, 2] = (stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[2, 3] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[2, 4] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[2, 5] = (stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[3, 0] = (stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[3, 1] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[3, 2] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[3, 3] = (stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q1_ * 2.0 - q4 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[3, 4] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[3, 5] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[4, 0] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[4, 1] = (stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[4, 2] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[4, 3] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q2 * 2.0 - q5 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[4, 4] = (stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q2 * 2.0 - q5 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[4, 5] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[5, 0] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[5, 1] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[5, 2] = (stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (-1.0 / 4.0)) / (
            q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6)) + stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) - (
                      stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    H[5, 3] = (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q1_ * 2.0 - q4 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[5, 4] = (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) + (stiffness * (q2 * 2.0 - q5 * 2.0) * (q3 * 2.0 - q6 * 2.0) * (
            l0 - np.sqrt(q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / pow(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6),
        3.0 / 2.0)) / 4.0
    H[5, 5] = (stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0)) / (
            q1_ * (q1_ - q4) * 4.0 + q2 * (q2 - q5) * 4.0 - q4 * (q1_ - q4) * 4.0 + q3 * (q3 - q6) * 4.0 - q5 * (
                q2 - q5) * 4.0 - q6 * (q3 - q6) * 4.0) - stiffness * (l0 - np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                q3 - q6))) * 1.0 / np.sqrt(
        q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (q3 - q6)) + (
                      stiffness * pow(q3 * 2.0 - q6 * 2.0, 2.0) * (l0 - np.sqrt(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6))) * 1.0 / pow(
                      q1_ * (q1_ - q4) + q2 * (q2 - q5) - q4 * (q1_ - q4) + q3 * (q3 - q6) - q5 * (q2 - q5) - q6 * (
                              q3 - q6), 3.0 / 2.0)) / 4.0
    return H


def mass_matrix_particles(mass: np.ndarray):
    return np.diag(mass.repeat(3))


def assemble_energy(q: np.ndarray, e: np.ndarray, l0: np.ndarray,
                    stiffness: np.ndarray):
    energy = 0
    for i in range(len(e)):
        v0 = e[i][0]
        v1 = e[i][1]
        q0 = np.array([q[3 * v0], q[3 * v0 + 1], q[3 * v0 + 2]])
        q1 = np.array([q[3 * v1], q[3 * v1 + 1], q[3 * v1 + 2]])
        this_l0 = l0[i]
        k = stiffness[i]
        energy += v_spring_particle_particle(q0, q1, this_l0, k)
    return energy


def assemble_forces(q: np.ndarray, e: np.ndarray, l0: np.ndarray,
                    stiffness: np.ndarray):
    f = np.zeros(len(q))
    for i in range(len(e)):
        v0 = e[i][0]
        v1 = e[i][1]
        q0 = np.array([q[3 * v0], q[3 * v0 + 1], q[3 * v0 + 2]])
        q1 = np.array([q[3 * v1], q[3 * v1 + 1], q[3 * v1 + 2]])
        this_l0 = l0[i]
        k = stiffness[i]
        # if k
        this_f = dv_spring_particle_particle_dq(q0, q1, this_l0, k)
        f[3 * v0: 3 * v0 + 3] = this_f[0:3]
        f[3 * v1: 3 * v1 + 3] = this_f[3:6]
    return f * -1.


def assemble_stiffness(q: np.ndarray, e: np.ndarray, l0: np.ndarray,
                       stiffness: np.ndarray):
    stiff_mat = np.zeros((len(q), len(q)))
    for i in range(len(e)):
        v0 = e[i][0]
        v1 = e[i][1]
        q0 = np.array([q[3 * v0], q[3 * v0 + 1], q[3 * v0 + 2]])
        q1 = np.array([q[3 * v1], q[3 * v1 + 1], q[3 * v1 + 2]])
        this_l0 = l0[i]
        k = stiffness[i]
        this_k = d2v_spring_particle_particle_dq2(q0, q1, this_l0, k)
        stiff_mat[v0 * 3:v0 * 3 + 3, v0 * 3:v0 * 3 + 3] = this_k[0:3, 0:3]
        stiff_mat[v0 * 3:v0 * 3 + 3, v1 * 3:v1 * 3 + 3] = this_k[0:3, 3:6]
        stiff_mat[v1 * 3:v1 * 3 + 3, v0 * 3:v0 * 3 + 3] = this_k[3:6, 0:3]
        stiff_mat[v1 * 3:v1 * 3 + 3, v1 * 3:v1 * 3 + 3] = this_k[3:6, 3:6]
    return stiff_mat * -1.


def fixed_point_constraints(q_size: int, indices: np.ndarray):
    P = np.zeros((q_size - len(indices) * 3, q_size))
    non_fixed = np.ones(q_size)
    non_fixed[3 * indices] = 0
    non_fixed[3 * indices + 1] = 0
    non_fixed[3 * indices + 2] = 0
    r = 0
    for c in range(len(non_fixed)):
        if non_fixed[c] != 0:
            P[r, c] = 1
            r += 1
    return P


def linearly_implicit_euler(q: np.ndarray, qdot: np.ndarray, dt: float, mass: np.ndarray, force: Callable,
                            stiffness: Callable):
    qt = q
    qdott = qdot
    f = force(qt)
    stiff = stiffness(qt)
    A = mass - dt * dt * stiff
    b = mass @ qdott + dt * f
    x = np.linalg.solve(A, b)
    return q + dt * x, x


def newtons_method(x0: np.ndarray, fprime: Callable, fprime2: Callable, maxSteps: int):
    p = 0.5
    tol = 1e-8
    v = x0
    tmp_g = fprime(v)
    tmp_H = fprime2(v)
    dEdv = np.linalg.norm(tmp_g)
    alpha = 1.0
    i = 0
    # Eigen::VectorXd d;
    # d.resizeLike(x0);
    while i < int(maxSteps) and dEdv >= tol:
        H_i = tmp_H
        d = np.linalg.solve(H_i, -1.0 * tmp_g)
        v += alpha * d
        i += 1
        tmp_g = fprime(v)
        tmp_H = fprime2(v)
        dEdv = np.linalg.norm(tmp_g)
        alpha *= p
    return v
    # dE/dv is observed decreasing in debugger.


def implicit_euler(q: np.ndarray, qdot: np.ndarray, dt: float, mass: np.ndarray, force: Callable, stiffness: Callable):

    def fprime(v):
        F = force(q + dt * v)
        return mass @ (v - qdot) - dt * F

    def fprime2(v):
        K = stiffness(q + dt * v)
        return mass - dt * dt * K

    new_qdot = newtons_method(qdot, fprime, fprime2, 50)
    return q + dt * new_qdot, new_qdot





