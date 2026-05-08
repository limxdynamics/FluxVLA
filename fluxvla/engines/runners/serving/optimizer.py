from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import time
import numpy as np
import osqp
import scipy.sparse as sp

class BaseOptimizer(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, inference_cfg):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, action_list: list) -> list:
        raise NotImplementedError


class PassThroughOptimizer(BaseOptimizer):
    @classmethod
    def from_config(cls, inference_cfg):
        return cls()

    def optimize(self, action_list: list) -> list:
        return action_list

class TimeParameterizationMPC(BaseOptimizer):
    @classmethod
    def from_config(cls, inference_cfg):
        return cls(
            dt_ref=inference_cfg.timeaxis_dt_ref_s,
            dt_min=inference_cfg.timeaxis_dt_min_s,
            dt_max=inference_cfg.timeaxis_dt_max_s,
            lambda_acc=inference_cfg.timeaxis_lambda_acc,
            lambda_time=inference_cfg.timeaxis_lambda_time,
            stride=inference_cfg.timeaxis_stride,
            optim_dims=inference_cfg.timeaxis_optdims,
            v_max=inference_cfg.timeaxis_v_max,
            lambda_v=inference_cfg.timeaxis_lambda_v,
            horizon=inference_cfg.timeaxis_horizon,
            logging=inference_cfg.timeaxis_logging,
        )

    def __init__(
        self,
        dt_ref: float = 0.05,
        dt_min: float = 0.01,
        dt_max: float = 0.3,
        lambda_acc: float = 1.0,
        lambda_time: float = 0.1,
        stride: int = 10,
        optim_dims: list[int] = [0, 1, 2, 3, 4, 5, 6],
        v_max: float | None = None,
        lambda_v: float = 10.0,
        horizon: int = 20,
        logging: bool = False,
    ) -> None:
        self.dt_ref = dt_ref
        self.s_ref = 1.0 / dt_ref
        self.s_min = 1.0 / dt_max
        self.s_max = 1.0 / dt_min
        self.lambda_acc = lambda_acc
        self.lambda_time = lambda_time
        self.solve_stride = stride
        self.optim_dims = optim_dims
        self.v_max = v_max
        self.lambda_v = lambda_v
        self.H = horizon
        self.logging = logging

    def solve_qp(self, k: int) -> np.ndarray:
        H = min(self.H, self.N - k - 1)
        dp = self.dp[k : k + H]
        dp_norm = np.linalg.norm(dp, axis=1)
        scale_time = self.s_ref ** 2 + 1e-6
        scale_acc = np.mean((dp_norm * self.s_ref) ** 2) + 1e-6
        lambda_time = self.lambda_time / scale_time
        lambda_acc = self.lambda_acc / scale_acc

        n_var = H
        P = np.zeros((n_var, n_var))
        P += 2 * lambda_time * np.eye(n_var)

        for i in range(H - 1):
            P[i, i] += 2 * lambda_acc * np.sum(dp[i] ** 2)
            P[i + 1, i + 1] += 2 * lambda_acc * np.sum(dp[i + 1] ** 2)
            P[i, i + 1] -= 2 * lambda_acc * np.dot(dp[i], dp[i + 1])
            P[i + 1, i] -= 2 * lambda_acc * np.dot(dp[i], dp[i + 1])
        P = sp.csc_matrix(P)

        q = -2 * lambda_time * self.s_ref * np.ones(n_var)

        A = sp.eye(n_var)
        l = self.s_min * np.ones(n_var)
        u = self.s_max * np.ones(n_var)

        if self.v_max is not None:
            A_v = sp.eye(n_var)
            dp_norm = np.linalg.norm(dp, axis=1)
            l_v = np.zeros(n_var)
            u_v = self.v_max / (dp_norm + 1e-8)
            A = sp.vstack([A, A_v])
            l = np.concatenate([l, l_v])
            u = np.concatenate([u, u_v])

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        res = prob.solve()
        if res.info.status != "solved":
            raise RuntimeError("OSQP failed")
        return np.asarray(res.x, dtype=np.float64)

    def re_allocate(self, waypoints: np.ndarray, ts: np.ndarray) -> np.ndarray:
        ts_out = np.arange(len(waypoints)) * self.dt_ref
        return np.apply_along_axis(lambda col: np.interp(ts_out, ts, col), axis=0, arr=waypoints)

    def solve(self, waypoints: np.ndarray, st_roll: int, end_roll: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.dp = waypoints[1:] - waypoints[:-1]
        self.N = len(self.dp)
        s_traj: list[float] = []
        k = st_roll

        while k < end_roll:
            st = time.time()
            s_opt = self.solve_qp(k)
            end = time.time()
            if self.logging:
                print(end - st)
            s_traj += s_opt[: self.solve_stride].tolist()
            k += self.solve_stride

        s_traj = np.asarray(s_traj)[: (end_roll - st_roll)]
        dt_traj = 1.0 / s_traj
        dt_traj = np.concatenate((dt_traj[:1], dt_traj, dt_traj[-1:]), axis=0)
        t = np.concatenate([[0.0], np.cumsum(dt_traj)])[:-1]
        optim_wp = self.re_allocate(waypoints[st_roll:end_roll], t[:(end_roll-st_roll)])
        return optim_wp, t, dt_traj

    def optimize(self, action_list: list) -> list:
        if not action_list:
            return action_list
        waypoints = np.asarray(action_list, dtype=np.float32)
        if waypoints.ndim == 1:
            waypoints = waypoints[None, :]
        optim_wp, _, _ = self.solve(waypoints[:, self.optim_dims], 0, self.H)
        waypoints[:len(optim_wp), self.optim_dims] = optim_wp
        return np.asarray(waypoints, dtype=np.float32).tolist()
