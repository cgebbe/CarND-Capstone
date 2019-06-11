import numpy as np
MIN_NUM = float('-inf')
MAX_NUM = float('inf')
import time


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        self.error_int = 0
        self.error_prev = 0.
        self.time_prev = None

    def reset(self):
        self.error_int = 0.0

    def step(self, error, sample_time=None):
        # measure time
        if sample_time is None:
            if self.time_prev is None:
                self.time_prev = time.time()
                return 0
            else:
                time_now = time.time()
                dt = time_now - self.time_prev
                self.time_prev = time_now
        else:
            dt = sample_time

        # calculate derivative and integrative
        self.error_int += error * dt;
        error_deriv = (error - self.error_prev) / dt;
        self.error_prev = error

        # determine output value
        accel = self.kp * error + self.kd * error_deriv + self.ki * self.error_int ;
        accel = np.clip(accel, self.min, self.max)
        return accel
