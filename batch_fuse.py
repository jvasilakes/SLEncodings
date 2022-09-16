import argparse
from time import perf_counter_ns

import numpy as np
import matplotlib.pyplot as plt

import sle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--check-is-correct", action="store_true",
                        default=False)
    parser.add_argument("-F", "--check-is-fast", action="store_true",
                        default=False)
    parser.add_argument("-K", "--max-num-dists", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=10000)
    parser.add_argument("--random-dists", action="store_true", default=False,
                        help="Initialize all distributions randomly.")
    return parser.parse_args()


def mytimeit(fn, *args, n=1000, **kwargs):
    def wrapper(fn, *args, **kwargs):
        s = perf_counter_ns()
        fn(*args, **kwargs)
        e = perf_counter_ns()
        return e - s
    times = [wrapper(fn, *args, **kwargs) for _ in range(n)]
    return np.mean(times), np.std(times)


def main(args):
    if args.random_dists is True:
        raise NotImplementedError("--random-dists not implemented")

    b = [0.1, 0.3, 0.4]
    u = 1. - sum(b)
    dist = sle.SLDirichlet(b, u)

    means_old_fuse = []
    means_batch_fuse = []
    for k in range(1, args.max_num_dists+1):
        print(k)
        dists = [dist for _ in range(k)]

        if args.check_is_correct is True:
            old_ans = sle.fuse(dists)
            batch_ans = batch_fuse(dists)
        if old_ans == batch_ans is False:
            raise ValueError(f"k={k}: old != batch")
        if args.check_is_fast is True:
            old = mytimeit(sle.fuse, dists, n=args.num_runs)[0]
            batch = mytimeit(batch_fuse, dists, n=args.num_runs)[0]
            means_old_fuse.append(old)
            means_batch_fuse.append(batch)

    if args.check_is_fast is True:
        xs = range(1, args.max_num_dists+1)
        plt.plot(xs, means_old_fuse, label="old fuse")
        plt.plot(xs, means_batch_fuse, label="batch fuse")
        plt.ylabel("Avg. runtime (nanoseconds)")
        plt.xlabel("Number of distributions")
        plt.legend()
        plt.show()


def batch_fuse(dists):
    """
    Currently just computes the fused belief parameter.
    """
    if len(dists) == 1:
        return dists[0]

    # if len(dists) != 3:
    #     raise ValueError("Testing 3 dists rn.")
    # a, b, c = dists
    # num = (c.u * a.b * b.u) + (c.u * b.b * a.u) + (c.b * a.u * b.u)
    # denom = (a.u * b.u) + (c.u * a.u) + (c.u * b.u) - (2 * c.u * a.u * b.u)
    # return num / denom

    def helper(dists):
        if len(dists) == 2:
            a, b = dists
            T = (a.b * b.u) + (b.b * a.u)
            V = a.u + b.u
            U = a.u * b.u
            return T, V, U

        T, V, U = helper(dists[:-1])
        c = dists[-1]
        new_T = (c.u * T) + (c.b * U)
        new_V = U + (c.u * V)
        new_U = c.u * U
        return new_T, new_V, new_U

    T, V, U = helper(dists)
    K = len(dists) - 1
    denom = (V - (K * U))
    b = T / denom
    u = U / denom
    return sle.SLDirichlet(b, u)


if __name__ == "__main__":
    args = parse_args()
    main(args)
