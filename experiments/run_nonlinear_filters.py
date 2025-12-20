import os
import argparse

import sys
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(this_dir)

from experiment_helper import CommonConfig
import run_sv
import run_rb


def main():
    common = CommonConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv", action="store_true", help="run SV experiment")
    parser.add_argument("--rb", action="store_true", help="run Range-Bearing experiment")
    parser.add_argument("--sv_obs_mode", choices=["y", "logy2"], default=None)
    parser.add_argument("--sv_obs_eps", type=float, default=None)
    parser.add_argument("--rb_motion", choices=["cv", "ctrv"], default=None)
    parser.add_argument("--T", type=int, default=common.T)
    parser.add_argument("--batch", type=int, default=common.batch_size)
    parser.add_argument("--seeds", type=int, nargs="*", default=common.seed)
    parser.add_argument("--no_seed", action="store_true")
    parser.add_argument("--out_dir", type=str, default=common.out_dir)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    if args.T is not None:
        common.T = args.T
    if args.batch is not None:
        common.batch_size = args.batch
    if args.out_dir is not None:
        common.out_dir = args.out_dir

    common.seed = None if args.no_seed else args.seeds
    common.save = (not args.no_save)
    common.show = (not args.no_show)

    run_sv_flag = args.sv or (not args.sv and not args.rb)
    run_rb_flag = args.rb or (not args.sv and not args.rb)
    seeds = args.seeds if not args.no_seed else [None]

    if run_sv_flag:
        sv_cfg = run_sv.SVConfig()
        if args.sv_obs_mode is not None:
            sv_cfg.obs_mode = args.sv_obs_mode
        if args.sv_obs_eps is not None:
            sv_cfg.obs_eps = args.sv_obs_eps
        run_sv.run(common, sv_cfg, seeds)

    if run_rb_flag:
        rb_cfg = run_rb.RBConfig()
        if args.rb_motion is not None:
            rb_cfg.motion_model = args.rb_motion
        run_rb.run(common, rb_cfg, seeds)


if __name__ == "__main__":
    main()
