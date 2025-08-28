import argparse
import torch
import torch.nn as nn

from iree.turbine.kernel.boo.op_exports.layer_norm import (
    LayerNormSignature,
    Mode,
    LayerNormForward,
    LayerNormBackwardInput,
    LayerNormBackwardWeight,
    LayerNormBackwardBias,
    LayerNormBackwardFull,
)
from typing import Optional, Sequence

from util import TorchModule, IREEModule, rel_error, torch_types

torch.set_default_device("cuda")


def _inverse_permutation(pi: Sequence[int]) -> list[int]:
    return [y for x, y in sorted({x: i for i, x in enumerate(pi)}.items())]


def run_boo(
    signature: LayerNormSignature,
    num_its: int,
    torch_compile: bool,
    use_boo: bool,
    permutation: Optional[Sequence[int]],
    scaling: Optional[Sequence[int]],
    module: torch.nn.Module,
):
    print(f"Layer norm[{signature.mode.name}]", flush=True)
    print(
        f"\tinput_shape={signature.input_shape}, normalized_shape={signature.normalized_shape}",
        flush=True,
    )
    print(f"\tdtypes={signature.dtype}, eps={signature.eps}", flush=True)
    print(f"\tnum_its={num_its}, torch_compile={torch_compile}", flush=True)

    # Get the sample args and model.
    args = list(signature.get_sample_args(device=torch.get_default_device()))
    args_to_permute = (
        (0,) if signature.mode in (Mode.FORWARD, Mode.BIAS_BACKWARD) else (0, 1)
    )
    if permutation is not None:
        for arg_index in args_to_permute:
            assert len(permutation) == len(args[0].shape)
            args[arg_index] = (
                args[arg_index]
                .permute(permutation)
                .clone(memory_format=torch.contiguous_format)
                .permute(_inverse_permutation(permutation))
            )
    if scaling is not None:
        assert len(args[0].shape) == len(scaling)
        for pos, scale in enumerate(scaling):
            if scale == 1:
                continue
            rank = len(args[0].shape)
            expansion = [scale if i == pos + 1 else -1 for i in range(rank + 1)]
            reshape = [args[0].shape[i] if i != pos else -1 for i in range(rank)]
            slices = [
                slice(None, None, scale) if i == pos else slice(None)
                for i in range(rank)
            ]
            args[0] = (
                args[0]
                .unsqueeze(pos + 1)
                .expand(expansion)
                .reshape(reshape)
                .clone(memory_format=torch.contiguous_format)[*slices]
            )

    args = tuple(args)

    print("\n" + ("=" * 40), flush=True)
    print("Inputs:", flush=True)
    for i, arg in enumerate(args):
        if arg is None:
            print(f"\tInput[{i}]: None", flush=True)
            continue
        print(
            f"\tInput[{i}]: shape={arg.shape}, strides={arg.stride()}, dtype={arg.dtype}",
            flush=True,
        )

    model = module(signature)
    if torch_compile:
        model_c = module(signature)

    # Compile IREE.
    if use_boo:
        print("\n" + ("=" * 40), flush=True)
        print("Compiling IREE...", flush=True)
        iree_run = IREEModule.from_torch(
            args[0].device, model, args, single_dispatch=True
        )
        print("Done compiling", flush=True)

    # Profile torch.
    torch_run = TorchModule(model, args)
    print("\n" + ("=" * 40), flush=True)
    print("Profiling torch:", flush=True)
    tet = torch_run.profile(num_its)

    if torch_compile:
        model_c.compile()
        torch_c_run = TorchModule(model_c, args)
        print("Profiling torch compile:", flush=True)
        tetc = torch_c_run.profile(num_its)

    # Profile IREE.
    if use_boo:
        print("\n" + ("=" * 40), flush=True)
        print("Profiling IREE:", flush=True)
        iet = iree_run.profile(num_its)

    # Print a summary.
    print("\n" + ("=" * 40), flush=True)
    print("Summary:", flush=True)
    print(f"Total torch time: {tet:.6f} ms", flush=True)
    if torch_compile:
        print(f"Total torch compile time: {tetc:.6f} ms", flush=True)
    if use_boo:
        print(f"Total IREE time: {iet:.6f} ms", flush=True)

    # Verify correctness.
    if not use_boo:
        return

    torch_r = torch_run.run()
    iree_r = iree_run.run()
    if isinstance(torch_r, tuple):
        for i, (t_r, i_r) in enumerate(zip(torch_r, iree_r)):
            print(f"Numeric error out[{i}]: {rel_error(t_r, i_r):.3e}", flush=True)
    else:
        print(f"Numeric error out: {rel_error(torch_r, iree_r):.3e}", flush=True)


def run_fwd(
    input_shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    dtype: torch.dtype,
    eps: float,
    elementwise_affine: bool,
    bias: bool,
    num_its: int,
    torch_compile: bool,
    permutation: Optional[Sequence[int]],
    scaling=Optional[Sequence[int]],
):
    print("*" * 80, flush=True)
    # Create the boo signature
    signature = LayerNormSignature(
        input_shape=input_shape,
        normalized_shape=normalized_shape,
        dtype=dtype,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias,
        mode=Mode.FORWARD,
    )
    run_boo(
        signature, num_its, torch_compile, True, permutation, scaling, LayerNormForward
    )
    print("\n")


def run_bwd_combined(
    input_shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    dtype: torch.dtype,
    forwarded_dtype: Optional[torch.dtype],
    eps: float,
    elementwise_affine: bool,
    bias: bool,
    num_its: int,
    torch_compile: bool,
    use_boo: bool,
    permutation: Optional[Sequence[int]],
    scaling=Optional[Sequence[int]],
):
    input_shape = list(input_shape)
    normalized_shape = list(normalized_shape)
    signature = LayerNormSignature(
        input_shape=input_shape,
        normalized_shape=normalized_shape,
        dtype=dtype,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias,
        mode=Mode.FULL_BACKWARD,
        forwarded_args_dtype=forwarded_dtype,
    )
    run_boo(
        signature,
        num_its,
        torch_compile,
        use_boo,
        permutation,
        scaling,
        LayerNormBackwardFull,
    )
    print("\n")


def run_bwd(
    input_shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    dtype: torch.dtype,
    forwarded_dtype: Optional[torch.dtype],
    eps: float,
    elementwise_affine: bool,
    bias: bool,
    num_its: int,
    torch_compile: bool,
    permutation: Optional[Sequence[int]],
    scaling=Optional[Sequence[int]],
):
    print("*" * 80, flush=True)
    print("-" * 40, flush=True)
    input_shape = list(input_shape)
    normalized_shape = list(normalized_shape)
    # Create the boo signature for the backward input.
    signature = LayerNormSignature(
        input_shape=input_shape,
        normalized_shape=normalized_shape,
        dtype=dtype,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias,
        mode=Mode.INPUT_BACKWARD,
        forwarded_args_dtype=forwarded_dtype,
    )
    run_boo(
        signature,
        num_its,
        torch_compile,
        True,
        permutation,
        scaling,
        LayerNormBackwardInput,
    )
    if elementwise_affine:
        print("-" * 40, flush=True)
        # Create the boo signature for the backward weight.
        signature = LayerNormSignature(
            input_shape=input_shape,
            normalized_shape=normalized_shape,
            dtype=dtype,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            mode=Mode.WEIGHT_BACKWARD,
        )
        run_boo(
            signature,
            num_its,
            torch_compile,
            True,
            permutation,
            scaling,
            LayerNormBackwardWeight,
        )
    if bias:
        print("-" * 40, flush=True)
        # Create the boo signature for the backward bias.
        signature = LayerNormSignature(
            input_shape=input_shape,
            normalized_shape=normalized_shape,
            dtype=dtype,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            mode=Mode.BIAS_BACKWARD,
        )
        run_boo(
            signature,
            num_its,
            torch_compile,
            True,
            permutation,
            scaling,
            LayerNormBackwardBias,
        )
    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        "-m",
        choices=["fwd", "bwd", "bwd_full"],
        type=str,
        required=True,
        help="the module to test",
    )
    parser.add_argument(
        "--batch-shape",
        "-b",
        metavar="<batch shape>",
        nargs="+",
        type=int,
        default=(),
        help="batch dimensions",
    )
    parser.add_argument(
        "--normalized-shape",
        "-n",
        metavar="<normalized shape>",
        nargs="+",
        type=int,
        default=(),
        help="normalized dimensions",
    )
    parser.add_argument(
        "--input-permutation",
        "-p",
        metavar="<permutation>",
        nargs="*",
        type=int,
        default=None,
        help="input permutation",
    )
    parser.add_argument(
        "--input-scaling",
        "-s",
        metavar="<scaling>",
        nargs="*",
        type=int,
        default=None,
        help="input stride scaling, applied after permutation",
    )
    parser.add_argument("--eps", "-e", type=float, default=1e-05, help="layer norm eps")
    parser.add_argument(
        "--elementwise-affine", "-a", action="store_true", help="add weights"
    )
    parser.add_argument("--bias", action="store_true", help="add bias")
    parser.add_argument(
        "--dtype", "-t", type=str, default="f16", choices=torch_types.keys()
    )
    parser.add_argument(
        "--forwarded-dtype", type=str, required=False, choices=torch_types.keys()
    )
    parser.add_argument(
        "--num-its",
        "-i",
        metavar="<num its>",
        type=int,
        default=10,
        help="number of iterations to execute",
    )
    parser.add_argument("-c", action="store_true", help="whether to use torch compile")
    parser.add_argument(
        "--disable-boo", action="store_true", help="whether to disable boo/iree"
    )
    args = parser.parse_args()
    if args.module == "fwd":
        run_fwd(
            input_shape=tuple(args.batch_shape + args.normalized_shape),
            normalized_shape=tuple(args.normalized_shape),
            dtype=torch_types[args.dtype],
            eps=args.eps,
            elementwise_affine=args.elementwise_affine,
            bias=args.bias,
            num_its=args.num_its,
            torch_compile=args.c,
            permutation=args.input_permutation,
            scaling=args.input_scaling,
        )
    elif args.module == "bwd":
        run_bwd(
            input_shape=tuple(args.batch_shape + args.normalized_shape),
            normalized_shape=tuple(args.normalized_shape),
            dtype=torch_types[args.dtype],
            forwarded_dtype=(
                torch_types[args.forwarded_dtype]
                if args.forwarded_dtype is not None
                else None
            ),
            eps=args.eps,
            elementwise_affine=args.elementwise_affine,
            bias=args.bias,
            num_its=args.num_its,
            torch_compile=args.c,
            permutation=args.input_permutation,
            scaling=args.input_scaling,
        )
    elif args.module == "bwd_full":
        run_bwd_combined(
            input_shape=tuple(args.batch_shape + args.normalized_shape),
            normalized_shape=tuple(args.normalized_shape),
            dtype=torch_types[args.dtype],
            forwarded_dtype=(
                torch_types[args.forwarded_dtype]
                if args.forwarded_dtype is not None
                else None
            ),
            eps=args.eps,
            elementwise_affine=args.elementwise_affine,
            bias=args.bias,
            num_its=args.num_its,
            torch_compile=args.c,
            use_boo=not args.disable_boo,
            permutation=args.input_permutation,
            scaling=args.input_scaling,
        )


if __name__ == "__main__":
    main()
