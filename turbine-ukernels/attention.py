import argparse
import torch
import torch.nn as nn

from dataclasses import dataclass
from util import TorchModule, IREEModule, rel_error, torch_types

torch.set_default_device("cuda")


@dataclass
class Dim:
    size: int
    dynamic: bool = False


def attention(
    b: Dim,
    m: Dim,
    n: Dim,
    k1: Dim,
    k2: Dim,
    is_causal: bool,
    enable_gqa: bool,
    batch: tuple[int],
    dtype: torch.dtype,
    num_its: int,
):
    class Attention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=None,
                enable_gqa=enable_gqa,
            )

    print("*" * 80, flush=True)
    print(
        f"Attention:\n\tb={b}, m={m}, n={n}, k1={k1}, k2={k2}\n"
        + f"\tbatch={batch}\n\tis_casual={is_causal}, enable_gqa={enable_gqa}, dtype={dtype}"
        + f"\n\tnum_its={num_its}\n",
        flush=True,
    )

    device = torch.get_default_device()
    model = Attention()

    q = torch.randn(batch + (b.size, m.size, k1.size), dtype=dtype).to(device)
    k = torch.randn(batch + (b.size, k2.size, k1.size), dtype=dtype).to(device)
    v = torch.randn(batch + (b.size, k2.size, n.size), dtype=dtype).to(device)
    dynamic_shapes = {"q": {}, "k": {}, "v": {}}

    if b.dynamic:
        dyn_len = torch.export.Dim("b_d")
        dynamic_shapes["q"][len(batch)] = dyn_len
        dynamic_shapes["k"][len(batch)] = dyn_len
        dynamic_shapes["v"][len(batch)] = dyn_len
    if m.dynamic:
        dyn_len = torch.export.Dim("m_d")
        dynamic_shapes["q"][len(batch) + 1] = dyn_len
    if n.dynamic:
        dyn_len = torch.export.Dim("n_d")
        dynamic_shapes["v"][len(batch) + 2] = dyn_len
    if k1.dynamic:
        dyn_len = torch.export.Dim("k1_d")
        dynamic_shapes["q"][len(batch) + 2] = dyn_len
        dynamic_shapes["k"][len(batch) + 2] = dyn_len
    if k2.dynamic:
        dyn_len = torch.export.Dim("k2_d")
        dynamic_shapes["k"][len(batch) + 1] = dyn_len
        dynamic_shapes["v"][len(batch) + 1] = dyn_len

    print("\n" + ("=" * 40), flush=True)
    print("Compiling IREE...", flush=True)
    iree_run = IREEModule.from_torch(
        device, model, (q, k, v), dynamic_shapes=dynamic_shapes
    )
    print("Done compiling", flush=True)

    with torch.no_grad():
        model.cuda().compile()
        torch_run = TorchModule(model, (q, k, v))

        print("\n" + ("=" * 40), flush=True)
        print("Profiling torch:", flush=True)
        tet = torch_run.profile(num_its=num_its)

    print("\n" + ("=" * 40), flush=True)
    print("Profiling IREE:", flush=True)
    iet = iree_run.profile(num_its=num_its)

    print("\n" + ("=" * 40), flush=True)
    print("Summary:", flush=True)
    print(f"Total torch time: {tet:.5f} ms", flush=True)
    print(f"Total IREE time: {iet:.5f} ms", flush=True)
    print(f"Numeric error: {rel_error(torch_run.run(), iree_run.run())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        "-B",
        metavar="<batch size>",
        nargs="*",
        type=int,
        default=(),
        help="batch dimensions",
    )
    parser.add_argument("-b", type=int, required=True, help="number of heads")
    parser.add_argument("-m", type=int, required=True, help="target seq length")
    parser.add_argument(
        "-n", type=int, required=True, help="embedding dimension of value"
    )
    parser.add_argument(
        "-k1",
        type=int,
        required=True,
        help="embedding dimension of query and key",
    )
    parser.add_argument("-k2", type=int, required=True, help="source seq length")
    parser.add_argument("--is-casual", action="store_true")
    parser.add_argument("--gqa", action="store_true")
    parser.add_argument(
        "--dynamic-dims",
        type=str,
        nargs="*",
        default=(),
        choices=["b", "m", "n", "k1", "k2"],
        help="dynamic dimensions",
    )
    parser.add_argument(
        "--num-its",
        "-i",
        metavar="<num its>",
        type=int,
        default=10,
        help="number of iterations to execute",
    )
    parser.add_argument(
        "--dtype", "-t", type=str, default="f16", choices=torch_types.keys()
    )
    args = parser.parse_args()
    attention(
        b=Dim(args.b, "b" in args.dynamic_dims),
        m=Dim(args.m, "m" in args.dynamic_dims),
        n=Dim(args.n, "n" in args.dynamic_dims),
        k1=Dim(args.k1, "k1" in args.dynamic_dims),
        k2=Dim(args.k2, "k2" in args.dynamic_dims),
        is_causal=args.is_causal,
        enable_gqa=args.gqa,
        batch=tuple(args.batch_size),
        dtype=torch_types[args.dtype],
        num_its=args.num_its,
    )


if __name__ == "__main__":
    main()
