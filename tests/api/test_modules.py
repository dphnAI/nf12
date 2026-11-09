# SPDX-License-Identifier: Apache-2.0
"""Module-state conformance for the torch modules (CUDA and MPS): NF12Linear and
NF12Embedding inside a parent model, their weights replaced through every
loading path (a parent's or a child's load_state_dict, strict or partial, files,
meta-device construction), checked on every observable (forward, weight,
decode, a torch.compile forward captured before the change, a save/reload).

Mirrors tests/mlx/test_mlx_modules.py. Testing module methods directly misses
paths that never call them, so state is changed the way models are loaded."""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

import nf12
from _device import DEV, requires_gpu
from portable_random import randn

pytestmark = requires_gpu

SHAPES = {"proj": (48, 128), "layers.0": (64, 64), "layers.1": (32, 64), "emb": (64, 128)}


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def weights(seed: int, scale: float, escapes: bool) -> dict[str, nf12.NF12Weight]:
    out = {}
    for i, (name, (n, k)) in enumerate(SHAPES.items()):
        w = randn(n, k, seed=seed + i, scale=scale, device=DEV)
        # Keep magnitudes >= scale / 64 so every value fits one exponent window (no escapes).
        floor = torch.tensor(scale / 64, dtype=w.dtype, device=w.device)
        w = torch.where(w.abs() < floor, torch.where(w < 0, -floor, floor), w)
        if escapes:
            w[0, : 8 * (i + 1)] = 1e30
        out[name] = nf12.encode(w)
    return out


@pytest.fixture(scope="module")
def sets():
    """A: initial. SAME: same shapes as A, different exp_base. ESC: different
    exp_base and escape counts."""
    a, same, esc = weights(10, 0.02, False), weights(50, 0.02 * 16, False), weights(30, 0.05, True)
    for name in SHAPES:  # SAME must differ from A in every array but keep A's shapes
        assert {k: tuple(v.shape) for k, v in same[name].state_dict().items()} == \
            {k: tuple(v.shape) for k, v in a[name].state_dict().items()}, name
        assert same[name].exp_base != a[name].exp_base, name
        assert not torch.equal(same[name].packed, a[name].packed), name
        assert esc[name].num_escapes != a[name].num_escapes, name
    return a, same, esc


BIAS_SEED = 99


def bias() -> torch.Tensor:
    return randn(48, seed=BIAS_SEED, device=DEV)


class Net(nn.Module):
    def __init__(self, w: dict[str, nf12.NF12Weight] | None = None, device=DEV):
        super().__init__()
        if w is None:  # freshly built (all -0.0 weights), e.g. before loading a checkpoint
            self.proj = nf12.NF12Linear(128, 48, bias=True, device=device)
            self.layers = nn.ModuleList([nf12.NF12Linear(64, 64, bias=False, device=device),
                                         nf12.NF12Linear(64, 32, bias=False, device=device)])
            self.emb = nf12.NF12Embedding(64, 128, device=device)
        else:
            self.proj = nf12.NF12Linear.from_weight(w["proj"], bias())
            self.layers = nn.ModuleList([nf12.NF12Linear.from_weight(w["layers.0"]),
                                         nf12.NF12Linear.from_weight(w["layers.1"])])
            self.emb = nf12.NF12Embedding.from_weight(w["emb"])

    def forward(self, x, h, ids):
        return self.proj(x), self.layers[1](self.layers[0](h)), self.emb(ids)


def inputs():
    return randn(3, 128, seed=1, device=DEV), randn(2, 64, seed=2, device=DEV), \
        torch.tensor([0, 5, 63], dtype=torch.int64, device=DEV)


def modules(net: Net) -> dict[str, nn.Module]:
    return {"proj": net.proj, "layers.0": net.layers[0], "layers.1": net.layers[1], "emb": net.emb}


def checkpoint(w: dict[str, nf12.NF12Weight]) -> dict[str, torch.Tensor]:
    sd = {"proj.bias": bias()}
    for name, weight in w.items():
        sd.update(weight.state_dict(name + ".weight."))
    return sd


def assert_holds(net: Net, w: dict[str, nf12.NF12Weight], compiled=None) -> None:
    for name, module in modules(net).items():
        assert module.weight.exp_base == w[name].exp_base, name
        assert module.weight.num_escapes == w[name].num_escapes, name
        assert bits_equal(module.weight.decode(), w[name].decode()), f"{name}: weight.decode()"
    x, h, ids = inputs()
    want = (nf12.linear(x, w["proj"], bias()), nf12.linear(nf12.linear(h, w["layers.0"]), w["layers.1"]),
            nf12.embedding(ids, w["emb"]))
    for i, (got, ref) in enumerate(zip(net(x, h, ids), want)):
        assert bits_equal(got, ref), f"forward output {i}"
    if compiled is not None:
        for i, (got, ref) in enumerate(zip(compiled(x, h, ids), want)):
            assert bits_equal(got, ref), f"compiled forward output {i} (compiled before the change)"


def via_parent_strict(net, sd, tmp_path):
    net.load_state_dict(sd)


def via_parent_file(net, sd, tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    path = str(tmp_path / "model.safetensors")
    safetensors_torch.save_file({k: v.contiguous().cpu() for k, v in sd.items()}, path)
    net.load_state_dict(safetensors_torch.load_file(path))  # CPU tensors into a GPU model


def via_child(net, sd, tmp_path):
    for name, module in modules(net).items():
        module.load_state_dict({k[len(name) + 1:]: v for k, v in sd.items() if k.startswith(name + ".")})


PATHS = [via_parent_strict, via_parent_file, via_child]


@pytest.mark.parametrize("target", ["SAME", "ESC"])
@pytest.mark.parametrize("path", PATHS, ids=lambda f: f.__name__)
def test_replacement_paths(path, target, sets, tmp_path):
    a, same, esc = sets
    new = same if target == "SAME" else esc
    net = Net(a)
    compiled = torch.compile(net)
    assert_holds(net, a, compiled)
    path(net, checkpoint(new), tmp_path)
    assert_holds(net, new, compiled)
    fresh = Net()
    fresh.load_state_dict(net.state_dict())
    assert_holds(fresh, new)


def test_checkpoint_into_meta_model(sets):
    """Build on the meta device (no allocation), then load a real checkpoint."""
    _, _, esc = sets
    net = Net(device="meta")
    net.load_state_dict(checkpoint(esc), assign=True)
    net.proj.bias = nn.Parameter(net.proj.bias.to(DEV), requires_grad=False)
    assert_holds(net, esc)


@pytest.mark.parametrize("keys", [c for r in (1, 2, 3) for c in itertools.combinations(
    ("nf12_packed", "nf12_escapes", "nf12_exp_base"), r)], ids="+".join)
def test_partial_checkpoint_strict_false(keys, sets):
    """strict=False applies exactly the supplied entries; the rest keep their values."""
    a, same, _ = sets
    net = Net(a)
    new = same["proj"].state_dict("proj.weight.")
    result = net.load_state_dict({f"proj.weight.{key}": new[f"proj.weight.{key}"] for key in keys}, strict=False)
    assert not result.unexpected_keys
    old = a["proj"].state_dict()
    merged = {k: (same["proj"].state_dict()[k] if k in keys else old[k]) for k in old}
    expect = nf12.NF12Weight.from_state_dict(merged)
    assert net.proj.weight.exp_base == expect.exp_base
    assert torch.equal(net.proj.weight.packed, expect.packed)
    x, _, _ = inputs()
    assert bits_equal(net.proj(x), nf12.linear(x, expect, bias()))
    for name in ("layers.0", "layers.1", "emb"):
        assert bits_equal(modules(net)[name].weight.decode(), a[name].decode())


@pytest.mark.parametrize("missing", ["nf12_packed", "nf12_escapes", "nf12_exp_base"])
def test_strict_partial_checkpoint(missing, sets):
    """torch passes strict=True to every module and checks strictness afterwards,
    so like torch's own modules the supplied entries are applied before a strict
    load raises. verify_weights reports a weight that was left mixed."""
    a, _, esc = sets
    net = Net(a)
    partial = {k: v for k, v in checkpoint(esc).items() if not k.endswith(f"proj.weight.{missing}")}
    with pytest.raises(RuntimeError, match="Missing key"):
        net.load_state_dict(partial)
    if missing == "nf12_exp_base":
        nf12.verify_weights(net)  # packed and escapes still describe one weight
    else:
        with pytest.raises(ValueError, match="proj.weight: nf12_packed has"):
            nf12.verify_weights(net)


@pytest.mark.parametrize("order", ["packed first", "escapes first"])
def test_sharded_checkpoint(order, sets):
    """One weight's tensors split across two shards, loaded one shard at a time."""
    a, _, esc = sets
    net = Net(a)
    sd = checkpoint(esc)
    first = {k: v for k, v in sd.items() if k.endswith("nf12_packed")}
    second = {k: v for k, v in sd.items() if not k.endswith("nf12_packed")}
    if order == "escapes first":
        first, second = second, first
    net.load_state_dict(first, strict=False)
    with pytest.raises(ValueError, match="escaped groups"):
        nf12.verify_weights(net)
    net.load_state_dict(second, strict=False)
    nf12.verify_weights(net)
    assert_holds(net, esc)


def test_truncated_escapes_rejected(sets):
    a, _, esc = sets
    net = Net(a)
    sd = checkpoint(esc)
    sd["proj.weight.nf12_escapes"] = sd["proj.weight.nf12_escapes"][:-1]
    with pytest.raises(RuntimeError, match="escaped groups"):
        net.load_state_dict(sd)
    assert bits_equal(net.proj.weight.decode(), a["proj"].decode())


def test_meta_model_skips_partial_weights_when_not_strict(sets):
    """Sharded loading into a meta model: a shard holding part of a weight is a
    tolerated missing-key case under strict=False, not an error."""
    _, _, esc = sets
    net = Net(device="meta")
    shard = {k: v for k, v in checkpoint(esc).items() if k.endswith("nf12_packed")}
    result = net.load_state_dict(shard, strict=False, assign=True)
    assert "proj.weight.nf12_escapes" in result.missing_keys


def test_non_tensor_entry_is_a_load_error(sets):
    a, _, _ = sets
    net = Net(a)
    sd = checkpoint(a)
    sd["proj.weight.nf12_exp_base"] = 5
    with pytest.raises(RuntimeError, match="must be a tensor"):
        net.load_state_dict(sd)


def test_verify_weights_flags_unloaded_meta_weights(sets):
    """A weight whose tensors were split across shards cannot load into a meta
    model one shard at a time; verify_weights must report it, not skip it."""
    _, _, esc = sets
    net = Net(device="meta")
    sd = checkpoint(esc)
    for shard in ({k: v for k, v in sd.items() if k.endswith("nf12_packed")},
                  {k: v for k, v in sd.items() if not k.endswith("nf12_packed")}):
        net.load_state_dict(shard, strict=False, assign=True)
    with pytest.raises(ValueError, match="not loaded"):
        nf12.verify_weights(net)


def test_checkpoint_tensors_at_odd_offsets(sets):
    """Packed bytes sliced from a larger buffer (any storage offset) load normally."""
    a, _, esc = sets
    net = Net(a)
    sd = {k: v.cpu() for k, v in checkpoint(esc).items()}
    for key in [k for k in sd if k.endswith("nf12_packed")]:
        buf = torch.empty(sd[key].numel() + 1, dtype=torch.uint8)
        buf[1:] = sd[key].reshape(-1)
        sd[key] = buf[1:].view(sd[key].shape)  # storage offset 1
    net.load_state_dict(sd)
    assert_holds(net, esc)
