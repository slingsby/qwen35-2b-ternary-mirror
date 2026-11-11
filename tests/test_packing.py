"""2-bit state-code payload (formats/packing.py)."""
import pytest
import torch

from openbonsai.formats.packing import pack_ternary_codes, unpack_ternary_codes


def test_bit_layout_and_non_multiple_of_four():
    codes = torch.tensor([-1, 0, 1, -1, 1], dtype=torch.int8)
    packed = pack_ternary_codes(codes)
    assert packed.tolist() == [146, 1]
    assert torch.equal(unpack_ternary_codes(packed, 5), codes)
    assert pack_ternary_codes(torch.empty(0)).numel() == 0
    assert unpack_ternary_codes(torch.empty(0, dtype=torch.uint8), 0).numel() == 0


@pytest.mark.parametrize("codes", [torch.tensor([2]), torch.tensor([0.5]), torch.tensor([float('nan')])])
def test_reject_nonternary_values(codes):
    with pytest.raises(ValueError, match="ternary codes"):
        pack_ternary_codes(codes)


def test_reject_reserved_code_and_noncanonical_padding():
    with pytest.raises(ValueError, match="reserved"):
        unpack_ternary_codes(torch.tensor([3], dtype=torch.uint8), 1)
    with pytest.raises(ValueError, match="padding"):
        unpack_ternary_codes(torch.tensor([4], dtype=torch.uint8), 1)
    with pytest.raises(ValueError, match="byte count"):
        unpack_ternary_codes(torch.tensor([0, 0], dtype=torch.uint8), 1)


@pytest.mark.parametrize("numel", [0, 1, 4, 127, 128, 1000])
def test_random_round_trip(numel):
    codes = torch.randint(-1, 2, (numel,), dtype=torch.int8, generator=torch.Generator().manual_seed(numel))
    packed = pack_ternary_codes(codes)
    assert packed.numel() == (numel + 3) // 4
    assert torch.equal(unpack_ternary_codes(packed, numel), codes)
