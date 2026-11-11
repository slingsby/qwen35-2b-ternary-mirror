import pytest
import torch
from openbonsai.core.data_recovery import pack_complete,target_windows,masked_kd_mean
from openbonsai.core.method_compare import full_kd_ce


def test_complete_conversations_are_not_cut_or_duplicated_and_padding_is_masked():
    records=[{'id':str(i),'ids':[i+1]*n} for i,n in enumerate([9,7,6,5,3,19])]
    rows,masks,sources,dropped=pack_complete(records,pad_id=99,length=16,min_fraction=.5)
    lookup={r['id']:r['ids'] for r in records}
    used=[]
    for row,mask,group in zip(rows,masks,sources):
        complete=sum((lookup[k] for k in group),[]);used.extend(group)
        assert row[:len(complete)]==complete
        assert mask==[True]*len(complete)+[False]*(16-len(complete))
        assert row[len(complete):]==[99]*(16-len(complete))
    assert len(used)==len(set(used)) and set(used)|set(dropped)==set(lookup)
    assert '5' in dropped


def test_target_budget_is_exact_and_primary_draws_match_despite_padding():
    full=torch.ones(12,512,dtype=torch.bool);padded=full.clone();padded[:,390:]=False
    def draw(m):return target_windows(m,torch.Generator().manual_seed(42),torch.Generator().manual_seed(842))
    a,b=draw(full),draw(padded)
    assert len(a)==4 and len(b)>4
    assert [i for i,_ in a]==[i for i,_ in b[:4]]
    assert sum(int(s.sum()) for _,s in a)==sum(int(s.sum()) for _,s in b)==2044
    for i,s in b:assert not bool((s & ~padded[i,1:]).any())


def test_masked_padding_has_zero_gradient_even_with_excluded_nonfinite_logits():
    raw=torch.randn(7,13);teacher=torch.randn_like(raw);mask=torch.tensor([1,1,1,0,0,0,0],dtype=torch.bool)
    raw[~mask]=float('nan');teacher[~mask]=float('nan');student=raw.requires_grad_()
    loss=masked_kd_mean(student,teacher,mask);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad[~mask])==0
    reference=full_kd_ce(student.detach()[mask],teacher[mask])
    torch.testing.assert_close(loss,reference)


def test_all_valid_loss_and_accumulation_match_original_token_normalization():
    s=torch.randn(4,511,17,requires_grad=True);t=torch.randn_like(s)
    old=sum(full_kd_ce(s[i],t[i])/4 for i in range(4))
    new=sum(masked_kd_mean(s[i],t[i],torch.ones(511,dtype=torch.bool))*(511/2044) for i in range(4))
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    a=torch.autograd.grad(old,s)[0];b=torch.autograd.grad(new,s)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_invalid_geometry_duplicate_ids_and_unfillable_budget_fail_closed():
    with pytest.raises(ValueError):pack_complete([],pad_id=0,length=1)
    with pytest.raises(ValueError):pack_complete([{'id':'x','ids':[1]},{'id':'x','ids':[2]}],pad_id=0)
    with pytest.raises(RuntimeError):target_windows(torch.zeros(2,512,dtype=torch.bool),torch.Generator(),torch.Generator())
