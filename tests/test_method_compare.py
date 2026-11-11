import torch
from openbonsai.core.method_compare import TrainTable, TrainLinear, TrainEmbedding, sparse_project, full_kd_ce
from openbonsai.core.ternary import TernaryLinear, TernaryEmbedding, shared_hadamard


def pair():
    torch.manual_seed(42)
    c = torch.randint(-1,2,(16,8),dtype=torch.int8)
    s = torch.rand(16,2).half()+.1
    h = shared_hadamard(8,'cpu',dtype=torch.float32)
    old = TernaryLinear(c,s,torch.ones(8),8,h)
    emb = TernaryEmbedding(c,s,torch.ones(8),8,h)
    t = TrainTable(old)
    return old,emb,t,TrainLinear(old,t),TrainEmbedding(emb,t)


def test_hard_forward_and_shared_gradients_sum():
    old,emb,t,head,lookup=pair()
    x=torch.randn(3,8)
    ids=torch.tensor([1,2,1])
    torch.testing.assert_close(head(x),old(x),rtol=0,atol=0)
    torch.testing.assert_close(lookup(ids),emb(ids),rtol=0,atol=0)
    a=torch.autograd.grad(head(x).square().sum(),(t.master,t.log_mult),retain_graph=False)
    b=torch.autograd.grad(lookup(ids).square().sum(),(t.master,t.log_mult),retain_graph=False)
    both=torch.autograd.grad(head(x).square().sum()+lookup(ids).square().sum(),(t.master,t.log_mult))
    for u,v,w in zip(a,b,both):torch.testing.assert_close(u+v,w)
    model=torch.nn.ModuleList([head,lookup])
    assert len(list(model.parameters()))==2
    assert head.table.master is lookup.table.master


def test_ste_derivative_matches_declared_surrogate():
    _,_,t,_,_=pair()
    g=torch.randn_like(t.master)
    q=t.weight()
    (q*g).sum().backward()
    expected=(g.reshape(16,2,4)*t.scales()[...,None]).reshape_as(g)
    torch.testing.assert_close(t.master.grad,expected)


def test_pv_nearest_projection_and_trust():
    old=torch.ones(4,128,dtype=torch.int8)
    s=torch.full((4,1),.25)
    p=old.float();p[0,:4]=-.8
    # With unrestricted bounds, compare against exhaustive 3-choice squared error.
    c,stats=sparse_project(p,old,s,fraction=1,trust=1)
    options=torch.tensor([-1.,0.,1.])
    exhaustive=options[(p[...,None]-options).square().argmin(-1)].to(torch.int8)
    torch.testing.assert_close(c,exhaustive)
    c,stats=sparse_project(p,old,s,fraction=.01,trust=.01)
    assert stats['fraction']<=.01 and stats['relative_l2']<=.0100001
    unchanged,_=sparse_project(old.float()+.1,old,s)
    assert torch.equal(unchanged,old)


def test_full_vocab_gradients_and_topk_bucket_is_lower_bound():
    torch.manual_seed(4)
    s=torch.randn(5,11,dtype=torch.double,requires_grad=True)
    t=torch.randn(5,11,dtype=torch.double)
    loss=full_kd_ce(s,t)
    grad=torch.autograd.grad(loss,s)[0]
    torch.testing.assert_close(grad.float(),(s.float().softmax(-1)-t.float().softmax(-1))/5)
    labels=torch.tensor([2,3,4,5,6])
    mix=full_kd_ce(s,t,labels,alpha=.2)
    assert torch.isfinite(mix)
    bucket_ids=t.float().topk(3,-1).indices
    p=t.float().softmax(-1);q=s.float().log_softmax(-1)
    pk=p.gather(-1,bucket_ids);qk=q.gather(-1,bucket_ids).exp()
    bucket=(pk*(pk.log()-qk.log())).sum(-1)+((1-pk.sum(-1))*((1-pk.sum(-1)).log()-(1-qk.sum(-1)).log()))
    assert float(bucket.mean().detach())<=float(loss.detach())+1e-6


def test_pv_trust_bound_accepts_a_nonempty_strict_subset():
    old=torch.ones(512,128,dtype=torch.int8)
    s=torch.full((512,1),.25)
    p=old.float();p[0,:100]=-1
    new,stats=sparse_project(p,old,s,fraction=.0001,trust=.01)
    assert stats['changed']==1
    assert int((new!=old).sum())==1
    assert stats['relative_l2']<=.01 and stats['fraction']<=.0001


def test_reusing_one_head_tensor_across_chunks_preserves_values_and_gradients():
    _,_,t,head,_=pair()
    x=torch.randn(9,8)
    targets=torch.randn(9,16)
    direct=sum(full_kd_ce(head(x[i:i+3]),targets[i:i+3])/3 for i in range(0,9,3))
    expected=torch.autograd.grad(direct,(t.master,t.log_mult))
    q=t.weight()
    xr=((x*head.signs).reshape(9,1,8)@head.h).reshape_as(x)
    cached=sum(full_kd_ce(torch.nn.functional.linear(xr[i:i+3],q),targets[i:i+3])/3 for i in range(0,9,3))
    actual=torch.autograd.grad(cached,(t.master,t.log_mult))
    torch.testing.assert_close(cached,direct,atol=0,rtol=0)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)


def test_factored_optimizer_resume_matches_uninterrupted():
    from transformers import Adafactor
    _,_,t,head,_=pair()
    opt=Adafactor([t.master],lr=.003,scale_parameter=False,relative_step=False,warmup_init=False)
    x=torch.randn(3,8)
    head(x).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    import copy
    saved=copy.deepcopy(t.state_dict());os=copy.deepcopy(opt.state_dict())
    head(x).square().sum().backward();opt.step()
    expected=t.master.detach().clone()
    t.load_state_dict(saved);opt.load_state_dict(os);opt.zero_grad(set_to_none=True)
    head(x).square().sum().backward();opt.step()
    torch.testing.assert_close(t.master,expected,rtol=0,atol=0)
