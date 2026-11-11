"""Opt-in data-ablation packing and token-accounting helpers; no model changes."""
import random
import torch
from .method_compare import full_kd_ce


def pack_complete(records, *, pad_id, length=512, min_fraction=.75, seed=42):
    """Best-fit whole conversations, never truncated; right padding has a false mask."""
    if length<2 or not 0<min_fraction<=1:raise ValueError('invalid packing geometry')
    records=list(records)
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate source ID')
    rng=random.Random(seed);rng.shuffle(records)
    records.sort(key=lambda r:len(r['ids']),reverse=True)
    bins=[];dropped=[]
    for record in records:
        ids=list(record['ids'])
        if not ids or len(ids)>length:
            dropped.append(record['id']);continue
        choices=[(length-len(b['ids'])-len(ids),i) for i,b in enumerate(bins) if len(b['ids'])+len(ids)<=length]
        if choices:
            i=min(choices)[1]
            bins[i]['ids'].extend(ids);bins[i]['sources'].append(record['id'])
        else:bins.append({'ids':ids,'sources':[record['id']]})
    rows=[];masks=[];sources=[]
    for item in bins:
        n=len(item['ids'])
        if n<int(length*min_fraction):
            dropped.extend(item['sources']);continue
        rows.append(item['ids']+[pad_id]*(length-n))
        masks.append([True]*n+[False]*(length-n))
        sources.append(item['sources'])
    order=list(range(len(rows)));rng.shuffle(order)
    return [rows[i] for i in order],[masks[i] for i in order],[sources[i] for i in order],dropped


def target_windows(masks, generator, fill_generator, *, targets=2044, base_microbatches=4, max_microbatches=8):
    """Same primary draws across arms; masked filler makes scored targets exact."""
    masks=torch.as_tensor(masks,dtype=torch.bool)
    if masks.ndim!=2 or masks.shape[1]<2:raise ValueError('masks must be [rows, tokens]')
    primary=torch.randint(len(masks),(base_microbatches,),generator=generator).tolist()
    result=[];remaining=targets
    for j in range(max_microbatches):
        index=primary[j] if j<len(primary) else int(torch.randint(len(masks),(1,),generator=fill_generator))
        score=masks[index,1:] & masks[index,:-1]
        positions=score.nonzero().flatten()
        if len(positions)>remaining:
            score=score.clone();score[positions[remaining:]]=False
        used=int(score.sum())
        result.append((index,score));remaining-=used
        if not remaining:return result
    raise RuntimeError('target budget cannot be filled within microbatch cap')


def masked_kd_mean(student,teacher,mask):
    """Full-vocab KL on scored positions only; excluded NaNs cannot contaminate it."""
    mask=mask.to(device=student.device,dtype=torch.bool)
    if mask.shape!=student.shape[:-1]:raise ValueError('mask/logit shape mismatch')
    if not bool(mask.any()):return student[mask].sum()
    return full_kd_ce(student[mask],teacher[mask])
