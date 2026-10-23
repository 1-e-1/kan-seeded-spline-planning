"""Conservative paired success bounds; does not collapse with zero discordances."""
import numpy as np
from scipy.stats import beta


def paired_success_interval(baseline, candidate, alpha=.05):
    a=np.asarray(baseline,dtype=bool);b=np.asarray(candidate,dtype=bool)
    assert a.shape==b.shape and a.ndim==1 and len(a)>0
    n=len(a);positive=int(sum(b & ~a));negative=int(sum(a & ~b))
    def bounds(x):
        # Four one-sided bounds: joint coverage at least 1-alpha by union bound.
        tail=alpha/4
        return (0. if x==0 else float(beta.ppf(tail,x,n-x+1)),
                1. if x==n else float(beta.ppf(1-tail,x+1,n-x)))
    pl,pu=bounds(positive);nl,nu=bounds(negative)
    return dict(difference=(positive-negative)/n,ci=[pl-nu,pu-nl],n=n,
                candidate_only_success=positive,baseline_only_success=negative,
                method='Bonferroni Clopper-Pearson bounds on paired discordance probabilities; assumes independent sampled task pairs')
