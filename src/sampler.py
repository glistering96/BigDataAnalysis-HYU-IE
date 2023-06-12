"""
An api for creating a undersampled/oversampled/SMOTE applied training set.

"""
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids


class Sampler:
    methods = {

        'random_over': RandomOverSampler,
        'smote': SMOTE,
        'adasyn': ADASYN,   # over sampling
        'cluster_centroid': ClusterCentroids,   # under sampling
        'random_under': RandomUnderSampler
    }

    def __init__(self,
                 method=None,
                 seed=43,
                 **kwargs
                 ) -> None:
        self.method = method
        self.seed = seed
        
        if method is not None and method not in self.methods.keys():
            raise ValueError(f"Sampling method must be one of: {self.methods.keys()}. Selected method: {self.method}")
        
        if method is not None:
            self._sampler = self.methods[method](random_state=self.seed, **kwargs)
            
        else:
            self._sampler = None
        
    def run(self, X, y):
        if self._sampler is None:
            return X, y
        return self._sampler.fit_resample(X, y)

# api style function
def run_sample(X, y, method=None, **kwargs):
    if method is None:
        return X, y
    
    sampler = Sampler(method, **kwargs)
    
    return sampler.run(X, y)
    
    
    
