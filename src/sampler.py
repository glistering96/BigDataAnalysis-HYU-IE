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
                 method_nm=None,
                 seed=43,
                 **kwargs
                 ) -> None:
        self.method_nm = method_nm
        self.seed = seed
        
        if self.method_nm is not None and self.method_nm not in self.methods.keys():
            raise ValueError(f"Sampling method_nm must be one of: {self.methods.keys()}. Selected method_nm: {self.method_nm}")
        
        if self.method_nm is not None:
            self._sampler = self.methods[method_nm](random_state=self.seed, **kwargs)
            
        else:
            self._sampler = None
        
    def run(self, X, y):
        if self._sampler is None:
            return X, y
        return self._sampler.fit_resample(X, y)
    
    def get_method_nm(self):
        return self.method_nm if self.method_nm is not None else 'None'

# api style function
def run_sample(X, y, method_nm=None, **kwargs):
    if method_nm is None:
        return X, y
    
    sampler = Sampler(method_nm, **kwargs)
    
    return sampler.run(X, y)
    
    
    
