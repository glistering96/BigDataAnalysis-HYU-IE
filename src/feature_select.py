"""
Feature selector module

supports F-test, mutual information, and chi-squared

Although f_classif is not for categorical data as it does not obey the assumptions for ANOVA*, 

we just give it a chane to see if it works.


*ANOVA: Analysis of Variance's assumptions:

1. The samples are independent.
2. Each sample is from a normally distributed population.
3. The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

"""
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif


class FeatureSelector:
    methods = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'chi2': chi2
    }
    def __init__(self, method_nm=None, **kwargs) -> None:
        self.method_nm = method_nm
        
        if method_nm is not None:
            self._method = self.methods[method_nm]
            
    def select(self, X, y, k):
        # select k best features
        
        if self.method_nm is None:
            return X
        
        selected = SelectKBest(score_func=self._method, k=k).fit(X, y)
        
        return selected.transform(X)
    
    def run(self, X, y, k):
        # alias for select method_nm
        return self.select(X, y, k)
    
    def get_method_nm(self):
        return self.method_nm if self.method_nm is not None else 'None'
    