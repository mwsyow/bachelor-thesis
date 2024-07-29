import numpy as np
import pandas as pd

class Encoder():
    def __init__(self, features, step_sizes=None):
        """
        Args:
        features is a dictionary of feature names and unique values of each feature.
        step_sizes (dictionary, optional):dictionary of feature names and its respective step sizes. Defaults to None.

        Objects:
        self.features_map is a dictionary of feature names and encodings of each value
        """
        self.features_map = {}
        self.features_is_interval = {}
        interval_feature_names = list(step_sizes.keys()) if not step_sizes is None  else []
        for key, data_unique in features.items():
            self.features_is_interval[key] = True if key in interval_feature_names else False
            interval = None
            if self.features_is_interval[key]:
                interval = self.make_interval(*step_sizes[key])
            mapping = self.map(data_unique, interval)  
            self.features_map[key] = mapping
                
                      
        
    def __call__(self, data_dict):
        """_summary_

        Args:
            data_dict (dictionary): dictionary of feature names and data values

        Returns:
            _type_: dictionary of feature names and encodings 
        """
        res_dict = {}
        for key, data in data_dict.items():
            res_dict[key] = self.encode(key, data)
        return res_dict
    
    
    def map(self, data_unique, interval=None):
        """_summary_
        if interval argument is given then the key would be a tuple (x, y) where x < y and a value z where x <= z < y would correspond to the interval.
        """
        if not interval is None:
            return {'INTERVAL': interval}
        return dict(zip(data_unique, range(len(data_unique))))
    
    
    def make_interval(self, start, stop, step_size):
        return np.arange(start, stop+1, step_size)
    
    
    def interval_index(self, feature_name, value):
        interval = self.features_map[feature_name]['INTERVAL']
        for i in range(len(interval)-1):
            if value >= interval[i] and value < interval[i+1]:
                return i
        return len(interval)-1
    
    def __str__(self=None):
        return "Encoder"
    # def __str__():
    #     return "Encoder"

class OneHotEncoder(Encoder):
    def __init__(self, features, step_sizes=None):
        super().__init__(features, step_sizes)
            
    def encode(self, feature_name, data):
        num_unique_values = len(self.features_map[feature_name]) if not self.features_is_interval[feature_name] else len(self.features_map[feature_name]['INTERVAL'])
        num_data = len(data)
        encoding = np.zeros((num_data, num_unique_values))
        for i, val in enumerate(data):
            if self.features_is_interval[feature_name]: 
                encoding[i, self.interval_index(feature_name, val)] += 1
            else: 
                encoding[i, self.features_map[feature_name][val]] += 1
        return encoding
    
    def __str__(self=None):
        return "OneHotEncoder"
    # def __str__():
    #     return "OneHotEncoder"
    
    
    
        
class IntegerEncoder(Encoder):
    def __init__(self, features, step_sizes=None, is_label=False):
        super().__init__(features, step_sizes)
        self.is_label = is_label
    
    def encode(self, feature_name, data):
        num_data = len(data)
        encoding = np.zeros((num_data, 1))
        for i, val in enumerate(data):
            if self.features_is_interval[feature_name]: 
                encoding[i] = self.interval_index(feature_name, val) if self.is_label else self.interval_index(feature_name, val)+1
            else: 
                encoding[i] = self.features_map[feature_name][val] if self.is_label else self.features_map[feature_name][val]+1
        return encoding
    
    def __str__(self=None):
        return "IntegerEncoder"
    # def __str__():
    #     return "IntegerEncoder"


class ThermometerEncoder(Encoder):
    def __init__(self, features, step_sizes=None):
        super().__init__(features, step_sizes)
    
    def encode(self, feature_name, data):      
        num_unique_values = len(self.features_map[feature_name]) if not self.features_is_interval[feature_name] else len(self.features_map[feature_name]['INTERVAL'])
        num_data = len(data)
        encoding = np.zeros((num_data, num_unique_values))
        for i, val in enumerate(data):
            if self.features_is_interval[feature_name]: 
                encoding[i, : self.interval_index(feature_name, val) + 1] += 1
            else: 
                encoding[i, : self.features_map[feature_name][val] + 1] += 1
        return encoding
        
    def __str__(self=None):
        return "ThermometerEncoder"
    # def __str__():
    #     return "ThermometerEncoder"

