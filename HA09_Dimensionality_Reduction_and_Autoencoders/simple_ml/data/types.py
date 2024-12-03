import numpy as np


class Variable:
    def __init__(self, value: np.ndarray = None, derivable: bool = True):
        self.value: np.ndarray = value # 0-th dimension is batch size (multiple samples), even if batch size is 1
        
        self._derivable: bool = derivable
        self.gradient: np.ndarray = None
    
    def set_derivable(self, derivable: bool):
        self._derivable = derivable
    
    @property
    def shape(self):
        return self.value.shape

    @property
    def G(self):
        return self.gradient

    @property
    def T(self):
        return self.transpose()
    
    def __getitem__(self, index: slice):
        return Variable(self.value[index], derivable = self._derivable)
    
    def __setitem__(self, index: slice, value): # TODO: should be allowed?
        if isinstance(value, Variable):
            v = value.value.copy()
        elif isinstance(value, np.ndarray):
            v = value.copy()
        elif isinstance(value, list):
            v = np.array(value)
        else:
            v = value
        self.value[index] = v

    def __repr__(self): # TODO: str or value?
        return f"Variable[value = {self.value}, gradient = {self.gradient}, derivable = {self._derivable}]"

    # TODO: 以下重载中（可能需要的）的计算图记录（以供自动求导）。目前属性只由外部直接赋值修改。
    # TODO: 其他重载方法。
    # TODO: 和常数（np.ndarray）的运算 & 异常处理。
    
    def transpose(self):
        return Variable(self.value.T, derivable = self._derivable)
    
    def __add__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value + other.value, derivable = self._derivable and other._derivable)
        else:
            return Variable(self.value + other, derivable = self._derivable) # TODO: or return modified `self`?
    
    def __sub__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value - other.value, derivable = self._derivable and other._derivable)
        else:
            return Variable(self.value - other, derivable = self._derivable)
    
    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value * other.value, derivable = self._derivable and other._derivable)
        else:
            return Variable(self.value * other, derivable = self._derivable)
    
    def __truediv__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value / other.value, derivable = self._derivable and other._derivable)
        else:
            return Variable(self.value / other, derivable = self._derivable)
    
    def __pow__(self, power, modulo = None): # TODO: modulo?
        return Variable(self.value ** power, derivable = self._derivable)
    
    def __neg__(self):
        return Variable(-self.value, derivable = self._derivable)
    
    def __matmul__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value @ other.value, derivable = self._derivable and other._derivable)
        else:
            return Variable(self.value @ other, derivable = self._derivable)

