class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        result = Value(self.data + other.data, (self, other), '+')
        return result
    
    def __mul__(self, other):
        result = Value(self.data * other.data, (self, other), '*')
        return result
    
    def __repr__(self):
        return f"Value(data={self.data})"