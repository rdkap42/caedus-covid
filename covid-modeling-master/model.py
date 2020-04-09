from typing import List, Sequence, Tuple, Union

class Comp():
    def __init__(self, v: str):
        self.v = v
    def __repr__(self):
        return "Comp(%s)" % self.v

class Param():
    def __init__(self, param_name: str, param_val: float):
        self.param_name = param_name
        self.param_val = param_val

    def __repr__(self):
        return "Param({}={})".format(self.param_name, self.param_val)

Rate = List[Union[Comp, Param]]

class Edge():
    def __init__(start: Comp, rate: Tuple[Comp, Rate, Comp], end: Comp):
        pass

class Model():
    def __init__(self, compartments: Sequence[str], edges: List[Tuple[str, str, str]]):
        self.compartments = compartments
        self.edges = []
        params = set()
        for source, rate, sink in edges:
            rate_components = [chunk.strip() for chunk in rate.split('*')]
            for rate_component in rate_components:
                if not rate_component in self.compartments:
                    params.add(rate_component)
            self.edges.append((source, rate_components, sink))
        self.params = list(sorted(params))


    def deriv(self, x, params):
        print(x, params)
        assert(len(x) == len(self.compartments))
        assert(len(params) == len(self.params))
        assert(np.sum(x) == 1)
        assert(np.all(0 <= x) and np.all(x <= 1))
        dx = np.zeros(len(x))
        for source, rate_comps, sink in self.edges:
            print("edge:", source, rate_comps, sink)
            rate = 1
            for rc in rate_comps:
                if rc in self.compartments:
                    comp_idx = self.compartments.index(rc)
                    val = x[comp_idx]
                else:
                    param_idx = self.params.index(rc)
                    val = params[param_idx]
                rate *= val
            print("final rate:", rate)
            source_idx = self.compartments.index(source)
            sink_idx = self.compartments.index(sink)
            dx[source_idx] -= rate
            dx[sink_idx] += rate
        if not (np.isclose(np.sum(dx), 0)):
            raise AssertionError("failed on: {}, {}".format(x, params))
        return dx

    def integrate(self, x0, params, T):
        assert(len(x0) == len(self.compartments))
        assert(np.isclose(sum(x0), 1))
        f = lambda t, x: self.deriv(x, params)
        sol = solve_ivp(f, (0, T), x0, max_step=1, t_eval=range(T))

sir = Model(
    "SIR",
    [
        ("S", "beta * I", "I"),
        ("I", "gamma", "R")
    ]
)
