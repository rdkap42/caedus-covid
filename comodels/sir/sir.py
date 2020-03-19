import numpy as np

class SIR:
    def __init__(self, S: int, I: int, R: int, beta: float,
                 gamma: float, beta_decay: float=0) -> None:
        if beta_decay < 0 or beta_decay > 1:
            raise ValueError('bad beta decay')
        self.S = S
        self.I = I
        self.R = R
        self.N = sum([S,I,R])
        self.beta = beta
        self.gamma = gamma
        self.beta_decay = beta_decay
        self.s = [S]
        self.i = [I]
        self.r = [R]

    def _rectify(self, x: float) -> float:
        out = 0 if x < 0 else x
        return out

    def __Sn(self, S: int, I:int) -> float:
        return self._rectify((-self.beta * S * self.I) + S)

    def __In(self, S: int, I:int) -> float:
        return self._rectify((self.beta * S * I - self.gamma * I) + I)

    def __Rn(self, I: int, R: int) -> float:
        return self._rectify((self.gamma * I + R))

    def __step(self, S: int, I: int, R: int) -> (float, float, float):
        Sn = self.__Sn(S, I)
        Rn = self.__Rn(I, R)
        In = self.__In(S, I)
        scale = self.N / (Sn + Rn + In)
        S = Sn * scale
        I = In * scale
        R = Rn * scale
        return S, I , R

    def sir(self, n_days: int) -> (np.ndarray, np.ndarray, np.ndarray):
        S = self.S
        I = self.I
        R = self.R
        self.s = [S]
        self.i = [I]
        self.r = [R]
        for day in range(n_days):
            S, I, R = self.__step(S, I, R)
            self.beta *= (1-self.beta_decay)
            self.s.append(S)
            self.i.append(I)
            self.r.append(R)
        return (np.asarray(x) for x in [self.s, self.i, self.r])

