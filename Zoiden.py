import numpy as np
from scipy.optimize import linprog

class ZoutendijkMethod:
    def __init__(self, f, constraints, beta=0.5, sigma=0.1, epsilon=1e-3, max_iter=100, h=1e-6):
        """
        Инициализация метода Зойтендейка с автоматическим расчетом градиентов
        
        Параметры:
        f - целевая функция (принимает x, возвращает скаляр)
        constraints - список функций ограничений (каждая принимает x, возвращает скаляр)
        beta - параметр уменьшения шага (0 < beta < 1)
        sigma - параметр достаточного уменьшения (0 < sigma < 1)
        epsilon - точность решения
        max_iter - максимальное число итераций
        h - шаг для численного дифференцирования
        """
        self.f = f
        self.constraints = constraints
        self.beta = beta
        self.sigma = sigma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.h = h
        self.history = []
        
        # Создаем функции для численного расчета градиентов
        self.grad_f = self._create_numerical_gradient(f)
        self.grad_constraints = [self._create_numerical_gradient(g) for g in constraints]
    
    def _create_numerical_gradient(self, func):
        """Создает функцию для численного расчета градиента"""
        def gradient(x):
            x = np.asarray(x, dtype=float)
            grad = np.zeros_like(x)
            h = self.h
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
            return grad
        return gradient
    
    def find_direction(self, x):
        """Находит направление спуска решая задачу ЛП"""
        n = len(x)
        grad_f = self.grad_f(x)
        
        # Определяем активные ограничения
        active = [i for i, g in enumerate(self.constraints) 
                 if g(x) >= -self.epsilon]
        
        if not active:
            # Если нет активных ограничений, используем антиградиент
            d = -grad_f
            return d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.zeros(n)
        
        # Формируем задачу ЛП
        c = np.zeros(n + 1)
        c[-1] = 1  # Минимизируем z
        
        # Ограничения
        A_ub = []
        b_ub = []
        
        # Ограничение для целевой функции
        row = np.zeros(n + 1)
        row[:n] = grad_f
        row[-1] = -1
        A_ub.append(row)
        b_ub.append(0)
        
        # Ограничения для активных условий
        for i in active:
            grad_g = self.grad_constraints[i](x)
            row = np.zeros(n + 1)
            row[:n] = grad_g
            row[-1] = -1
            A_ub.append(row)
            b_ub.append(0)
        
        # Ограничения на направление (нормировка)
        bounds = [(-1, 1) for _ in range(n)] + [(None, None)]
        
        # Решаем задачу ЛП
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        
        if not res.success:
            return np.zeros(n)
        
        d = res.x[:-1]
        z = res.x[-1]
        
        if abs(z) < self.epsilon:
            return np.zeros(n)
        
        return d

    def line_search(self, x, d):
        """Поиск шага вдоль направления d"""
        alpha = 1.0
        f_x = self.f(x)
        grad_f_d = np.dot(self.grad_f(x), d)
        
        while alpha > 1e-10:
            x_new = x + alpha * d
            # Проверяем допустимость
            feasible = all(g(x_new) <= self.epsilon for g in self.constraints)
            # Проверяем достаточное уменьшение
            decrease = self.f(x_new) <= f_x + self.sigma * alpha * grad_f_d
            
            if feasible and decrease:
                return alpha
            
            alpha *= self.beta
        
        return 0.0

    def solve(self, x0):
        """Решает задачу оптимизации"""
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]
        
        for _ in range(self.max_iter):
            d = self.find_direction(x)
            
            if np.linalg.norm(d) < self.epsilon:
                break
                
            alpha = self.line_search(x, d)
            
            if alpha < 1e-10:
                break
                
            x = x + alpha * d
            self.history.append(x.copy())
            
        return x

# Пример использования
if __name__ == "__main__":
    # Пример 1: Простая квадратичная задача с линейными ограничениями
    def f1(x):
        return (x[0])**2 - (x[1])**2
    
    def g1(x):
        return 2*x[0] - x[1] - 2
    
    # Создаем решатель 
    solver = ZoutendijkMethod(
        f=f1,
        constraints=[g1],
        epsilon=1e-4
    )
    
    # Начальная точка 
    x0 = [1, 0]
    
    # Решаем задачу
    solution = solver.solve(x0)
    
    print("Решение:", solution)
    print("Значение функции:", f1(solution))
    print("Ограничения:", g1(solution))
    print("Число итераций:", len(solver.history))
    