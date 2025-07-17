import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class QuadraticFunction:
    """
    Matrix-free implementation of quadratic function family f_k(x)
    f_k(x) = (L/4) * [(1/2) * x^T * A_k * x - e_1^T * x]
    where A_k is a k×k tridiagonal matrix embedded in n×n space
    """
    
    def __init__(self, L: float, k: int, n: int = None):
        """
        Parameters:
        L: Lipschitz constant (L > 0)
        k: function index (k >= 1)
        n: vector dimension (defaults to k if not specified)
        """
        self.L = L
        self.k = k
        self.n = n if n is not None else k
        if self.n < self.k:
            raise ValueError(f"Vector dimension n={self.n} must be at least k={self.k}")
    
    def _matvec_A_k(self, x: np.ndarray) -> np.ndarray:
        """Matrix-free computation of A_k * x for tridiagonal matrix A_k"""
        result = np.zeros_like(x)
        
        # A_k has tridiagonal structure: main diagonal = 2, sub/super diagonals = -1
        # Only the first k×k block is non-zero
        for i in range(self.k):
            result[i] = 2.0 * x[i]  # Main diagonal
            if i > 0:
                result[i] -= x[i-1]  # Super diagonal
            if i < self.k - 1:
                result[i] -= x[i+1]  # Sub diagonal
                
        return result
    
    def _quadratic_form_A_k(self, x: np.ndarray) -> float:
        """Matrix-free computation of x^T * A_k * x"""
        quadratic_form = 0.0
        
        # Direct computation without storing the matrix
        for i in range(self.k):
            # Main diagonal contribution: 2 * x[i]^2
            quadratic_form += 2.0 * x[i]**2
            
            # Off-diagonal contributions: -2 * x[i] * x[i+1] for adjacent pairs
            if i < self.k - 1:
                quadratic_form -= 2.0 * x[i] * x[i+1]
                
        return quadratic_form
    
    def function_value(self, x: np.ndarray) -> float:
        """Compute function value f_k(x) = (L/4) * [(1/2) * x^T * A_k * x - e_1^T * x]"""
        if len(x) != self.n:
            raise ValueError(f"Vector x must have dimension {self.n}")
        
        quadratic_term = 0.5 * self._quadratic_form_A_k(x)
        linear_term = x[0]  # e_1^T * x = x[0]
        
        return (self.L / 4) * (quadratic_term - linear_term)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient ∇f_k(x) = (L/4) * (A_k * x - e_1)"""
        if len(x) != self.n:
            raise ValueError(f"Vector x must have dimension {self.n}")
        
        grad = self._matvec_A_k(x)
        grad[0] -= 1.0  # Subtract e_1
        
        return (self.L / 4) * grad
    
    def theoretical_optimum(self) -> tuple:
        """Return theoretical optimal solution and optimal function value"""
        x_opt = np.zeros(self.n)
        
        # Theoretical formula: x_k^(i) = 1 - i/(k+1) for i = 1, ..., k
        for i in range(self.k):
            x_opt[i] = 1 - (i + 1) / (self.k + 1)
        
        # Optimal function value: f_k^* = (L/8) * (-1 + 1/(k+1))
        f_opt = (self.L / 8) * (-1 + 1 / (self.k + 1))
        
        return x_opt, f_opt
    
    def get_matrix_A_k(self) -> np.ndarray:
        """Get explicit matrix A_k for verification purposes only"""
        A = np.zeros((self.n, self.n))
        for i in range(self.k):
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < self.k - 1:
                A[i, i+1] = -1.0
        return A

def gradient_descent(func: QuadraticFunction, x0: np.ndarray, 
                    learning_rate: float, max_iter: int = 1000, 
                    tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Standard gradient descent algorithm
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    x = x0.copy()
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for i in range(max_iter):
        f_val = func.function_value(x)
        grad = func.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(x.copy())
        
        if grad_norm < tol:
            break
        
        x = x - learning_rate * grad
    
    return x, f_history, grad_norm_history, x_history

def nesterov_accelerated_gradient(func: QuadraticFunction, x0: np.ndarray,
                                learning_rate: float, max_iter: int = 1000,
                                tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Nesterov accelerated gradient algorithm
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    x = x0.copy()
    y = x0.copy()
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for i in range(max_iter):
        f_val = func.function_value(x)
        grad = func.gradient(y)
        grad_norm = np.linalg.norm(grad)
        
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(x.copy())
        
        if grad_norm < tol:
            break
        
        x_new = y - learning_rate * grad
        
        # Nesterov momentum parameter
        beta = i / (i + 3)
        y = x_new + beta * (x_new - x)
        
        x = x_new
    
    return x, f_history, grad_norm_history, x_history

def explicit_euler_ode_method(func: QuadraticFunction, x0: np.ndarray,
                            learning_rate: float, max_iter: int = 1000,
                            tol: float = 1e-8, alpha: float = 4.0, 
                            t0: float = 1.0, h: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Explicit Euler scheme for ODE-based optimization
    
    The scheme:
    (x_{k+1} - x_k)/h = v_k - β(t_k)∇f(x_k)
    (v_{k+1} - v_k)/h = -(α/t_k)(v_k - β(t_k)∇f(x_k)) + (β(t_k) - γ(t_k))∇f(x_k)
    
    where:
    α > 3, β(t) = (4-2αh/t)/L, γ(t) = β(t)/h
    t_k = t_0 + kh, k ≥ 0
    
    Parameters:
    alpha: parameter α (should be > 3)
    t0: initial time t_0
    h: step size (defaults to 1/sqrt(L) if not specified)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # if h is None:
    h = 1/np.sqrt(func.L)/1.2
    
    beta_0 = (4 - 2 * alpha * h / t0) / func.L
    grad_0 = func.gradient(x0)
    x = x0.copy()
    v = -beta_0 * grad_0
    f_history = []
    grad_norm_history = []
    x_history = []
    
    L = func.L  # Get Lipschitz constant from function
    
    for k in range(max_iter):
        # Current time
        t_k = t0 + k * h
        
        # Compute β(t_k) and γ(t_k)
        beta_k = (4 - 2 * alpha * h / t_k) / L
        gamma_k = beta_k / h
        diff_beta_k = 2*alpha*h/t_k/t_k/L
        
        # Compute function value and gradient
        f_val = func.function_value(x)
        grad = func.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        # Store history
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(x.copy())
        
        if grad_norm < tol:
            break
        
        # Update equations
        # (x_{k+1} - x_k)/h = v_k - β(t_k)∇f(x_k)
        x_new = x + h * (v - beta_k * grad)
        
        # (v_{k+1} - v_k)/h = -(α/t_k)(v_k - β(t_k)∇f(x_k)) + (β(t_k) - γ(t_k))∇f(x_k)
        v_update_term1 = -(alpha / t_k) * (v - beta_k * grad)
        v_update_term2 = (diff_beta_k-gamma_k) * grad
        v_new = v + h * (v_update_term1 + v_update_term2)
        
        # Update variables
        x = x_new
        v = v_new
    
    return x, f_history, grad_norm_history, x_history

def create_ode_method(alpha: float = 4.0, t0: float = 1.0, h: float = None):
    """Create ODE method with specific parameters"""
    def ode_method(func, x0, learning_rate, max_iter, tol=1e-8):
        return explicit_euler_ode_method(func, x0, learning_rate, max_iter, 
                                       tol, alpha, t0, h)
    return ode_method

def compare_methods(L: float, k: int, dim: int, learning_rate: float, 
                   methods: dict, max_iter: int = 1000, random_seed: int = 42):
    """
    Compare performance of multiple optimization methods
    
    Parameters:
    methods: dict with method_name: method_function pairs
             Each method_function should have signature (func, x0, learning_rate, max_iter, tol)
             and return (solution, f_history, grad_norm_history, x_history)
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create function instance
    func = QuadraticFunction(L, k, dim)
    
    # Get theoretical optimal solution
    x_theory, f_theory = func.theoretical_optimum()
    
    # Random initialization
    # x0 = np.zeros(dim)
    x0 = np.random.randn(dim)
    
    print(f"Quadratic function parameters: L={L}, k={k}, dimension={dim}")
    print(f"Theoretical optimal solution (first {min(k, 10)} components): {x_theory[:min(k, 10)]}")
    print(f"Theoretical optimal function value: {f_theory:.8f}")
    print(f"Initial point (first 5 components): {x0[:5]}")
    print(f"Initial function value: {func.function_value(x0):.6f}")
    print("-" * 50)
    
    # Run all methods and collect results
    results = {}
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-', '--', '-.']
    
    for i, (method_name, method_func) in enumerate(methods.items()):
        print(f"\nRunning {method_name}...")
        start_time = time.time()
        x_opt, f_history, grad_history, x_history = method_func(func, x0, learning_rate, max_iter)
        computation_time = time.time() - start_time
        
        # Store results
        results[method_name] = {
            'solution': x_opt,
            'f_history': f_history,
            'grad_history': grad_history,
            'x_history': x_history,
            'time': computation_time,
            'color': colors[i % len(colors)],
            'linestyle': linestyles[i % len(linestyles)]
        }
        
        # Print results
        print(f"{method_name} Results:")
        print(f"  Final solution (first {min(k, 10)} components): {x_opt[:min(k, 10)]}")
        print(f"  Final function value: {func.function_value(x_opt):.8f}")
        print(f"  Difference from theoretical optimum: {abs(func.function_value(x_opt) - f_theory):.2e}")
        print(f"  Final gradient norm: {np.linalg.norm(func.gradient(x_opt)):.8f}")
        print(f"  Number of iterations: {len(f_history)}")
        print(f"  Computation time: {computation_time:.4f} seconds")
    
    # Plot convergence curves
    plt.figure(figsize=(15, 5))
    
    # Relative distance to optimal point convergence
    plt.subplot(1, 3, 1)
    initial_distance = np.linalg.norm(x0 - x_theory)
    for method_name, result in results.items():
        distance_history = [np.linalg.norm(x - x_theory) for x in result['x_history']]
        relative_distance_history = [d / initial_distance for d in distance_history]
        plt.plot(relative_distance_history, 
                color=result['color'], 
                linestyle=result['linestyle'],
                label=method_name, 
                linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|x_k - x^*\| / \|x_0 - x^*\|$')
    plt.title('Relative Distance to Optimal Point')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Gradient norm convergence
    plt.subplot(1, 3, 2)
    for method_name, result in results.items():
        plt.plot(result['grad_history'], 
                color=result['color'], 
                linestyle=result['linestyle'],
                label=method_name, 
                linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f_k(x)\|$')
    plt.title('Gradient Norm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Relative performance comparison
    plt.subplot(1, 3, 3)
    for method_name, result in results.items():
        relative_error = [max(abs(f - f_theory) / max(abs(f_theory), 1e-16), 1e-16) for f in result['f_history']]
        plt.plot(relative_error, 
                color=result['color'], 
                linestyle=result['linestyle'],
                label=method_name, 
                linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(r'$|f_k(x) - f_k^*|/|f_k^*|$')
    plt.title('Relative Function Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('convergence_curves.pdf')
    
    return results

def main():
    """Main function: run multiple experiments"""
    
    print("=" * 60)
    print("Quadratic Function Family Optimization Algorithm Comparison")
    print("=" * 60)
    
    # Define optimization methods to compare
    methods = {
        'Gradient Descent': gradient_descent,
        'Nesterov Accelerated Gradient': nesterov_accelerated_gradient,
        'ODE Method (α=4.0)': create_ode_method(alpha=3.0, t0=1.0)
    }
    
    # Experiment 3: Large-scale problem
    print("\n" + "="*60)
    print("\nExperiment 3: Large-scale problem (k=800, dim=1601)")
    results3 = compare_methods(L=1.0, k=200, dim=401, learning_rate=1.0, 
                              methods=methods, max_iter=400)
    
    # Verify theoretical optimal solution
    print("\n" + "="*60)
    print("\nTheoretical Verification:")
    
    for k_val, L_val, dim_val in [(200, 1.0, 401)]:
        func_test = QuadraticFunction(L=L_val, k=k_val, n=dim_val)
        x_opt, f_opt = func_test.theoretical_optimum()
        
        print(f"\nk={k_val}, L={L_val}, dim={dim_val}:")
        print(f"  Theoretical optimal solution: {x_opt}")
        print(f"  Theoretical optimal function value: {f_opt:.8f}")
        print(f"  Verified function value: {func_test.function_value(x_opt):.8f}")
        print(f"  Verified gradient norm: {np.linalg.norm(func_test.gradient(x_opt)):.2e}")
        
        # Verify matrix A_k structure
        A_k = func_test.get_matrix_A_k()
        print(f"  Matrix A_{k_val}:\n{A_k}")

if __name__ == "__main__":
    main() 
