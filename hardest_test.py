import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class QuadraticFunction:
    """
    Matrix-free implementation of quadratic function family f_k(x)
    f_k(x) = (L/4) * [(1/2) * x^T * A_k * x - e_1^T * x]
    where A_k is a k \times k tridiagonal matrix embedded in n \times n space
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
        # Only the first k \times k block is non-zero
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
        """Compute gradient \nabla f_k(x) = (L/4) * (A_k * x - e_1)"""
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


class SmoothedSVM:
    """
    Smoothed SVM implementation with smoothed hinge loss
    
    Objective: f_D(x) = (1/|D|) * \sum_{(a_i, b_i) \in D} \ell_\gamma(b_i \langle a_i, x \rangle) + (\gamma/2)||x||^2
    
    Smoothed hinge loss \ell_\gamma(t):
    - t \geq 1: \ell_\gamma(t) = 0
    - 0 \leq t < 1: \ell_\gamma(t) = (1/2\gamma)(1-t)^2  (when \gamma > 0)
    - t < 0: \ell_\gamma(t) = 1/2 - t
    
    When \gamma = 0, this reduces to standard hinge loss:
    - t \geq 1: \ell(t) = 0
    - t < 1: \ell(t) = 1 - t
    """
    
    def __init__(self, data: List[Tuple[np.ndarray, int]], gamma: float):
        """
        Parameters:
        data: List of (feature_vector, label) pairs where label is +1 or -1
        gamma: Regularization parameter (\gamma \geq 0)
        """
        self.data = data
        self.gamma = gamma
        self.n_samples = len(data)
        self.n_features = len(data[0][0]) if data else 0
        
        if gamma < 0:
            raise ValueError("Gamma must be non-negative")
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Convert to numpy arrays for efficiency
        self.A = np.array([a for a, b in data])  # Feature matrix (n_samples \times n_features)
        self.b = np.array([b for a, b in data])  # Labels (n_samples,)
    
    def _smoothed_hinge_loss(self, t: float) -> float:
        """Compute smoothed hinge loss \ell_\gamma(t)"""
        if t >= 1:
            return 0.0
        elif self.gamma == 0:
            # Standard hinge loss when gamma = 0
            return 1.0 - t
        elif t >= 0:
            return (1.0 / (2 * self.gamma)) * (1 - t)**2
        else:
            return 0.5 - t
    
    def _smoothed_hinge_loss_derivative(self, t: float) -> float:
        """Compute derivative of smoothed hinge loss \ell'_\gamma(t)"""
        if t >= 1:
            return 0.0
        elif self.gamma == 0:
            # Standard hinge loss derivative when gamma = 0
            return -1.0 if t < 1 else 0.0
        elif t >= 0:
            return -(1.0 / self.gamma) * (1 - t)
        else:
            return -1.0
    
    def function_value(self, x: np.ndarray) -> float:
        """Compute objective function value"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute predictions: b_i * \langle a_i, x \rangle
        predictions = self.b * (self.A @ x)  # Element-wise multiplication
        
        # Compute average smoothed hinge loss
        loss_sum = sum(self._smoothed_hinge_loss(pred) for pred in predictions)
        avg_loss = loss_sum / self.n_samples
        
        # Add regularization term (only if gamma > 0)
        regularization = (self.gamma / 2) * np.dot(x, x) if self.gamma > 0 else 0.0
        
        return avg_loss + regularization
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute predictions: b_i * \langle a_i, x \rangle
        predictions = self.b * (self.A @ x)
        
        # Compute gradient of loss term
        grad_loss = np.zeros(self.n_features)
        for i in range(self.n_samples):
            loss_derivative = self._smoothed_hinge_loss_derivative(predictions[i])
            grad_loss += loss_derivative * self.b[i] * self.A[i]
        
        grad_loss /= self.n_samples
        
        # Add gradient of regularization term (only if gamma > 0)
        grad_reg = self.gamma * x if self.gamma > 0 else np.zeros(self.n_features)
        
        return grad_loss + grad_reg
    
    @staticmethod
    def generate_synthetic_data(n_samples: int, n_features: int, 
                              noise_level: float = 0.1, random_seed: int = 42) -> List[Tuple[np.ndarray, int]]:
        """
        Generate synthetic linearly separable data for testing
        
        Parameters:
        n_samples: Number of data points
        n_features: Number of features
        noise_level: Amount of noise to add to the data
        random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Generate a random separating hyperplane
        w_true = np.random.randn(n_features)
        w_true /= np.linalg.norm(w_true)
        
        # Generate random points
        X = np.random.randn(n_samples, n_features)
        
        # Determine labels based on separating hyperplane
        y = np.sign(X @ w_true + noise_level * np.random.randn(n_samples))
        
        # Ensure we have both classes
        if np.all(y == 1) or np.all(y == -1):
            # Force some points to be on the other side
            flip_indices = np.random.choice(n_samples, size=n_samples//4, replace=False)
            y[flip_indices] *= -1
        
        # Convert to list of tuples format
        data = [(X[i], int(y[i])) for i in range(n_samples)]
        
        return data


class LogisticRegression:
    """
    Logistic Regression implementation
    
    Objective: f_D(x) = (1/|D|) * \sum_{(a_i, b_i) \in D} \log(1 + \exp(-y_i \langle a_i, x \rangle))
    where y_i = 2*b_i - 1 (converting {0,1} labels to {-1,1})
    """
    
    def __init__(self, data: List[Tuple[np.ndarray, int]]):
        """
        Parameters:
        data: List of (feature_vector, label) pairs where label is 0 or 1
        """
        self.data = data
        self.n_samples = len(data)
        self.n_features = len(data[0][0]) if data else 0
        
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Convert to numpy arrays for efficiency
        self.A = np.array([a for a, b in data])  # Feature matrix (n_samples \times n_features)
        self.b = np.array([b for a, b in data])  # Labels (n_samples,) in {0,1}
        
        # Convert labels from {0,1} to {-1,1}
        self.y = 2 * self.b - 1  # Labels in {-1,1}
        
        # Estimate Lipschitz constant: L <= ||AA^T|| / |D|
        self.L = np.linalg.norm(self.A @ self.A.T) / self.n_samples
    
    def function_value(self, x: np.ndarray) -> float:
        """Compute objective function value"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute y_i * \langle a_i, x \rangle
        predictions = self.y * (self.A @ x)
        
        # Compute average logistic loss
        loss_sum = np.sum(np.log(1 + np.exp(-predictions)))
        
        return loss_sum / self.n_samples
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute y_i * \langle a_i, x \rangle
        predictions = self.y * (self.A @ x)
        
        # Compute sigmoid values
        sigmoid_vals = 1 / (1 + np.exp(predictions))  # sigma(-y_i * <a_i, x>)
        
        # Compute gradient
        grad = np.zeros(self.n_features)
        for i in range(self.n_samples):
            grad += (-self.y[i] * sigmoid_vals[i]) * self.A[i]
        
        return grad / self.n_samples
    
    @staticmethod
    def generate_synthetic_data(n_samples: int, n_features: int, 
                              noise_level: float = 0.1, random_seed: int = 42) -> List[Tuple[np.ndarray, int]]:
        """
        Generate synthetic data for logistic regression
        
        Parameters:
        n_samples: Number of data points
        n_features: Number of features
        noise_level: Amount of noise to add to the data
        random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Generate a random separating hyperplane
        w_true = np.random.randn(n_features)
        w_true /= np.linalg.norm(w_true)
        
        # Generate random points
        X = np.random.randn(n_samples, n_features)
        
        # Generate probabilities using logistic function
        logits = X @ w_true + noise_level * np.random.randn(n_samples)
        probs = 1 / (1 + np.exp(-logits))
        
        # Generate binary labels
        y = (np.random.rand(n_samples) < probs).astype(int)
        
        # Convert to list of tuples format
        data = [(X[i], int(y[i])) for i in range(n_samples)]
        
        return data


class LppMinimization:
    """
    l_p^p minimization problem
    
    Objective: f_D(x) = (1/|D|) * \sum_{(a_i, b_i) \in D} (1/p) * (\langle a_i, x \rangle - b_i)^p
    where p >= 4 is an even integer
    """
    
    def __init__(self, data: List[Tuple[np.ndarray, float]], p: int = 4):
        """
        Parameters:
        data: List of (feature_vector, target_value) pairs
        p: Even integer >= 4 for l_p^p norm
        """
        self.data = data
        self.p = p
        self.n_samples = len(data)
        self.n_features = len(data[0][0]) if data else 0
        
        if p < 4 or p % 2 != 0:
            raise ValueError("p must be an even integer >= 4")
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Convert to numpy arrays for efficiency
        self.A = np.array([a for a, b in data])  # Feature matrix (n_samples \times n_features)
        self.b = np.array([b for a, b in data])  # Target values (n_samples,)
    
    def function_value(self, x: np.ndarray) -> float:
        """Compute objective function value"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute residuals: \langle a_i, x \rangle - b_i
        residuals = self.A @ x - self.b
        
        # Compute l_p^p loss
        loss_sum = np.sum(np.abs(residuals) ** self.p) / self.p
        
        return loss_sum / self.n_samples
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function"""
        if len(x) != self.n_features:
            raise ValueError(f"Vector x must have dimension {self.n_features}")
        
        # Compute residuals: \langle a_i, x \rangle - b_i
        residuals = self.A @ x - self.b
        
        # Compute gradient: (1/|D|) * \sum_i (residual_i)^{p-1} * sign(residual_i) * a_i
        grad = np.zeros(self.n_features)
        for i in range(self.n_samples):
            if residuals[i] != 0:  # Avoid division by zero
                sign_residual = np.sign(residuals[i])
                grad += (np.abs(residuals[i]) ** (self.p - 1)) * sign_residual * self.A[i]
        
        return grad / self.n_samples
    
    @staticmethod
    def generate_synthetic_data(n_samples: int, n_features: int, 
                              noise_level: float = 0.1, random_seed: int = 42) -> List[Tuple[np.ndarray, float]]:
        """
        Generate synthetic regression data
        
        Parameters:
        n_samples: Number of data points
        n_features: Number of features
        noise_level: Amount of noise to add to the targets
        random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Generate a random weight vector
        w_true = np.random.randn(n_features)
        
        # Generate random points
        X = np.random.randn(n_samples, n_features)
        
        # Generate target values with noise
        y = X @ w_true + noise_level * np.random.randn(n_samples)
        
        # Convert to list of tuples format
        data = [(X[i], float(y[i])) for i in range(n_samples)]
        
        return data

def gradient_descent(func, x0: np.ndarray, 
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

def nesterov_accelerated_gradient(func, x0: np.ndarray,
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

def explicit_euler_ode_method(func, x0: np.ndarray,
                            learning_rate: float, max_iter: int = 1000,
                            tol: float = 1e-8, alpha: float = 4.0, 
                            t0: float = 1.0, h: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Explicit Euler scheme for ODE-based optimization
    
    The scheme:
    (x_{k+1} - x_k)/h = v_k - \beta(t_k)\nabla f(x_k)
    (v_{k+1} - v_k)/h = -(\alpha/t_k)(v_k - \beta(t_k)\nabla f(x_k)) + (\beta(t_k) - \gamma(t_k))\nabla f(x_k)
    
    where:
    \alpha > 3, \beta(t) = (4-2\alpha h/t)/L, \gamma(t) = \beta(t)/h
    t_k = t_0 + kh, k \geq 0
    
    Parameters:
    alpha: parameter \alpha (should be > 3)
    t0: initial time t_0
    h: step size (defaults to 1/sqrt(L) if not specified)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # Get Lipschitz constant from function (if available) or use default
    L = getattr(func, 'L', 1.0)  # Default to 1.0 if L not available
    
    # if h is None:
    h = 1/np.sqrt(L)/1.2
    
    beta_0 = (4 - 2 * alpha * h / t0) / L
    grad_0 = func.gradient(x0)
    x = x0.copy()
    v = -beta_0 * grad_0
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for k in range(max_iter):
        # Current time
        t_k = t0 + k * h
        
        # Compute \beta(t_k) and \gamma(t_k)
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
        # (x_{k+1} - x_k)/h = v_k - \beta(t_k)\nabla f(x_k)
        x_new = x + h * (v - beta_k * grad)
        
        # (v_{k+1} - v_k)/h = -(\alpha/t_k)(v_k - \beta(t_k)\nabla f(x_k)) + (\beta(t_k) - \gamma(t_k))\nabla f(x_k)
        v_update_term1 = -(alpha / t_k) * (v - beta_k * grad)
        v_update_term2 = (diff_beta_k-gamma_k) * grad
        v_new = v + h * (v_update_term1 + v_update_term2)
        
        # Update variables
        x = x_new
        v = v_new
    
    return x, f_history, grad_norm_history, x_history

def new_ode_variant_method(func, x0: np.ndarray,
                         learning_rate: float, max_iter: int = 1000,
                         tol: float = 1e-8, omega: float = 2.0, r: float = 2.0, 
                         s: float = None, alpha: float = 4.0, t0: float = 1.0, 
                         h: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    新的ODE变体方法，使用不同的系数
    
    The scheme:
    (x_{k+1} - x_k)/h = v_k - \beta\nabla f(x_k)
    (v_{k+1} - v_k)/h = -(\alpha/t_k)(v_k - \beta\nabla f(x_k)) + (\beta - \gamma_k)\nabla f(x_k)
    
    where:
    \beta = \omega\sqrt{s} (constant)
    \gamma_k = 1 + (r+1)\omega/(k+r+1) (iteration-dependent)
    t_k = t_0 + kh, k \geq 0
    
    Parameters:
    omega: parameter \omega (default: 2.0)
    r: parameter r (default: 2.0)
    s: step size parameter (default: 1/L if L available, else 1.0)
    alpha: parameter \alpha (default: 4.0)
    t0: initial time t_0 (default: 1.0)
    h: step size (defaults to 1/sqrt(L) if not specified)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # Get Lipschitz constant from function (if available) or use default
    L = getattr(func, 'L', 1.0)  # Default to 1.0 if L not available
    
    if s is None:
        s = 1.0 / L
    if h is None:
        h = 1/np.sqrt(L)/1.2
    
    # Compute constant beta
    beta = omega * np.sqrt(s)
    
    # Initialize
    grad_0 = func.gradient(x0)
    x = x0.copy()
    v = -beta * grad_0  # Initial velocity
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for k in range(max_iter):
        # Current time
        t_k = t0 + k * h
        
        # Compute \gamma_k = 1 + (r+1)\omega/(k+r+1)
        gamma_k = 1 + (r + 1) * omega / (k + r + 1)
        
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
        # (x_{k+1} - x_k)/h = v_k - \beta\nabla f(x_k)
        x_new = x + h * (v - beta * grad)
        
        # (v_{k+1} - v_k)/h = -(\alpha/t_k)(v_k - \beta\nabla f(x_k)) + (\beta - \gamma_k)\nabla f(x_k)
        v_update_term1 = -(alpha / t_k) * (v - beta * grad)
        v_update_term2 = (beta - gamma_k) * grad
        v_new = v + h * (v_update_term1 + v_update_term2)
        
        # Update variables
        x = x_new
        v = v_new
    
    return x, f_history, grad_norm_history, x_history

def igahd_method(func, x0: np.ndarray,
                learning_rate: float, max_iter: int = 1000,
                tol: float = 1e-8, alpha: float = 3.0, beta: float = 2.0, 
                s: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    IGAHD (Inertial Gradient Algorithm with Hessian Damping) method
    
    Algorithm:
    Initialization: x_0, x_1 given
    Set α_k := 1 - α/k
    for k = 1, ... do:
        y_k = x_k + α_k(x_k - x_{k-1}) - β√s(∇f(x_k) - ∇f(x_{k-1})) - (β√s/k)∇f(x_{k-1})
        x_{k+1} = y_k - s∇f(y_k)
    
    Parameters:
    alpha: parameter α (controls inertial term decay, default: 3.0)
    beta: parameter β (controls Hessian damping, default: 2.0)
    s: step size parameter (default: 1/L if L available, else 1.0)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # Get Lipschitz constant from function (if available) or use default
    L = getattr(func, 'L', 1.0)  # Default to 1.0 if L not available
    if s is None:
        s = 1.0 / L
    
    sqrt_s = np.sqrt(s)
    
    # Initialize x_0 and x_1
    # x_0 is given as x0
    # x_1 = x_0 - s * ∇f(x_0) (simple gradient step for initialization)
    x_prev = x0.copy()  # x_0
    grad_prev = func.gradient(x_prev)
    x_curr = x_prev - s * grad_prev  # x_1
    
    f_history = []
    grad_norm_history = []
    x_history = []
    
    # Record initial point
    f_val = func.function_value(x_prev)
    grad_norm = np.linalg.norm(grad_prev)
    f_history.append(f_val)
    grad_norm_history.append(grad_norm)
    x_history.append(x_prev.copy())
    
    for k in range(1, max_iter):
        # Current gradient
        grad_curr = func.gradient(x_curr)
        
        # α_k = 1 - α/k
        alpha_k = 1 - alpha / k if k > alpha else 0  # Ensure α_k >= 0
        
        # y_k = x_k + α_k(x_k - x_{k-1}) - β√s(∇f(x_k) - ∇f(x_{k-1})) - (β√s/k)∇f(x_{k-1})
        inertial_term = alpha_k * (x_curr - x_prev)
        hessian_damping_term = -beta * sqrt_s * (grad_curr - grad_prev)
        additional_gradient_term = -(beta * sqrt_s / k) * grad_prev
        
        y_k = x_curr + inertial_term + hessian_damping_term + additional_gradient_term
        
        # x_{k+1} = y_k - s∇f(y_k)
        grad_y = func.gradient(y_k)
        x_new = y_k - s * grad_y
        
        # Compute function value and gradient norm at current point
        f_val = func.function_value(x_curr)
        grad_norm = np.linalg.norm(grad_curr)
        
        # Store history
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(x_curr.copy())
        
        if grad_norm < tol:
            break
        
        # Update for next iteration
        x_prev = x_curr.copy()
        grad_prev = grad_curr.copy()
        x_curr = x_new.copy()
    
    # Add final point
    f_val = func.function_value(x_curr)
    grad_norm = np.linalg.norm(func.gradient(x_curr))
    f_history.append(f_val)
    grad_norm_history.append(grad_norm)
    x_history.append(x_curr.copy())
    
    return x_curr, f_history, grad_norm_history, x_history

def create_ode_method(alpha: float = 4.0, t0: float = 1.0, h: float = None):
    """Create ODE method with specific parameters"""
    def ode_method(func, x0, learning_rate, max_iter, tol=1e-8):
        return explicit_euler_ode_method(func, x0, learning_rate, max_iter, 
                                       tol, alpha, t0, h)
    return ode_method

def create_new_ode_variant_method(omega: float = 2.0, r: float = 2.0, s: float = None, 
                                 alpha: float = 4.0, t0: float = 1.0, h: float = None):
    """Create new ODE variant method with specific parameters"""
    def new_ode_variant(func, x0, learning_rate, max_iter, tol=1e-8):
        return new_ode_variant_method(func, x0, learning_rate, max_iter, 
                                    tol, omega, r, s, alpha, t0, h)
    return new_ode_variant

def create_igahd_method(alpha: float = 3.0, beta: float = 2.0, s: float = None):
    """Create IGAHD method with specific parameters"""
    def igahd(func, x0, learning_rate, max_iter, tol=1e-8):
        return igahd_method(func, x0, learning_rate, max_iter, 
                          tol, alpha, beta, s)
    return igahd

def phase_space_method_1(func, x0: np.ndarray,
                        learning_rate: float, max_iter: int = 1000,
                        tol: float = 1e-8, omega: float = 2.0, r: float = 2.0, 
                        s: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Phase-space method 1 implementation
    
    Algorithm:
    y_k - y_{k-1} = sqrt(s) * v_{k-1}
    v_k - v_{k-1} = -(r+1)/(k+r+1) * v_{k-1} - omega*sqrt(s)*(∇f(y_k) - ∇f(y_{k-1})) 
                    - (1 + omega*(r+1)/(k+r+1))*sqrt(s)*∇f(y_{k-1})
    
    Parameters:
    omega: parameter omega (default: 2.0)
    r: parameter r (default: 2.0) 
    s: step size parameter (default: 1/L if L available, else 1.0)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # Get Lipschitz constant from function (if available) or use default
    L = getattr(func, 'L', 1.0)  # Default to 1.0 if L not available
    if s is None:
        s = 1.0 / L
    
    sqrt_s = np.sqrt(s)
    
    # Initialize
    y = x0.copy()
    grad_y = func.gradient(y)
    v = -sqrt_s * grad_y  # v_0 = -sqrt(s) * ∇f(y_0)
    
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for k in range(1, max_iter + 1):
        # Store current values
        y_prev = y.copy()
        v_prev = v.copy()
        grad_y_prev = grad_y.copy()
        
        # Update y: y_k = y_{k-1} + sqrt(s) * v_{k-1}
        y = y_prev + sqrt_s * v_prev
        
        # Compute new gradient
        grad_y = func.gradient(y)
        
        # Update v: v_k = v_{k-1} - (r+1)/(k+r+1) * v_{k-1} 
        #                         - omega*sqrt(s)*(∇f(y_k) - ∇f(y_{k-1}))
        #                         - (1 + omega*(r+1)/(k+r+1))*sqrt(s)*∇f(y_{k-1})
        momentum_coeff = (r + 1) / (k + r + 1)
        gradient_diff = grad_y - grad_y_prev
        
        v = (v_prev 
             - momentum_coeff * v_prev
             - omega * sqrt_s * gradient_diff
             - (1 + omega * momentum_coeff) * sqrt_s * grad_y_prev)
        
        # Compute function value and gradient norm
        f_val = func.function_value(y)
        grad_norm = np.linalg.norm(grad_y)
        
        # Store history
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(y.copy())
        
        if grad_norm < tol:
            break
    
    return y, f_history, grad_norm_history, x_history

def phase_space_method_2(func, x0: np.ndarray,
                        learning_rate: float, max_iter: int = 1000,
                        tol: float = 1e-8, r: float = 2.0, 
                        s: float = None) -> Tuple[np.ndarray, List[float], List[float], List[np.ndarray]]:
    """
    Phase-space method 2 implementation
    
    Algorithm:
    y_k - y_{k-1} = sqrt(s) * v_{k-1}
    v_k - v_{k-1} = -(r+1)/k * v_k - (1 + (r+1)/k)*sqrt(s)*∇f(y_k) 
                    - sqrt(s)*(∇f(y_k) - ∇f(y_{k-1}))
    
    Note: This is an implicit equation in v_k. We need to solve for v_k.
    Rearranging: v_k + (r+1)/k * v_k = v_{k-1} - (1 + (r+1)/k)*sqrt(s)*∇f(y_k) 
                                                - sqrt(s)*(∇f(y_k) - ∇f(y_{k-1}))
    So: v_k = [v_{k-1} - (1 + (r+1)/k)*sqrt(s)*∇f(y_k) - sqrt(s)*(∇f(y_k) - ∇f(y_{k-1}))] / [1 + (r+1)/k]
    
    Parameters:
    r: parameter r (default: 2.0)
    s: step size parameter (default: 1/L if L available, else 1.0)
    
    Returns: (optimal solution, function value history, gradient norm history, point history)
    """
    # Get Lipschitz constant from function (if available) or use default
    L = getattr(func, 'L', 1.0)  # Default to 1.0 if L not available
    if s is None:
        s = 1.0 / L
    
    sqrt_s = np.sqrt(s)
    
    # Initialize
    y = x0.copy()
    grad_y = func.gradient(y)
    v = -sqrt_s * grad_y  # v_0 = -sqrt(s) * ∇f(y_0)
    
    f_history = []
    grad_norm_history = []
    x_history = []
    
    for k in range(1, max_iter + 1):
        # Store current values
        y_prev = y.copy()
        v_prev = v.copy()
        grad_y_prev = grad_y.copy()
        
        # Update y: y_k = y_{k-1} + sqrt(s) * v_{k-1}
        y = y_prev + sqrt_s * v_prev
        
        # Compute new gradient
        grad_y = func.gradient(y)
        
        # Update v (solving the implicit equation):
        # v_k + (r+1)/k * v_k = v_{k-1} - (1 + (r+1)/k)*sqrt(s)*∇f(y_k) 
        #                                - sqrt(s)*(∇f(y_k) - ∇f(y_{k-1}))
        momentum_coeff = (r + 1) / k
        gradient_diff = grad_y - grad_y_prev
        
        rhs = (v_prev 
               - (1 + momentum_coeff) * sqrt_s * grad_y
               - sqrt_s * gradient_diff)
        
        v = rhs / (1 + momentum_coeff)
        
        # Compute function value and gradient norm
        f_val = func.function_value(y)
        grad_norm = np.linalg.norm(grad_y)
        
        # Store history
        f_history.append(f_val)
        grad_norm_history.append(grad_norm)
        x_history.append(y.copy())
        
        if grad_norm < tol:
            break
    
    return y, f_history, grad_norm_history, x_history

def create_phase_space_method_1(omega: float = 2.0, r: float = 2.0, s: float = None):
    """Create phase-space method 1 with specific parameters"""
    def phase_space_1(func, x0, learning_rate, max_iter, tol=1e-8):
        return phase_space_method_1(func, x0, learning_rate, max_iter, 
                                   tol, omega, r, s)
    return phase_space_1

def create_phase_space_method_2(r: float = 2.0, s: float = None):
    """Create phase-space method 2 with specific parameters"""
    def phase_space_2(func, x0, learning_rate, max_iter, tol=1e-8):
        return phase_space_method_2(func, x0, learning_rate, max_iter, 
                                   tol, r, s)
    return phase_space_2

def compare_methods(func, learning_rate: float, 
                   methods: dict, max_iter: int = 1000, random_seed: int = 42, 
                   title: str = "Optimization Comparison"):
    """
    Compare performance of multiple optimization methods
    
    Parameters:
    func: Function object with function_value and gradient methods
    methods: dict with method_name: method_function pairs
             Each method_function should have signature (func, x0, learning_rate, max_iter, tol)
             and return (solution, f_history, grad_norm_history, x_history)
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get dimension from function
    if hasattr(func, 'n'):
        dim = func.n
    elif hasattr(func, 'n_features'):
        dim = func.n_features
    else:
        raise ValueError("Function must have dimension attribute 'n' or 'n_features'")
    
    # Get theoretical optimal solution if available
    x_theory, f_theory = None, None
    if hasattr(func, 'theoretical_optimum'):
        x_theory, f_theory = func.theoretical_optimum()
    
    # Random initialization
    x0 = np.random.randn(dim)
    
    print(f"{title}")
    print(f"Function type: {type(func).__name__}")
    print(f"Dimension: {dim}")
    if x_theory is not None:
        print(f"Theoretical optimal solution (first {min(10, len(x_theory))} components): {x_theory[:min(10, len(x_theory))]}")
        print(f"Theoretical optimal function value: {f_theory:.8f}")
    print(f"Initial point (first 5 components): {x0[:5]}")
    print(f"Initial function value: {func.function_value(x0):.6f}")
    print("-" * 50)
    
    # Run all methods and collect results
    results = {}
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown', 'gray', 'pink', 'lime']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '--']
    
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
        print(f"  Final solution (first {min(10, len(x_opt))} components): {x_opt[:min(10, len(x_opt))]}")
        print(f"  Final function value: {func.function_value(x_opt):.8f}")
        if f_theory is not None:
            print(f"  Difference from theoretical optimum: {abs(func.function_value(x_opt) - f_theory):.2e}")
        print(f"  Final gradient norm: {np.linalg.norm(func.gradient(x_opt)):.8f}")
        print(f"  Number of iterations: {len(f_history)}")
        print(f"  Computation time: {computation_time:.4f} seconds")
    
    # Plot convergence curves
    plt.figure(figsize=(15, 5))
    
    # Relative distance to optimal point convergence (if theoretical optimum available)
    if x_theory is not None:
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
        
        subplot_idx = 2
    else:
        subplot_idx = 1
    
    # Gradient norm convergence
    n_plots = 3 if x_theory is not None else 2
    plt.subplot(1, n_plots, subplot_idx)
    for method_name, result in results.items():
        plt.plot(result['grad_history'], 
                color=result['color'], 
                linestyle=result['linestyle'],
                label=method_name, 
                linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f(x)\|$')
    plt.title('Gradient Norm Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Relative performance comparison (if theoretical optimum available)
    if x_theory is not None:
        plt.subplot(1, n_plots, subplot_idx + 1)
        for method_name, result in results.items():
            relative_error = [max(abs(f - f_theory) / max(abs(f_theory), 1e-16), 1e-16) for f in result['f_history']]
            plt.plot(relative_error, 
                    color=result['color'], 
                    linestyle=result['linestyle'],
                    label=method_name, 
                    linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel(r'$|f(x) - f^*|/|f^*|$')
        plt.title('Relative Function Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'convergence_curves_{type(func).__name__.lower()}.pdf')
    
    return results

def create_unified_convergence_plot(all_experiments: dict, methods: dict):
    """
    Create a unified convergence plot showing function value vs iteration 
    for all test problems in one figure
    """
    print("\n" + "="*60)
    print("Creating Unified Convergence Plot for All Test Problems")
    print("="*60)
    
    # Define colors for different test problems
    problem_colors = {
        'Quadratic Function': 'blue',
        'Smoothed SVM': 'red', 
        'Logistic Regression': 'green',
        'LppMinimization (p=4)': 'purple'
    }
    
    # Define line styles for different methods
    method_linestyles = {
        'Gradient Descent': '-',
        'Nesterov Accelerated Gradient': '--',
        'ODE Method (alpha=3.0)': '-.',
        'New ODE Variant (ω=0.5, r=2.0)': ':',
        'IGAHD (α=3.0, β=2.0)': '-',
        'Phase-Space Method 1': '--',
        'Phase-Space Method 2': '-.'
    }
    
    # Create a unified plot showing all methods for each problem
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Method colors
    method_colors = {
        'Gradient Descent': 'blue',
        'Nesterov Accelerated Gradient': 'red',
        'ODE Method (alpha=3.0)': 'green', 
        'New ODE Variant (ω=0.5, r=2.0)': 'magenta',
        'IGAHD (α=3.0, β=2.0)': 'cyan',
        'Phase-Space Method 1': 'orange',
        'Phase-Space Method 2': 'purple'
    }
    
    for i, (problem_name, (func, results)) in enumerate(all_experiments.items()):
        ax = axes[i]
        
        for method_name, method_result in results.items():
            f_history = method_result['f_history']
            color = method_colors.get(method_name, 'black')
            linestyle = method_linestyles.get(method_name, '-')
            
            # Plot function values
            ax.plot(f_history, color=color, linestyle=linestyle,
                   label=method_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Function Value')
        ax.set_title(f'{problem_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('unified_convergence_all_problems.pdf', dpi=300, bbox_inches='tight')
    print("Unified convergence plot saved as 'unified_convergence_all_problems.pdf'")
    
    # Create a second figure showing all methods for each problem
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (problem_name, (func, results)) in enumerate(all_experiments.items()):
        ax = axes[i]
        
        for method_name, method_result in results.items():
            f_history = method_result['f_history']
            color = method_result['color']
            linestyle = method_result['linestyle']
            
            # Plot absolute function values
            ax.plot(f_history, color=color, linestyle=linestyle,
                   label=method_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Function Value')
        ax.set_title(f'{problem_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('all_problems_convergence_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Detailed convergence comparison saved as 'all_problems_convergence_comparison.pdf'")
    
    # Summary table
    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY FOR ALL TEST PROBLEMS")
    print("="*80)
    
    for problem_name, (func, results) in all_experiments.items():
        print(f"\n{problem_name}:")
        print("-" * len(problem_name))
        
        # Get theoretical optimum if available
        f_theory = None
        if hasattr(func, 'theoretical_optimum'):
            _, f_theory = func.theoretical_optimum()
        
        for method_name, method_result in results.items():
            final_f = method_result['f_history'][-1]
            final_grad_norm = method_result['grad_history'][-1]
            n_iters = len(method_result['f_history'])
            comp_time = method_result['time']
            
            print(f"  {method_name}:")
            print(f"    Final function value: {final_f:.6e}")
            if f_theory is not None:
                print(f"    Distance to theoretical optimum: {abs(final_f - f_theory):.2e}")
            print(f"    Final gradient norm: {final_grad_norm:.6e}")
            print(f"    Iterations: {n_iters}")
            print(f"    Time: {comp_time:.4f}s")

def create_specific_convergence_plots(all_experiments: dict):
    """
    Create two specific plots:
    1. Gradient norm vs iteration
    2. Relative function value vs iteration (relative to 1000-step Nesterov result)
    """
    print("\n" + "="*60)
    print("Creating Specific Convergence Plots")
    print("="*60)
    
    # Define method colors and styles
    method_colors = {
        'Gradient Descent': 'blue',
        'Nesterov Accelerated Gradient': 'red',
        'ODE Method (alpha=3.0)': 'green', 
        'New ODE Variant (ω=0.5, r=2.0)': 'magenta',
        'IGAHD (α=3.0, β=2.0)': 'cyan',
        'Phase-Space Method 1': 'orange',
        'Phase-Space Method 2': 'purple'
    }
    
    method_linestyles = {
        'Gradient Descent': '-',
        'Nesterov Accelerated Gradient': '--',
        'ODE Method (alpha=3.0)': '-.',
        'New ODE Variant (ω=0.5, r=2.0)': ':',
        'IGAHD (α=3.0, β=2.0)': '-',
        'Phase-Space Method 1': '--',
        'Phase-Space Method 2': '-.'
    }
    
    # Create Figure 1: Gradient norm vs iteration
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    axes1 = axes1.flatten()
    
    for i, (problem_name, (func, results)) in enumerate(all_experiments.items()):
        ax = axes1[i]
        
        for method_name, method_result in results.items():
            grad_history = method_result['grad_history']
            color = method_colors.get(method_name, 'black')
            linestyle = method_linestyles.get(method_name, '-')
            
            # Plot gradient norm
            ax.plot(grad_history, color=color, linestyle=linestyle,
                   label=method_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(r'$\|\nabla f(x_k)\|$', fontsize=12)
        ax.set_title(f'{problem_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('gradient_norm_convergence.pdf', dpi=300, bbox_inches='tight')
    print("Gradient norm convergence plot saved as 'gradient_norm_convergence.pdf'")
    
    # Create Figure 2: Relative function value vs iteration
    # First, get reference values by running 1000-step Nesterov for each problem
    print("Computing reference values using 1000-step Nesterov method...")
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    for i, (problem_name, (func, results)) in enumerate(all_experiments.items()):
        ax = axes2[i]
        
        # Get initial point (same seed as in main experiments)
        np.random.seed(42)
        if hasattr(func, 'n'):
            dim = func.n
        elif hasattr(func, 'n_features'):
            dim = func.n_features
        x0 = np.random.randn(dim)
        
        # Run 1000-step Nesterov to get reference value f*
        _, f_ref_history, _, _ = nesterov_accelerated_gradient(func, x0, 1.0 if problem_name == 'Quadratic Function' else 0.1, 1000, 1e-12)
        f_star = min(f_ref_history)  # Best value achieved by Nesterov
        print(f"  {problem_name}: f* = {f_star:.8e}")
        
        for method_name, method_result in results.items():
            f_history = method_result['f_history']
            color = method_colors.get(method_name, 'black')
            linestyle = method_linestyles.get(method_name, '-')
            
            # Compute relative function value: (f - f*) / (1 + |f*|)
            relative_f = [(f - f_star) / (1 + abs(f_star)) for f in f_history]
            
            # Plot relative function value (only positive values for log scale)
            relative_f_positive = [max(rf, 1e-16) for rf in relative_f]  # Ensure positive for log scale
            
            ax.plot(relative_f_positive, color=color, linestyle=linestyle,
                   label=method_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(r'$(f(x_k) - f^*) / (1 + |f^*|)$', fontsize=12)
        ax.set_title(f'{problem_name}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('relative_function_value_convergence.pdf', dpi=300, bbox_inches='tight')
    print("Relative function value convergence plot saved as 'relative_function_value_convergence.pdf'")
    
    return fig1, fig2

def main():
    """Main function: run multiple experiments"""
    
    print("=" * 60)
    print("Optimization Algorithm Comparison")
    print("=" * 60)
    
    # Define optimization methods to compare
    methods = {
        'Gradient Descent': gradient_descent,
        'Nesterov Accelerated Gradient': nesterov_accelerated_gradient,
        'ODE Method (alpha=3.0)': create_ode_method(alpha=3.0, t0=1.0),
        'New ODE Variant (ω=0.5, r=2.0)': create_new_ode_variant_method(omega=0.5, r=2.0),
        'IGAHD (α=3.0, β=2.0)': create_igahd_method(alpha=3.0, beta=2.0),
        'Phase-Space Method 1': create_phase_space_method_1(omega=2.0, r=2.0),
        'Phase-Space Method 2': create_phase_space_method_2(r=2.0)
    }
    
    # Store all experiment results for unified visualization
    all_experiments = {}
    
    # Experiment 1: Quadratic Function Family
    print("\n" + "="*60)
    print("Experiment 1: Quadratic Function Family (k=200, dim=401)")
    func1 = QuadraticFunction(L=1.0, k=200, n=401)
    results1 = compare_methods(func1, learning_rate=1.0, methods=methods, max_iter=400,
                              title="Quadratic Function Family Optimization")
    all_experiments['Quadratic Function'] = (func1, results1)
    
    # Experiment 2: Smoothed SVM
    print("\n" + "="*60)
    print("Experiment 2: Smoothed SVM")
    
    # Generate synthetic data for SVM
    n_samples, n_features = 100, 20
    gamma = 0.1
    svm_data = SmoothedSVM.generate_synthetic_data(n_samples, n_features, 
                                                  noise_level=0.1, random_seed=42)
    func2 = SmoothedSVM(svm_data, gamma)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Regularization parameter gamma = {gamma}")
    
    results2 = compare_methods(func2, learning_rate=0.1, methods=methods, max_iter=500,
                              title="Smoothed SVM Optimization")
    all_experiments['Smoothed SVM'] = (func2, results2)
    
    # Experiment 3: Logistic Regression
    print("\n" + "="*60)
    print("Experiment 3: Logistic Regression")
    
    # Generate synthetic data for Logistic Regression
    n_samples, n_features = 100, 20
    noise_level = 0.1
    lr_data = LogisticRegression.generate_synthetic_data(n_samples, n_features, 
                                                       noise_level=noise_level, random_seed=42)
    func3 = LogisticRegression(lr_data)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Noise level for logistic regression: {noise_level}")
    
    results3 = compare_methods(func3, learning_rate=0.1, methods=methods, max_iter=500,
                              title="Logistic Regression Optimization")
    all_experiments['Logistic Regression'] = (func3, results3)
    
    # Experiment 4: LppMinimization
    print("\n" + "="*60)
    print("Experiment 4: LppMinimization (p=4)")
    
    # Generate synthetic data for LppMinimization
    n_samples, n_features = 100, 20
    noise_level = 0.1
    lpp_data = LppMinimization.generate_synthetic_data(n_samples, n_features, 
                                                       noise_level=noise_level, random_seed=42)
    func4 = LppMinimization(lpp_data, p=4)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Noise level for LppMinimization: {noise_level}")
    
    results4 = compare_methods(func4, learning_rate=0.1, methods=methods, max_iter=500,
                              title="LppMinimization Optimization")
    all_experiments['LppMinimization (p=4)'] = (func4, results4)
    
    # Create unified visualization for all four test problems
    create_unified_convergence_plot(all_experiments, methods)
    
    # Create the two specific plots requested by user
    create_specific_convergence_plots(all_experiments)
    
    # Verify theoretical optimal solution for Quadratic Function
    print("\n" + "="*60)
    print("Theoretical Verification for Quadratic Function:")
    
    func_test = QuadraticFunction(L=1.0, k=200, n=401)
    x_opt, f_opt = func_test.theoretical_optimum()
    
    print(f"k=200, L=1.0, dim=401:")
    print(f"  Theoretical optimal solution (first 10 components): {x_opt[:10]}")
    print(f"  Theoretical optimal function value: {f_opt:.8f}")
    print(f"  Verified function value: {func_test.function_value(x_opt):.8f}")
    print(f"  Verified gradient norm: {np.linalg.norm(func_test.gradient(x_opt)):.2e}")
    
    # Verify SVM implementation
    print("\n" + "="*60)
    print("Smoothed SVM Verification:")
    
    # Simple test case with known data
    simple_data = [
        (np.array([1.0, 0.0]), 1),
        (np.array([-1.0, 0.0]), -1),
        (np.array([0.0, 1.0]), 1),
        (np.array([0.0, -1.0]), -1)
    ]
    simple_svm = SmoothedSVM(simple_data, gamma=0.1)
    
    test_point = np.zeros(2)
    print(f"Test point: {test_point}")
    print(f"Function value at origin: {simple_svm.function_value(test_point):.6f}")
    print(f"Gradient at origin: {simple_svm.gradient(test_point)}")
    print(f"Gradient norm at origin: {np.linalg.norm(simple_svm.gradient(test_point)):.6f}")

if __name__ == "__main__":
    main() 
