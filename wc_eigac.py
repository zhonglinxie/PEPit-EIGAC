from math import sqrt
import pickle
import os

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction, SmoothStronglyConvexFunction


def wc_optimized_gradient(L, n, wrapper="cvxpy", solver=None, verbose=1):
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the optimized gradient method (OGM) method
    theta_new = 1
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        theta_old = theta_new
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new ** 2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new ** 2 + 1)) / 2

        y = x_new + (theta_old - 1) / theta_new * (x_new - x_old) + theta_old / theta_new * (x_new - y)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(y) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * theta_new ** 2)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of optimized gradient method ***')
        print('\tPEPit guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


def wc_accelerated_gradient_convex(mu, L, n, wrapper="cvxpy", solver=None, verbose=1):
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the fast gradient method
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
        y = x_new + i / (i + 3) * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x_new) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (n ** 2 + 5 * n + 6)  # tight only for mu=0, see [2], Table 1 (column 1, line 1)
    if mu != 0:
        print('Warning: momentum is tuned for non-strongly convex functions.')

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of accelerated gradient method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


def wc_eigac_convex_modified(L, n, alpha=3.0, t0=1.0, wrapper="cvxpy", solver=None, verbose=1):
    from PEPit import PEP
    from PEPit.functions import SmoothConvexFunction

    # Step size
    s = 2/3*1/L
    h = sqrt(s)
    t0 = (alpha+12)*h

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Define optimal point and function value
    xs = func.stationary_point()
    fs = func(xs)

    # Initial conditions
    x0 = problem.set_initial_point()
    # v0 = 2.5/ h /L * func.gradient(x0)
    v0 = x0 - x0
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Initialize
    x_k = x0
    v_k = v0

    for k in range(n):
        t_k = t0 + k * h
        beta_k = 2.5*h
        dbeta_k = alpha / t_k ** 2  # derivative of beta(t)
        gamma_k = 1 + alpha * beta_k / t_k

        grad_fk = func.gradient(x_k)

        # Update x_{k+1}
        x_k1 = x_k + h * (v_k - beta_k * grad_fk)

        # Update v_{k+1}
        v_k1 = v_k - h * (alpha / t_k) * (
            v_k - beta_k * grad_fk) - h * gamma_k * grad_fk

        # Prepare for next iteration
        x_k = x_k1
        v_k = v_k1

    # Set the performance metric
    problem.set_performance_metric(func(x_k + h * (v_k - beta_k * func.gradient(x_k))) - fs)

    # Solve the PEP
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=max(verbose, 0))

    # Compute theoretical guarantee (for comparison)
    gamma = 1 / L
    theoretical_agd = 2 * L / (n ** 2 + 5 * n + 6)
    theoretical_gd = L / (2 * (2 * n * L * gamma + 1))


    # No known theoretical bound given, but we print the result
    if verbose != -1:
        print('*** Modified Euler Scheme (from paper) ***')
        if pepit_tau is not None:
            print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        else:
            print('\tPEPit guarantee:\t 问题无界 (unbounded)')

    return pepit_tau, theoretical_agd, theoretical_gd



if __name__ == "__main__":
  import numpy as np
  import matplotlib.pyplot as plt
  from PEPit.examples.unconstrained_convex_minimization import wc_gradient_descent

  # Set the parameters
  L = 5          # smoothness parameter
  mu = 0       # strong convexity parameter
  gamma = 1 / L  # step-size

  # 缓存文件名，包含参数信息以确保缓存正确性
  cache_filename = f"eigac_cache_L{L}_mu{mu}.pkl"
  
  # 尝试加载缓存的结果
  cached_results = {}
  if os.path.exists(cache_filename):
      try:
          with open(cache_filename, 'rb') as f:
              cached_results = pickle.load(f)
          print(f"已加载缓存文件: {cache_filename}")
          print(f"缓存中包含 {len(cached_results)} 个已计算结果")
      except Exception as e:
          print(f"加载缓存文件失败: {e}")
          cached_results = {}

  # Set a list of iteration counter to test
  n_list = np.array([1, 2, 4, 6, 8, 10, 20, 30, 40, 50])
  # , 80, 90, 120, 150, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300, 400, 500, 750, 1000
  
  # 检查哪些n值需要计算
  n_to_compute = []
  for n in n_list:
      if n not in cached_results:
          n_to_compute.append(n)
      else:
          print(f"n={n} 已在缓存中，跳过计算")
  
  print(f"需要计算的n值: {n_to_compute}")

  # Compute numerical and theoretical (analytical) worst-case guarantees for each iteration count
  pepit_taus_ogm = list()
  theoretical_taus_ogm = list()
  pepit_taus_agm = list()
  theoretical_taus_agm = list()
  pepit_taus_gd = list()
  theoretical_taus_gd = list()
  pepit_taus_eigac = list()
  theoretical_taus_agd = list()
  theoretical_taus_gd = list()

  # 只计算未缓存的结果
  for n in n_to_compute:
      print(f"正在计算 n={n}...")
      
      # Explicit Inertial Gradient Algorithm with Correction (EIGAC)
      pepit_tau_eigac, theoretical_agd, theoretical_gd = wc_eigac_convex_modified(L=L, n=n, verbose=1, wrapper="mosek", 
        solver={
            'MSK_IPAR_NUM_THREADS': 0,  # 0 表示使用所有可用核心
            'MSK_IPAR_INTPNT_MULTI_THREAD': 1,  # 启用内点法多线程
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-5,  # 可选：调整精度
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
        })
    #   pepit_tau_eigac, theoretical_agd, theoretical_gd = wc_eigac_convex_modified(L=L, n=n, verbose=1, wrapper="mosek")
      
      # 将结果存储到缓存中
      cached_results[n] = {
          'pepit_tau_eigac': pepit_tau_eigac,
          'theoretical_agd': theoretical_agd,
          'theoretical_gd': theoretical_gd
      }
      
      # 如果PEPit结果为None，打印警告
      if pepit_tau_eigac is None:
          print(f"警告: n={n} 的PEPit求解失败，结果为None")
      
      # 每计算完一个结果就保存缓存（防止中途中断丢失结果）
      try:
          with open(cache_filename, 'wb') as f:
              pickle.dump(cached_results, f)
          print(f"已保存 n={n} 的计算结果到缓存")
      except Exception as e:
          print(f"保存缓存失败: {e}")

  # 从缓存中提取所有结果
  for n in n_list:
      if n in cached_results:
          result = cached_results[n]
          # 检查pepit_tau_eigac是否为None
          if result['pepit_tau_eigac'] is not None:
              pepit_taus_eigac.append(result['pepit_tau_eigac'])
          else:
              print(f"警告: n={n} 的PEPit结果为None，跳过绘图")
              continue
          theoretical_taus_agd.append(result['theoretical_agd'])
          theoretical_taus_gd.append(result['theoretical_gd'])

  print(f"所有计算完成！总共有 {len(pepit_taus_eigac)} 个结果")

  # Plot theoretical and PEPit (numerical) worst-case performance bounds as functions of the iteration count

  plt.figure(figsize=(10, 6))

#   plt.plot(n_list, theoretical_taus_ogm, '--', label='Theoretical tight bound (OGM)')
#   plt.plot(n_list, pepit_taus_ogm, 'x', label='PEPit worst-case bound (OGM)')

  # plt.plot(n_list, theoretical_taus_ogm, '--', label='Theoretical tight bound (OGM)')
  plt.plot(n_list, pepit_taus_eigac, '-o', label='PEPit worst-case bound (EIGAC)')

  plt.plot(n_list, theoretical_taus_agd, '--', label='Theoretical tight bound (AGD)')
#   plt.plot(n_list, pepit_taus_agm, 'o', label='PEPit worst-case bound (AGM)')

  plt.plot(n_list, theoretical_taus_gd, '--', label='Theoretical tight bound (GD)')
#   plt.plot(n_list, pepit_taus_gd, '*', label='PEPit worst-case bound (GD)')

  plt.loglog()
  plt.legend()
  plt.xlabel('Interation count n')
  plt.ylabel('Worst-case guarantee')
  plt.title('Worst-case performance of Optimization Methods')
  plt.grid(True)
  plt.savefig(f"eigac_cache_L{L}_mu{mu}.pdf")
#   plt.show()