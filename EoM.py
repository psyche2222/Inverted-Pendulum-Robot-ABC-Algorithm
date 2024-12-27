from sympy import symbols, Function, Eq, diff, solve

# Parameter
M, m, g, l, b, I, r, T = symbols('M m g l b I r T')  # Parameter sistem
t = symbols('t')  # Waktu sebagai variabel independen

# Variabel dinamis
theta = Function('theta')(t)  # Sudut pendulum
dphi = Function('phi')(t)  # Sudut roda (turunan pertama phi terhadap waktu)

# Turunan pertama dan kedua dari theta dan phi
theta_dot = diff(theta, t)
theta_ddot = diff(theta_dot, t)
phi_dot = diff(dphi, t)
phi_ddot = diff(phi_dot, t)

# Persamaan gerak untuk pendulum (menggunakan hukum Euler dan dinamika)
pendulum_eq = Eq(I * theta_ddot, -m * g * l * theta - b * theta_dot + m * l * r * phi_ddot)

# Persamaan gerak untuk roda (menggunakan hukum Newton dan dinamika rotasi)
wheel_eq = Eq(M * r * phi_ddot, T - b * theta_dot - m * g * l * theta - m * l * theta_ddot)

# Penyelesaian simultan
solutions = solve([pendulum_eq, wheel_eq], (theta_ddot, phi_ddot))

# Output persamaan gerak
print("Percepatan sudut pendulum (ddtheta):")
print(solutions[theta_ddot])

print("\nPercepatan sudut roda (ddphi):")
print(solutions[phi_ddot])
