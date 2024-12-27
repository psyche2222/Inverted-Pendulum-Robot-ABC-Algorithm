import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Inverted Pendulum Robot Class
class InvertedPendulumRobot:
    def __init__(self):
        # Parameter sistem
        self.g = 9.81  # gravitasi
        self.M = 0.5   # massa robot
        self.m = 0.2   # massa pendulum
        self.l = 0.3   # panjang pendulum
        self.b = 0.1   # koefisien gesek
        self.I = 0.006 # momen inersia pendulum
        self.r = 0.05  # jari-jari roda
        self.dt = 0.02 # time step

        # State awal
        self.phi = 0       # sudut rotasi roda
        self.theta = np.radians(30)   # sudut pendulum (rad)
        self.dphi = 0      # kecepatan rotasi roda
        self.dtheta = 0    # kecepatan sudut pendulum
        self.x = 0         # posisi linier roda
        self.dx = 0        # kecepatan linier roda

    def update_state(self, torque):
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)

        # Denominator for equations of motion
        D = self.I * (self.M + self.m) + self.M * self.m * self.l ** 2 + self.m ** 2 * self.l ** 2 * cos_theta ** 2
        if D == 0:
            D = 1e-10  # Avoid division by zero

        # Percepatan sudut pendulum (ddtheta) dan rotasi roda (ddphi)
        ddtheta = (1 / D) * (
            self.m * self.l * sin_theta * (self.M * self.g + self.m * self.g) -
            self.m ** 2 * self.l ** 2 * sin_theta * cos_theta * self.dtheta ** 2 -
            self.b * self.dtheta * self.m * self.l + torque * self.m * self.l * cos_theta / self.r
        )

        ddphi = (1 / D) * (
            torque * (self.I + self.m * self.l ** 2) / self.r -
            self.m * self.l * sin_theta * (self.g * (self.M + self.m) + self.m * self.l * self.dtheta ** 2)
        )

        # Update state dengan metode integrasi
        self.phi += self.dphi * self.dt
        self.theta += self.dtheta * self.dt
        self.dphi += ddphi * self.dt
        self.dtheta += ddtheta * self.dt

        # Update posisi linier dan kecepatannya
        self.dx = self.r * self.dphi
        self.x += self.dx * self.dt

        # Batasi nilai kecepatan untuk menghindari ketidakstabilan
        self.dphi = np.clip(self.dphi, -20, 20)
        self.dtheta = np.clip(self.dtheta, -20, 20)


# Artificial Bee Colony Optimization for PID
class ABC:
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.solutions = []
        self.trials = np.zeros(20)
        self.best_solution = None
        self.best_fitness = float('inf')

        # Inisialisasi solusi acak (Kp, Ki, Kd)
        for _ in range(20):
            solution = {
                'Kp': np.random.uniform(0, 100),
                'Ki': np.random.uniform(0, 100),
                'Kd': np.random.uniform(0, 100)
            }
            self.solutions.append(solution)

    def calculate_fitness(self, solution):
        # Reset state pendulum
        self.pendulum.theta = 0.1
        self.pendulum.phi = 0
        self.pendulum.dtheta = 0
        self.pendulum.dphi = 0
        self.pendulum.x = 0
        self.pendulum.dx = 0

        total_error = 0

        # Simulasi sistem
        for _ in range(200):
            error = 0 - self.pendulum.theta
            torque = (
                solution['Kp'] * error +
                solution['Kd'] * (-self.pendulum.dtheta)
            )
            torque = np.clip(torque, -10, 10)

            # Update state sistem
            self.pendulum.update_state(torque)

            # Akumulasi error
            total_error += abs(self.pendulum.theta)

        return total_error

    def optimize(self):
        for iteration in range(100):
            # Employed Bee Phase
            for i in range(len(self.solutions)):
                new_solution = self.solutions[i].copy()
                param = np.random.choice(['Kp', 'Ki', 'Kd'])
                new_solution[param] += (np.random.random() - 0.5) * 20

                old_fitness = self.calculate_fitness(self.solutions[i])
                new_fitness = self.calculate_fitness(new_solution)

                if new_fitness < old_fitness:
                    self.solutions[i] = new_solution
                    self.trials[i] = 0
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_solution.copy()
                else:
                    self.trials[i] += 1

            # Onlooker Bee Phase
            fitness_values = [1 / (self.calculate_fitness(s) + 1e-10) for s in self.solutions]
            total_fitness = sum(fitness_values)
            probabilities = [f / total_fitness for f in fitness_values]

            for i in range(len(self.solutions)):
                if np.random.random() < probabilities[i]:
                    new_solution = self.solutions[i].copy()
                    param = np.random.choice(['Kp', 'Ki', 'Kd'])
                    new_solution[param] += (np.random.random() - 0.5) * 20

                    old_fitness = self.calculate_fitness(self.solutions[i])
                    new_fitness = self.calculate_fitness(new_solution)

                    if new_fitness < old_fitness:
                        self.solutions[i] = new_solution
                        self.trials[i] = 0
                        if new_fitness < self.best_fitness:
                            self.best_fitness = new_fitness
                            self.best_solution = new_solution.copy()
                    else:
                        self.trials[i] += 1

            # Scout Bee Phase
            for i in range(len(self.solutions)):
                if self.trials[i] > 30:
                    self.solutions[i] = {
                        'Kp': np.random.uniform(0, 100),
                        'Ki': np.random.uniform(0, 100),
                        'Kd': np.random.uniform(0, 100)
                    }
                    self.trials[i] = 0

            print(f"Iteration {iteration + 1}/100, Best Fitness: {self.best_fitness}")


# Plot grafik sudut terhadap waktu
def simulate_system(pendulum, abc):
    state_data = {
        'time': [],
        'theta': [],
        'phi': [],
        'x': []
    }

    pendulum.theta = 0.1
    pendulum.phi = 0
    pendulum.dtheta = 0
    pendulum.dphi = 0
    pendulum.x = 0
    pendulum.dx = 0

    for t in range(int(10 / pendulum.dt)):
        state_data['time'].append(t * pendulum.dt)
        state_data['theta'].append(pendulum.theta)
        state_data['phi'].append(pendulum.phi)
        state_data['x'].append(pendulum.x)

        error = 0 - pendulum.theta
        torque = (
            abc.best_solution['Kp'] * error +
            abc.best_solution['Kd'] * (-pendulum.dtheta)
        )
        torque = np.clip(torque, -10, 10)

        pendulum.update_state(torque)

    return state_data

def plot_theta_vs_time(state_data):
    plt.plot(state_data['time'], state_data['theta'])
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Theta vs Time")
    plt.grid()
    plt.show()

# Animasi sistem
def animate_system(state_data, pendulum):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.grid()

    # Lingkaran roda
    wheel = Circle((0, 0), pendulum.r, color='black', fill=False, lw=2)
    marker = Circle((pendulum.r, 0), 0.02, color='red', fill=True)

    # Garis pendulum
    pendulum_line, = ax.plot([], [], 'o-', lw=2, color='black')

    # Tambahan teks
    time_text = ax.text(-1.8, 1.8, '', fontsize=12)
    theta_text = ax.text(0.5, 1.8, '', fontsize=12)
    phi_text = ax.text(-1.8, 1.6, '', fontsize=12)

    ax.add_patch(wheel)
    ax.add_patch(marker)

    def init():
        pendulum_line.set_data([], [])
        time_text.set_text('')
        theta_text.set_text('')
        phi_text.set_text('')
        return pendulum_line, time_text, theta_text, phi_text, wheel, marker

    def update(frame):
        x_base = state_data['x'][frame]
        y_base = pendulum.r
        x_pendulum = x_base + pendulum.l * np.sin(state_data['theta'][frame])
        y_pendulum = y_base + pendulum.l * np.cos(state_data['theta'][frame])

        # Update lingkaran roda
        wheel.center = (x_base, y_base)
        marker.set_center((x_base + pendulum.r * np.cos(state_data['phi'][frame]), y_base + pendulum.r * np.sin(state_data['phi'][frame])))

        # Update garis pendulum
        pendulum_line.set_data([x_base, x_pendulum], [y_base, y_pendulum])

        # Update teks
        time_text.set_text(f"time = {state_data['time'][frame]:.1f}s")
        theta_text.set_text(f"θ = {np.degrees(state_data['theta'][frame]):.2f}°")
        phi_text.set_text(f"φ = {state_data['phi'][frame]:.2f}")

        return pendulum_line, time_text, theta_text, phi_text, wheel, marker

    anim = FuncAnimation(fig, update, frames=len(state_data['time']), init_func=init, blit=True, interval=pendulum.dt * 1000)
    plt.show()

# Jalankan optimasi dan simulasi
pendulum = InvertedPendulumRobot()
abc = ABC(pendulum)
abc.optimize()

print("\nBest Solution:")
print(f"Kp: {abc.best_solution['Kp']:.2f}")
print(f"Ki: {abc.best_solution['Ki']:.2f}")
print(f"Kd: {abc.best_solution['Kd']:.2f}")

# Simulasi dan plot hasil
state_data = simulate_system(pendulum, abc)
plot_theta_vs_time(state_data)

# Jalankan animasi
animate_system(state_data, pendulum)
