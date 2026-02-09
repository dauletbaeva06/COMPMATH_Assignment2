import numpy as np
import matplotlib.pyplot as plt

#Defining the SIR Model ODEs
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I                # Rate of change for Susceptible
    dIdt = beta * S * I - gamma * I     # Rate of change for Infected
    dRdt = gamma * I                    # Rate of change for Recovered
    return np.array([dSdt, dIdt, dRdt])

# RK4 Solver Implementation
def solve_rk4(S0, I0, R0, beta, gamma, h, days):
    steps = int(days / h)
    t_vals = np.linspace(0, days, steps + 1)
    results = np.zeros((steps + 1, 3))
    results[0] = [S0, I0, R0]
    
    for i in range(steps):
        ti = t_vals[i]
        yi = results[i]

        k1 = h * sir_model(ti, yi, beta, gamma)
        k2 = h * sir_model(ti + h/2, yi + k1/2, beta, gamma)
        k3 = h * sir_model(ti + h/2, yi + k2/2, beta, gamma)
        k4 = h * sir_model(ti + h, yi + k3, beta, gamma)
        
        results[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
        
    return t_vals, results

# Initial Parameters
population = 1000000             # Total population
S0_orig = 999000                 # Initial Susceptible
I0_orig = 1000                   # Initial Infected
R0_orig = 0                      # Initial Recovered
beta_orig = 0.0003 / population  # Infection rate
gamma = 0.1                      # Recovery rate
h = 0.1                          # Step size
days = 100                       # Total simulation days

# Simulations
# Original Scenario
t, res_orig = solve_rk4(S0_orig, I0_orig, R0_orig, beta_orig, gamma, h, days)

# Vaccination Scenario
S0_vac = S0_orig * 0.5
t, res_vac = solve_rk4(S0_vac, I0_orig, R0_orig, beta_orig, gamma, h, days)

# Social Distancing Scenario
beta_sd = beta_orig * 0.5
t, res_sd = solve_rk4(S0_orig, I0_orig, R0_orig, beta_sd, gamma, h, days)

# Analysis and Plotting
# Plotting Original Curves
plt.figure(figsize=(10, 6))
plt.plot(t, res_orig[:, 0], label='Susceptible (S)', color='blue')
plt.plot(t, res_orig[:, 1], label='Infected (I)', color='red')
plt.plot(t, res_orig[:, 2], label='Recovered (R)', color='green')
plt.title("SIR Model: Original Outbreak Simulation")
plt.xlabel("Time (Days)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# Peak Analysis
peak_idx = np.argmax(res_orig[:, 1])
peak_val = res_orig[peak_idx, 1]
peak_day = t[peak_idx]

# Total Infected
total_infected = S0_orig - res_orig[-1, 0]

print(f"--- Original Scenario Results ---")
print(f"Peak Infected Individuals: {int(peak_val):,}")
print(f"Peak Occurs on Day: {peak_day:.1f}")
print(f"Total People Ever Infected: {int(total_infected):,}")

# Comparison Plot for Scenarios
plt.figure(figsize=(10, 6))
plt.plot(t, res_orig[:, 1], label='Original Infected', color='red')
plt.plot(t, res_vac[:, 1], label='Vaccination Infected (50% S reduction)', linestyle='--')
plt.plot(t, res_sd[:, 1], label='Social Distancing Infected (50% Beta reduction)', linestyle=':')
plt.title("Comparison of Infected Curves Across Scenarios")
plt.xlabel("Days")
plt.ylabel("Number of Infected")
plt.legend()
plt.grid(True)
plt.show()