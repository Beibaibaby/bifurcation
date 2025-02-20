using DifferentialEquations, Plots

# ---------------------------
# Parameters
# ---------------------------
const jee = 10
const jei = 2
const jie = 10
const jii = 1
const Ie = 0.05
const Ii = -0.2
const N = 500 
const taue = 1
const taui = 2
const tree = 20
const tdee = 5
const trie = 20
const tdie = 5

# Initial values for thresholds (will be overwritten in the grid search)
threshee = 0.9  #0.9
threshie = 0.1

const bee = 50
const bie = 50
const mee = 4
const mie = 4

# ---------------------------
# Transfer function and helper functions
# ---------------------------
f(u) = 1 / (1 + exp(-sqrt(N) * u))
ar(u, a, b, m) = m / (1 + exp(-b * (u - a)))
pr(u, a, b, m, c, d) = d / (d + c * ar(u, a, b, m))

# ---------------------------
# ODE system definition
# ---------------------------
function neural_dynamics!(du, u, p, t)
    # u[1] = v_e, u[2] = v_i, u[3] = pree, u[4] = prie
    ve, vi, pree, prie = u
    du[1] = (-ve + f(jee * pree * ve - jei * vi + Ie)) / taue
    du[2] = (-vi + f(jie * prie * ve - jii * vi + Ii)) / taui
    du[3] = ((1 - pree) / tree) - pree * ar(ve, threshee, bee, mee) / tdee
    du[4] = ((1 - prie) / trie) - prie * ar(ve, threshie, bie, mie) / tdie
end

# ---------------------------
# Initial conditions and time span
# ---------------------------
u0 = [0.1, 0.1, 1.0, 1.0]
tspan = (0.0, 200.0)

# ---------------------------
# Grid search over threshee and threshie values
# ---------------------------
threshold_values = 0.0:0.1:1.0  # Creates the range [0, 0.25, 0.5, 0.75, 1.0]

for threshee_val in threshold_values
    for threshie_val in threshold_values
        # Update the global threshold parameters (scope can be tricky in Julia!)
        global threshee = threshee_val
        global threshie = threshie_val
        
        # Define and solve the ODE problem
        prob = ODEProblem(neural_dynamics!, u0, tspan)
        sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10)
        
        # Create the plot for this parameter combination
        
        p = plot(sol.t, sol[1, :],
                 label="v_e", xlabel="Time", ylabel="Activity", lw=2,
                 title="threshee=$(threshee_val), threshie=$(threshie_val)",
                 color=["red"])
        plot!(p, sol.t, sol[2, :], label="v_i", lw=2,color=["blue"])
        
        # Save the plot with a unique filename
        filename = "./plots_test/threshee_$(threshee_val)_threshie_$(threshie_val).png"
        savefig(p, filename)
        println("Saved plot to ", filename)
    end
end
