############################################
# Bifurcation Analysis with BifurcationKit
# for the system described.
############################################

using DifferentialEquations
using Plots
using BifurcationKit
using Accessors

#############################
# 1) Define constants
#############################
const jee = 1
const jei = 1
const jie = 5
const jii = 2
const Ie = -0.12
const Ii = -0.2
const N = 325
const taue = 1
const taui = 2
const tree = 40
const tdee = 10
const trie = 40
const tdie = 10


# We treat 'threshee' as fixed, 
# and do bifurcation analysis in terms of 'threshie'
const threshee_init = 0.3
const threshie_init = 0.4

const bee = 50
const bie = 50
const mee = 4
const mie = 4

#############################
# 2) Define transfer functions
#############################
f(u) = 1 / (1 + exp(-sqrt(N) * u))

ar(u, a, b, m) = m / (1 + exp(-b * (u - a)))
pr(u, a, b, m, c, d) = d / (d + c * ar(u, a, b, m))

#############################
# 3) ODE system
#############################
"""
    neural_dynamics!(du, u, p, t)

where u = [v_e, v_i, pree, prie], and p is a named tuple containing:
    p.threshee
    p.threshie
"""
function neural_dynamics!(du, u, p, t)
    ve, vi, pree, prie = u
    # unpack params
    threshee = p.threshee
    threshie = p.threshie
    # ODEs
    du[1] = (-ve + f(jee * pree * ve - jei * vi + Ie)) / taue
    du[2] = (-vi + f(jie * prie * ve - jii * vi + Ii)) / taui
    du[3] = ((1 - pree) / tree) - pree * ar(ve, threshee, bee, mee) / tdee
    du[4] = ((1 - prie) / trie) - prie * ar(ve, threshie, bie, mie) / tdie
end

#############################
# 4) Run bifurcation analysis
#############################
function run_bifurcation_analysis(; threshee_fixed=threshee_init, threshie_start=threshie_init)

    # We'll do continuation in 'threshie'
    param = (
        threshee = threshee_fixed, 
        threshie = threshie_start
    )

    # initial condition near equilibrium
    u0_guess = [0.8, 0.5, 1.0, 1.0]

    # Use an ODE solve to get a better guess of the equilibrium
    prob_for_guess = ODEProblem(neural_dynamics!, u0_guess, (0., 200.), param)
    sol_guess = DifferentialEquations.solve(prob_for_guess, Tsit5(); abstol=1e-11, reltol=1e-11)
    u0_eq = sol_guess[end]  # last point ~ equilibrium
    print("Initial guess for equilibrium: ", u0_eq)

    # Create a function for BifurcationProblem
    # IMPORTANT: use `similar(u)` instead of `zeros(...)`
    function sysBK(u, p)
        du = similar(u) # same element type as u (handles Dual numbers properly)
        neural_dynamics!(du, u, p, 0)
        return du
    end

    # We vary threshie, so we use @optic _.threshie
    #bk_prob = BifurcationProblem(sysBK, u0_eq, param, @optic _.threshie)
    bk_prob = BifurcationProblem(
    sysBK, 
    u0_eq, 
    param, 
    @optic _.threshie; 
    # This "record_from_solution" callback assigns names to each component
    record_from_solution = (x, p; k...) -> (
        v_e = x[1], 
        v_i = x[2], 
        pree = x[3], 
        prie = x[4]
    )
    )

    # continuation parameters
    cont_params = ContinuationPar(
        p_min  = 0.005,
        p_max  = 0.995,
        ds     = 0.05,
        dsmin  = 1e-5,     # Minimum step size allowed
        dsmax  = 0.1,     # Maximum step size allowed
    )

    # run equilibrium continuation
    br = continuation(
        bk_prob,
        PALC(),
        cont_params;
        bothside = true
    )

    # show info
    println(br)

    # find Hopf points
    hopfpts = findall(pt -> pt.type == HopfPoint, br.specialpoint)
    println("Hopf points found at indices: ", hopfpts)

    # plot equilibrium branch
    p1 = plot(
        br,
        title  = "Equilibrium Branch vs threshie\n(threshee=$(threshee_fixed))"
    )


    #save the plot
    savefig(p1, "./try_equilibrium_branch_03.png")
    display(p1)


    ##################
    # If Hopf found, continue limit cycles from Hopf
    ##################
    if !isempty(hopfpts)
        hopf_index = hopfpts[1]  # take the first Hopf

        # create an ODEProblem for shooting
        # Guess a period = 10.0
        prob_shoot = ODEProblem(
            (du, u, p, t) -> neural_dynamics!(du, u, p, t),
            copy(u0_eq),
            (0., 10.),
            param
        )

        # smaller steps for limit cycle continuation
        cont_params_po = ContinuationPar(
            ds=0.001,
            dsmin=1e-6,
            dsmax=0.05,
            p_min=0.0,
            p_max=1.0,
            max_steps=200,
            newton_options = NewtonPar(tol=1e-8),
            detect_bifurcation=0,
        )

        # use multiple shooting with 20 slices
        multshoot = ShootingProblem(
            20,
            prob_shoot,
            Tsit5();
            parallel = false
        )

        # continue the limit cycle branch from the Hopf index
        br_po = continuation(
            br,
            hopf_index,
            cont_params_po,
            multshoot
        )

        # plot the resulting periodic-orbit branch
        p2 = plot(
            br_po,
            title = "Limit Cycle Branch vs threshie",
            linewidth = 2
        )
        display(p2)
        #save plot
        savefig(p2, "./try_limit_cycle_branch_03.png")
        println(br_po)
    end
end

#############################
# 5) Main script logic
#############################
function main()
    println("\nPerforming bifurcation analysis in 'threshie', with threshee fixed.")
    run_bifurcation_analysis(threshee_fixed=0.3, threshie_start=0.1)
end

# Execute main when script is run
main()
