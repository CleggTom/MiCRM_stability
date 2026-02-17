using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2
using .GC

# functions for simulations
function get_exponential_parameters(N::Int64,M::Int64,σ::Float64)
    gx = rand(Uniform(0,1),N) 
    gs = rand(Uniform(0,2), N)
    mx = gx
    
    fy = rand(Uniform(1,2),N,M) 
    λy = zeros(N)

    iy = zeros(M)
    oy = rand(Uniform(0.5,1.5), M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end


function random_community(N,M,f)
    #fixed consumption per consumer
    #fixed leakage per resource
    c = MiCRM_stability.random_community(N,M, 10/N, 10/M)

    Λ = rand(Uniform(0.1,0.8)) .+ rand(Uniform(-0.1,0.1),N) 
    G = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),N) 
    α = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    β = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    C_pert = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    
    s = MiCRM_stability.get_structural_params(c.N, c.M, c.U,c.D, Λ, G, α, β, C_pert)
    e = f(N,M, 0.1)
    
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re

#params
nN = 50
nM = 50
Np = 500

#size vects
n_vec = Int.(floor.(10 .^ range(log10(20),log10(200), length = nN)))
m_vec = Int.(floor.(10 .^ range(log10(20),log10(200), length = nM)))

d_max = maximum(n_vec .+ m_vec)

#allocate per-thread buffers
jac_buffers = [zeros(Float64, d_max, d_max) for _ in 1:Threads.nthreads()]

#allocate results
stability = zeros(Complex, nN, nM, Np)

#parellelise
tasks = vec(collect(Iterators.product(1:nN, 1:nM, 1:Np)))
total_tasks = length(tasks)
completed = Threads.Atomic{Int}(0)

#warmup
begin
    p = random_community(10,10, get_exponential_parameters)

    J_full = jac_buffers[1]
    J_view = @view J_full[1:20, 1:20]
    MiCRM_stability.jacobian!(p,J_view)

    # Only store the real part as Float64 to save memory
    real(eigsolve(J_view, 1, :LR)[1][1])
end


Threads.@threads for t = 1:total_tasks
    i,j,r = tasks[t]
    tid = Threads.threadid()

    N = n_vec[i]
    M = m_vec[j]
    dim = N + M
    
    p = random_community(N,M, get_exponential_parameters)

    J_full = jac_buffers[tid]
    J_view = @view J_full[1:dim, 1:dim]

    MiCRM_stability.jacobian!(p,J_view)

    try 
        # Only store the real part as Float64 to save memory
        # @time stability[i,j,r] = real(eigsolve(J_view, 1, :LR)[1][1])
       stability[i,j,r] = maximum(real.(eigvals(J_view)))
    catch
        stability[i,j,r] = NaN
    end

    # 5. Thread-safe counter
    new_val = Threads.atomic_add!(completed, 1)
    if new_val % 100 == 0
        print("\rProgress: ", new_val)
    end
end


save("./Results/data/new_sims/size_stability.jld2", Dict("l" => stability))