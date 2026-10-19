using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2
using .GC

# functions for simulations
function get_exponential_parameters(N::Int64,M::Int64,σ::Float64, λ0::Float64)
    gx = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ), N)
    mx = gx 
    
    fy = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ),N,M) 
    λg = λ0 .+ rand(Uniform(-σ,σ),N) 

    iy = rand(Uniform(σ,0.5)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(0.5,1.0)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λg,iy,oy)
end



function niche_community(N,M,f,C,d,λ0,σ)
    #fixed consumption per consumer
    #fixed leakage per resource
    
    c = MiCRM_stability.niche_model(N,M,C,d,σ)

    #supply params
    Λ = 0.5 .+ rand(Uniform(-0.1,0.1),N) 
    G = 1.0 .+ rand(Uniform(-0.1,0.1),N) 
    α = 1.0 .+ rand(Uniform(-0.1,0.1),M) 
    β = 1.0 .+ rand(Uniform(-0.1,0.1),M) 
    C_pert = zeros(M)
    
    s = MiCRM_stability.get_structural_params(c.N, c.M, c.U, c.D, Λ, G, α, β, C_pert)
    e = f(N,M, 0.1, λ0)
    
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re


#params
N = 50
M = 75

Nr = 2000
Nσ = 25
NΛ = 5

#paramter vectors
Λ_vec = range(0.5, 1.0, length = NΛ)
σ_vec = range(0.0, 1.0, length = Nσ)

#allocate per-thread buffers
d_max = N+M
jac_buffers = [zeros(Float64, d_max, d_max) for _ in 1:Threads.nthreads()]

#allocate results
stability = zeros(Float64, Nσ, NΛ, Nr)

#parellelise
tasks = vec(collect(Iterators.product(1:Nσ, 1:NΛ, 1:Nr)))
total_tasks = length(tasks)
completed = Threads.Atomic{Int}(0)

#warmup
begin
    p = niche_community(N, M, get_exponential_parameters, 0.3, 0.1, 1.0, 0.0)
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p, J)
    try 
        # real(eigsolve(J, 1, :LR)[1][1])
        real(eigen(J).values[end])
    catch
        println("test didnt work")
    end
end


Threads.@threads for t = 1:total_tasks
    if completed[] % 1000 ==0
        println("\r Total: ", completed[])
    end
    
    i,j,r = tasks[t]
    
    tid = Threads.threadid() - 1

    local p = niche_community(N, M, get_exponential_parameters, 0.3, 0.1, Λ_vec[j], σ_vec[i])
    local J = jac_buffers[tid]

    MiCRM_stability.jacobian!(p,J)

    try 
        # Only store the real part as Float64 to save memory
        # @time stability[i,j,r] = real(eigsolve(J_view, 1, :LR)[1][1])
       stability[i,j,r] = real(eigen(J).values[end])
    catch
        stability[i,j,r] = NaN
    end
    
    Threads.atomic_add!(completed, 1)
end


save("./Results/data/trophic_stability.jld2", Dict("l" => stability,"σ" => σ_vec, "Λ" => Λ_vec))