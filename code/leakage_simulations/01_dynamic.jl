using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2
using Random

#functions for simulations
# #functions for simulations
function get_exponential_parameters(N::Int64,M::Int64,σ::Float64)
    gx = rand(Uniform(σ,0.5)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(0.5,1.0)) .+ rand(Uniform(-σ,σ), N)
    
    fy = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ),N,M) 
    λg = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ),N) 

    iy = rand(Uniform(σ,0.5)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(0.5,1.0)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λg,iy,oy)
end

function random_community(N::Int64,M::Int64,f::Function,Cu::Float64,Cd::Float64)
    #generate community
    c = MiCRM_stability.random_community(N,M,Cu,Cd)
    
    #supply params
    Λ = fill(rand(Uniform(0.1,0.8)), N) .+ rand(Uniform(-0.1,0.1),N) 
    G = fill(rand(Uniform(0.1,1.0)), N) .+ rand(Uniform(-0.1,0.1),N) 
    α = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    β = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    C_pert = rand(Uniform(0.1,1.0)) .+ rand(Uniform(-0.1,0.1),M) 
    
    s = MiCRM_stability.get_structural_params(c.N,c.M,c.U, c.D, Λ, G, α, β, C_pert)
    e = f(N,M, 0.2)
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_param_mean(p::MiCRM_stability.Parameters)
    fe = fieldnames(MiCRM_stability.exponential_params)
    fs = fieldnames(MiCRM_stability.structural_params)

    fs = filter(x -> x ∉ [:χ,:ϕ,:γ,:η], fs)
    
    ue = mean.(getfield.(Ref(p.e), fe))
    us = mean.(getfield.(Ref(p.s), fs))

    vcat(ue..., us...) 
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re

function inner(i)
    #connectance
    Cu = rand() 
    Cd = rand() 
    #param
    p = random_community(N,M, get_exponential_parameters, Cu, Cd)
    #save
    p_vec[i] = vcat(get_param_mean(p), Cu, Cd)
    #jaccobian
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p, J)

    e = eigsolve(J, 1, (:LR))
    stability[i] = get_real(e[1][1]) 
    stability_vec[i] = e[2][1]
    con_stability[i] = get_real(eigsolve(J[1:N, 1:N], 1, (:LR))[1][1]) 
    res_stability[i] = get_real(eigsolve(J[N+1:end, N+1:end], 1, (:LR))[1][1]) 
end

#params
N=100
M=100
Np = 100000

p_vec = Vector{Any}(undef,Np)
stability = zeros(Complex, Np)
stability_vec = fill([], Np)
con_stability = zeros(Complex, Np)
res_stability = zeros(Complex, Np)

k = [0]
Threads.@threads for i = 1:Np
    k[1] += 1
    print("\r",k)
    inner(i)
end

save("./Results/data/new_sims/dynamic_stabiltiy.jld2", Dict("p" => p_vec, "l" => stability,"v" => stability_vec, "c" => con_stability, "r" => res_stability))