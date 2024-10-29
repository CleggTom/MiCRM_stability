using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2
using Random

#functions for simulations
function get_exponential_parameters(N::Int64,M::Int64,σ::Float64)
    gx = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(σ,2.0)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(0.5,1.5)) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M)

    λy = zeros(N,M)

    iy = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(0.5,1.5)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end

function random_community(N::Int64,M::Int64,f::Function,Cu::Float64,Cd::Float64)
    c = MiCRM_stability.random_community(N,M,Cu,Cd)
    # c.U[c.U .!= 0] .= abs.( rand(Normal(1,s), sum( (c.U .!= 0)[:] ) ) )
    Λ = fill(rand(),N)
    
    s = MiCRM_stability.get_structural_params(c.U, c.D, Λ, 2rand())
    e = f(N,M, 0.1)

    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_param_mean(p::MiCRM_stability.Parameters)
    fe = fieldnames(MiCRM_stability.exponential_params)
    fs = fieldnames(MiCRM_stability.structural_params)

    fs = filter(x -> x ∉ [:χ,:ϕ,:γ, :η], fs)
    
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
    
    eig = eigen(J)
    stability[i] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
    # vectors[i] = eig.vectors[:,end]
 
end

#params
N=100
M=100
Np = 10000

p_vec = Vector{Any}(undef,Np)
stability = zeros(Complex, Np)
# vectors = Vector{Vector{Complex}}(undef,Np)
vectors = 0

k = [0]
Threads.@threads for i = 1:Np
    k[1] += 1
    print("\r",k)
    inner(i)
end

save("./Results/data/new_sims/dynamic_stabiltiy.jld2", Dict("p" => p_vec, "l" => stability, "v" => vectors))