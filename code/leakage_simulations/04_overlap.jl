using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2

#functions for simulations
function get_exponential_parameters(N::Int64,M::Int64,σ::Float64)
    gx = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), N) 
    gs = rand(Uniform(σ,2.0), N)
    mx = gx .+ rand(Uniform(0,σ), N)
    
    fy = ones(N,M) .+ rand(Uniform(-σ,σ), N)
    λy = zeros(N)

    iy = zeros(M) 
    oy = ones(M) .+ rand(Uniform(-σ,σ), N) 

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end

function jac(c)
    x = eachrow(c.U)
    [sum(min.(x[i],x[j])) / sum(max.(x[i],x[j])) for i = 1:(c.N-1) for j = (i+1):c.N] |> mean
end

function random_community(N,M,f,μ)
    pvec = [μ, 0.0]
    c = MiCRM_stability.niche_model(N,M,μ)
    
    U = c.U .> 0
    pvec[2] = [mean(maximum(U[:,U[i,:]], dims = 2)[:]) for i = 1:N] |> mean
    
    Λ = fill(rand() * 0.25, N)
    s = MiCRM_stability.get_structural_params(c.U, c.D, Λ)
    e = f(N,M, 0.1)
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p, pvec)
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


#params
N=100
M=100
Np = 10000

p_vec = Vector{Any}(undef,Np)
stability = zeros(Complex, Np)

k = [0]

Threads.@threads for i = 1:Np
    k[1] += 1
    print("\r",k)
    p,q = random_community(N,M, get_exponential_parameters, rand()/2)
    p_vec[i] = q
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p,J)
    stability[i] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
end

save("./Results/data/new_sims/overlap_stabiltiy.jld2", Dict("p" => p_vec, "l" => stability))