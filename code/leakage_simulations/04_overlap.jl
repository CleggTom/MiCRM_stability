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
    mx = gx 
    
    fy = ones(N,M) 
    λy = zeros(N,M)

    iy = zeros(M)
    oy = ones(M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end

function random_community(N,M,f,C,o)
    c = MiCRM_stability.overlap_community(N,M,C,o)
    Λ = fill(rand(),N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D,Λ, rand())
    e = f(N,M, 0.1)

    p = MiCRM_stability.Parameters(N,M,s,e,[])
    push!(p.tmp, MiCRM_stability.calculate_g_s(p))

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
    C = rand()
    o = rand()
    p = random_community(N,M, get_exponential_parameters, C,o)
    p_vec[i] = [o]
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p,J)
    stability[i] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
end

save("./Results/data/new_sims/overlap_stabiltiy.jld2", Dict("p" => p_vec, "l" => stability))