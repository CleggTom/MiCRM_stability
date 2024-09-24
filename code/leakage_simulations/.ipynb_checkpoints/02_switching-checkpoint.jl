using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2

#switching
function get_switching_parameters(N,M,σ)

    gx = rand(Uniform(0.75,1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(1,2)) .+ rand(Uniform(-σ,σ), N)
    gw = -rand(Uniform(1,2)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(1.0,1.25)) .+ rand(Uniform(-σ,σ), N)
    
    fy = rand(Uniform(σ,2)) .+ rand(Uniform(-σ,σ),N,M)
    hy = rand(Uniform(σ,2)) .+ rand(Uniform(-σ,σ),N,M)

    # fy = ones(N,M)
    # hy = ones(N,M)
    
    λy = rand(Uniform(-1,1)) .+ rand(Uniform(-σ,σ),N,M)
    # ωy = rand(Uniform(-1,1)) .+ rand(Uniform(-σ,σ),N,M)
    # λy = zeros(N,M)

    iy = rand(Uniform(σ,0.5)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(1.0,2.0)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,iy,oy)
end

function random_community(N,M, f)
    U = rand(N,M) ./ (N*M)
    U[:,end] .= 0
    
    D = rand(M,M)
    D[:,1] .= 0
    [D[i,:] .= D[i,:] ./ sum(D[i,:]) for i = 1:M]
    
    Λ = fill(rand(),N)
    
    s = MiCRM_stability.get_structural_params(U,D,Λ)
    e = f(N,M, 0.1)
    u = MiCRM_stability.util_params(N,M,s,e)
    
    p = MiCRM_stability.Parameters(N,M,s,e,u)

    return(p)
end

function get_param_mean(p::MiCRM_stability.Parameters)
    fe = fieldnames(MiCRM_stability.exponential_params)
    fs = fieldnames(MiCRM_stability.structural_params)
    
    ue = mean.(getfield.(Ref(p.e), fe))
    us = mean.(getfield.(Ref(p.s), fs))

    vcat(ue..., us...) 
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re


#params
N=50
M=100
Np = 10000

p_vec = Vector{Any}(undef,Np)
stability = zeros(Complex, Np)

k = [0]

Threads.@threads for i = 1:Np
    k[1] += 1
    print("\r",k)
    p = random_community(N,M, get_switching_parameters)
    p_vec[i] = get_param_mean(p)
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p,J)
    stability[i] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
end

save("./Results/data/new_sims/switching_stabiltiy.jld2", Dict("p" => p_vec, "l" => stability))