using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2
using .GC

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

function modular_community(N, M, f, C, r)
    c = MiCRM_stability.modular_community(N,M,C,5,r)
    # c.U[c.U .!= 0] .= abs.( rand(Normal(1,s), sum( (c.U .!= 0)[:] ) ) )
    Λ = fill(rand(),N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D,Λ)
    e = f(N,M, 0.1)

    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p,c.U)
end

overlap_(x,y) = sum(y) != 0 ? sum(minimum.(zip(x,y))) / sum(y) : 0

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re


#params
Np = 25
Nr = 1000

r_vec = range(0.1, 10.0, length = Np)

itt = collect(Iterators.product(1:Np,1:Nr))
p_itt = collect(Iterators.partition(itt, 50))

stability = zeros(Complex, Np, Nr)
pmat = zeros(Np,Nr)

N=100
M=100

k = [0,0]
Threads.@threads for pi = p_itt
    k[1] += 1
    for (i,r) = pi
        k[2] += 1
        print("\r",k)
        p,U = modular_community(N, M, get_exponential_parameters, 0.3, r_vec[i])
        J = zeros(N+M, N+M)
        MiCRM_stability.jacobian!(p,J)
        stability[i,r] = eigsolve(J, 1, (:LR))[1][1] |> get_real
    end
    GC.safepoint()
end

save("./Results/data/new_sims/modular_stabiltiy.jld2", Dict("l" => stability))