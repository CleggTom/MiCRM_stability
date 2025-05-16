using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2
using .GC

#functions for simulations
# function get_exponential_parameters(N::Int64,M::Int64,σ::Float64)
#     gx = ones(N) 
#     gs = rand(Uniform(0,2), N)
#     mx = ones(N)
    
#     fy = rand(Uniform(1,2),N,M) 
#     λy = zeros(N,M)

#     iy = zeros(M)
#     oy = rand(Uniform(0.5,1.5), M)

#     return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
# end

function get_exponential_parameters(N::Int64,M::Int64,σ::Float64, K::Float64)
    gx = rand(Uniform(0.8,1.2), N)
    gs = rand(Uniform(0.5,2.0), N) 
    mx = gx .+ rand(Normal(0.0,K), N)
    
    fy = rand(Uniform(1,2), N, M)
    λy = zeros(N,M)

    iy = zeros(M)
    oy = rand(Uniform(1.0,1.5),M) 

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end

function random_community(N,M,f, K)
    c = MiCRM_stability.random_community(N,M, rand(), rand())

    while any(sum(c.U, dims = 2) .== 0)
        c = MiCRM_stability.random_community(N,M, rand(), rand())
    end
        # #evey consumer has a link
    # for i = 1:N
    #     if sum(U[i,:]) == 0
    #         U[i, rand(1:M)]
    #     end
    # end
    Λ = rand(N) * 0.25
    I = rand(M) * 0.25
    s = MiCRM_stability.get_structural_params(c.N, c.M, c.U,c.D, Λ, I)
    e = f(N,M, 0.1, K)
    
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re

function inner(i, j, r, k, N, M, K)
    #param
    p = random_community(N, M, get_exponential_parameters, K)
    #jaccobian
    J = zeros(N + M, N + M)
    MiCRM_stability.jacobian!(p, J)
    try 
    stability[i,j,r,k] = get_real(eigsolve(J, 1, (:LR))[1][1])  
    catch
        stability[i,j,r,k] = 0.0
    end
end


#params
nN = 10
nM = 10
Np = 50
nK = 10

n_vec = Int.(floor.(10 .^ range(log10(20),log10(100), length = nN)))
m_vec = Int.(floor.(10 .^ range(log10(20),log10(100), length = nM)))
k_vec = 10 .^ range(-5,-2, nK)

stability = zeros(Complex, nN, nM, Np, nK)

itt = collect(Iterators.product(1:nN,1:nM,1:Np, 1:nK))

counter = [0]
Threads.@threads for (i,j,r,k) = itt
    counter[1] += 1
    N = n_vec[i]
    M = m_vec[j]
    K = k_vec[k]
    print("\r",counter,"  ",N,"  ",M)
    
    inner(i, j, r, k, N, M, K)
end

save("./Results/data/new_sims/size_stability.jld2", Dict("l" => stability))