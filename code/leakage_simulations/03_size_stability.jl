using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
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

#create two communties one structured and one shuffled
function random_community(N,M,f)
    c = MiCRM_stability.random_community(N,M, rand(), rand())
    Λ = rand(N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D, Λ)
    e = f(N,M, 0.1)
    
    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p)
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re

function inner(i, j, r, N, M)
    #param
    p = random_community(N, M, get_exponential_parameters)
    #jaccobian
    J = zeros(N + M, N + M)
    MiCRM_stability.jacobian!(p, J)
    try 
    stability[i,j,r] = get_real(eigsolve(J, 1, (:LR))[1][1])  
    catch
        stability[i,j,r] = 0.0
    end
end


#params
nN = 10
nM = 10
Np = 100

n_vec = Int.(floor.(10 .^ range(log10(20),log10(100), length = nN)))
m_vec = Int.(floor.(10 .^range(log10(20),log10(100), length = nM)))

stability = zeros(Complex, nN, nM, Np)

itt = collect(Iterators.product(1:nN,1:nM,1:Np))

counter = [0]
Threads.@threads for (i,j,r) = itt
    counter[1] += 1
    N = n_vec[i]
    M = m_vec[j]
    print("\r",counter,"  ",N,"  ",M)
    
    inner(i, j, r, N, M)
end

save("./Results/data/new_sims/size_stability2.jld2", Dict("l" => stability))