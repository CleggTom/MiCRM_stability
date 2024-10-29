using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2
using .GC

#switching
function get_exponential_parameters(N,M,σ)
    gx = rand(Uniform(0.9, 1.1), N)
    gs = rand(Uniform(0.0, 2.0), N)
    mx = rand(Uniform(0.9, 1.1), N)
    
    fy = ones(N,M)
    λy = zeros(N,M) #.+ rand(Uniform(-σ,σ),N,M)

    iy = zeros(M) 
    oy = ones(M) + rand(M)/10

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end


#create two communties one structured and one shuffled
function random_community(N,M,f)
    c = MiCRM_stability.random_community(N,M, 0.3, 0.3)
    Λ = fill(0.25,N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D, Λ, rand())
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

    stability[i,j,r] = get_real(eigsolve(J, 1, (:LR))[1][1])  
end


#params
nN= 10
nM= 10
Np = 10

n_vec = Int.(floor.(10 .^ range(1,log10(200), length = nN)))
m_vec = Int.(floor.(10 .^range(1,log10(200), length = nM)))

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

save("./Results/data/new_sims/size_stability3.jld2", Dict("l" => stability))