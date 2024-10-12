using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2

#switching
function get_exponential_parameters(N,M,σ)
    gx = rand(Uniform(0.75,1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(1,2)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(1.0,1.25)) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N) .+ rand(Uniform(-σ,σ),N,M) 
    λy = zeros(N,M) #.+ rand(Uniform(-σ,σ),N,M)

    iy = zeros(M)
    oy = ones(M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end


#create two communties one structured and one shuffled
function random_community(N,M,f)
    c = MiCRM_stability.random_community(N,M, 0.3)
    Λ = fill(0.3,N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D,Λ, 0.0)
    e = f(N,M, 0.1)
    
    p = MiCRM_stability.Parameters(N,M,s,e,[])
    push!(p.tmp, MiCRM_stability.calculate_g_s(p))

    return(p)
end

function get_real(x::T) where T <: AbstractFloat
    x
end

get_real(x::Complex) = x.re


#params
N=30
M=30
Np = 500

n_vec = Int.(floor.(range(10,150, length = N)))
m_vec = Int.(floor.(range(10,150, length = M)))

stability = zeros(Complex, N, M, Np)
# p_vec = Vector{Any}(undef,n, m, Np)

counter = [0]
for n = 1:N
    for m = 1:M
  Threads.@threads for i = 1:Np
            counter[1] += 1
            N_ = n_vec[n]
            M_ = m_vec[m]
            print("\r",counter,"  ",N_,"  ",M_)
            
            p = random_community(N_,M_, get_exponential_parameters)
            J = zeros(N_+ M_, N_ + M_)
            MiCRM_stability.jacobian!(p,J)
            stability[n,m,i] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
        end
    end
end

save("./Results/data/new_sims/size_stability.jld2", Dict("l" => stability))