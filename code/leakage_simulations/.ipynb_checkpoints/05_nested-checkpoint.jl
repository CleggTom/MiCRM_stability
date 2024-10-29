using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase
using Distributions
using JLD2
using .GC

#functions for simulations
function get_exponential_parameters(N,M,σ)
    gx = rand(Uniform(0,1),N) 
    gs = rand(Uniform(0,2),N)
    mx = ones(N)
    
    fy = ones(N) 
    λy = zeros(N,M) #.+ rand(Uniform(-σ,σ),N,M)

    iy = zeros(M) 
    oy = ones(M)

    return MiCRM_stability.exponential_params(gx,gs,mx,fy,λy,iy,oy)
end

function nested_community(N, M, f, k, v, Cd)
    c = MiCRM_stability.nested_community(N, M, k, v, Cd)
    Λ = fill(0.3,N)
    
    s = MiCRM_stability.get_structural_params(c.U,c.D,Λ, 2rand())
    e = f(N,M, 0.1)

    p = MiCRM_stability.Parameters(N,M,s,e)

    return(p,c.U)
end

overlap_(x,y) = sum(y) != 0 ? sum(minimum.(zip(x,y))) / sum(y) : 0

function nodf(A)
    N,M = size(A)
    #rows
    #get ones
    row_ind = [findall(A[i,:] .> 0) for i = 1:N]

    x_row = [[length(row_ind[i]) > length(row_ind[j]) ? overlap_(A[i,:], A[j,:]) : 0 for j = i+1:N] for i = 1:N] 
    
    col_ind = [findall(A[:,i] .> 0) for i = 1:M]
    x_col = [[length(col_ind[i]) > length(col_ind[j]) ? overlap_(A[:,i], A[:,j]) : 0 for j = i+1:M] for i = 1:M] 
    
    return mean(vcat(x_row..., x_col...))
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
Np = 10
Nr = 100

v_vec = range(0.0001, 10.0, length = Np)

itt = collect(Iterators.product(1:Np, 1:Nr))

stability = zeros(Complex, Np, Nr)
pmat = zeros(Np, Nr)

N=100
M=100

k = [0,0]
Threads.@threads for (i,r) = itt
    k[1] += 1
    print("\r",k)
    p,U = nested_community(N, M, get_exponential_parameters, 1.0, v_vec[i], 0.3)
    J = zeros(N+M, N+M)
    MiCRM_stability.jacobian!(p,J)
    stability[i,r] = eigsolve(J, 1, (:LR))[1][1] |> get_real
    pmat[i,r] = nodf(U)
    GC.safepoint()
end

save("./Results/data/new_sims/nested_stabiltiy.jld2", Dict("l" => stability,"p" => pmat))