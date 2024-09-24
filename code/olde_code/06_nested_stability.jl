using Pkg

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Distances, Distributions
using CairoMakie
using DelimitedFiles
using Random
using Graphs, GraphMakie, NetworkLayout, SimpleWeightedGraphs
using LsqFit

println(pwd())
println(Threads.nthreads())

include("./functions.jl")

function get_param_e_mean(p::MiCRM_stability.Parameters)
    fe = fieldnames(MiCRM_stability.exponential_params)
    ue = mean.(getfield.(Ref(p.e), fe))
    vcat(ue...) 
end

function get_param_s_mean(p::MiCRM_stability.Parameters)
    fs = fieldnames(MiCRM_stability.structural_params)
    us = mean.(getfield.(Ref(p.s), fs))
    vcat(us...) 
end


function get_exponential_parameters(N,M)
    σ = 0.1
    gx = 0.95 .* ones(N) .+ rand(Uniform(-σ,σ), N)
    gs = ones(N) .+ rand(Uniform(-0.25,0.25), N)
    gw = zeros(N) 
    mx = 1.1ones(N) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M)
    hy = -ones(N,M)
    
    λy = zeros(N,M)
    ωy = zeros(N,M)

    iy = zeros(M)
    oy = ones(M) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,ωy,iy,oy)
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

N,M = 100,100
C = 0.3
Ncom = 10000
Nrep = 100

k = [0]

λ_mat = zeros(Ncom, Nrep)
n_mat = zeros(Ncom)
# p_mat = Array{Float64,4}(undef, Ncom, 4, Nrep, length(p_names))

#modularity
Threads.@threads for i = 1:Ncom
    k[1] += 1
    if k[1] % 100 == 0
        print("\r", k)
    end

    #allocate jacobian
    J = zeros(N+M,N+M)    
    com = MiCRM_stability.nested_community(N,M, -0.25, 10rand(), 10, 10)
    n_mat[i] = nodf(com.U)
    
    sp = MiCRM_stability.get_structural_params(com.U,com.L, com.N, com.M)
    
    for r = 1:Nrep
        J .= 0
        ep = get_exponential_parameters(com.N, com.M)
        p = MiCRM_stability.Parameters(com.N,com.M,sp,ep)      
            
        MiCRM_stability.jacobian!(p, J)
        λ_mat[i,r] = get_real(eigsolve(J, 1, (:LR))[1][1])    
    end
end

save("./Results/data/nestedness_stabiltiy.jld2", Dict("l" => λ_mat, "n" => n_mat))