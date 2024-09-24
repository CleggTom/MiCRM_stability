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
z
function get_overlap_U(com)
    I = [cor(com.U[i,:], com.U[j,:]) for i = 1:com.N, j = 1:com.N]
    return(filter(x -> !isnan(x), I) |> mean)
end

function get_overlap_L(com)
    I = [cor(com.U[i,:], com.L[j,:]) for i = 1:com.N, j = 1:com.N]
    return(filter(x -> !isnan(x), I) |> mean)
end

function get_overlap(com)
    K = com.L .- com.U
    I = [cor(com.U[i,:], K[j,:]) for i = 1:com.N, j = 1:com.N]
    return(filter(x -> !isnan(x), I) |> mean)
end

#simulate

N = 25
M = 50

J = zeros(N+M, N+M)

Nover = 50
Ncom = 500

minO = 1 ./ 0.1
maxO = 1 ./ 10.0

overlap_vec = 1 ./ range(minO, maxO, length = Nover)
overlap = zeros(Nover, Ncom, 3)

psw_mat = zeros(Nover, Ncom)

#allocate jacobian
J = zeros(N+M, N+M)    

k = [0]
for t = 1:(Nover * Ncom)
    i = ((t - 1) .÷ Ncom) + 1
    j = ((t - 1) .% Ncom) + 1
    
    k[1] += 1
    if k[1] % 100 == 0
        print(k, "\r")
    end
        
    Nc = get_n(overlap_vec[i], N)
    Nr = range(0,1,length = M)

    c = rand_com(N, M, 0.2, Nc, Nr)
    # while !check_connected(c)
    #     Nc = get_n(1 / overlap_vec[i], N)
    #     Nr = get_n(1 / overlap_vec[i], M)
        
    #     c = rand_com(N, M, 0.2, Nc, Nr)
    # end
    
     #generate parameters
    sp = MiCRM_stability.get_structural_params(c.U, c.L, c.N, c.M)
    ep = get_exponential_parameters(c.N, c.M)
    p = MiCRM_stability.Parameters(c.N,c.M,sp,ep)

    #calculate stabiltiy
    MiCRM_stability.jacobian!(p, J)
    psw_mat[i,j] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
    overlap[i,j,1] = get_overlap_U(c)
    overlap[i,j,2] = get_overlap_L(c)
    overlap[i,j,3] = get_overlap(c)
end


save("./Results/data/interaction_stabiltiy.jld2", Dict("psw" => psw_mat, "int" => overlap))