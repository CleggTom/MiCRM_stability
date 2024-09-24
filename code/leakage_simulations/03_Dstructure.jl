using Pkg 

Pkg.activate(".")

using MiCRM_stability
using LinearAlgebra, KrylovKit
using StatsBase, Random
using Distributions
using JLD2

#switching
function get_exponential_parameters(N,M,σ)
    gx = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), N)
    gs = rand(Uniform(1.0,2.0)) .+ rand(Uniform(-σ,σ), N)
    gw = -rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), N)
    mx = rand(Uniform(0.5,1.0)) .+ rand(Uniform(-σ,σ), N)
    
    fy = ones(N,M)
    hy = ones(N,M)

    λy = zeros(N,M)

    iy = rand(Uniform(σ,1.0)) .+ rand(Uniform(-σ,σ), M)
    oy = rand(Uniform(1.0,2.0)) .+ rand(Uniform(-σ,σ), M)

    return MiCRM_stability.exponential_params(gx,gs,gw,mx,fy,hy,λy,iy,oy)
end

function coherence_D(M,λd,h,T)
    #create skeleton
    TL = fill(-Inf,M)
    TL[1] = 1
    
    D = zeros(M,M)
    p = 0
    for i = 2:M
        ind = 1:(i-1)
    #choose base
        pb = (TL[ind] / maximum(TL[ind])) .^ h
        b = sample(ind, Weights(pb[ind]))
        D[b,i] = 1
        TL[i] = TL[b] + 1
    end
    
    #sort by TL
    TLind = sortperm(TL)
    D = D[TLind, TLind]
    TL = TL[TLind]
    
    #add extra links
    k = rand(Poisson(λd), M)
    x = TL .- TL'
    pl = exp.(-abs.(x .- 1)/ T)
    pl[findall(D .== 1)] .= 0
    pl[diagind(pl)] .= 0
    
    for i = 2:M
        D[sample(1:M, Weights(pl[i,:]), k[i]), i] .= 1
    end
    
    return(D)
end

#create two communties one structured and one shuffled
function random_community(N,M,f,h,T)
    U = rand(N,M) / (N*M)
    [U[i,i] = 1.0 for i = 1:N]
    # U[:,end] .= 0
    
    D = coherence_D(M,Int(0.1 * M),h,T)
    D_shuff = shuffle(D)

    [D_shuff[i,:] .= D_shuff[i,:] ./ sum(D_shuff[i,:]) for i = 1:M]
    D_shuff[isnan.(D_shuff)] .= 0
    
    [D[i,:] .= D[i,:] ./ sum(D[i,:]) for i = 1:M]
    D[isnan.(D)] .= 0
    
    Λ = 0.45 .+ (rand(N) ./ 10)
    
    s = MiCRM_stability.get_structural_params(U,D,Λ)
    s_shuff = MiCRM_stability.get_structural_params(U,D_shuff,Λ)

    e = f(N,M, 0.1)
    u = MiCRM_stability.util_params(N,M,s,e)
    
    p = MiCRM_stability.Parameters(N,M,s,e,u)
    p_shuff = MiCRM_stability.Parameters(N,M,s_shuff,e,u)

    return(p,p_shuff)
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

Np = 1000


stability = zeros(Complex, Np, 2)
p_vec = Vector{Any}(undef,Np)

counter = [0]
Threads.@threads for i = 1:Np
    counter[1] += 1
    print("\r",counter)
    h,T = 1,1
    p_vec[i] = [h,T]
    
    p = random_community(N,M, get_exponential_parameters, h, T)
    J = zeros(N+M, N+M)
    for j = 1:2
        MiCRM_stability.jacobian!(p[j],J)
        stability[i,j] = get_real(eigsolve(J, 1, (:LR))[1][1]) 
    end
end

save("./Results/data/new_sims/D_struct_stabiltiy.jld2", Dict("l" => stability, "p" => p_vec))