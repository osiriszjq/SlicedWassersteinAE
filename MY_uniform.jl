
using Flux, Flux.Data.MNIST
using Flux.Tracker
using Flux: onehotbatch, argmax, binarycrossentropy, throttle
using Base.Iterators: repeated, partition
using Flux: @epochs, mse
using Juno: @progress
#using CuArrays

# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

imgs = MNIST.images()

# Partition into batches of size 500
train = [(cat(4, float.(imgs[i])...))
         for i in partition(1:60_000, 500)]
train = gpu.(train)

function generateTheta(L,endim)
    theta = randn(L,endim)
    theta = theta./sqrt.(sum(theta.^2,2))
    return theta
end
function generateZ(batchsize,endim)
    z = 2*(rand(batchsize,endim)-0.5)
    return z
end

#encoder
encoder = Chain(
    Conv((3,3), 1=>16, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 16=>16, x->leakyrelu(x,0.2), pad = (1,1)),
    x -> meanpool(x, (2,2)),
    Conv((3,3), 16=>32, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 32=>32, x->leakyrelu(x,0.2), pad = (1,1)),
    x -> meanpool(x, (2,2)),
    Conv((3,3), 32=>64, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 64=>64, x->leakyrelu(x,0.2), pad = (1,1)),
    x -> meanpool(x, (2,2), pad = (1,1)),
    x -> reshape(x, :, size(x, 4)),
    Dense(1024, 128,relu),
    Dense(128,2)) |> gpu

#decoder
decoder = Chain(
    Dense(2,128),
    Dense(128,1024 ,relu),
    x -> reshape(x, 4,4,64,:),
    x -> repeat(x,inner=(2,2,1,1),outer=(1,1,1,1)),
    Conv((3,3), 64=>64, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 64=>64, x->leakyrelu(x,0.2), pad = (1,1)),
    x -> repeat(x,inner=(2,2,1,1),outer=(1,1,1,1)),
    Conv((3,3), 64=>64, x->leakyrelu(x,0.2)),
    Conv((3,3), 64=>64, x->leakyrelu(x,0.2), pad = (1,1)),
    x -> repeat(x,inner=(2,2,1,1),outer=(1,1,1,1)),
    Conv((3,3), 64=>32, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 32=>32, x->leakyrelu(x,0.2), pad = (1,1)),
    Conv((3,3), 32=>1, σ, pad = (1,1))  ) |> gpu

autoencoder=Chain(encoder,decoder)

function loss(x,theta,z)    
    x_hat = vec(autoencoder(x))
    projae = transpose(encoder(x).tracker.data)*theta'
    projz = z*theta'
    a = sort(projae,1)
    b = sort(projz,1)
    W2 = (a - b).^2
    W2Loss = 10.0*mean(W2)
    x = vec(x)
    crosscost = 1.0*mean(binarycrossentropy.(x_hat,x))
    L1Loss = 1.0*mean(abs.(x_hat.tracker.data-x))
    return crosscost+L1Loss+W2Loss
end
Loss = zeros(1200,1)
for epoch in 1:10
    ind = randperm(length(train))
    for i in 1:length(train)
        X = train[ind[i]]
        theta = generateTheta(50,2)
        z = generateZ(500,2)
        #evalcb = throttle(() -> @show(loss(X,theta,z)), 5)
        opt = RMSProp(params(autoencoder))
        #Flux.train!(x->loss(x,theta,z), zip([X]), opt, cb=evalcb)
        Flux.train!(x->loss(x,theta,z), zip([X]), opt)
        Loss[(epoch-1)*120+i]=loss(X,theta,z).tracker.data
    end
    print(mean(Loss[(epoch-1)*120:epoch*120]))
end



using Plots;pyplot()
plot(Loss,  title = "Loss", xlabel="iterations", ylabel="Loss")
