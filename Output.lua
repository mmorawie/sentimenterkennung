require 'torch'
require 'nn'
require 'nngraph'

local Output = {}

function Output.create(inp, out)
    input = nn.Dropout()()
    lin = nn.Linear(inp, out)(input)
    lsm = nn.LogSoftMax()(lin)
    return nn.gModule({input}, {lsm})
end

return Output