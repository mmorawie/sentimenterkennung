require 'torch'
require 'nn'
require 'nngraph'

local Single = {}

function Single.create(inp, out)
    input = nn.Linear(inp, out)()
    tan = nn.Tanh()(input)
    return nn.gModule({input}, {tan})
end

return Single