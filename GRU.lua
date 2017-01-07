require 'torch'
require 'nn'
require 'nngraph'

local GRU = {}

function GRU.create(input_size)
	local m = input_size
	local input1 = nn.Identity()()
	local input2 = nn.Identity()()
	
	local rgate = nn.Sigmoid()( nn.CAddTable()({ nn.Linear(m, m)(input1), nn.Linear(m, m)(input2) }) )
	local z1gate = nn.Sigmoid()( nn.CAddTable()({ nn.Linear(m, m)(input1), nn.Linear(m, m)(input2) }) )
	local z2gate = nn.Sigmoid()( nn.CAddTable()({ nn.Linear(m, m)(input1), nn.Linear(m, m)(input2) }) )
	
	local one = nn.CMulTable()({rgate, input1})
	local two = nn.CMulTable()({rgate, input2})
	--local one = input1
	--local two = input2
	local hidden_candidate = nn.Tanh()( nn.CAddTable()({ nn.Linear(m, m)(one), nn.Linear(m, m)(two) }) )

	local zgate = nn.CAddTable()({ nn.Linear(m, m)(z1gate), nn.Linear(m, m)(z2gate) })
	local z_m = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(zgate)), hidden_candidate})
	local z_1 = nn.CMulTable()({ z1gate, input1 })
	local z_2 = nn.CMulTable()({ z2gate, input2 })
	local output = nn.CAddTable()({z_m, z_1, z_2})

	return nn.gModule({input1, input2}, {output})
end

return GRU


