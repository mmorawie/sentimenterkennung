require 'torch'
require 'nn'
require 'nngraph'

local GRU = {}

function GRU.create(input_size, output_size)
	local input = nn.Identity()()
	local prev_h = nn.Identity()()
	--	local prev_h1 = nn.Identity()()

	local input_prev_h = nn.JoinTable(1,1)({input, prev_h})
	--	local input_prev_h = nn.JoinTable(1,1)({input_prev_h, prev_h1})
	local gates = nn.Sigmoid()(nn.Linear(input_size + output_size, 2*output_size)(input_prev_h))
	local update_gate, reset_gate = nn.SplitTable(1,2)(nn.Reshape(2, output_size)(gates)):split(2)

	local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
	local input_g = nn.JoinTable(1,1)({input, gated_hidden})
	local hidden_candidate = nn.Tanh()(nn.Linear(input_size + output_size, output_size)(input_g))

	local z = nn.CMulTable()({update_gate, hidden_candidate})
	local z_m = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
	local nextprev_h = nn.CAddTable()({z, z_m})

	return nn.gModule({input, prev_h}, {nextprev_h})
	--	return nn.gModule({input, prev_h, prev_h1}, {nextprev_h, nn.Identity()(nextprev_h)})
end

return GRU
