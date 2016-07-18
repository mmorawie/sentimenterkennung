require 'torch'
require 'nn'
require 'nngraph'

local GRU1 = {}

function GRU1.create(input_size, output_size)
	input_size = 1
	output_size = 1
	local input1 = nn.Identity()()
	local input2 = nn.Identity()()

	local input_prev_h = nn.JoinTable(1,1)({input1, input2})
	local gates = nn.Sigmoid()(nn.Linear(input_size + output_size, 2*output_size)(input_prev_h))
	local update_gate1, update_gate, reset_gate = nn.SplitTable(1,2)(nn.Reshape(2, output_size)(gates)):split(2)

	--local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
	local input_g = nn.JoinTable(1,1)({input1, input2}) --nn.JoinTable(1,1)({input, gated_hidden})
	local hidden_candidate = nn.Tanh()(nn.Linear(input_size + output_size, output_size)(input_g))


	local z1 = nn.CMulTable()({update_gate1, input1})
	local z_m1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate1)), input2})
	local hidden2 = nn.CAddTable()({z1, z_m1})

	local z = nn.CMulTable()({update_gate, hidden_candidate})
	local z_m = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), hidden2})
	local output = nn.CAddTable()({z, z_m})

	return nn.gModule({input1, input2}, {output})
	--	return nn.gModule({input, prev_h, prev_h1}, {nextprev_h, nn.Identity()(nextprev_h)})
end

return GRU1


