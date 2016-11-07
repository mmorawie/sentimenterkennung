require 'torch'
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, gate_o)
	local lc, lh = nn.Identity()(), nn.Identity()()
  	local rc, rh = nn.Identity()(), nn.Identity()()
	local new_gate = function()
		return nn.CAddTable(){
		nn.Linear(input_size, input_size)(lh),
		nn.Linear(input_size, input_size)(rh)
	}
  	end

	local i = nn.Sigmoid()(new_gate())
	local lf = nn.Sigmoid()(new_gate())
	local rf = nn.Sigmoid()(new_gate())
	local update = nn.Tanh()(new_gate())
	local c = nn.CAddTable(){
    	nn.CMulTable(){i, update},
      	nn.CMulTable(){lf, lc},
      	nn.CMulTable(){rf, rc}
    }

	local h
	if gate_o then
		local o = nn.Sigmoid()(new_gate())
		h = nn.CMulTable(){o, nn.Tanh()(c)}
	else
		h = nn.Tanh()(c)
	end
	return nn.gModule({lc, lh, rc, rh},{c, h})
end

return LSTM


