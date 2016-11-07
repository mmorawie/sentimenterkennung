require 'torch'
require 'nn'
require 'nngraph'

local Double = {}

function Double.create(inp, out, hid1, hid2)
	h1 = nn.Linear(inp, hid1)()
	h2 = nn.Linear(hid2, 5)(nn.Tanh()(nn.Linear(hid1, hid2)(nn.Tanh()(h1))))
	
	--h1 = nn.Linear(inp, hid1)()
	--h2 = nn.Linear(hid1, 5)(nn.Tanh()(h1))
	
	--local h1 = nn.Identity()()
  	--local c = nn.Linear(inp, hid1)(h1)
  	--local h2
	--h2 = nn.Tanh()(c)

	--h1 = nn.Linear(inp, hid1)()
	--h2 = nn.Tanh()(nn.Linear(hid1, hid2)(nn.Tanh()(h1)))
	--h3 = nn.Tanh()(nn.Linear(hid2, hid3)(h2))
	--h4 = nn.Tanh()(nn.Linear(hid3, hid4)(h3))
	--h5 = nn.Tanh()(nn.Linear(hid4, hid5)(h4))
	--h6 = nn.Linear(hid5, out)(h5)
	return nn.gModule({h1}, {h2})--:add(nn.Linear(hid1, 5))
end

return Double
