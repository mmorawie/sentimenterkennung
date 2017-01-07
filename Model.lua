
local Model, parent = torch.class('Model', 'nn.Module')

function Model:__init(config)
  parent.__init(self)
  self.in_dim = config.in_dim
  if self.in_dim == nil then error('input dimension must be specified') end
  self.mem_dim = config.mem_dim or 150
  self.mem_zeros = torch.zeros(self.mem_dim)
  self.train = false

  --parent.__init(self, config)
  self.gate_output = config.gate_output
  if self.gate_output == nil then self.gate_output = true end
  self.output_module_fn = config.output_module_fn
  self.criterion = config.criterion
  
  self.leaf_module = self:new_leaf_module()
  self.leaf_modules = {}
  self.composer = self:new_composer()
  self.composers = {}
  self.output_module = self:new_output_module()
  self.output_modules = {}
end

function Model:evaluate()
  self.train = false
end


function Model:allocate_module(tree, module)
  local modules = module .. 's'
  local num_free = #self[modules]
  if num_free == 0 then
    tree[module] = self['new_' .. module](self)
  else
    tree[module] = self[modules][num_free]
    self[modules][num_free] = nil
  end

  -- necessary for dropout to behave properly
  if self.train then tree[module]:training() else tree[module]:evaluate() end
end

function Model:free_module(tree, module)
  if tree[module] == nil then return end
  table.insert(self[module .. 's'], tree[module])
  tree[module] = nil
end

function Model:new_leaf_module()
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {h})
  if self.leaf_module ~= nil then
    share_params(leaf_module, self.leaf_module)
  end
  return leaf_module
end

function Model:new_composer()
  local composer = GRU4.create(self.mem_dim)
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function Model:new_output_module()
  if self.output_module_fn == nil then return nil end
  local output_module = self.output_module_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function Model:forward(no)--(tree, inputs)
  local lloss, rloss = 0, 0
  if #no.children == 0 then
    self:allocate_module(no, 'leaf_module')
    no.state = no.leaf_module:forward( no.input )
    --no:printout()
  else
    self:allocate_module(no, 'composer')

    local lh, lloss = self:forward(no.children[1])
    local rh, rloss = self:forward(no.children[2])

    -- compute state and output
    no.state = no.composer:forward{lh, rh}
  end

  local loss
  if self.output_module ~= nil then
    self:allocate_module(no, 'output_module')
    --no.output = no.output_module:forward(no.state[2])
    no.output = no.output_module:forward(no.state)
    --if self.train then
      --loss = self.criterion:forward(no.output, tree.gold_label) + lloss + rloss
      loss = self.criterion:forward( no.output, bucket(no.correct) + 1 ) + lloss + rloss
    --end
  end

  return no.state, loss
end

function Model:backward(no, grad)
  local grad_inputs = torch.Tensor(no.size)
  self:_backward(no, grad, grad_inputs)
  return grad_inputs
end

function Model:_backward(no, grad, grad_inputs)
  local output_grad = self.mem_zeros
  if no.output ~= nil and no.correct ~= nil then
    output_grad = no.output_module:backward(no.state, self.criterion:backward(no.output, bucket(no.correct) + 1 ))
  end
  self:free_module(no, 'output_module')

  --print("----------------------------------" .. no.name)
  --print_r(grad)
  

  if #no.children == 0 then
    --no.leaf_module:backward(no.input, {grad[1], grad[2] + output_grad})
    no.leaf_module:backward(no.input, grad + output_grad)
    self:free_module(no, 'leaf_module')
  else
    local lh, rh = self:get_child_states(no)
    
    --local composer_grad = no.composer:backward( {lh, rh}, {grad[1], grad[2] + output_grad})
    local composer_grad = no.composer:backward( {lh, rh}, grad + output_grad)
    self:free_module(no, 'composer')
    --print("composer")
    --print_r(composer_grad)

    -- backward propagate to children
    --self:_backward(no.children[1], {composer_grad[1], composer_grad[2]}, grad_inputs)
    --self:_backward(no.children[2], {composer_grad[3], composer_grad[4]}, grad_inputs)
    self:_backward(no.children[1], composer_grad[1], grad_inputs)
    self:_backward(no.children[2], composer_grad[2], grad_inputs)
  end
  no.state = nil
  no.output = nil
end

function Model:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
  local lp, lg = self.leaf_module:parameters()
  tablex.insertvalues(params, lp)
  tablex.insertvalues(grad_params, lg)
  if self.output_module ~= nil then
    local op, og = self.output_module:parameters()
    tablex.insertvalues(params, op)
    tablex.insertvalues(grad_params, og)
  end
  return params, grad_params
end

function Model:get_child_states(tree)
  local lh, rh
  if tree.children[1] ~= nil then lh = tree.children[1].state end
  if tree.children[2] ~= nil then rh = tree.children[2].state end
  return lh, rh
end

function Model:clean(no)
  no.state = nil
  no.output = nil
  self:free_module(no, 'leaf_module')
  self:free_module(no, 'composer')
  self:free_module(no, 'output_module')
  for i = 1, #no.children do
    self:clean(no.children[i])
  end
end







