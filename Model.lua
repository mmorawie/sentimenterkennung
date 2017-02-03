
local Model, parent = torch.class('Model', 'nn.Module')

function Model:__init(config)

  parent.__init(self)
  self.train = false

  self.dimension = config.dimension
  self.ft = config.ft
  self.fine_tunning = config.fine_tunning
  self.bucket = config.bucket
  self.binary = (config.classes == 3)
  self.mem = config.mem
  self.classes = config.classes
  self.criterion = nn.ClassNLLCriterion()
  
  self.leaf_module = self:new_leaf_module()
  self.leaf_modules = {}
  self.composer = self:new_composer()
  self.composers = {}
  self.output_module = self:new_output_module()
  self.output_modules = {}
end

function Model:training()
  self.train = true
end

function Model:evaluate()
  self.train = false
end

function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
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
  
  if self.train then tree[module]:training() else tree[module]:evaluate() end
end

function Model:free_module(tree, module)
  if tree[module] == nil then return end
  table.insert(self[module .. 's'], tree[module])
  tree[module] = nil
end

function Model:fine_tune(sst, no)
  local sent = {} 
  for i = 1, no.wsize do
    sent[i] = sst.ivocabulary[no[i].name]
  end
  local inputs = self.ft:forward(torch.IntTensor(sent)) 
  for i = 1, #sent do 
      for j = 1, self.dimension do no[i].input[1][j] = inputs[i][j] end
  end
  no.fine_tuned = true
  no.sentence = sent
end

function Model:new_sentiment_module()
  return Output.create(self.mem, self.classes)
end

function Model:new_leaf_module()
  local leaf_module = Single.create(self.dimension, self.mem)
  if self.leaf_module ~= nil then
    share_params(leaf_module, self.leaf_module)
  end
  return leaf_module
end

function Model:new_composer()
  local composer = GRU4.create(self.mem)
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function Model:new_output_module()
  local output_module = Output.create(self.mem, self.classes)
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function Model:forward(no, sst)
  if self.fine_tunning then self:fine_tune(sst, no) end
  return self:forward2(no[#no])
end

function Model:forward2(no)
  local lloss, rloss = 0, 0
  if #no.children == 0 then
    self:allocate_module(no, 'leaf_module')
    no.state = no.leaf_module:forward( no.input )
  else
    self:allocate_module(no, 'composer')

    local h1, j1 = self:forward2(no.children[1])
    local h2, j2 = self:forward2(no.children[2])
    no.state = no.composer:forward{h1, h2}
  end

  local j
  if self.output_module ~= nil then
    self:allocate_module(no, 'output_module')
    no.output = no.output_module:forward(no.state)
    j = self.criterion:forward( no.output, self.bucket(no.correct) + 1 ) + j1 + j2
  end

  return no.state, j
end

function Model:backward(no, grad)
  local grad_inputs = torch.Tensor(no.size)
  self:_backward(no, grad, grad_inputs)
  if no.fine_tuned then self.ft:backward(torch.IntTensor(no.sentence) , grad_inputs) end
  return grad_inputs
end

function Model:_backward(no, grad, grad_inputs)
  local output_grad = torch.zeros(self.mem)
  if no.output ~= nil and no.correct ~= nil then
    output_grad = no.output_module:backward(no.state, self.criterion:backward(no.output, self.bucket(no.correct) + 1 ))
  end
  self:free_module(no, 'output_module')
  if #no.children == 0 then
    no.leaf_module:backward(no.input, grad + output_grad)
    self:free_module(no, 'leaf_module')
  else
    local lh, rh
    if tree.children[1] ~= nil then lh = tree.children[1].state end
    if tree.children[2] ~= nil then rh = tree.children[2].state end
    
    local composer_grad = no.composer:backward( {lh, rh}, grad + output_grad)
    self:free_module(no, 'composer')
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







