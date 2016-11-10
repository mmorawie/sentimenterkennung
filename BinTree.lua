
local BinTree, parent = torch.class('BinTree', 'nn.Module')

function BinTree:__init(config)
  parent.__init(BinTree)
  BinTree.dim = 50
  BinTree.mem = 30
  BinTree.mem_zeros = torch.zeros(BinTree.mem)
  BinTree.train = false
  
  BinTree.gate_output = config.gate_output
  if BinTree.gate_output == nil then BinTree.gate_output = true end
  BinTree.outputmodule = function() 
      return nn.Sequential():add(nn.Dropout()):add(nn.Linear(Trainer.mem, Trainer.classes)):add(nn.LogSoftMax())
  end
  BinTree.criterion = nn.ClassNLLCriterion()
  
  BinTree.leaf = BinTree:leaf()
  BinTree.leafs = {}
  BinTree.node = BinTree:newnode()
  BinTree.nodes = {}
  BinTree.outputmodule = BinTree:newoutput()
  BinTree.outputmodules = {}
end

function BinTree:evaluate()
  BinTree.train = false
end


function BinTree:allocate_module(tree, module)
  local modules = module .. 's'
  local num_free = #BinTree[modules]
  if num_free == 0 then
    tree[module] = BinTree['new_' .. module](BinTree)
  else
    tree[module] = BinTree[modules][num_free]
    BinTree[modules][num_free] = nil
  end

  if BinTree.train then tree[module]:training() else tree[module]:evaluate() end
end

function BinTree:free(tree, module)
  if tree[module] == nil then return end
  table.insert(BinTree[module .. 's'], tree[module])
  tree[module] = nil
end

function BinTree:leaf()
  local input = nn.Identity()()
  local h
  h = nn.Tanh()(nn.Linear(BinTree.dim, BinTree.mem)(input))
  local leaf = nn.gModule({input}, {h})
  if BinTree.leaf ~= nil then share_param(leaf, BinTree.leaf) end
  return leaf
end

function BinTree:newnode()
  local node = GRU4.create(BinTree.mem)
  if BinTree.node ~= nil then
    share_param(node, BinTree.node)
  end
  return node
end

function BinTree:newoutput()
  if BinTree.outputmodule == nil then return nil end
  local outputmodule = BinTree.outputmodule()
  if BinTree.outputmodule ~= nil then
    share_param(outputmodule, BinTree.outputmodule)
  end
  return outputmodule
end

function BinTree:forward(no)
  local lloss, rloss = 0, 0
  if #no.children == 0 then
    BinTree:allocate_module(no, 'leaf')
    no.state = no.leaf:forward( no.input )
    --no:printout()
  else
    BinTree:allocate_module(no, 'node')

    local lvecs, lloss = BinTree:forward(no.children[1])
    local rvecs, rloss = BinTree:forward(no.children[2])
    local lh = BinTree:unpack_state(lvecs)
    local rh = BinTree:unpack_state(rvecs)
    no.state = no.node:forward{lh, rh}
  end

  local loss
  if BinTree.outputmodule ~= nil then
    BinTree:allocate_module(no, 'outputmodule')
    no.output = no.outputmodule:forward(no.state[2])
    --if BinTree.train then
      --loss = BinTree.criterion:forward(no.output, tree.gold_label) + lloss + rloss
      loss = BinTree.criterion:forward( no.output, bucket(no.correct) + 1 ) + lloss + rloss
    --end
  end

  return no.state, loss
end

function BinTree:backward(no, grad)
  local grad_inputs = torch.Tensor(no.size)
  BinTree:_backward(no, grad, grad_inputs)
  return grad_inputs
end

function BinTree:_backward(no, grad, grad_inputs)
  local output_grad = BinTree.mem_zeros
  if no.output ~= nil and no.correct ~= nil then
    output_grad = no.outputmodule:backward(
      no.state[2], BinTree.criterion:backward(no.output, bucket(no.correct) + 1 ))
  end
  BinTree:free(no, 'outputmodule')
  if #no.children == 0 then
    no.leaf:backward(no.input, {grad[1], grad[2] + output_grad})
    BinTree:free(no, 'leaf')
  else
    local lh, rh = BinTree:get_child_states(no)
    local node_grad = no.node:backward( {lh, rh}, {grad[1], grad[2] + output_grad})
    BinTree:free(no, 'node')

    BinTree:_backward(no.children[1], {node_grad[1], node_grad[2]}, grad_inputs)
    BinTree:_backward(no.children[2], {node_grad[3], node_grad[4]}, grad_inputs)
  end
  no.state = nil
  no.output = nil
end

function BinTree:parameters()
  local param, gradParam = {}, {}
  local cp, cg = BinTree.node:parameters()
  tablex.insertvalues(param, cp)
  tablex.insertvalues(gradParam, cg)
  local lp, lg = BinTree.leaf:parameters()
  tablex.insertvalues(param, lp)
  tablex.insertvalues(gradParam, lg)
  if BinTree.outputmodule ~= nil then
    local op, og = BinTree.outputmodule:parameters()
    tablex.insertvalues(param, op)
    tablex.insertvalues(gradParam, og)
  end
  return param, gradParam
end

function BinTree:unpack_state(state)
  local h
  if state == nil then h = BinTree.mem_zeros
  else h = unpack(state) end
  return h
end

function BinTree:get_child_states(tree)
  local lh, rh
  if tree.children[1] ~= nil then
    lh = BinTree:unpack_state(tree.children[1].state)
  end
  if tree.children[2] ~= nil then
    rh = BinTree:unpack_state(tree.children[2].state)
  end
  return lh, rh
end

function BinTree:clean(no)
  no.state = nil
  no.output = nil
  BinTree:free(no, 'leaf')
  BinTree:free(no, 'node')
  BinTree:free(no, 'outputmodule')
  for i = 1, #no.children do
    BinTree:clean(no.children[i])
  end
end







