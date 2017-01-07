
local Trainer = torch.class('Trainer')

function Trainer:__init(mem, lr, regularization)
  self.batch_size = 25
  self.reg  = regularization or 1e-4
  --self.dropout           = (config.dropout == nil) and true or config.dropout

  self.dimension = vectorSize
  self.in_zeros = torch.zeros(self.dimension)
  self.classes = classnumber
  self.mem_dim = mem
  self.optim_state = { learningRate = lr } --learning rate -- optimizer configuration 0.05 main_learning_rate
  
  self.bTree = Model{
    in_dim  = self.dimension,
    mem_dim = mem,
    output_module_fn = function() return self:new_sentiment_module() end,
    criterion = nn.ClassNLLCriterion()
  }
  self.params, self.grad_params = self.bTree:getParameters()
end

function Trainer:new_sentiment_module()
  local sentiment_module = nn.Sequential()
  --if self.dropout then
  sentiment_module:add(nn.Dropout())
  --end
  sentiment_module
    :add(nn.Linear(self.mem_dim, self.classes))
    :add(nn.LogSoftMax())
  return sentiment_module
end

function Trainer:train(n)
  self.bTree:training()
  n = n or #trainset
  local indices = torch.randperm(#trainset)
  local zeros = torch.zeros(self.mem_dim)

  for i = 1, n, self.batch_size do
    local batch_size = math.min(i + self.batch_size - 1, #trainset) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local no = Node.setup(sentences[trainset[idx]]:split('|'), trees[trainset[idx]])
        
        local _, tree_loss = self.bTree:forward(no[#no])
        loss = loss + tree_loss
        local input_grad = self.bTree:backward( no[#no], zeros)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
end
