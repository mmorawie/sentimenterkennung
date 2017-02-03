
local Trainer = torch.class('Trainer')

function Trainer:__init(mem,  classes, lr, regularization, options)
  options = options or {}
  self.algorithm = options.algorithm or optim.adagrad
  self.fine_tunning = true --options.fine_tunning or false
  self.ft_layer = options.glove_T
  self.batch_size = 25
  
  self.dimension = options.vector_size
  self.reg  = regularization
  self.classes =  classes
  self.mem = mem
  self.epsilon = lr
  self.bucket = options.bucket
  

  local ftl = nil
  if self.fine_tunning then 
    ftl = nn.LookupTable(self.ft_layer:size()[1] , options.vector_size)
    ftl.weight:copy( self.ft_layer ) 
  end

  self.Model = Model{
    in_dim  = self.dimension,
    mem_dim = mem,
    classes = self.classes,
    dimension = self.dimension,
    ft = ftl,
    fine_tunning = self.fine_tunning,
    bucket = options.bucket
  }
  self.params, self.grad_params = self.Model:getParameters()

  

end

function Trainer:train(sst)
  self.Model:training()
  local indices = torch.randperm(#sst.trainset)
  
  for i = 1, #sst.trainset, self.batch_size do
    local batch_size = math.min(i + self.batch_size - 1, #sst.trainset) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local no = Node.setup(sst.sentences[sst.trainset[idx]]:split('|'), sst.trees[sst.trainset[idx]], sst)
        local _, tree_loss = self.Model:forward(no, sst)
        loss = loss + tree_loss
        local input_grad = self.Model:backward( no[#no], torch.zeros(self.mem))
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    self.algorithm(feval, self.params, { learningRate = self.epsilon })
    self.Model.ft:updateParameters(self.epsilon)
  end
end
