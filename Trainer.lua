
local Trainer = torch.class('Trainer')

function Trainer:__init(config, lr)
  Trainer.batch        =  25
  Trainer.reg               = 1e-4

  Trainer.dimension = vectorSize
  Trainer.in_zeros = torch.zeros(Trainer.dimension)
  Trainer.classes = classnumber
  Trainer.mem_dim = 30
  Trainer.optim_state = { learningRate = 0.05 }
  
  Trainer.bTree = BinTree{}
  Trainer.params, Trainer.grad_params = Trainer.bTree:getParameters()
end

function Trainer:train(n)
  Trainer.bTree:training()
  n = n or #trainset
  local indices = torch.randperm(#trainset)
  local zeros = torch.zeros(Trainer.mem_dim)

  for i = 1, n, Trainer.batch do
    local batch = math.min(i + Trainer.batch - 1, #trainset) - i + 1

    local feval = function(x)
      Trainer.grad_params:zero()
      local loss = 0
      for j = 1, batch do
        local idx = indices[i + j - 1]
        local no = Node.setup(sentences[trainset[idx]]:split('|'), trees[trainset[idx]])
        
        local _, tree_loss = Trainer.bTree:forward(no[#no])
        loss = loss + tree_loss
        local input_grad = Trainer.bTree:backward( no[#no], {zeros, zeros})
      end

      loss = loss / batch
      Trainer.grad_params:div(batch)

      
      loss = loss + 0.5 * Trainer.reg * Trainer.params:norm() ^ 2
      Trainer.grad_params:add(Trainer.reg, Trainer.params)
    --print("> ", i, loss)
      return loss, Trainer.grad_params
    end

    optim.adagrad(feval, Trainer.params, Trainer.optim_state)
  end
end
