
Node = require 'Node'
require 'torch'

function trainWords(sentindex, network, uc, verbose)
	local words = sentences[sentindex]:split("|")
	local correct = {}
	for i = 1,#words do correct[i] = torch.Tensor({{results[phrases[words[i]]]}}) end 
	local nwords = #words
	local outputs = {}
	local inputs = map(t, words)
	local tot1=0;
	local criterion = nn.MSECriterion()
	for j = 1,nwords do -- Single words phase
		network:forward(inputs[j])
		a = criterion:forward(network.output, correct[j])
		outputs[j] = network.output[1][1]
		tot1 = tot1 + a
		network:zeroGradParameters()
		b = criterion:backward( network.output, correct[j] )
		network:backward(inputs[j], b)
		network:updateParameters(uc) end 
		if verbose == true then print("-->", tot1/nwords) end
	return outputs
end

function testWords(sentindex, network, verbose)
	local words = sentences[sentindex]:split("|")
	local correct = {}; for i = 1,#words do correct[i] = results[phrases[words[i]]] end 
	local total = 0; local totalok = 0; local inputs = map(t, words)
	for j = 1,#words do
		network:forward(inputs[j]); total = total + 1
		if bucket(network.output[1][1]) == bucket(correct[j]) then totalok = totalok + 1 end
		if verbose == true then print("-->", words[j], network.output[1][1], correct[j]) end
	end
	return {total, totalok}
end

function trainII(words, tree, netwrk1, netwrk2, uc1, uc2)
	nwords = #words; ntree = #tree; local correct = getcorrectvalues(words, tree)
	nodes = {}
	local layer = {}
	for j = nwords + 1,ntree do 	nodes[j] = Node:new("" .. j) end --input[j]= {u({0}),u({0})} 
	for j = 1,nwords do 			nodes[j] = Node:newleaf(j, words[j]); nodes[tree[j]]:addchild(nodes[j]) end
	for j = 1, ntree do 			correct[j] = torch.Tensor({{correct[j][1][1]}}); nodes[j].correct = correct[j][1][1] end
	for j = nwords+1,ntree-1 do 	nodes[tree[j]]:addchild(nodes[j]) end
	for j = nwords+1, ntree do 		nodes[j].name = nodes[j].children[1].name .. " " .. nodes[j].children[2].name end
	for j = 1,nwords do 			nodes[j].output = correct[j][1][1] end

	local criterion = nn.MSECriterion()
	
	tot1=0;
	
	for j = nwords + 1,ntree do
		layer[j] = netwrk2:clone('weight', 'bias')
		nodes[j].input = { u({nodes[j].children[1].output}) , u({nodes[j].children[2].output}) }
		nodes[j].output = layer[j]:forward(nodes[j].input)[1][1]
		a = criterion:forward(layer[j].output[1], correct[j][1])
		tot1 = tot1 + a;
		layer[j]:zeroGradParameters() 
	end 
	print(tot1/(ntree - nwords) .. ", ")
	for j = ntree, nwords + 1, -1 do
		b = criterion:backward( layer[j].output[1], correct[j][1] )
		layer[j]:backward(nodes[j].input, b)
		layer[j]:updateParameters(uc2)
		netwrk2:updateParameters(uc2) 
		end
	return {nil, netwrk2}
end




function testII(words, tree, netwrk1, netwrk2, verbose)
	nwords = #words; ntree = #tree; local correct = getcorrectvalues(words, tree)
	nodes = {}
	for j = nwords + 1,ntree do 	nodes[j] = Node:new("" .. j) end 
	for j = 1,nwords do 			nodes[j] = Node:newleaf(j, words[j]); nodes[tree[j]]:addchild(nodes[j]) end
	for j = 1, ntree do 			correct[j] = torch.Tensor({{correct[j][1][1]}}) end
	for j = nwords+1,ntree-1 do 	nodes[tree[j]]:addchild(nodes[j]) end
	for j = nwords+1, ntree do 		nodes[j].name = nodes[j].children[1].name .. " " .. nodes[j].children[2].name end

	for j = 1,nwords do -- Single words phase
		netwrk1:forward(nodes[j].input)
		nodes[j].output = correct[j][1][1]
		nodes[j].correct = correct[j][1][1]
		statistics:words(nodes[j].output, correct[j][1][1])
		end
	
	for j = nwords + 1,ntree do
		input = { u({nodes[j].children[1].output}) , u({nodes[j].children[2].output}) }
		nodes[j].output = netwrk2:forward(input)[1][1]; nodes[j].correct = correct[j][1][1]
		statistics:phrases(nodes[j].output, correct[j][1][1], nodes[j].size)
		if j == ntree then 
			statistics:sentences(nodes[ntree].output, correct[j][1][1]) 
			--print("test ", nodes[j].output, correct[j][1][1], nodes[j].size)
		end
	end 
	--nodes[ntree]:printout()
	
end


function wordtrainer(name, uc, verbose, vverbose)
	local wnet = Double.create(vectorSize, 1, vectorSize - 10, vectorSize - 20)
	for i = 1,11000 do
		if set[i] == "1" then trainWords(i, wnet, uc, vverbose) end
	end
	local total = 0; local totalok = 0;
	for i = 1,11000 do 
		local ok = testWords(i, wnet, total, totalok, vverbose) 
		total = total + ok[1]; totalok = totalok + ok[2]
	end
	torch.save("./networks/" .. name, wnet)
	if verbose == true then print(totalok/total, "saved to " .. name) end
	return wnet
end

function treetrainer(n, name, uc, verbose, vverbose)
	local network1 = Double.create(vectorSize, 1, 50, 40)
	local network2 = GRU1.create()
	for i = 1, n do
		local words = sentences[i]:split("|"); local tree = trees[i]
		if set[i] == "1" then
			networks = trainII(words, tree, network1, network2, 0.15, uc)
			network1 = networks[1]
			network2 = networks[2]
		end
	end
	statistics = require 'Stat'
	statistics:zero()
	for i = 1, 11000 do
		local words = sentences[i]:split("|"); local tree = trees[i]
		if set[i] == "2" then networks = testII(words, tree, network1, network2) end
	end
	torch.save("./networks/" .. name, network2)
	if verbose==true then statistics:printout() end
	return network2
end




function execute(words, tree, netwrk1, netwrk2, verbose)
	nwords = #words; ntree = #tree
	nodes = {}
	for j = nwords + 1,ntree do 	nodes[j] = Node:new("" .. j) end 
	for j = 1,nwords do 			nodes[j] = Node:newleaf(j, words[j]); nodes[tree[j]]:addchild(nodes[j]) end
	for j = nwords+1,ntree-1 do 	nodes[tree[j]]:addchild(nodes[j]) end
	for j = nwords+1, ntree do 		nodes[j].name = nodes[j].children[1].name .. " " .. nodes[j].children[2].name end

	for j = 1,nwords do -- Single words phase
		netwrk1:forward(nodes[j].input)
		nodes[j].output = netwrk1.output[1][1]
	end
	
	for j = nwords + 1,ntree do
		input = { u({nodes[j].children[1].output}) , u({nodes[j].children[2].output}) }
		nodes[j].output = netwrk2:forward(input)[1][1]
	end 
	--nodes[ntree]:printout()
	if verbose then nodes[#tree]:printout() end
	return nodes[#tree].output
end



